"""
session_router.py — WebSocket handler for shared sessions + REST session endpoints.

Structure
─────────
SessionManager          in-process WS registry (connect / disconnect / broadcast)
Background tasks        _update_thread_summary, _send_resolution_to_thread, etc.
WS handler              shared_session_ws — broken into clear named stages:
                          • _deliver_pending_messages   on-connect queued delivery
                          • _handle_consent             bridge_consent payload
                          • _persist_message            save + increment count
                          • _check_phase_change         advance mediation arc
                          • _run_investigation          story/extraction state machine
                          • _post_response_tasks        bg tasks, beat-2, closing
REST endpoints          sessions CRUD
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from models.database import SessionLocal, Message, Session, Thread, generate_id
from agents.therapist import get_ai_response
from agents.mediation import (
    check_bridge_readiness,
    generate_bridge_insight,
    generate_thread_summary,
    extract_core_need,
    check_phase_transition,
    record_bridge_consent,
    generate_resolution_message,
    generate_resolution_beat_2,
    generate_bridge_lead_in,
    should_offer_close,
    generate_closing_reflection,
    analyze_both_threads,
    get_latest_analysis,
    generate_story_summary,
    generate_investigation_brief,
    get_investigation_state,
    mark_intention_done,
    classify_post_confirmation_intent,
)
from auth import decode_token
from datetime import datetime
import json
import asyncio
import random

router = APIRouter()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

async def _save_ai_message(session_id: str, thread_id: str, content: str) -> None:
    """Persist an AI message to the DB."""
    db = SessionLocal()
    try:
        db.add(Message(
            id=generate_id(),
            session_id=session_id,
            thread_id=thread_id,
            sender_id="ai",
            content=content,
            is_private=True,
        ))
        db.commit()
    finally:
        db.close()


async def _send_ws(websocket: WebSocket, payload: dict) -> bool:
    """Send JSON to a WebSocket. Returns False if the socket is dead."""
    try:
        await websocket.send_text(json.dumps(payload))
        return True
    except Exception:
        return False


def _combined_message_count(session_id: str) -> int:
    db = SessionLocal()
    try:
        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        return sum(t.message_count or 0 for t in threads)
    finally:
        db.close()


# ─────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────

class SessionManager:
    def __init__(self):
        # thread_id → WebSocket
        self.threads: dict[str, WebSocket] = {}
        # session_id → set of user_ids currently connected
        self.session_users: dict[str, set] = {}
        # (session_id, user_id) → thread_id
        self.user_thread: dict[tuple, str] = {}

    async def connect(self, thread_id: str, session_id: str, user_id: str, ws: WebSocket):
        await ws.accept()
        self.threads[thread_id] = ws
        self.user_thread[(session_id, user_id)] = thread_id
        self.session_users.setdefault(session_id, set()).add(user_id)
        await self._notify_partners(session_id, actor=user_id, online=True)

    def disconnect(self, thread_id: str, session_id: str, user_id: str):
        self.threads.pop(thread_id, None)
        self.user_thread.pop((session_id, user_id), None)
        if session_id in self.session_users:
            self.session_users[session_id].discard(user_id)

    async def notify_offline(self, session_id: str, left_user_id: str):
        await self._notify_partners(session_id, actor=left_user_id, online=False)

    async def _notify_partners(self, session_id: str, actor: str, online: bool):
        """Tell every OTHER connected user in the session about a status change."""
        for uid in list(self.session_users.get(session_id, set())):
            if uid == actor:
                continue
            tid = self.user_thread.get((session_id, uid))
            if tid:
                await self.send(tid, {"type": "partner_status", "online": online})

    async def send(self, thread_id: str, payload: dict):
        ws = self.threads.get(thread_id)
        if ws:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass

    def partner_online(self, session_id: str, user_id: str) -> bool:
        return any(uid != user_id for uid in self.session_users.get(session_id, set()))

    def get_partner_thread_id(self, session_id: str, user_id: str) -> str | None:
        """Returns the thread_id of the other connected user in this session."""
        for uid in self.session_users.get(session_id, set()):
            if uid != user_id:
                return self.user_thread.get((session_id, uid))
        return None

    def partner_ws(self, session_id: str, user_id: str):
        """Return (partner_thread_id, WebSocket) or (None, None)."""
        for uid in self.session_users.get(session_id, set()):
            if uid != user_id:
                tid = self.user_thread.get((session_id, uid))
                ws = self.threads.get(tid) if tid else None
                return tid, ws
        return None, None


manager = SessionManager()


# ─────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────

async def _update_thread_summary(thread_id: str):
    try:
        summary = await generate_thread_summary(thread_id)
        if not summary:
            return
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if t:
                prefix = "[BRIDGE_SENT] " if (t.thread_summary or "").startswith("[BRIDGE_SENT]") else ""
                t.thread_summary = f"{prefix}{summary}"
                db.commit()
                print(f"[THREAD SUMMARY] updated thread={thread_id[:8]}")
        finally:
            db.close()
    except Exception as e:
        print(f"[THREAD SUMMARY ERROR] {e}")


async def _send_resolution_to_thread(
    session_id: str, thread_id: str, user_id: str, websocket: WebSocket
):
    """Generate and deliver resolution beat-1. Idempotent via DB flag."""
    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        if t and t.resolution_message:
            return  # already delivered
    finally:
        db.close()

    try:
        message = await generate_resolution_message(session_id, user_id)
        if not message:
            return

        await _save_ai_message(session_id, thread_id, message)

        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if t:
                t.resolution_message = message
                t.resolution_pending = message
            db.commit()
        finally:
            db.close()

        sent = await _send_ws(websocket, {"type": "resolution", "role": "ai", "content": message})
        print(f"[RESOLUTION] {'sent' if sent else 'queued'} thread={thread_id[:8]}")
    except Exception as e:
        print(f"[RESOLUTION ERROR] {e}")


async def _send_bridge_lead_in(
    session_id: str, thread_id: str, user_id: str, websocket: WebSocket
):
    """Send the bridge lead-in question. Idempotent via DB flag."""
    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        if t and t.bridge_pending:
            return  # already queued / sent
    finally:
        db.close()

    try:
        lead_in = await generate_bridge_lead_in(session_id, user_id)
        if not lead_in:
            return

        await _save_ai_message(session_id, thread_id, lead_in)

        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if t:
                t.bridge_pending = lead_in
            db.commit()
        finally:
            db.close()

        sent = await _send_ws(websocket, {"type": "bridge", "role": "ai", "content": lead_in})
        print(f"[BRIDGE LEAD-IN] {'sent' if sent else 'queued'} thread={thread_id[:8]}")
    except Exception as e:
        print(f"[BRIDGE LEAD-IN ERROR] {e}")


async def _send_first_extraction_question(
    thread_id: str, session_id: str, couple_id: str,
    user_id: str, user_name: str, websocket: WebSocket,
):
    """After story confirmation — wait for brief then push the first extraction question."""
    for _ in range(30):
        state = get_investigation_state(thread_id)
        if state.get("phase") == "extracting" and state.get("next_intention"):
            break
        await asyncio.sleep(0.5)
    else:
        print(f"[EXTRACTION] brief not ready after 15s thread={thread_id[:8]}")
        return

    intention = state.get("next_intention", "")
    next_key  = state.get("next_key", "")

    db = SessionLocal()
    try:
        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        partner = next((t for t in threads if t.user_id != user_id), None)
        partner_still_going = partner and (
            (partner.investigation_phase or "story") == "story" or not partner.story_confirmed
        )
    finally:
        db.close()

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config import get_settings
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.primary_model, api_key=settings.openai_api_key, temperature=0.6,
        )
        prompt = (
            "You are BOND — a warm relationship support counsellor.\n"
            "You've just confirmed someone's story and are about to go deeper.\n"
            "Write one natural opening question to start exploring this:\n\n"
            f"INTENTION: {intention}\n\n"
            "Rules:\n"
            "- Start with a brief warm transition: \"I want to understand this a bit better.\" or similar\n"
            "- Then ask one specific question that would naturally surface this understanding\n"
            "- Do not reveal the intention or use clinical language\n"
            "- Warm, conversational, 2 sentences max\n"
            "- Never start with \"It sounds like\", \"I can understand\", fragment starters"
        )
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        question = resp.content.strip()
        if not question:
            return

        if next_key:
            asyncio.create_task(mark_intention_done(thread_id, next_key, "answered"))
        if partner_still_going:
            question += "\n\n*(I'm still listening to the other side too — take your time.)*"

        await _save_ai_message(session_id, thread_id, question)
        await _send_ws(websocket, {"type": "message", "role": "ai", "content": question})
        print(f"[EXTRACTION] first question sent thread={thread_id[:8]}")
    except Exception as e:
        print(f"[EXTRACTION FIRST Q ERROR] {e}")


async def _increment_integration_count(thread_id: str) -> int:
    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        if t:
            t.integration_count = (t.integration_count or 0) + 1
            db.commit()
            return t.integration_count
        return 0
    finally:
        db.close()


async def _check_and_send_closing(thread_id: str, integration_count: int, websocket: WebSocket):
    """Offer a closing reflection once the person is winding down. DB-idempotent."""
    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        session_id = t.session_id if t else None
        resolution_msg = t.resolution_message if t else None
    finally:
        db.close()

    if not resolution_msg or not session_id:
        return

    try:
        if not await should_offer_close(thread_id, integration_count):
            return
        closing = await generate_closing_reflection(thread_id, resolution_msg)
        if not closing:
            return
        await _save_ai_message(session_id, thread_id, closing)
        await _send_ws(websocket, {"type": "closing", "role": "ai", "content": closing})
        print(f"[CLOSING] sent thread={thread_id[:8]}")
    except Exception as e:
        print(f"[CLOSING ERROR] {e}")


# ─────────────────────────────────────────────
# WS STAGE HELPERS
# ─────────────────────────────────────────────

async def _deliver_pending_messages(
    thread_id: str, session_id: str, user_id: str,
    mediation_phase: str, websocket: WebSocket,
) -> str:
    """On connect — deliver queued messages and check for missed phase transitions."""
    if mediation_phase == "listening":
        try:
            new_phase = await check_phase_transition(session_id)
            if new_phase == "bridging":
                mediation_phase = "bridging"
                await _send_ws(websocket, {"type": "phase_change", "phase": "bridging"})
                asyncio.create_task(_send_bridge_lead_in(session_id, thread_id, user_id, websocket))
        except Exception as e:
            print(f"[ON-CONNECT PHASE ERROR] {e}")

    try:
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if not t:
                return mediation_phase
            if t.bridge_pending and not t.bridge_consented:
                await _send_ws(websocket, {
                    "type": "bridge", "role": "ai", "content": t.bridge_pending,
                })
                print(f"[ON-CONNECT] delivered pending bridge thread={thread_id[:8]}")
            elif t.resolution_pending and not t.resolution_message:
                t.resolution_message = t.resolution_pending
                t.resolution_pending = None
                db.commit()
                await _send_ws(websocket, {
                    "type": "resolution", "role": "ai", "content": t.resolution_message,
                })
                print(f"[ON-CONNECT] delivered pending resolution thread={thread_id[:8]}")
        finally:
            db.close()
    except Exception as e:
        print(f"[ON-CONNECT DELIVERY ERROR] {e}")

    return mediation_phase


async def _handle_consent(
    consented: bool,
    thread_id: str, session_id: str, user_id: str,
    websocket: WebSocket,
):
    if not consented:
        await _send_ws(websocket, {
            "type": "message",
            "role": "ai",
            "content": "That's okay — we can keep going at your pace. I'm here.",
        })
        return

    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        s = db.query(Session).filter(Session.id == session_id).first()
        if t:
            t.bridge_consented = True
            t.bridge_pending = None
        if s:
            consents = json.loads(s.bridge_consents or "[]")
            if user_id not in consents:
                consents.append(user_id)
            s.bridge_consents = json.dumps(consents)
        db.commit()
    finally:
        db.close()

    asyncio.create_task(_send_resolution_to_thread(session_id, thread_id, user_id, websocket))

    db = SessionLocal()
    try:
        partner_t = db.query(Thread).filter(
            Thread.session_id == session_id,
            Thread.user_id != user_id,
        ).first()
        if partner_t and partner_t.bridge_consented:
            _, partner_ws = manager.partner_ws(session_id, user_id)
            if partner_ws and not partner_t.resolution_message:
                asyncio.create_task(
                    _send_resolution_to_thread(session_id, partner_t.id, partner_t.user_id, partner_ws)
                )
            s = db.query(Session).filter(Session.id == session_id).first()
            if s and s.mediation_phase == "bridging":
                s.mediation_phase = "resolution"
                db.commit()
                print(f"[PHASE] {session_id[:8]} → resolution (both consented)")
    finally:
        db.close()


async def _persist_message(
    session_id: str, thread_id: str, user_id: str, content: str,
) -> tuple[int, list, str | None, str]:
    """Save user message, increment count. Returns (count, prior_msgs, partner_summary, phase)."""
    db = SessionLocal()
    try:
        raw_prior = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()

        class _M:
            __slots__ = ("sender_id", "content", "created_at")
            def __init__(self, m):
                self.sender_id = m.sender_id
                self.content = m.content
                self.created_at = m.created_at

        prior_messages = [_M(m) for m in raw_prior]

        db.add(Message(
            id=generate_id(), session_id=session_id, thread_id=thread_id,
            sender_id=user_id, content=content, is_private=True,
        ))
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        t.message_count = (t.message_count or 0) + 1
        user_msg_count = t.message_count
        db.commit()

        partner_t = db.query(Thread).filter(
            Thread.session_id == session_id, Thread.user_id != user_id,
        ).first()
        partner_summary = partner_t.thread_summary if partner_t else None

        s = db.query(Session).filter(Session.id == session_id).first()
        phase = s.mediation_phase or "listening" if s else "listening"
    finally:
        db.close()

    return user_msg_count, prior_messages, partner_summary, phase


async def _check_phase_change(
    session_id: str, thread_id: str, user_id: str,
    current_phase: str, websocket: WebSocket,
) -> tuple[str, bool]:
    """Check for phase transition and broadcast. Returns (new_phase, changed_to_bridging)."""
    if current_phase in ("resolution", "integration"):
        return current_phase, False

    new_phase = await check_phase_transition(session_id)
    if new_phase == current_phase:
        return current_phase, False

    # Collect session thread IDs in one query then broadcast
    db = SessionLocal()
    try:
        session_thread_ids = {
            t.id for t in db.query(Thread).filter(Thread.session_id == session_id).all()
        }
    finally:
        db.close()

    phase_msg = json.dumps({"type": "phase_change", "phase": new_phase})
    for tid, ws in list(manager.threads.items()):
        if tid in session_thread_ids:
            try:
                await ws.send_text(phase_msg)
            except Exception:
                pass

    if new_phase == "bridging":
        async def _bridge_sequence():
            if not get_latest_analysis(session_id):
                await analyze_both_threads(session_id)
            db = SessionLocal()
            try:
                for t in db.query(Thread).filter(Thread.session_id == session_id).all():
                    ws = manager.threads.get(manager.user_thread.get((session_id, t.user_id)))
                    if ws:
                        await _send_bridge_lead_in(session_id, t.id, t.user_id, ws)
            finally:
                db.close()
        asyncio.create_task(_bridge_sequence())
        return new_phase, True

    if new_phase == "resolution":
        asyncio.create_task(
            _send_resolution_to_thread(session_id, thread_id, user_id, websocket)
        )

    return new_phase, False


_STORY_CLOSERS = [
    "That helps me understand what's been going on.",
    "Thank you for sharing that with me.",
    "I hear you — that gives me a clearer picture.",
    "That makes sense. Give me a moment.",
    "Got it — I appreciate you sharing that.",
]

_DONE_SIGNALS = {
    "that's it", "thats it", "that's all", "thats all",
    "nothing else", "that's everything", "thats everything",
    "that's about it", "thats about it", "i think that's it",
    "i guess that's it", "yeah that's it", "that's basically it",
    "thats basically it", "and that's it", "and thats it",
}


async def _run_investigation(
    session_id: str, thread_id: str, couple_id: str,
    user_id: str, user_name: str,
    user_message: str, user_msg_count: int,
    prior_messages: list, websocket: WebSocket,
) -> bool:
    """Story/extraction state machine. Returns True if turn was handled."""
    inv_state = get_investigation_state(thread_id)
    inv_phase = inv_state.get("phase", "story")

    async def _send_direct(text: str):
        await _save_ai_message(session_id, thread_id, text)
        await _send_ws(websocket, {"type": "message", "role": "ai", "content": text})

    async def _confirm_story():
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if t:
                t.story_confirmed = True
                db.commit()
        finally:
            db.close()
        await _send_direct(random.choice(_STORY_CLOSERS))
        print(f"[STORY] confirmed thread={thread_id[:8]}")
        asyncio.create_task(
            _send_first_extraction_question(
                thread_id, session_id, couple_id, user_id, user_name, websocket,
            )
        )

    async def _reset_story(keep_summary: bool = False):
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            if t:
                t.investigation_phase = "story"
                t.investigation_brief_json = None
                t.brief_answered_json = None
                if not keep_summary:
                    t.story_summary = None
                db.commit()
        finally:
            db.close()

    async def _resummary() -> bool:
        await _reset_story(keep_summary=False)
        new_summary = await generate_story_summary(thread_id)
        if new_summary:
            await _send_direct(f"{new_summary} Is that right?")
            asyncio.create_task(generate_investigation_brief(thread_id))
            print(f"[STORY] re-summarised thread={thread_id[:8]}")
            return True
        return False

    async def _classify_and_dispatch(story_summary: str) -> bool:
        intent = await classify_post_confirmation_intent(user_message, story_summary)
        print(f"[STORY] post-confirm intent={intent} phase={inv_phase} thread={thread_id[:8]}")
        if intent == "confirm":
            await _confirm_story()
            return True
        elif intent == "continuation":
            await _reset_story()
            print(f"[STORY] continuation thread={thread_id[:8]}")
            return False
        elif intent == "meta":
            await _send_direct(
                "I'm just taking a moment to think about what you've shared. "
                "I'll have something for you shortly."
            )
            return True
        else:
            return await _resummary()

    # ── Story phase ──────────────────────────────────────────────────────────
    if inv_phase == "story":
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            summary_sent = bool(t.story_summary) if t else False
            already_confirmed = bool(t.story_confirmed) if t else False
            story_summary_text = t.story_summary if t else ""
        finally:
            db.close()

        if summary_sent and not already_confirmed:
            return await _classify_and_dispatch(story_summary_text)

        if not summary_sent:
            msg_lower = user_message.lower().strip()
            explicit_done = any(sig in msg_lower for sig in _DONE_SIGNALS)
            count_done = user_msg_count >= 6

            repetition_done = False
            if user_msg_count >= 3 and not explicit_done and not count_done:
                prev_user = [m.content for m in prior_messages if m.sender_id != "ai"]
                def _sim(a: str, b: str) -> float:
                    aw, bw = set(a.lower().split()), set(b.lower().split())
                    return len(aw & bw) / min(len(aw), len(bw)) if aw and bw else 0.0
                for prev in prev_user[-3:]:
                    if _sim(user_message, prev) > 0.65:
                        repetition_done = True
                        print(f"[STORY] repetition msg={user_msg_count} thread={thread_id[:8]}")
                        break

            if explicit_done or count_done or repetition_done:
                reason = "explicit" if explicit_done else ("count" if count_done else "repetition")
                print(f"[STORY] trigger={reason} msg={user_msg_count} thread={thread_id[:8]}")
                summary = await generate_story_summary(thread_id)
                if summary:
                    await _send_direct(f"{summary} Is that right?")
                    asyncio.create_task(generate_investigation_brief(thread_id))
                    return True

    # ── Extracting phase ─────────────────────────────────────────────────────
    elif inv_phase == "extracting":
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            already_confirmed = bool(t.story_confirmed) if t else False
            has_summary = bool(t.story_summary) if t else False
            story_summary_text = t.story_summary if t else ""
            if not has_summary:
                last_ai = db.query(Message).filter(
                    Message.thread_id == thread_id, Message.sender_id == "ai",
                ).order_by(Message.created_at.desc()).first()
                has_summary = bool(last_ai and "is that right" in last_ai.content.lower())
        finally:
            db.close()

        if has_summary and not already_confirmed:
            return await _classify_and_dispatch(story_summary_text)

    return False


async def _post_response_tasks(
    session_id: str, thread_id: str, user_id: str,
    user_message: str, user_msg_count: int,
    mediation_phase: str, websocket: WebSocket,
) -> str:
    """
    Background work after every AI response.
    Returns the (possibly updated) mediation_phase so the outer loop stays current.
    """
    if user_msg_count % 3 == 0:
        asyncio.create_task(_update_thread_summary(thread_id))

    combined = _combined_message_count(session_id)
    if combined >= 10 and mediation_phase == "listening":
        asyncio.create_task(analyze_both_threads(session_id))

    # Resolution beat-2 — uses DB flag so it fires exactly once per thread
    if mediation_phase == "resolution":
        db = SessionLocal()
        try:
            t = db.query(Thread).filter(Thread.id == thread_id).first()
            beat_1 = t.resolution_message if t else None
            beat2_sent = bool(t.resolution_beat2_sent) if t else True
        finally:
            db.close()

        if beat_1 and not beat2_sent:
            beat_2 = await generate_resolution_beat_2(
                session_id=session_id, user_id=user_id,
                beat_1=beat_1, reaction=user_message,
            )
            db = SessionLocal()
            try:
                t = db.query(Thread).filter(Thread.id == thread_id).first()
                if t:
                    t.resolution_beat2_sent = True
                s = db.query(Session).filter(Session.id == session_id).first()
                if s and s.mediation_phase == "resolution":
                    s.mediation_phase = "integration"
                    print(f"[PHASE] {session_id[:8]} → integration")
                if beat_2:
                    db.add(Message(
                        id=generate_id(), session_id=session_id, thread_id=thread_id,
                        sender_id="ai", content=beat_2, is_private=True,
                    ))
                db.commit()
            finally:
                db.close()

            mediation_phase = "integration"
            if beat_2:
                await _send_ws(websocket, {"type": "resolution", "role": "ai", "content": beat_2})
            await _send_ws(websocket, {"type": "phase_change", "phase": "integration"})

        elif beat2_sent:
            mediation_phase = "integration"

    # Closing check
    db = SessionLocal()
    try:
        t = db.query(Thread).filter(Thread.id == thread_id).first()
        thread_has_resolution = bool(t and t.resolution_message)
    finally:
        db.close()

    if mediation_phase == "integration" or (mediation_phase == "bridging" and thread_has_resolution):
        integration_count = await _increment_integration_count(thread_id)
        asyncio.create_task(_check_and_send_closing(thread_id, integration_count, websocket))

    return mediation_phase


# ─────────────────────────────────────────────
# SHARED SESSION WEBSOCKET
# ─────────────────────────────────────────────

@router.websocket("/ws/shared/{session_id}")
async def shared_session_ws(websocket: WebSocket, session_id: str, token: str):
    # ── Auth & setup ─────────────────────────────────────────────────────────
    user_id = decode_token(token)
    if not user_id:
        await websocket.accept()
        await websocket.close(code=4001)
        return

    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            await websocket.accept()
            await websocket.close(code=4004)
            return

        thread = db.query(Thread).filter(
            Thread.session_id == session_id, Thread.user_id == user_id,
        ).first()
        if not thread:
            thread = Thread(
                id=generate_id(), session_id=session_id,
                user_id=user_id, message_count=0,
            )
            db.add(thread)
            db.commit()
            db.refresh(thread)

        thread_id       = thread.id
        couple_id       = session.couple_id
        mediation_phase = session.mediation_phase or "listening"

        from models.database import User
        user_obj = db.query(User).filter(User.id == user_id).first()
        user_name = user_obj.name if user_obj else "User"
    finally:
        db.close()

    await manager.connect(thread_id, session_id, user_id, websocket)

    # ── Send history ─────────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        history = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()

        if not history:
            _openers = [
                "Hey — what's been going on?",
                "Good to have you here. What happened?",
                "Hey — tell me what's going on.",
                "What's been on your mind?",
                "Hey — what's going on with you?",
            ]
            opening_msg = Message(
                id=generate_id(), session_id=session_id, thread_id=thread_id,
                sender_id="ai", content=random.choice(_openers), is_private=True,
            )
            db.add(opening_msg)
            db.commit()
            history = [opening_msg]

        await websocket.send_text(json.dumps({
            "type": "history",
            "messages": [
                {
                    "role": "ai" if m.sender_id == "ai" else "user",
                    "sender": "BOND" if m.sender_id == "ai" else user_name,
                    "content": m.content,
                }
                for m in history
            ],
            "mediation_phase": mediation_phase,
        }))
        await websocket.send_text(json.dumps({
            "type": "partner_status",
            "online": manager.partner_online(session_id, user_id),
        }))
        print(
            f"[WS] {user_name} connected | session={session_id[:8]} "
            f"thread={thread_id[:8]} phase={mediation_phase}"
        )
    finally:
        db.close()

    # ── On-connect pending delivery ───────────────────────────────────────────
    mediation_phase = await _deliver_pending_messages(
        thread_id, session_id, user_id, mediation_phase, websocket,
    )

    # ── Message loop ──────────────────────────────────────────────────────────
    try:
        while True:
            data = json.loads(await websocket.receive_text())

            if data.get("type") == "bridge_consent":
                await _handle_consent(
                    data.get("consent", False),
                    thread_id, session_id, user_id, websocket,
                )
                continue

            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            user_msg_count, prior_messages, partner_summary, mediation_phase = (
                await _persist_message(session_id, thread_id, user_id, user_message)
            )

            await _send_ws(websocket, {"type": "typing"})

            mediation_phase, phase_to_bridging = await _check_phase_change(
                session_id, thread_id, user_id, mediation_phase, websocket,
            )
            if phase_to_bridging:
                continue

            investigation_handled = False
            if mediation_phase == "listening":
                investigation_handled = await _run_investigation(
                    session_id, thread_id, couple_id,
                    user_id, user_name,
                    user_message, user_msg_count,
                    prior_messages, websocket,
                )

            if investigation_handled:
                if user_msg_count % 3 == 0:
                    asyncio.create_task(_update_thread_summary(thread_id))
                combined = _combined_message_count(session_id)
                if combined >= 10 and mediation_phase == "listening":
                    asyncio.create_task(analyze_both_threads(session_id))
                continue

            ai_response = await get_ai_response(
                session_id=session_id,
                couple_id=couple_id,
                speaker_name=user_name,
                message=user_message,
                session_type="shared",
                recent_messages=prior_messages,
                user_id=user_id,
                partner_summary=partner_summary,
                mediation_phase=mediation_phase,
                thread_id=thread_id,
                user_msg_count=user_msg_count,
            )

            await _save_ai_message(session_id, thread_id, ai_response)
            await _send_ws(websocket, {"type": "message", "role": "ai", "content": ai_response})

            # _post_response_tasks returns updated phase — reassign so next
            # iteration of the loop sees the correct phase without a DB re-read
            mediation_phase = await _post_response_tasks(
                session_id, thread_id, user_id,
                user_message, user_msg_count,
                mediation_phase, websocket,
            )

    except WebSocketDisconnect:
        print(f"[WS] {user_name} disconnected from session {session_id[:8]}")
    except Exception as e:
        if "WebSocket is not connected" not in str(e) and "close message" not in str(e):
            print(f"[WS ERROR] {e}")
    finally:
        manager.disconnect(thread_id, session_id, user_id)
        await manager.notify_offline(session_id, user_id)


# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────

@router.get("/sessions/{couple_id}")
def get_sessions(couple_id: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        sessions = db.query(Session).filter(
            Session.couple_id == couple_id
        ).order_by(Session.created_at.desc()).limit(30).all()

        result = []
        for s in sessions:
            if s.session_type == "individual":
                if s.initiated_by != user_id:
                    continue
                last_msg = db.query(Message).filter(
                    Message.session_id == s.id
                ).order_by(Message.created_at.desc()).first()
            elif s.session_type == "shared":
                thread = db.query(Thread).filter(
                    Thread.session_id == s.id, Thread.user_id == user_id,
                ).first()
                if not thread:
                    # No thread yet — only include if active and partner started it
                    # so Niranjana can see Yogul's session as a notification
                    if s.is_active and s.initiated_by != user_id:
                        result.append({
                            "id": s.id,
                            "session_type": s.session_type,
                            "created_at": s.created_at.isoformat(),
                            "last_message": None,
                            "is_active": True,
                            "mediation_phase": s.mediation_phase,
                            "initiated_by": s.initiated_by,
                        })
                    continue
                last_msg = db.query(Message).filter(
                    Message.thread_id == thread.id
                ).order_by(Message.created_at.desc()).first()
            else:
                last_msg = db.query(Message).filter(
                    Message.session_id == s.id
                ).order_by(Message.created_at.desc()).first()

            result.append({
                "id": s.id,
                "session_type": s.session_type,
                "created_at": s.created_at.isoformat(),
                "last_message": (
                    last_msg.content[:60] + "..."
                    if last_msg and len(last_msg.content) > 60
                    else last_msg.content if last_msg else None
                ),
                "is_active": s.is_active,
                "mediation_phase": s.mediation_phase,
                "initiated_by": s.initiated_by,
            })

        return {"sessions": result[:10]}
    finally:
        db.close()


@router.post("/sessions/create")
def create_session(couple_id: str, session_type: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        existing = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.session_type == session_type,
            Session.is_active == True,
        ).first()
        if existing:
            return {"session_id": existing.id, "mediation_phase": existing.mediation_phase or "listening"}

        past_count = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.session_type == session_type,
            Session.is_active == False,
        ).count()

        session = Session(
            id=generate_id(),
            couple_id=couple_id,
            session_type=session_type,
            initiated_by=user_id,
            is_active=True,
            mediation_phase="listening",
            session_number=past_count + 1,
        )
        db.add(session)
        db.commit()
        return {"session_id": session.id, "mediation_phase": "listening"}
    finally:
        db.close()


@router.post("/sessions/{session_id}/end")
async def end_session(session_id: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session:
            session.is_active = False
            session.ended_at = datetime.utcnow()
            db.commit()

        from memory.patterns_out import generate_session_summary, update_pattern_memory, update_couple_pattern
        from models.database import Couple

        _couple_id    = session.couple_id if session else None
        _session_type = session.session_type if session else None
        _user_ids: list[str] = []
        if session:
            couple = db.query(Couple).filter(Couple.id == _couple_id).first()
            if couple:
                _user_ids = [u.id for u in couple.users]
        if user_id not in _user_ids:
            _user_ids.append(user_id)
    finally:
        db.close()

    async def _save_summary_and_patterns():
        try:
            result = await generate_session_summary(session_id, _session_type)
            print(f"[SUMMARY] session={session_id[:8]} depth={result.get('depth') if result else 'none'}")
            if result:
                db2 = SessionLocal()
                try:
                    s = db2.query(Session).filter(Session.id == session_id).first()
                    if s:
                        s.summary = result["readable"]
                        if result["json_data"]:
                            s.summary_json = json.dumps(result["json_data"])
                        db2.commit()
                finally:
                    db2.close()

                if _couple_id:
                    for uid in _user_ids:
                        await update_pattern_memory(_couple_id, uid)
                    if _session_type == "shared":
                        await update_couple_pattern(_couple_id)

            try:
                from agents.rag import embed_session
                db3 = SessionLocal()
                try:
                    s3 = db3.query(Session).filter(Session.id == session_id).first()
                    session_num = s3.session_number if s3 else 1
                    ended_at    = s3.ended_at or datetime.utcnow()
                finally:
                    db3.close()
                await embed_session(
                    session_id=session_id,
                    session_type=_session_type,
                    couple_id=_couple_id,
                    user_ids=_user_ids,
                    session_number=session_num,
                    ended_at=ended_at,
                )
            except Exception as rag_err:
                print(f"[RAG] embedding failed (non-fatal): {rag_err}")
        except Exception as e:
            import traceback
            print(f"[SUMMARY ERROR] {e}")
            traceback.print_exc()

    asyncio.create_task(_save_summary_and_patterns())
    return {"success": True}


@router.get("/sessions/history/{session_id}")
def get_session_history(session_id: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session and session.session_type == "shared":
            thread = db.query(Thread).filter(
                Thread.session_id == session_id, Thread.user_id == user_id,
            ).first()
            if not thread:
                return {"messages": []}
            messages = db.query(Message).filter(
                Message.thread_id == thread.id
            ).order_by(Message.created_at).all()
        else:
            messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at).all()

        return {
            "messages": [
                {
                    "sender_id": m.sender_id,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ]
        }
    finally:
        db.close()


@router.get("/sessions/{session_id}/end")
async def end_session_beacon(session_id: str, token: str):
    """GET version — for sendBeacon compatibility."""
    return await end_session(session_id, token)