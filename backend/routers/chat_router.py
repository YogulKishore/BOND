from fastapi import APIRouter
from pydantic import BaseModel
from models.database import SessionLocal, Message, Session, Thread, generate_id
from agents.therapist import get_ai_response
from agents.mediation import (
    get_investigation_state,
    generate_story_summary,
    generate_investigation_brief,
    generate_depth_brief,
    mark_intention_done,
    classify_post_confirmation_intent,
    should_offer_close,
    generate_closing_reflection,
    generate_individual_reflection,
    detect_integration_reaction,
)
from auth import decode_token
import asyncio
import random

router = APIRouter(prefix="/chat", tags=["chat"])

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


class MessageRequest(BaseModel):
    message: str
    session_type: str
    speaker_name: str
    couple_id: str
    session_id: str = ""
    token: str = ""
    history: list = []


@router.post("/message")
async def send_message(req: MessageRequest):
    user_id = decode_token(req.token) if req.token else None
    db = SessionLocal()
    try:
        session = None
        thread_id = None
        all_messages = []
        user_msg_count = 0

        if user_id and req.couple_id != "solo":
            # Resume or create session
            if req.session_id:
                session = db.query(Session).filter(
                    Session.id == req.session_id,
                    Session.is_active == True
                ).first()

            if not session:
                session = Session(
                    id=generate_id(),
                    couple_id=req.couple_id,
                    session_type=req.session_type,
                    initiated_by=user_id,
                    is_active=True
                )
                db.add(session)
                db.commit()

            # Find or create thread for this user
            thread = db.query(Thread).filter(
                Thread.session_id == session.id,
                Thread.user_id == user_id
            ).first()
            if not thread:
                thread = Thread(
                    id=generate_id(),
                    session_id=session.id,
                    user_id=user_id,
                    message_count=0,
                )
                db.add(thread)
                db.commit()
                db.refresh(thread)

            thread_id = thread.id

            # History before saving current message
            all_messages = db.query(Message).filter(
                Message.thread_id == thread_id
            ).order_by(Message.created_at).all()

            # Increment message count and save user message
            thread.message_count = (thread.message_count or 0) + 1
            user_msg_count = thread.message_count
            db.commit()

            user_msg = Message(
                id=generate_id(),
                session_id=session.id,
                thread_id=thread_id,
                sender_id=user_id,
                content=req.message,
                is_private=True
            )
            db.add(user_msg)
            db.commit()

    finally:
        db.close()

    # ── Investigation state machine (individual sessions) ────────────────────
    if thread_id and req.session_type == "individual":
        inv_state = get_investigation_state(thread_id)
        inv_phase = inv_state.get("phase", "story")

        # ── Story phase ──────────────────────────────────────────────────────
        if inv_phase == "story":
            db2 = SessionLocal()
            try:
                t = db2.query(Thread).filter(Thread.id == thread_id).first()
                summary_sent = bool(t.story_summary) if t else False
                already_confirmed = bool(t.story_confirmed) if t else False
                story_summary_text = t.story_summary if t else ""
            finally:
                db2.close()

            if summary_sent and not already_confirmed:
                # Waiting for user to confirm or continue
                intent = await classify_post_confirmation_intent(req.message, story_summary_text)
                if intent == "confirm":
                    db3 = SessionLocal()
                    try:
                        t = db3.query(Thread).filter(Thread.id == thread_id).first()
                        if t:
                            t.story_confirmed = True
                            db3.commit()
                    finally:
                        db3.close()
                    closer = random.choice(_STORY_CLOSERS)
                    asyncio.create_task(generate_investigation_brief(thread_id))
                    db4 = SessionLocal()
                    try:
                        ai_msg = Message(
                            id=generate_id(),
                            session_id=session.id,
                            thread_id=thread_id,
                            sender_id="ai",
                            content=closer,
                            is_private=True
                        )
                        db4.add(ai_msg)
                        db4.commit()
                    finally:
                        db4.close()
                    return {"response": closer, "session_id": session.id}

                elif intent == "continuation":
                    # They're still sharing — reset summary, keep listening
                    db3 = SessionLocal()
                    try:
                        t = db3.query(Thread).filter(Thread.id == thread_id).first()
                        if t:
                            t.story_summary = None
                            db3.commit()
                    finally:
                        db3.close()

                elif intent == "meta":
                    meta_reply = "I'm just taking a moment to think about what you've shared. I'll have something for you shortly."
                    db3 = SessionLocal()
                    try:
                        ai_msg = Message(
                            id=generate_id(),
                            session_id=session.id,
                            thread_id=thread_id,
                            sender_id="ai",
                            content=meta_reply,
                            is_private=True
                        )
                        db3.add(ai_msg)
                        db3.commit()
                    finally:
                        db3.close()
                    return {"response": meta_reply, "session_id": session.id}

            elif not summary_sent:
                msg_lower = req.message.lower().strip()
                explicit_done = any(sig in msg_lower for sig in _DONE_SIGNALS)
                count_done = user_msg_count >= 6

                repetition_done = False
                if user_msg_count >= 3:
                    db_rep = SessionLocal()
                    try:
                        prev_msgs = db_rep.query(Message).filter(
                            Message.thread_id == thread_id,
                            Message.sender_id == user_id
                        ).order_by(Message.created_at).all()
                        prev_user = [m.content for m in prev_msgs]
                    finally:
                        db_rep.close()

                    def _sim(a, b):
                        aw, bw = set(a.lower().split()), set(b.lower().split())
                        return len(aw & bw) / min(len(aw), len(bw)) if aw and bw else 0.0

                    for prev in prev_user[-3:]:
                        if _sim(req.message, prev) > 0.65:
                            repetition_done = True
                            break

                if explicit_done or count_done or repetition_done:
                    summary = await generate_story_summary(thread_id)
                    if summary:
                        summary_msg = f"{summary} Is that right?"
                        db2 = SessionLocal()
                        try:
                            ai_msg = Message(
                                id=generate_id(),
                                session_id=session.id,
                                thread_id=thread_id,
                                sender_id="ai",
                                content=summary_msg,
                                is_private=True
                            )
                            db2.add(ai_msg)
                            db2.commit()
                        finally:
                            db2.close()
                        asyncio.create_task(generate_investigation_brief(thread_id))
                        return {"response": summary_msg, "session_id": session.id}

        # ── Extracting complete → kick off depth ────────────────────────────
        elif inv_phase == "extracting_complete":
            asyncio.create_task(generate_depth_brief(thread_id))

        # ── Complete → deliver individual reflection ─────────────────────────
        elif inv_phase == "complete":
            db_ref = SessionLocal()
            try:
                t = db_ref.query(Thread).filter(Thread.id == thread_id).first()
                has_reflection = bool(t.resolution_message) if t else False
            finally:
                db_ref.close()

            if not has_reflection:
                reflection = await generate_individual_reflection(thread_id)
                if reflection:
                    db_ref2 = SessionLocal()
                    try:
                        t = db_ref2.query(Thread).filter(Thread.id == thread_id).first()
                        if t:
                            t.resolution_message = reflection
                            db_ref2.commit()
                        ai_msg = Message(
                            id=generate_id(),
                            session_id=session.id,
                            thread_id=thread_id,
                            sender_id="ai",
                            content=reflection,
                            is_private=True
                        )
                        db_ref2.add(ai_msg)
                        db_ref2.commit()
                    finally:
                        db_ref2.close()
                    return {"response": reflection, "session_id": session.id, "type": "reflection"}

    # ── Map investigation phase → mediation phase for get_ai_response ────────
    inv_state = get_investigation_state(thread_id) if thread_id else {"phase": "story"}
    inv_phase = inv_state.get("phase", "story")
    next_key = inv_state.get("next_key", "")

    if req.session_type == "individual":
        if inv_phase in ("story", "extracting", "extracting_complete", "depth"):
            # All investigation stages use "listening" so SHARED_LISTENING_PROMPT
            # activates the correct EXTRACTING / DEPTH sub-sections
            mediation_phase = "listening"
        elif inv_phase == "complete":
            # Reflection already delivered — now in integration
            db_ph = SessionLocal()
            try:
                t = db_ph.query(Thread).filter(Thread.id == thread_id).first()
                mediation_phase = "integration" if (t and t.resolution_message) else "listening"
            finally:
                db_ph.close()
        else:
            mediation_phase = "listening"
    else:
        # Shared session — phase comes from the session-level mediation arc
        mediation_phase = "listening"

    # ── Generate AI response ─────────────────────────────────────────────────
    response = await get_ai_response(
        session_id=session.id if session else req.session_id,
        couple_id=req.couple_id,
        speaker_name=req.speaker_name,
        message=req.message,
        session_type=req.session_type,          # now correctly "individual" or "shared"
        recent_messages=all_messages,
        user_id=user_id,
        mediation_phase=mediation_phase,
        thread_id=thread_id,
    )

    # ── Mark brief intention done after extracting/depth response ────────────
    if thread_id and next_key and inv_phase in ("extracting", "depth"):
        if "[SKIP]" in response:
            response = response.replace("[SKIP]", "").strip()
            asyncio.create_task(mark_intention_done(thread_id, next_key, "skipped"))
        else:
            asyncio.create_task(mark_intention_done(thread_id, next_key, "answered"))

    # ── Integration closing check ─────────────────────────────────────────────
    if thread_id and mediation_phase == "integration":
        db_int = SessionLocal()
        try:
            t = db_int.query(Thread).filter(Thread.id == thread_id).first()
            int_count = (t.integration_count or 0) + 1 if t else 1
            if t:
                t.integration_count = int_count
                db_int.commit()
            res_msg = t.resolution_message if t else None
        finally:
            db_int.close()

        if res_msg and await should_offer_close(thread_id, int_count):
            closing = await generate_closing_reflection(thread_id, res_msg)
            if closing:
                response = closing

    # ── Save AI message ───────────────────────────────────────────────────────
    if session:
        db5 = SessionLocal()
        try:
            ai_msg = Message(
                id=generate_id(),
                session_id=session.id,
                thread_id=thread_id,
                sender_id="ai",
                content=response,
                is_private=True
            )
            db5.add(ai_msg)
            db5.commit()
        finally:
            db5.close()

    return {
        "response": response,
        "session_id": session.id if session else req.session_id
    }
