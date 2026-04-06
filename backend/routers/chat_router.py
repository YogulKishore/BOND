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
        thread = None

        if user_id and req.couple_id != 'solo':
            # Resume existing session
            if req.session_id:
                session = db.query(Session).filter(
                    Session.id == req.session_id,
                    Session.is_active == True
                ).first()

            # Fresh session
            if not session:
                session = Session(
                    id=generate_id(),
                    couple_id=req.couple_id,
                    session_type="shared",
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

            # Get prior messages
            all_messages = db.query(Message).filter(
                Message.thread_id == thread_id
            ).order_by(Message.created_at).all()

            # Increment message count
            thread.message_count = (thread.message_count or 0) + 1
            user_msg_count = thread.message_count

            # Save user message
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

        else:
            all_messages = []
            thread_id = None
            user_msg_count = 0

        # ── Investigation state machine ──────────────────────────────────────
        if thread_id:
            inv_state = get_investigation_state(thread_id)
            inv_phase = inv_state.get("phase", "story")

            # Story phase — check if ready to summarise
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
                    # Classify what they said after the summary
                    intent = await classify_post_confirmation_intent(req.message, story_summary_text)
                    if intent == "confirm":
                        # Confirm and start extraction
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
                        # Save and return closer
                        db4 = SessionLocal()
                        try:
                            ai_msg = Message(id=generate_id(), session_id=session.id, thread_id=thread_id, sender_id="ai", content=closer, is_private=True)
                            db4.add(ai_msg)
                            db4.commit()
                        finally:
                            db4.close()
                        return {"response": closer, "session_id": session.id}
                    elif intent == "continuation":
                        # Reset story
                        db3 = SessionLocal()
                        try:
                            t = db3.query(Thread).filter(Thread.id == thread_id).first()
                            if t:
                                t.investigation_phase = "story"
                                t.investigation_brief_json = None
                                t.brief_answered_json = None
                                t.story_summary = None
                                db3.commit()
                        finally:
                            db3.close()
                    elif intent == "meta":
                        meta_reply = "I'm just taking a moment to think about what you've shared. I'll have something for you shortly."
                        db3 = SessionLocal()
                        try:
                            ai_msg = Message(id=generate_id(), session_id=session.id, thread_id=thread_id, sender_id="ai", content=meta_reply, is_private=True)
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
                        prev_user = [m.content for m in all_messages if m.sender_id != "ai"]
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
                            db2 = SessionLocal()
                            try:
                                ai_msg = Message(id=generate_id(), session_id=session.id, thread_id=thread_id, sender_id="ai", content=f"{summary} Is that right?", is_private=True)
                                db2.add(ai_msg)
                                db2.commit()
                            finally:
                                db2.close()
                            asyncio.create_task(generate_investigation_brief(thread_id))
                            return {"response": f"{summary} Is that right?", "session_id": session.id}

            # Extracting complete — trigger depth
            elif inv_phase == "extracting_complete":
                asyncio.create_task(generate_depth_brief(thread_id))

        # ── Generate AI response ─────────────────────────────────────────────
        inv_state = get_investigation_state(thread_id) if thread_id else {"phase": "story"}
        inv_phase = inv_state.get("phase", "story")
        next_intention = inv_state.get("next_intention", "")
        next_key = inv_state.get("next_key", "")
        pacing = inv_state.get("pacing", "normal")
        handle_with_care = inv_state.get("handle_with_care", "")

        response = await get_ai_response(
            session_id=session.id if session else req.session_id,
            couple_id=req.couple_id,
            speaker_name=req.speaker_name,
            message=req.message,
            session_type="shared",
            recent_messages=all_messages,
            user_id=user_id,
            mediation_phase="listening",
            thread_id=thread_id,
            user_msg_count=user_msg_count,
        )

        # Mark intention done if in extraction/depth
        if thread_id and next_key and inv_phase in ("extracting", "depth"):
            if "[SKIP]" in response:
                response = response.replace("[SKIP]", "").strip()
                asyncio.create_task(mark_intention_done(thread_id, next_key, "skipped"))
            else:
                asyncio.create_task(mark_intention_done(thread_id, next_key, "answered"))

        # Check for closing
        if thread_id and inv_phase in ("complete",):
            db_int = SessionLocal()
            try:
                t = db_int.query(Thread).filter(Thread.id == thread_id).first()
                int_count = (t.integration_count or 0) + 1 if t else 0
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

        # Save AI message
        if session and thread_id:
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
    finally:
        db.close()