from fastapi import APIRouter
from pydantic import BaseModel
from models.database import SessionLocal, Message, Session, generate_id
from agents.therapist import get_ai_response
from auth import decode_token

router = APIRouter(prefix="/chat", tags=["chat"])

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

        if user_id and req.couple_id != 'solo':
            # Resuming a specific session from recent sessions
            if req.session_id:
                session = db.query(Session).filter(
                    Session.id == req.session_id,
                    Session.is_active == True
                ).first()

            # No session_id = fresh start from dashboard, always new
            if not session:
                session = Session(
                    id=generate_id(),
                    couple_id=req.couple_id,
                    session_type="individual",
                    initiated_by=user_id,
                    is_active=True
                )
                db.add(session)
                db.commit()

            # Get history BEFORE saving current message — avoids sending it twice
            # (get_ai_response appends the current message explicitly)
            all_messages = db.query(Message).filter(
                Message.session_id == session.id
            ).order_by(Message.created_at).all()

            # Save user message
            user_msg = Message(
                id=generate_id(),
                session_id=session.id,
                sender_id=user_id,
                content=req.message,
                is_private=True
            )
            db.add(user_msg)
            db.commit()

        else:
            all_messages = []

        # Get AI response
        response = await get_ai_response(
            session_id=session.id if session else req.session_id,
            couple_id=req.couple_id,
            speaker_name=req.speaker_name,
            message=req.message,
            session_type=req.session_type,
            recent_messages=all_messages,
            user_id=user_id
        )

        # Save AI message
        if session:
            ai_msg = Message(
                id=generate_id(),
                session_id=session.id,
                sender_id="ai",
                content=response,
                is_private=True
            )
            db.add(ai_msg)
            db.commit()

        return {
            "response": response,
            "session_id": session.id if session else req.session_id
        }
    finally:
        db.close()