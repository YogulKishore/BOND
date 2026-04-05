from fastapi import APIRouter
from pydantic import BaseModel
from models.database import SessionLocal, Message, Session, User, generate_id
from auth import decode_token

router = APIRouter(prefix="/async", tags=["async"])

class AsyncMessageRequest(BaseModel):
    token: str
    couple_id: str
    content: str


def get_or_create_async_session(db, couple_id: str, user_id: str) -> str:
    # Always reuse the most recent async session for this couple — active or not
    existing = db.query(Session).filter(
        Session.couple_id == couple_id,
        Session.session_type == "async",
    ).order_by(Session.created_at.desc()).first()

    if existing:
        # Reactivate if it was closed
        if not existing.is_active:
            existing.is_active = True
            db.commit()
        return existing.id

    session = Session(
        id=generate_id(),
        couple_id=couple_id,
        session_type="async",
        initiated_by=user_id,
        is_active=True
    )
    db.add(session)
    db.commit()
    return session.id


@router.post("/send")
def send_async_message(req: AsyncMessageRequest):
    user_id = decode_token(req.token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        session_id = get_or_create_async_session(db, req.couple_id, user_id)

        msg = Message(
            id=generate_id(),
            session_id=session_id,
            sender_id=user_id,
            content=req.content,
            is_private=False
        )
        db.add(msg)
        db.commit()

        return {
            "message": {
                "id": msg.id,
                "sender_id": user_id,
                "sender_name": user.name,
                "content": req.content,
                "created_at": msg.created_at.isoformat()
            }
        }
    finally:
        db.close()


@router.get("/messages/{couple_id}")
def get_async_messages(couple_id: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        # Get most recent async session regardless of active status
        session = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.session_type == "async",
        ).order_by(Session.created_at.desc()).first()

        if not session:
            return {"messages": [], "has_unread": False}

        messages = db.query(Message).filter(
            Message.session_id == session.id
        ).order_by(Message.created_at).all()

        has_unread = any(m.sender_id != user_id for m in messages)

        result = []
        for m in messages:
            sender = db.query(User).filter(User.id == m.sender_id).first()
            result.append({
                "id": m.id,
                "sender_id": m.sender_id,
                "sender_name": sender.name if sender else "Unknown",
                "content": m.content,
                "created_at": m.created_at.isoformat()
            })

        return {"messages": result, "has_unread": has_unread}
    finally:
        db.close()


@router.get("/summary/{couple_id}")
async def get_async_summary(couple_id: str, token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        session = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.session_type == "async",
        ).order_by(Session.created_at.desc()).first()

        if not session:
            return {"summary": None}

        messages = db.query(Message).filter(
            Message.session_id == session.id,
            Message.sender_id != user_id
        ).order_by(Message.created_at).all()

        if not messages:
            return {"summary": None}

        combined = "\n".join([m.content for m in messages])
        prompt = f"Your partner left you these messages:\n\n{combined}\n\nSummarize what they're trying to say in 2-3 warm, empathetic sentences. Don't take sides."

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            from config import get_settings as _get_settings
            _settings = _get_settings()
            llm = ChatOpenAI(
                model=_settings.primary_model,
                api_key=_settings.openai_api_key,
                temperature=0.5
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            summary = resp.content.strip()
        except Exception as e:
            print(f"[ASYNC SUMMARY ERROR] {e}")
            summary = None
        return {"summary": summary}
    finally:
        db.close()