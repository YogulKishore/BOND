from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from models.database import User, Couple, generate_id, SessionLocal
from sqlalchemy import text
from config import get_settings
import bcrypt
import random
import string

settings = get_settings()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=7)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        settings.secret_key,
        algorithm="HS256"
    )

def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload.get("sub")
    except JWTError:
        return None

def generate_invite_code() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def format_couples(user, user_id):
    return [
        {
            "id": c.id,
            "invite_code": c.invite_code,
            "label": user.name,
            "is_relationship_profiled": c.is_relationship_profiled,
            "partner": next((u.name for u in c.users if u.id != user_id), None)
        }
        for c in user.couples
    ]

def signup(db: Session, email: str, password: str, name: str) -> dict:
    if db.query(User).filter(User.email == email).first():
        return {"error": "Email already registered"}

    couple = Couple(
        id=generate_id(),
        invite_code=generate_invite_code(),
        label="My relationship"
    )
    db.add(couple)

    user = User(
        id=generate_id(),
        email=email,
        hashed_password=hash_password(password),
        name=name,
    )
    user.couples.append(couple)
    db.add(user)
    db.commit()

    return {
        "token": create_token(user.id),
        "user_id": user.id,
        "name": user.name,
        "is_onboarded": user.is_onboarded,
        "couples": format_couples(user, user.id)
    }

def login(db: Session, email: str, password: str) -> dict:
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return {"error": "Invalid credentials"}

    return {
        "token": create_token(user.id),
        "user_id": user.id,
        "name": user.name,
        "is_onboarded": user.is_onboarded,
        "couples": format_couples(user, user.id)
    }

def join_couple(db: Session, user_id: str, invite_code: str) -> dict:
    couple = db.query(Couple).filter(Couple.invite_code == invite_code).first()
    if not couple:
        return {"error": "Invalid invite code"}

    user = db.query(User).filter(User.id == user_id).first()

    if couple in user.couples:
        return {"error": "Already in this relationship"}

    for c in user.couples[:]:
        if len(c.users) == 1:
            user.couples.remove(c)
            # Delete dependent rows first to avoid FK violations in Postgres
            db.execute(text("DELETE FROM memories WHERE couple_id = :cid"), {"cid": c.id})
            db.execute(text("DELETE FROM user_couple WHERE couple_id = :cid"), {"cid": c.id})
            db.delete(c)

    user.couples.append(couple)
    db.commit()
    db.expire_all()  # force fresh load so partner relationship is visible immediately

    # Re-query to get fresh state
    couple = db.query(Couple).filter(Couple.id == couple.id).first()

    return {
        "success": True,
        "couple": {
            "id": couple.id,
            "invite_code": couple.invite_code,
            "label": user.name,
            "is_relationship_profiled": couple.is_relationship_profiled,
            "partner": next((u.name for u in couple.users if u.id != user_id), None)
        }
    }

def create_new_couple(db: Session, user_id: str, label: str) -> dict:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "User not found"}

    couple = Couple(
        id=generate_id(),
        invite_code=generate_invite_code(),
        label=label
    )
    user.couples.append(couple)
    db.add(couple)
    db.commit()

    return {
        "id": couple.id,
        "invite_code": couple.invite_code,
        "label": couple.label,
        "is_relationship_profiled": False,
        "partner": None
    }