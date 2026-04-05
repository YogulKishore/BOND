from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db, User, Couple
from auth import signup, login, join_couple, create_new_couple, decode_token
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class JoinRequest(BaseModel):
    token: str
    invite_code: str

class NewCoupleRequest(BaseModel):
    token: str
    label: str

class RefreshRequest(BaseModel):
    token: str

class LeaveCoupleRequest(BaseModel):
    token: str
    couple_id: str

class EditLabelRequest(BaseModel):
    token: str
    couple_id: str
    label: str

@router.post("/signup")
def signup_route(req: SignupRequest, db: Session = Depends(get_db)):
    result = signup(db, req.email, req.password, req.name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/login")
def login_route(req: LoginRequest, db: Session = Depends(get_db)):
    result = login(db, req.email, req.password)
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return result

@router.post("/join")
def join_route(req: JoinRequest, db: Session = Depends(get_db)):
    user_id = decode_token(req.token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    result = join_couple(db, user_id, req.invite_code)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/new-couple")
def new_couple_route(req: NewCoupleRequest, db: Session = Depends(get_db)):
    user_id = decode_token(req.token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    result = create_new_couple(db, user_id, req.label)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/login-refresh")
def refresh_route(req: RefreshRequest, db: Session = Depends(get_db)):
    user_id = decode_token(req.token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    db.expire_all()  # force fresh load from DB
    user = db.query(User).filter(User.id == user_id).first()
    couples = [
        {
            "id": c.id,
            "invite_code": c.invite_code,
            "label": user.name,
            "is_relationship_profiled": c.is_relationship_profiled,
            "partner": next((u.name for u in c.users if u.id != user.id), None)
        }
        for c in user.couples
    ]
    return {"couples": couples}

@router.post("/leave-couple")
def leave_couple_route(req: LeaveCoupleRequest, db: Session = Depends(get_db)):
    user_id = decode_token(req.token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    couple = db.query(Couple).filter(Couple.id == req.couple_id).first()
    if not couple or couple not in user.couples:
        raise HTTPException(status_code=404, detail="Connection not found")
    user.couples.remove(couple)
    if len(couple.users) == 0:
        db.delete(couple)
    db.commit()
    return {"success": True}

@router.put("/couple/label")
def edit_label_route(req: EditLabelRequest, db: Session = Depends(get_db)):
    user_id = decode_token(req.token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    couple = db.query(Couple).filter(Couple.id == req.couple_id).first()
    if not couple:
        raise HTTPException(status_code=404, detail="Not found")
    couple.label = req.label
    db.commit()
    return {"success": True, "label": req.label}