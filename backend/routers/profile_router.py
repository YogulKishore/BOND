from fastapi import APIRouter
from pydantic import BaseModel
from models.database import SessionLocal, Memory, generate_id
from auth import decode_token
import json

router = APIRouter(prefix="/profile", tags=["profile"])

class OnboardingRequest(BaseModel):
    token: str
    answers: dict

class RelationshipRequest(BaseModel):
    token: str
    couple_id: str
    answers: dict

class CheckInRequest(BaseModel):
    token: str
    couple_id: str
    mood_score: int          # 1-5
    mood_label: str          # distressed / low / neutral / okay / good
    intention: str = ""      # optional free text
    session_type: str = ""   # "shared" or "individual"

@router.get("/me")
def get_profile(token: str):
    user_id = decode_token(token)
    if not user_id:
        return {"error": "Invalid token"}
    db = SessionLocal()
    try:
        memory = db.query(Memory).filter(
            Memory.owner_id == user_id,
            Memory.memory_type == "profile"
        ).order_by(Memory.created_at.desc()).first()
        if not memory:
            return {"profile": None}
        # Return parsed JSON if possible, else raw content
        try:
            return {"profile": json.loads(memory.content)}
        except Exception:
            return {"profile": memory.content}
    finally:
        db.close()

@router.post("/onboarding")
def save_onboarding(req: OnboardingRequest):
    user_id = decode_token(req.token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        from models.database import User
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}

        # Store as structured JSON — not flat string
        communication_map = {
            'talk':     'addresses issues directly and promptly',
            'withdraw': 'needs time and space before discussing issues',
            'indirect': 'tends to hint rather than say things directly',
            'suppress': 'keeps feelings internalized',
        }
        love_map = {
            'words': 'words of affirmation',
            'time':  'quality time',
            'acts':  'acts of service',
            'touch': 'physical touch',
        }
        conflict_map = {
            'escalate': 'tends to escalate quickly when upset',
            'shutdown': 'goes quiet and withdraws during conflict',
            'deflect':  'uses humor to avoid difficult moments',
            'resolve':  'pushes to reach resolution quickly',
        }
        support_map = {
            'listen':  'feels most supported when simply listened to without advice',
            'advice':  'appreciates practical advice and solutions',
            'reframe': 'values help seeing situations from a new angle',
            'space':   'needs gentle space to process things alone',
        }

        profile = {
            "communication_style": communication_map.get(
                req.answers.get('communication_style', ''), 
                req.answers.get('communication_style', '')
            ),
            "love_language": love_map.get(
                req.answers.get('love_language', ''),
                req.answers.get('love_language', '')
            ),
            "conflict_style": conflict_map.get(
                req.answers.get('conflict_style', ''),
                req.answers.get('conflict_style', '')
            ),
            "support_style": support_map.get(
                req.answers.get('support_style', ''),
                req.answers.get('support_style', '')
            ),
            "hope": req.answers.get('hope', ''),
        }

        memory = Memory(
            id=generate_id(),
            couple_id=user.couples[0].id if user.couples else 'default',
            owner_id=user_id,
            content=json.dumps(profile),
            memory_type="profile"
        )
        db.add(memory)
        user.is_onboarded = True
        db.commit()
        return {"success": True}
    finally:
        db.close()

@router.post("/relationship")
def save_relationship(req: RelationshipRequest):
    user_id = decode_token(req.token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        status_map = {
            'great':   'things are generally going well',
            'rough':   'going through a rough patch',
            'stuck':   'feeling stuck in recurring patterns',
            'healing': 'recovering from something difficult',
        }
        duration_map = {
            'new':         'less than 6 months',
            'growing':     '6 months to 2 years',
            'established': '2 to 5 years',
            'long':        'more than 5 years',
        }
        goal_map = {
            'communication': 'communicate more openly',
            'conflict':      'handle conflict without escalating',
            'reconnect':     'reconnect and feel closer',
            'understand':    'understand each other better',
        }

        # Store as structured JSON
        profile = {
            "duration": duration_map.get(
                req.answers.get('duration', ''),
                req.answers.get('duration', '')
            ),
            "current_status": status_map.get(
                req.answers.get('status', ''),
                req.answers.get('status', '')
            ),
            "primary_goal": goal_map.get(
                req.answers.get('goal', ''),
                req.answers.get('goal', '')
            ),
            "biggest_challenge": req.answers.get('biggest_challenge', ''),
        }

        memory = Memory(
            id=generate_id(),
            couple_id=req.couple_id,
            owner_id=None,
            content=json.dumps(profile),
            memory_type="relationship_profile"
        )
        db.add(memory)

        from models.database import Couple
        couple = db.query(Couple).filter(Couple.id == req.couple_id).first()
        if couple:
            couple.is_relationship_profiled = True

        db.commit()
        return {"success": True}
    finally:
        db.close()

@router.post("/checkin")
def save_checkin(req: CheckInRequest):
    user_id = decode_token(req.token)
    if not user_id:
        return {"error": "Invalid token"}

    db = SessionLocal()
    try:
        checkin = {
            "mood_score": req.mood_score,       # 1-5
            "mood_label": req.mood_label,       # distressed/low/neutral/okay/good
            "intention": req.intention.strip() if req.intention else "",
            "session_type": req.session_type,
        }

        memory = Memory(
            id=generate_id(),
            couple_id=req.couple_id,
            owner_id=user_id,
            content=json.dumps(checkin),
            memory_type="checkin"
        )
        db.add(memory)
        db.commit()
        return {"success": True}
    finally:
        db.close()