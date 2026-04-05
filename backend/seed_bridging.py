"""
seed_bridging.py — seeds a session where both threads have completed investigation,
ready to test the bridging phase.

Run from backend directory:
    python seed_bridging.py --token_a <token_a> --token_b <token_b> --couple_id <couple_id>

You need tokens for BOTH users (dev and sia).
Get from browser: localStorage.getItem('token')
"""

import sys, json, argparse
sys.path.insert(0, '.')

from models.database import SessionLocal, Session, Thread, Message, generate_id
from auth import decode_token

# ── Person A story (the one who waits) ───────────────────────────────────────
STORY_A = [
    ("user", "She read my message but didn't reply for hours. I saw the read receipt and just waited."),
    ("ai",   "What did you do while you were waiting?"),
    ("user", "I kept checking my phone. After a while I just put it down and tried to distract myself."),
    ("ai",   "What happened when she finally replied?"),
    ("user", "It was just a one-word answer. Nothing about the delay, nothing about what I'd said."),
    ("ai",   "What did you say in response?"),
    ("user", "I just replied normally. Didn't bring up the wait. Didn't want to make it a thing."),
    ("ai",   "Got it."),
    # Layer 2 extraction
    ("ai",   "What do you think keeps you from bringing it up?"),
    ("user", "I don't want to seem needy. I figure if it bothers her she'd say something."),
    ("ai",   "What do you think her delayed replies mean?"),
    ("user", "Honestly I think she just doesn't realise how it lands. She's not great at that kind of thing."),
    ("ai",   "What happens after one of these moments — does it get resolved?"),
    ("user", "It just fades. We move on and it never gets talked about. It builds up over time."),
    ("ai",   "What does that build-up cost you?"),
    ("user", "I end up feeling like what I say doesn't really matter to her. Like I'm less of a priority."),
]

SUMMARY_A = "So what's been happening is — she reads your messages but takes hours to reply with minimal responses, and you've been choosing not to bring it up even though it's been building."

BRIEF_A = {
    "emotional_register": "guarded",
    "pacing": "normal",
    "bucket_a_their_side": [
        "Understand what staying neutral protects them from",
        "Understand what they tell themselves when they decide not to bring it up",
    ],
    "bucket_b_their_read": [
        "Find out what they think the delayed replies mean",
        "Understand what they assume about her awareness of the delay",
    ],
    "bucket_c_the_dynamic": [
        "Map what typically happens after these moments",
        "Find out what the silence costs them",
    ]
}

# ── Person B story (the one who delays) ───────────────────────────────────────
STORY_B = [
    ("user", "He messaged me yesterday. I saw it but I was in the middle of something and figured I'd reply later."),
    ("ai",   "What happened after that?"),
    ("user", "I got caught up and forgot about it for a while. When I came back to it I just replied to what he said."),
    ("ai",   "What did you say when you replied?"),
    ("user", "Just a short reply. I was still a bit distracted so I kept it brief."),
    ("ai",   "How did the conversation go after that?"),
    ("user", "Normal I think. He didn't say anything about the wait so I assumed it was fine."),
    ("ai",   "Got it."),
    # Layer 2 extraction
    ("ai",   "What goes through your mind when you see a message and don't reply right away?"),
    ("user", "Honestly not much. I know I'll get to it. I don't think about the timing that much."),
    ("ai",   "Do you think he notices when replies are delayed?"),
    ("user", "I mean maybe, but he never says anything so I assume it's okay. If it bothered him he'd tell me."),
    ("ai",   "What usually happens after these exchanges?"),
    ("user", "Things just carry on. We talk about other stuff. It's not a big deal to me."),
    ("ai",   "What do you think he makes of the short replies?"),
    ("user", "I don't know. I guess I haven't really thought about how it comes across from his side."),
]

SUMMARY_B = "So what's been happening is — you saw his message, got caught up, replied briefly when you remembered, and assumed everything was fine since he didn't say anything."

BRIEF_B = {
    "emotional_register": "analytical",
    "pacing": "normal",
    "bucket_a_their_side": [
        "Understand how much thought they give to reply timing",
        "Understand what they assume silence from him means",
    ],
    "bucket_b_their_read": [
        "Find out what they think he makes of delayed or short replies",
        "Understand whether they've considered how it lands for him",
    ],
    "bucket_c_the_dynamic": [
        "Map what happens between them after these moments",
        "Find out whether they notice any pattern in how things fade",
    ]
}

# ── Completed investigation state ─────────────────────────────────────────────
ALL_ANSWERED = {
    "bucket_a_their_side_0": "answered",
    "bucket_a_their_side_1": "answered",
    "bucket_b_their_read_0": "answered",
    "bucket_b_their_read_1": "answered",
    "bucket_c_the_dynamic_0": "answered",
    "bucket_c_the_dynamic_1": "answered",
    "depth_0": "answered",
    "depth_1": "answered",
    "depth_2": "answered",
    "depth_3": "answered",
}

DEPTH_BRIEF = {
    "depth_intentions": [
        "What this situation costs them emotionally",
        "What they tell themselves about why it keeps happening",
        "What they actually want that they haven't said",
        "What they're afraid would happen if they said it",
    ],
    "handle_with_care": "Avoids conflict — needs space to arrive at things themselves"
}


def seed(token_a: str, token_b: str, couple_id: str) -> dict:
    user_a = decode_token(token_a)
    user_b = decode_token(token_b)
    if not user_a or not user_b:
        print("ERROR: Invalid token")
        sys.exit(1)

    db = SessionLocal()
    try:
        session = Session(
            id=generate_id(),
            couple_id=couple_id,
            session_type="shared",
            initiated_by=user_a,
            is_active=True,
            mediation_phase="listening",
        )
        db.add(session)
        db.flush()

        # Thread A
        thread_a = Thread(
            id=generate_id(),
            session_id=session.id,
            user_id=user_a,
            message_count=len([m for m in STORY_A if m[0] == "user"]),
            story_summary=SUMMARY_A,
            story_confirmed=True,
            investigation_phase="complete",
            investigation_brief_json=json.dumps(BRIEF_A),
            depth_brief_json=json.dumps(DEPTH_BRIEF),
            brief_answered_json=json.dumps(ALL_ANSWERED),
        )
        db.add(thread_a)
        db.flush()

        for sender_type, content in STORY_A:
            db.add(Message(
                id=generate_id(),
                session_id=session.id,
                thread_id=thread_a.id,
                sender_id=user_a if sender_type == "user" else "ai",
                content=content,
                is_private=True,
            ))

        # Thread B
        thread_b = Thread(
            id=generate_id(),
            session_id=session.id,
            user_id=user_b,
            message_count=len([m for m in STORY_B if m[0] == "user"]),
            story_summary=SUMMARY_B,
            story_confirmed=True,
            investigation_phase="complete",
            investigation_brief_json=json.dumps(BRIEF_B),
            depth_brief_json=json.dumps(DEPTH_BRIEF),
            brief_answered_json=json.dumps(ALL_ANSWERED),
        )
        db.add(thread_b)
        db.flush()

        for sender_type, content in STORY_B:
            db.add(Message(
                id=generate_id(),
                session_id=session.id,
                thread_id=thread_b.id,
                sender_id=user_b if sender_type == "user" else "ai",
                content=content,
                is_private=True,
            ))

        db.commit()

        print(f"\n{'='*55}")
        print(f"  Bridging seed complete")
        print(f"{'='*55}")
        print(f"  session_id : {session.id}")
        print(f"  user_a     : {user_a[:8]}... ({len([m for m in STORY_A if m[0]=='user'])} messages)")
        print(f"  user_b     : {user_b[:8]}... ({len([m for m in STORY_B if m[0]=='user'])} messages)")
        print(f"  Both threads: investigation_phase=complete")
        print(f"\n  Test with:")
        print(f"  python test_bridging.py --token_a <token_a> --token_b <token_b> --couple_id {couple_id} --session_id {session.id}")
        print(f"{'='*55}\n")

        return {"session_id": session.id}

    except Exception as e:
        db.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_a",   required=True, help="Token for user A (dev)")
    parser.add_argument("--token_b",   required=True, help="Token for user B (sia)")
    parser.add_argument("--couple_id", required=True)
    args = parser.parse_args()
    seed(args.token_a, args.token_b, args.couple_id)
