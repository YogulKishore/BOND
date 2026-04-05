"""
setup.py — One-time setup for BOND persona sim.

Creates two accounts, links them as a couple, runs onboarding
and relationship profiling for both, then saves everything to
config.json so persona_sim.py can just read it and go.

Run once after wiping the DB:
    python setup.py

Then run the sim as many times as you want:
    python persona_sim.py
"""

import asyncio, json, httpx
from pathlib import Path

BASE = "http://localhost:8000"
CONFIG_FILE = "config.json"

# ── Account details ───────────────────────────────────────────────────────────
USERS = {
    "yogul": {
        "name":     "Yogul",
        "email":    "yogul@bond.test",
        "password": "test1234",
    },
    "niranjana": {
        "name":     "Niranjana",
        "email":    "niranjana@bond.test",
        "password": "test1234",
    },
}

# ── Onboarding profiles ───────────────────────────────────────────────────────
PROFILES = {
    "yogul": {
        "communication_style": "withdraw",   # needs time before discussing
        "love_language":       "time",        # quality time
        "conflict_style":      "shutdown",    # goes quiet
        "support_style":       "listen",      # just wants to be heard
        "hope":                "understand why things feel distant sometimes",
    },
    "niranjana": {
        "communication_style": "indirect",   # hints rather than says directly
        "love_language":       "acts",        # acts of service
        "conflict_style":      "deflect",     # uses humor to avoid
        "support_style":       "space",       # needs space to process
        "hope":                "feel less pressure when I'm already overwhelmed",
    },
}

# ── Relationship profile ──────────────────────────────────────────────────────
RELATIONSHIP = {
    "duration":          "growing",       # 6 months to 2 years
    "status":            "stuck",         # feeling stuck in recurring patterns
    "goal":              "understand",    # understand each other better
    "biggest_challenge": "We go quiet when things feel off and it just builds up",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def ok(label, data):
    print(f"  ✓  {label}")
    return data

def fail(label, data):
    print(f"  ✗  {label}: {data}")
    raise SystemExit(1)


async def setup():
    print(f"\n{'='*55}")
    print(f"  BOND Setup")
    print(f"{'='*55}\n")

    async with httpx.AsyncClient(timeout=15) as c:

        # ── 1. Register both users ────────────────────────────────────────────
        print("Creating accounts...")
        tokens = {}
        for key, u in USERS.items():
            r = await c.post(f"{BASE}/auth/signup", json={
                "email":    u["email"],
                "password": u["password"],
                "name":     u["name"],
            })
            if r.status_code != 200:
                fail(f"signup {u['name']}", r.text)
            data = r.json()
            tokens[key] = data["token"]
            ok(f"Registered {u['name']}", None)

        # ── 2. Get Yogul's couple from signup (created automatically) ──────────
        print("\nLinking couple...")
        r = await c.post(f"{BASE}/auth/login-refresh", json={"token": tokens["yogul"]})
        if r.status_code != 200:
            fail("login-refresh yogul", r.text)
        yogul_couples = r.json().get("couples", [])
        if not yogul_couples:
            fail("get couple", "No couple found for Yogul")
        couple_id   = yogul_couples[0]["id"]
        invite_code = yogul_couples[0]["invite_code"]
        ok(f"Couple found (invite: {invite_code})", None)

        # ── 3. Niranjana joins Yogul's couple ─────────────────────────────────
        r = await c.post(f"{BASE}/auth/join", json={
            "token":       tokens["niranjana"],
            "invite_code": invite_code,
        })
        if r.status_code != 200:
            fail("join couple", r.text)
        ok("Niranjana joined the couple", None)

        # ── 4. Onboarding for both ────────────────────────────────────────────
        print("\nSaving profiles...")
        for key in ["yogul", "niranjana"]:
            r = await c.post(f"{BASE}/profile/onboarding", json={
                "token":   tokens[key],
                "answers": PROFILES[key],
            })
            if r.status_code != 200 or r.json().get("error"):
                fail(f"onboarding {key}", r.text)
            ok(f"Profile saved for {USERS[key]['name']}", None)

        # ── 5. Relationship profile ───────────────────────────────────────────
        print("\nSaving relationship profile...")
        r = await c.post(f"{BASE}/profile/relationship", json={
            "token":     tokens["yogul"],
            "couple_id": couple_id,
            "answers":   RELATIONSHIP,
        })
        if r.status_code != 200 or r.json().get("error"):
            fail("relationship profile", r.text)
        ok("Relationship profile saved", None)

        # ── 6. Check-ins for both ─────────────────────────────────────────────
        print("\nSaving check-ins...")
        checkins = {
            "yogul": {
                "mood_score": 2,
                "mood_label": "low",
                "intention":  "figure out why things feel off with her",
                "session_type": "shared",
            },
            "niranjana": {
                "mood_score": 3,
                "mood_label": "neutral",
                "intention":  "just talk through what's been going on",
                "session_type": "shared",
            },
        }
        for key, ci in checkins.items():
            r = await c.post(f"{BASE}/profile/checkin", json={
                "token":    tokens[key],
                "couple_id": couple_id,
                **ci,
            })
            if r.status_code != 200 or r.json().get("error"):
                fail(f"checkin {key}", r.text)
            ok(f"Check-in saved for {USERS[key]['name']}", None)

    # ── 7. Save config ────────────────────────────────────────────────────────
    config = {
        "token_a":   tokens["yogul"],
        "token_b":   tokens["niranjana"],
        "couple_id": couple_id,
    }
    Path(CONFIG_FILE).write_text(json.dumps(config, indent=2))

    print(f"\n{'='*55}")
    print(f"  Setup complete")
    print(f"  Saved to {CONFIG_FILE}")
    print(f"\n  couple_id : {couple_id}")
    print(f"  token_a   : {tokens['yogul'][:30]}...")
    print(f"  token_b   : {tokens['niranjana'][:30]}...")
    print(f"\n  Run the sim:")
    print(f"  python persona_sim.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    asyncio.run(setup())