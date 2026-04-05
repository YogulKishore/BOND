"""
persona_sim.py — Simulate a full BOND shared session with two local LLM personas.

Architecture reality:
  - Shared sessions are WebSocket ONLY — no HTTP endpoint exists for them
  - Both users must be connected simultaneously (BOND tracks partner presence)
  - Each user has a private thread — messages stay separate until bridging
  - Phase flow: story → extraction → bridging → resolution → integration
  - BOND controls all transitions — the sim just responds naturally

SETUP:
    pip install httpx websockets
    ollama pull mistral

USAGE:
    1. Create a session:
       python persona_sim.py create --token <tok> --couple_id <id>

    2. Run the sim:
       python persona_sim.py run --token_a <tok_a> --token_b <tok_b> --session_id <id>

    Note: token_a and token_b must be different user tokens (two accounts in the couple).
    Use seed_bridging.py to get two tokens if needed.

WHAT YOU SEE:
    Cyan   = Yogul
    Yellow = Niranjana
    White  = BOND messages
    Green  = system events
    Magenta = phase transitions
"""

import asyncio, json, argparse, httpx, websockets
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BASE_HTTP = "http://localhost:8000"
BASE_WS   = "ws://localhost:8000"
OLLAMA    = "http://localhost:11434/api/chat"
MODEL     = "mistral"
MAX_HIST  = 6
MAX_TURNS = 16
TIMEOUT   = 30

# ── Colours ───────────────────────────────────────────────────────────────────
class C:
    A     = "\033[96m"
    B     = "\033[93m"
    BOND  = "\033[37m"
    SYS   = "\033[92m"
    PHASE = "\033[95m"
    WARN  = "\033[91m"
    RESET = "\033[0m"

def ts():
    return datetime.now().strftime("%H:%M:%S")

def print_msg(label, text, colour):
    print(f"\n{colour}[{ts()}] {label}{C.RESET}")
    for line in text.strip().split("\n"):
        print(f"{colour}  {line}{C.RESET}")

def print_sys(text):
    print(f"{C.SYS}[{ts()}] ── {text}{C.RESET}")

def print_phase(text):
    print(f"\n{C.PHASE}[{ts()}] ⟳ PHASE: {text}{C.RESET}\n")

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS = {
    "yogul": {
        "name": "Yogul",
        "colour": C.A,
        "intention": "He's scared Niranjana is losing interest but won't say it directly — interprets her silence as rejection",
        "system": (
            "You are Yogul. You're talking to a counsellor called BOND. "
            "You're a casual, laid-back guy. You type like you talk — short, lowercase sometimes, no over-explaining. "
            "\n"
            "THE SITUATION: "
            "You and Niranjana have been close. Last week you asked her to hang out, she said she was busy. "
            "You said 'no worries' but it stung. Since then her replies have been one word and slow. "
            "You haven't said anything about it. You don't want to seem needy. "
            "Deep down you're scared she's losing interest but you'd never put it that way. "
            "\n"
            "HOW YOU TALK TO BOND: "
            "Tell the story in your own casual voice. Short sentences. One thing at a time. "
            "Don't volunteer your feelings — just say what happened. "
            "If BOND catches something real, you might open up a little — but slowly. "
            "Sound like a real person, not a summary. "
            "ONLY talk about what's happening between you and Niranjana. Nothing else. "
            "1-2 sentences per reply. Casual, slightly guarded."
        ),
    },
    "niranjana": {
        "name": "Niranjana",
        "colour": C.B,
        "intention": "She's not avoiding Yogul — exams have her completely overwhelmed and she has no emotional bandwidth for anyone",
        "system": (
            "You are Niranjana. You're talking to a counsellor called BOND. "
            "You're direct, a bit tired, not in the mood for long conversations. "
            "You give short answers. You don't over-explain. "
            "\n"
            "THE SITUATION: "
            "You have exams in two weeks. You're stressed and behind. "
            "Yogul texted asking to hang out. You saw it, forgot to reply, then hours later said you were busy. "
            "He said 'no worries' but seemed a bit off. You haven't texted since — not because you're avoiding him, "
            "you just genuinely have no bandwidth for anyone right now. "
            "\n"
            "HOW YOU TALK TO BOND: "
            "Tell BOND what happened — the texts, what was said. One thing at a time. "
            "Don't mention exams upfront. Let BOND ask. "
            "If pushed on why you've been distant, you might mention being swamped — but briefly, not as an excuse. "
            "You get slightly defensive if someone implies you're ignoring people on purpose. "
            "Sound like a real person — short, slightly impatient, honest. "
            "ONLY talk about what's been happening between you and Yogul. Nothing else. "
            "1-2 sentences per reply. Terse, real."
        ),
    },
}

# ── Ollama ────────────────────────────────────────────────────────────────────
async def ollama_reply(persona_key: str, bond_msg: str, history: list) -> str:
    persona  = PERSONAS[persona_key]
    trimmed  = history[-(MAX_HIST * 2):]

    # Detect semantic repetition across last 3 persona messages
    last_persona_msgs = [m["content"] for m in trimmed if m["role"] == "assistant"]
    repeat_warning = ""
    if len(last_persona_msgs) >= 2:
        # Check if last 2-3 messages are all saying the same thing in different words
        recent = last_persona_msgs[-3:]
        # Extract key nouns/verbs as topic fingerprint
        import re as _re
        def topic_words(s):
            return set(_re.findall(r'\b[a-z]{4,}\b', s.lower())) - {
                "that", "this", "with", "from", "have", "been", "they", "them",
                "said", "just", "didn", "wasn", "hadn", "doesn", "after", "then",
                "when", "what", "back", "again", "since", "will", "would", "could"
            }
        if len(recent) >= 2:
            last_topics = topic_words(recent[-1])
            prev_topics = topic_words(recent[-2])
            if last_topics and prev_topics:
                overlap = len(last_topics & prev_topics) / min(len(last_topics), len(prev_topics))
                if overlap > 0.6:
                    repeat_warning = (
                        "[WARNING: You are repeating the same point. "
                        "You have already said this. Do NOT repeat it. "
                        "Share something NEW that happened — the next event, "
                        "a different moment, something you haven't mentioned yet. "
                        "If there is nothing new to add, say so briefly.]"
                    )

    # Build a topic-lock based on last persona message
    last_topic = ""
    if last_persona_msgs:
        last_topic = f' Your last message was: "{last_persona_msgs[-1][:80]}". Build on that or move forward from it.'

    reminder = (
        f"[REMINDER: You are {persona['name']}. Stay in character completely."
        f" ONLY talk about what has happened between you and the other person."
        f" Do NOT mention: siblings, friends, coworkers, parties, yoga, meditation, games, sports, travel, food, work, movies, or any inner philosophical thoughts."
        f" If BOND asks about something unrelated, redirect back to the relationship."
        f" Give ONE new piece of information you haven't mentioned yet. 1-2 sentences."
        f"{last_topic}]"
    )

    messages = [{"role": "system", "content": persona["system"]}]
    messages += trimmed
    messages.append({"role": "system", "content": reminder})
    if repeat_warning:
        messages.append({"role": "system", "content": repeat_warning})
    messages.append({"role": "user", "content": f"BOND: {bond_msg}"})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 80, "temperature": 0.75},
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(OLLAMA, json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except Exception as e:
        print_sys(f"Ollama error ({persona['name']}): {e}")
        return "I'm not sure."

# ── WebSocket helpers ─────────────────────────────────────────────────────────
async def recv_bond(ws, timeout=30.0):
    """Wait for next meaningful event. Returns (type, content)."""
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None, None
        d = json.loads(raw)
        t = d.get("type")

        if t == "typing":
            continue
        elif t == "history":
            msgs = d.get("messages", [])
            bond_msgs = [m["content"] for m in msgs if m.get("role") == "ai"]
            opening = bond_msgs[-1] if bond_msgs else "Hey — what's been going on?"
            return "history", opening
        elif t == "message":
            return "message", d.get("content", "")
        elif t == "phase_change":
            return "phase_change", d.get("phase", "")
        elif t == "partner_status":
            return "partner_status", d.get("online", False)
        elif t == "bridge":
            return "bridge", d.get("content", "")
        elif t == "resolution":
            return "resolution", d.get("content", "")
        elif t == "closing":
            return "closing", d.get("content", "")

async def drain(ws, timeout=1.0):
    events = []
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            events.append(json.loads(raw))
        except asyncio.TimeoutError:
            break
    return events

async def send_msg(ws, text):
    await ws.send(json.dumps({"message": text}))
    await asyncio.sleep(0.4)

async def send_consent(ws, yes=True):
    await ws.send(json.dumps({"type": "bridge_consent", "consent": yes}))

# ── Per-persona coroutine ─────────────────────────────────────────────────────
async def persona_loop(
    token: str,
    session_id: str,
    persona_key: str,
    ready_event: asyncio.Event,
    partner_ready: asyncio.Event,
):
    persona  = PERSONAS[persona_key]
    name     = persona["name"]
    colour   = persona["colour"]
    url      = f"{BASE_WS}/ws/shared/{session_id}?token={token}"
    history: list[dict] = []
    consented = False

    print_sys(f"Connecting {name}...")

    async with websockets.connect(url) as ws:
        # Get opening from history
        evt_type, content = await recv_bond(ws, timeout=10.0)
        bond_opening = content if evt_type == "history" else "Hey — what's been going on?"
        print_msg(f"BOND → {name}", bond_opening, colour)

        await asyncio.sleep(0.3)
        await drain(ws)

        # Signal ready, wait for partner to also connect
        ready_event.set()
        print_sys(f"{name} connected — waiting for partner...")
        await partner_ready.wait()
        print_sys(f"{name}: partner online — starting")

        # Opening message
        opening = await ollama_reply(persona_key, bond_opening, history)
        print_msg(name, opening, colour)
        history.append({"role": "assistant", "content": opening})
        await send_msg(ws, opening)

        for _ in range(MAX_TURNS):
            evt_type, content = await recv_bond(ws, timeout=40.0)

            if evt_type is None:
                print_sys(f"{name}: timeout — stopping")
                break

            elif evt_type == "phase_change":
                print_phase(f"{name} → {content}")
                continue

            elif evt_type == "partner_status":
                print_sys(f"{name}: partner {'online' if content else 'offline'}")
                continue

            elif evt_type in ("message", "bridge", "resolution", "closing"):
                label = "BOND" if evt_type == "message" else f"BOND [{evt_type.upper()}]"
                print_msg(f"{label} → {name}", content, colour)
                history.append({"role": "user", "content": content})

                if evt_type == "bridge" and not consented:
                    await asyncio.sleep(1.5)
                    print_sys(f"{name} consenting...")
                    await send_consent(ws, yes=True)
                    consented = True
                    continue

                # Respond to all message types
                await asyncio.sleep(1.0)
                reply = await ollama_reply(persona_key, content, history)
                print_msg(name, reply, colour)
                history.append({"role": "assistant", "content": reply})
                await send_msg(ws, reply)

                if evt_type == "closing":
                    break

        print_sys(f"{name} done")

# ── Session creation ──────────────────────────────────────────────────────────
async def create_session(token: str, couple_id: str) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        # Get recent sessions and end any active shared ones
        existing = await client.get(
            f"{BASE_HTTP}/sessions/{couple_id}",
            params={"token": token}
        )
        if existing.status_code == 200:
            sessions = existing.json().get("sessions", [])
            for s in sessions:
                if s.get("is_active") and s.get("session_type") == "shared":
                    end_r = await client.post(
                        f"{BASE_HTTP}/sessions/{s['id']}/end",
                        params={"token": token}
                    )
                    print_sys(f"Ended old session: {s['id'][:12]}... ({end_r.status_code})")

        r = await client.post(
            f"{BASE_HTTP}/sessions/create",
            params={"token": token, "couple_id": couple_id, "session_type": "shared"}
        )
        r.raise_for_status()
        return r.json()["session_id"]

async def run_sim(token_a: str, token_b: str, couple_id: str, session_id: str = ""):
    if not session_id:
        print_sys("No session_id given -- creating one...")
        session_id = await create_session(token_a, couple_id)
        print_sys(f"Session created: {session_id}")

    print(f"\n{'='*60}")
    print(f"  BOND Persona Sim -- Shared Session")
    print(f"  Session: {session_id[:16]}...")
    print(f"\n  Intentions to assess:")
    for p in PERSONAS.values():
        print(f"  [{p['name']}] {p['intention']}")
    print(f"{'='*60}\n")

    yogul_ready     = asyncio.Event()
    niranjana_ready = asyncio.Event()

    await asyncio.gather(
        persona_loop(token_a, session_id, "yogul",     yogul_ready,     niranjana_ready),
        persona_loop(token_b, session_id, "niranjana", niranjana_ready, yogul_ready),
    )

    print(f"\n{'='*60}")
    print(f"  Sim complete. Did BOND surface these?")
    for p in PERSONAS.values():
        print(f"  [{p['name']}] {p['intention']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="BOND shared session persona simulator")
    parser.add_argument("--token_a",    default="",  help="Token for Yogul (reads config.json if omitted)")
    parser.add_argument("--token_b",    default="",  help="Token for Niranjana (reads config.json if omitted)")
    parser.add_argument("--couple_id",  default="",  help="Couple ID (reads config.json if omitted)")
    parser.add_argument("--session_id", default="",  help="Existing session ID (auto-creates if omitted)")
    parser.add_argument("--turns",      type=int, default=16)
    parser.add_argument("--model",      default="mistral")
    args = parser.parse_args()

    # Read from config.json if args not provided
    token_a   = args.token_a
    token_b   = args.token_b
    couple_id = args.couple_id

    if not token_a or not token_b or not couple_id:
        config_path = Path("config.json")
        if not config_path.exists():
            print("No config.json found. Run setup.py first, or pass --token_a --token_b --couple_id manually.")
            sys.exit(1)
        cfg = json.loads(config_path.read_text())
        token_a   = token_a   or cfg.get("token_a", "")
        token_b   = token_b   or cfg.get("token_b", "")
        couple_id = couple_id or cfg.get("couple_id", "")

    if not token_a or not token_b or not couple_id:
        print("Missing token_a, token_b or couple_id. Run setup.py first.")
        sys.exit(1)

    MAX_TURNS = args.turns
    MODEL     = args.model
    asyncio.run(run_sim(token_a, token_b, couple_id, args.session_id))