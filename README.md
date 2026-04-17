# BOND

AI relationship counselling for couples. Both partners talk to BOND separately — BOND listens to both sides, finds the dynamic keeping them stuck, and delivers a shared insight to each person privately.

**Live:** https://bond-sable.vercel.app

---

## What it does

Most relationship apps are journaling tools or generic advice. BOND does something different — it hears both sides of a situation privately, synthesises what neither person can see on their own, and names each person's contribution to the loop without blame.

**Shared session** — both partners join the same session. Each gets a private thread. BOND listens independently, runs a combined analysis, and when it has enough signal delivers a bridge question and a personalised resolution to each person.

**Individual session** — one person talks to BOND about a relationship situation. BOND investigates through structured questioning and delivers a one-sided reflection — what it notices about this person's pattern and what it might be costing them.

---

## Session arc

```
Story → Extraction → Depth → Bridging → Resolution → Integration → Closing
```

- **Story** — BOND follows events. No feelings questions. Summary fires at 6+ messages.
- **Extraction** — 3 questions from a private brief generated from the story.
- **Depth** — 2 deeper questions into emotional reality.
- **Bridging** — each person gets a private inward-facing question, then consents to hear the insight. Shared sessions only.
- **Resolution** — the main insight. Names their experience, the loop, their specific contribution using their actual words. gpt-4o.
- **Integration** — helps them process what landed. Reflects, pushes toward ownership, allows direction.
- **Closing** — fires when winding down detected. Picks a real moment from their conversation. gpt-4o.

---

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite → Vercel |
| Backend | FastAPI (Python) → Render |
| Database | PostgreSQL → Render |
| AI | OpenAI gpt-4o-mini + gpt-4o |
| Real-time | WebSocket |

---

## Key files

```
backend/
  main.py                    # FastAPI app, CORS, router registration
  config.py                  # Settings — models, keys, DB URL
  auth.py                    # JWT, password hashing, couple logic
  models/database.py         # All DB models — User, Couple, Session, Thread, Message, Memory
  routers/session_router.py  # Shared session WebSocket, phase transitions, resolution delivery
  routers/chat_router.py     # Individual session HTTP handler
  routers/auth_router.py     # Signup, login, onboarding, couple management
  agents/therapist.py        # Response generation — prompt selection, LLM call, post-processing
  agents/mediation.py        # Arc intelligence — analysis, briefs, resolution, integration, closing
  rag.py                     # Longitudinal memory across sessions

frontend/src/
  pages/Chat.jsx             # Chat interface — WebSocket, message rendering, phase-specific UI
  pages/Dashboard.jsx        # Session management, partner notification
  pages/CheckIn.jsx          # Pre-session mood check
```

---

## Local setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY, SECRET_KEY
uvicorn main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Environment variables**

| Variable | Required | Default |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | — |
| `SECRET_KEY` | Yes | — |
| `DATABASE_URL` | No | SQLite locally, Postgres on Render |
| `PRIMARY_MODEL` | No | gpt-4o-mini |
| `STRONG_MODEL` | No | gpt-4o |

---

## Cost

~$0.15–0.25 per full shared session. gpt-4o only fires for resolution and closing — the moments that define whether the product works. Everything else runs on mini.

---

## Limitations (current)

- Render free tier — server restarts after inactivity, 512MB RAM limit
- WebSocket drops during heavy gpt-4o calls — frontend falls back to HTTP
- Single server instance — no horizontal scaling without a pub/sub layer

---

Built for the conversations couples can't quite start.
