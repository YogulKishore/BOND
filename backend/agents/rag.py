"""
BOND RAG — Cross-Session Memory

Four document types:
  1. behavioral_signal   — per session: how this person engages (triggers, avoidance, what worked)
  2. bond_approach       — per session: what BOND tried and how it landed
  3. synthesized_profile — cross-session: consolidated profile rebuilt every 3 sessions
  4. couple_dynamic      — per session: what played out between them (shared only)

Three retrieval additions over v1:
  1. Smart query — LLM transforms surface text into pattern-level query before search
  2. Synthesized profile — always pinned if exists, never diluted by old per-session docs
  3. BOND approach memory — retrieves what worked / backfired with this specific person
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI
from config import get_settings

settings = get_settings()

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "bond_rag")
_chroma_client: Optional[chromadb.PersistentClient] = None
_openai_client: Optional[AsyncOpenAI] = None


def get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(_CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=_CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
    return _chroma_client


def _personal_collection(user_id: str) -> chromadb.Collection:
    return get_chroma().get_or_create_collection(
        name=f"personal_{user_id[:32]}",
        metadata={"hnsw:space": "cosine"}
    )


def _couple_collection(couple_id: str) -> chromadb.Collection:
    return get_chroma().get_or_create_collection(
        name=f"couple_{couple_id[:32]}",
        metadata={"hnsw:space": "cosine"}
    )


def get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature
    )


async def embed(text: str) -> list[float]:
    response = await get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=text.strip()[:8000]
    )
    return response.data[0].embedding


def _parse_json(raw: str) -> dict | None:
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None
    return None


def _age_label(dt: datetime) -> str:
    delta = datetime.utcnow() - dt
    if delta.days >= 30:
        return f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
    if delta.days >= 7:
        return f"{delta.days // 7} week{'s' if delta.days // 7 > 1 else ''} ago"
    if delta.days >= 1:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    if delta.seconds >= 3600:
        return f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
    return "earlier today"


# ─────────────────────────────────────────────
# EXTRACTION PROMPTS
# ─────────────────────────────────────────────

_BEHAVIORAL_SIGNAL_PROMPT = """\
You are reading a private conversation between one person and BOND.
Extract HOW this person engages — not what topic they discussed, but their behavioral patterns.

Return ONLY valid JSON:
{{
  "triggers": ["specific situations or behaviours that activated a reaction — precise"],
  "avoidance_moves": ["what they do when pulling back — specific actions or phrases"],
  "what_opened_them_up": ["what made them go deeper — BOND moves or their own moments"],
  "what_shut_them_down": ["what produced shorter answers or deflection"],
  "recurring_language": ["their exact phrases verbatim"],
  "core_position": "one sentence: their central stance about this relationship right now",
  "trajectory": "opening_up / stuck / deflecting / processing / defended"
}}

CONVERSATION:
{conversation}

Be specific. Empty list if no evidence for a field."""

_BOND_APPROACH_PROMPT = """\
Review this conversation between one person and BOND.
Extract what BOND tried and how well it landed — from the person's response, not BOND's intent.

Return ONLY valid JSON:
{{
  "worked_well": ["specific things BOND did or asked that produced more depth or openness"],
  "backfired": ["specific things that produced shutdown, shorter answers, or deflection"],
  "missed_openings": ["moments the person signalled something deeper but BOND missed it"],
  "person_responded_best_to": "one sentence: the style that consistently opened this person up"
}}

CONVERSATION:
{conversation}

Empty lists if no clear evidence. Don't invent."""

_SYNTHESIS_PROMPT = """\
Synthesize multiple sessions of behavioral data about one person. Most recent sessions last.

{signals}

Generate a consolidated profile weighted toward recent sessions.
If early patterns contradict recent ones, trust recent.

Return ONLY valid JSON:
{{
  "stable_triggers": ["triggers appearing in 2+ sessions"],
  "stable_avoidance": ["avoidance moves repeating across sessions"],
  "what_reliably_opens_them": ["approaches that worked in multiple sessions"],
  "what_reliably_shuts_them_down": ["things that backfired more than once"],
  "their_language": ["phrases they use repeatedly — verbatim"],
  "current_position": "one sentence: where they are in this relationship now, based on the arc",
  "trajectory_arc": "how engagement evolved across sessions — e.g. gradually more open, stuck in same loop"
}}

Patterns appearing only once don't belong here."""

_COUPLE_DYNAMIC_PROMPT = """\
You have listened privately to both people in a relationship.

PERSON A's conversation:
{thread_a}

PERSON B's conversation:
{thread_b}

Extract what pattern played out between them — what you can see from outside both perspectives.

Return ONLY valid JSON:
{{
  "pattern_this_session": "one sentence: the core dynamic — pursue/withdraw, over-explain/shut-down etc",
  "how_far_they_got": "surface_venting / pattern_emerging / real_insight / near_resolution / resolved",
  "a_effect_on_b": "how A's behaviour landed on B — the gap between intention and impact",
  "b_effect_on_a": "how B's behaviour landed on A — the gap between intention and impact",
  "what_broke_down": "specific moment that blocked deeper progress, or null",
  "what_worked": "what created the most movement or openness",
  "unresolved_thread": "what kept circling but was never fully surfaced"
}}"""

_SMART_QUERY_PROMPT = """\
Someone is starting a relationship support session.
Generate a SHORT search query capturing the underlying relational PATTERN — not the surface topic.

Checkin: {checkin}
First message: {message}

Examples:
"feeling unheard again" + "she just stops talking mid-conversation"
→ "partner withdrawal during conversation, pursuit and distance loop"

"overwhelmed" + "he texts constantly and expects me to reply right away"
→ "pressure to match energy, contact frequency mismatch, communication overwhelm"

"don't know what to do" + "every time we fight she brings up old stuff"
→ "conflict escalation, past grievances resurfacing, unresolved resentment"

Return only the query. One sentence, no punctuation at end."""


# ─────────────────────────────────────────────
# TEXT FORMATTERS
# ─────────────────────────────────────────────

def _signal_to_text(signal: dict, ended_at: datetime, n: int) -> str:
    p = [f"Session {n} ({_age_label(ended_at)})."]
    if signal.get("core_position"): p.append(signal["core_position"])
    if signal.get("triggers"): p.append(f"Triggered by: {'; '.join(signal['triggers'][:3])}.")
    if signal.get("avoidance_moves"): p.append(f"Pulls back by: {'; '.join(signal['avoidance_moves'][:2])}.")
    if signal.get("what_opened_them_up"): p.append(f"Opened up when: {'; '.join(signal['what_opened_them_up'][:2])}.")
    if signal.get("what_shut_them_down"): p.append(f"Shut down when: {'; '.join(signal['what_shut_them_down'][:2])}.")
    if signal.get("recurring_language"): p.append(f"Their words: \"{'; '.join(signal['recurring_language'][:3])}\".")
    if signal.get("trajectory"): p.append(f"Trajectory: {signal['trajectory']}.")
    return " ".join(p) if len(p) > 2 else ""


def _approach_to_text(approach: dict, ended_at: datetime, n: int) -> str:
    p = [f"Session {n} ({_age_label(ended_at)}) BOND approach:"]
    if approach.get("worked_well"): p.append(f"Worked: {'; '.join(approach['worked_well'][:2])}.")
    if approach.get("backfired"): p.append(f"Backfired: {'; '.join(approach['backfired'][:2])}.")
    if approach.get("missed_openings"): p.append(f"Missed: {approach['missed_openings'][0]}.")
    if approach.get("person_responded_best_to"): p.append(f"Responds best to: {approach['person_responded_best_to']}.")
    return " ".join(p) if len(p) > 2 else ""


def _synthesis_to_text(s: dict, ended_at: datetime, n: int) -> str:
    p = [f"Consolidated profile across {n} sessions ({_age_label(ended_at)})."]
    if s.get("current_position"): p.append(s["current_position"])
    if s.get("stable_triggers"): p.append(f"Consistent triggers: {'; '.join(s['stable_triggers'][:3])}.")
    if s.get("stable_avoidance"): p.append(f"Consistent avoidance: {'; '.join(s['stable_avoidance'][:2])}.")
    if s.get("what_reliably_opens_them"): p.append(f"Reliably opens up when: {'; '.join(s['what_reliably_opens_them'][:2])}.")
    if s.get("what_reliably_shuts_them_down"): p.append(f"Reliably shuts down when: {'; '.join(s['what_reliably_shuts_them_down'][:2])}.")
    if s.get("their_language"): p.append(f"Their words: \"{'; '.join(s['their_language'][:4])}\".")
    if s.get("trajectory_arc"): p.append(f"Arc: {s['trajectory_arc']}.")
    return " ".join(p) if len(p) > 2 else ""


def _dynamic_to_text(d: dict, ended_at: datetime, n: int) -> str:
    p = [f"Session {n} ({_age_label(ended_at)})."]
    if d.get("pattern_this_session"): p.append(d["pattern_this_session"])
    if d.get("how_far_they_got"): p.append(f"Depth: {d['how_far_they_got']}.")
    if d.get("a_effect_on_b"): p.append(f"A on B: {d['a_effect_on_b']}.")
    if d.get("b_effect_on_a"): p.append(f"B on A: {d['b_effect_on_a']}.")
    if d.get("what_broke_down"): p.append(f"Broke down: {d['what_broke_down']}.")
    if d.get("unresolved_thread"): p.append(f"Unresolved: {d['unresolved_thread']}.")
    return " ".join(p) if len(p) > 2 else ""


# ─────────────────────────────────────────────
# EMBEDDING PIPELINE
# ─────────────────────────────────────────────

async def embed_session(
    session_id: str,
    session_type: str,
    couple_id: str,
    user_ids: list[str],
    session_number: int,
    ended_at: datetime,
) -> None:
    """
    Full embedding pipeline. Runs as background task after session end.
    Per person: behavioral_signal + bond_approach + synthesis (every 3rd session)
    Per couple (shared): couple_dynamic
    """
    from models.database import SessionLocal, Message, Thread, Session as SessionModel
    llm = get_llm()
    db = SessionLocal()
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            return

        for user_id in user_ids:
            try:
                if session_type == "shared":
                    thread = db.query(Thread).filter(
                        Thread.session_id == session_id,
                        Thread.user_id == user_id
                    ).first()
                    if not thread:
                        continue
                    msgs = db.query(Message).filter(
                        Message.thread_id == thread.id
                    ).order_by(Message.created_at).all()
                else:
                    msgs = db.query(Message).filter(
                        Message.session_id == session_id
                    ).order_by(Message.created_at).all()

                if not msgs or len([m for m in msgs if m.sender_id != "ai"]) < 2:
                    continue

                conv = "\n".join([
                    f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
                    for m in msgs
                ])
                base_meta = {
                    "user_id": user_id, "couple_id": couple_id,
                    "session_id": session_id, "session_number": session_number,
                    "session_date": ended_at.isoformat(), "session_type": session_type,
                }
                coll = _personal_collection(user_id)

                # behavioral_signal
                try:
                    resp = await llm.ainvoke([HumanMessage(content=_BEHAVIORAL_SIGNAL_PROMPT.format(conversation=conv))])
                    signal = _parse_json(resp.content)
                    if signal:
                        text = _signal_to_text(signal, ended_at, session_number)
                        if text:
                            coll.upsert(
                                ids=[f"{session_id}_{user_id}_signal"],
                                embeddings=[await embed(text)],
                                documents=[text],
                                metadatas=[{**base_meta, "doc_type": "behavioral_signal",
                                            "trajectory": signal.get("trajectory", "unknown")}]
                            )
                            print(f"[RAG] behavioral_signal user={user_id[:8]}")
                except Exception as e:
                    print(f"[RAG] behavioral_signal failed: {e}")

                # bond_approach
                try:
                    resp = await llm.ainvoke([HumanMessage(content=_BOND_APPROACH_PROMPT.format(conversation=conv))])
                    approach = _parse_json(resp.content)
                    if approach:
                        text = _approach_to_text(approach, ended_at, session_number)
                        if text:
                            coll.upsert(
                                ids=[f"{session_id}_{user_id}_approach"],
                                embeddings=[await embed(text)],
                                documents=[text],
                                metadatas=[{**base_meta, "doc_type": "bond_approach"}]
                            )
                            print(f"[RAG] bond_approach user={user_id[:8]}")
                except Exception as e:
                    print(f"[RAG] bond_approach failed: {e}")

                # cross-session synthesis every 3rd session
                if session_number >= 3 and session_number % 3 == 0:
                    asyncio.create_task(
                        _synthesize_profile(user_id, couple_id, session_number, ended_at)
                    )

            except Exception as e:
                print(f"[RAG] per-person failed user={user_id[:8]}: {e}")

        # couple_dynamic
        if session_type == "shared":
            try:
                threads = db.query(Thread).filter(Thread.session_id == session_id).all()
                if len(threads) >= 2:
                    def fmt(t):
                        ms = db.query(Message).filter(Message.thread_id == t.id).order_by(Message.created_at).all()
                        return "\n".join([f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}" for m in ms])

                    resp = await llm.ainvoke([HumanMessage(content=_COUPLE_DYNAMIC_PROMPT.format(
                        thread_a=fmt(threads[0]), thread_b=fmt(threads[1])
                    ))])
                    dynamic = _parse_json(resp.content)
                    if dynamic:
                        text = _dynamic_to_text(dynamic, ended_at, session_number)
                        if text:
                            coll = _couple_collection(couple_id)
                            coll.upsert(
                                ids=[f"{session_id}_couple_dynamic"],
                                embeddings=[await embed(text)],
                                documents=[text],
                                metadatas=[{
                                    "couple_id": couple_id, "session_id": session_id,
                                    "doc_type": "couple_dynamic", "session_number": session_number,
                                    "session_date": ended_at.isoformat(),
                                    "how_far": dynamic.get("how_far_they_got", "unknown"),
                                }]
                            )
                            print(f"[RAG] couple_dynamic couple={couple_id[:8]}")
            except Exception as e:
                print(f"[RAG] couple_dynamic failed: {e}")

    except Exception as e:
        print(f"[RAG] embed_session failed: {e}")
    finally:
        db.close()


async def _synthesize_profile(user_id: str, couple_id: str, up_to: int, ended_at: datetime) -> None:
    """
    Runs every 3rd session. Reads all behavioral_signal docs, generates a
    consolidated profile, upserts with stable ID (replaces previous synthesis).
    """
    try:
        coll = _personal_collection(user_id)
        results = coll.get(where={"doc_type": "behavioral_signal"}, include=["documents", "metadatas"])
        if not results["documents"] or len(results["documents"]) < 2:
            return

        paired = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1].get("session_number", 0)
        )
        signals_text = "\n\n---\n\n".join([
            f"Session {meta.get('session_number', '?')}:\n{doc}"
            for doc, meta in paired
        ])

        llm = get_llm()
        resp = await llm.ainvoke([HumanMessage(content=_SYNTHESIS_PROMPT.format(signals=signals_text))])
        synthesized = _parse_json(resp.content)
        if not synthesized:
            return

        text = _synthesis_to_text(synthesized, ended_at, up_to)
        if not text:
            return

        coll.upsert(
            ids=[f"{user_id}_synthesized_profile"],  # stable ID — always replaces
            embeddings=[await embed(text)],
            documents=[text],
            metadatas=[{
                "user_id": user_id, "couple_id": couple_id,
                "doc_type": "synthesized_profile",
                "sessions_covered": up_to,
                "synthesized_at": ended_at.isoformat(),
            }]
        )
        print(f"[RAG] synthesized_profile user={user_id[:8]} sessions=1-{up_to}")

    except Exception as e:
        print(f"[RAG] synthesis failed user={user_id[:8]}: {e}")


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

async def retrieve_context(
    user_id: str,
    couple_id: str,
    session_type: str,
    query_text: str,
) -> str | None:
    """
    Three-stage retrieval:
    1. Smart query — LLM transforms surface text to pattern-level search query
    2. Personal: pin synthesized_profile (if exists) + semantic search for
       behavioral_signal (top 2) + bond_approach (top 1)
    3. Couple: semantic search for couple_dynamic (top 2, shared only)
    Relevance threshold: cosine distance < 0.65
    """
    if not query_text or not query_text.strip():
        query_text = "relationship patterns communication"

    # Smart query generation
    smart_query = query_text
    try:
        llm = get_llm(temperature=0.1)
        parts = query_text.split("\n", 1)
        checkin = parts[0].strip() if len(parts) > 1 else ""
        message = parts[-1].strip()
        resp = await llm.ainvoke([HumanMessage(
            content=_SMART_QUERY_PROMPT.format(checkin=checkin or message, message=message)
        )])
        smart_query = resp.content.strip()
        print(f"[RAG] smart query: {smart_query[:80]}")
    except Exception as e:
        print(f"[RAG] smart query failed (using raw): {e}")

    try:
        qv = await embed(smart_query)
    except Exception as e:
        print(f"[RAG] embed failed: {e}")
        return None

    sections = []

    # Personal
    try:
        coll = _personal_collection(user_id)
        if coll.count() > 0:
            personal_lines = []

            # Synthesized profile — pinned, no semantic search needed
            try:
                synth = coll.get(ids=[f"{user_id}_synthesized_profile"], include=["documents"])
                if synth["documents"]:
                    personal_lines.append(f"[PROFILE] {synth['documents'][0][:350]}")
            except Exception:
                pass

            # Behavioral signals
            try:
                r = coll.query(
                    query_embeddings=[qv], n_results=min(2, coll.count()),
                    where={"doc_type": "behavioral_signal"},
                    include=["documents", "distances"]
                )
                for doc, dist in zip(r["documents"][0], r["distances"][0]):
                    if dist < 0.65:
                        personal_lines.append(doc[:280])
            except Exception:
                pass

            # BOND approach memory
            try:
                r = coll.query(
                    query_embeddings=[qv], n_results=min(1, coll.count()),
                    where={"doc_type": "bond_approach"},
                    include=["documents", "distances"]
                )
                for doc, dist in zip(r["documents"][0], r["distances"][0]):
                    if dist < 0.65:
                        personal_lines.append(f"[APPROACH] {doc[:220]}")
            except Exception:
                pass

            if personal_lines:
                sections.append(
                    "PERSONAL HISTORY — use silently, NEVER quote back:\n" +
                    "\n".join(f"- {l}" for l in personal_lines)
                )
    except Exception as e:
        print(f"[RAG] personal retrieval failed: {e}")

    # Couple dynamic
    if session_type == "shared":
        try:
            ccoll = _couple_collection(couple_id)
            if ccoll.count() > 0:
                r = ccoll.query(
                    query_embeddings=[qv], n_results=min(2, ccoll.count()),
                    include=["documents", "distances"]
                )
                couple_lines = []
                for doc, dist in zip(r["documents"][0], r["distances"][0]):
                    if dist < 0.65:
                        couple_lines.append(doc[:280])
                if couple_lines:
                    sections.append(
                        "COUPLE PATTERN HISTORY — background only, NEVER reference directly:\n" +
                        "\n".join(f"- {l}" for l in couple_lines)
                    )
        except Exception as e:
            print(f"[RAG] couple retrieval failed: {e}")

    return "\n\n".join(sections) if sections else None