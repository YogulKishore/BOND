"""
BOND Memory — Session Summaries & Pattern Memory

Handles all post-session intelligence:
- Session summary generation (brief / standard / rich depth)
- Individual behavioral pattern memory
- Couple pattern memory
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.database import SessionLocal, Message, Memory, Session, Thread
from config import get_settings
import json
import re

settings = get_settings()


def get_llm(temperature: float = 0.65):
    return ChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature
    )


def _extract_json(text: str) -> str:
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


SUMMARY_PROMPT_BRIEF = """Analyze this short relationship support conversation.
Return ONLY valid JSON — no preamble, no markdown.

{{
  "main_topic": "one sentence",
  "emotional_arc": "one phrase",
  "key_insights": ["1 insight — empty if none"],
  "patterns": [],
  "avoidance_patterns": [],
  "attachment_signals": "",
  "resolution": "resolved / ongoing / unresolved / just venting"
}}

CONVERSATION:
{conversation}"""

SUMMARY_PROMPT_STANDARD = """Analyze this relationship support conversation.
Return ONLY valid JSON — no preamble, no markdown.

{{
  "main_topic": "one sentence — core issue",
  "emotional_arc": "how tone shifted",
  "key_insights": ["2-3 specific things learned"],
  "patterns": ["recurring themes or behaviors"],
  "avoidance_patterns": ["feelings or topics steered away from"],
  "attachment_signals": "one sentence — any attachment style signs",
  "emotional_shifts": ["moments where tone changed and what triggered them"],
  "resolution": "resolved / ongoing / unresolved / just venting"
}}

CONVERSATION:
{conversation}"""

SUMMARY_PROMPT_RICH = """Analyze this in-depth relationship support conversation. Extract maximum signal.
Return ONLY valid JSON — no preamble, no markdown.

{{
  "main_topic": "one sentence — core issue",
  "emotional_arc": "detailed tone shift description",
  "key_insights": ["3-5 specific things learned"],
  "patterns": ["all recurring themes and behaviors"],
  "avoidance_patterns": ["consistently deflected topics or feelings"],
  "attachment_signals": "one sentence — attachment style signs",
  "emotional_shifts": ["specific moments of openness or closure and triggers"],
  "self_awareness_moments": ["moments of genuine insight about own role"],
  "breakthroughs": ["moments of clarity or shift — empty if none"],
  "resolution": "resolved / ongoing / unresolved / just venting",
  "follow_up": "one thing worth revisiting next session"
}}

CONVERSATION:
{conversation}"""


def get_summary_depth(messages: list, session_type: str = "individual") -> str:
    user_msg_count = sum(1 for m in messages if m.sender_id != "ai")
    minimum = 6 if session_type == "shared" else 4
    if user_msg_count < minimum:
        return "none"
    elif user_msg_count < 8:
        return "brief"
    elif user_msg_count < 15:
        return "standard"
    else:
        return "rich"


async def generate_session_summary(session_id: str, session_type: str = "individual") -> dict | None:
    db = SessionLocal()
    try:
        if session_type == "shared":
            threads = db.query(Thread).filter(Thread.session_id == session_id).all()
            if not threads:
                return None
            messages = db.query(Message).filter(
                Message.thread_id.in_([t.id for t in threads])
            ).order_by(Message.created_at).all()
            user_ids = list({t.user_id for t in threads})
            from models.database import User
            users = db.query(User).filter(User.id.in_(user_ids)).all()
            id_to_name = {u.id: u.name for u in users}
        else:
            messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.created_at).all()
            id_to_name = {}

        if not messages:
            return None

        depth = get_summary_depth(messages, session_type)
        if depth == "none":
            return None

        lines = []
        for m in messages:
            if m.sender_id == "ai":
                speaker = "BOND"
            elif session_type == "shared":
                speaker = id_to_name.get(m.sender_id, m.sender_id)
            else:
                speaker = "User"
            lines.append(f"{speaker}: {m.content}")
        conversation = "\n".join(lines)

        if depth == "brief":
            prompt = SUMMARY_PROMPT_BRIEF.format(conversation=conversation)
        elif depth == "standard":
            prompt = SUMMARY_PROMPT_STANDARD.format(conversation=conversation)
        else:
            prompt = SUMMARY_PROMPT_RICH.format(conversation=conversation)

        llm = get_llm(temperature=0.2)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            print(f"[SUMMARY LLM ERROR] {e}")
            return None

        def extract_json(text):
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
            return text

        cleaned = extract_json(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            try:
                fixed = cleaned
                if not fixed.rstrip().endswith("}"):
                    fixed += "]" * (fixed.count("[") - fixed.count("]"))
                    if not fixed.rstrip().endswith("}"):
                        fixed += "}"
                parsed = json.loads(fixed)
            except Exception:
                import re
                parsed = {}
                for field, pat in [
                    ('main_topic', r'"main_topic":\s*"([^"]+)"'),
                    ('emotional_arc', r'"emotional_arc":\s*"([^"]+)"'),
                    ('resolution', r'"resolution":\s*"([^"]+)"'),
                ]:
                    m = re.search(pat, cleaned)
                    if m:
                        parsed[field] = m.group(1)
                if not parsed:
                    return None

        parts = []
        field_map = [
            ('main_topic', 'Topic'), ('emotional_arc', 'Arc'),
            ('key_insights', 'Insights'), ('patterns', 'Patterns'),
            ('avoidance_patterns', 'Avoidance'), ('attachment_signals', 'Attachment'),
            ('emotional_shifts', 'Shifts'), ('self_awareness_moments', 'Self-awareness'),
            ('breakthroughs', 'Breakthroughs'), ('resolution', 'Resolution'),
            ('follow_up', 'Follow up'),
        ]
        for key, label in field_map:
            val = parsed.get(key)
            if not val:
                continue
            if isinstance(val, list):
                # Safely convert any item to string — model sometimes returns dicts
                items = [str(i) if not isinstance(i, str) else i for i in val if i]
                if items:
                    parts.append(f"{label}: {'; '.join(items)}")
            else:
                parts.append(f"{label}: {val}")

        if not parts:
            return None

        return {"readable": " | ".join(parts), "json_data": parsed, "depth": depth}
    finally:
        db.close()


# ─────────────────────────────────────────────
# PATTERN MEMORY UPDATER
# ─────────────────────────────────────────────

PATTERN_PROMPT = """Analyze relationship support session summaries for one person.
Extract behavioral patterns. Only note what the data shows. No invention.

Return ONLY valid JSON — no preamble, no markdown.

{{
  "recurring_themes": ["key themes this person returns to"],
  "triggers": ["things that escalate or upset them"],
  "what_helps": ["things that calm or support them"],
  "avoidance_patterns": ["topics or feelings consistently avoided"],
  "attachment_signals": "one sentence — attachment style signs",
  "self_awareness_trajectory": "one sentence — is self-awareness growing, stable, or inconsistent",
  "progress": "one sentence — improving, stable, or harder ('too early to tell' if one session)",
  "watch_for": "one sentence — most important thing to watch for"
}}

SESSION SUMMARIES (most recent first):
{summaries}"""


async def update_pattern_memory(couple_id: str, user_id: str) -> None:
    db = SessionLocal()
    try:
        past_sessions = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.is_active == False,
            Session.summary_json != None
        ).order_by(Session.created_at.desc()).limit(10).all()

        if not past_sessions:
            return

        summaries_list = []
        for s in past_sessions:
            try:
                data = json.loads(s.summary_json)
                parts = []
                for key, label in [
                    ('main_topic', 'Topic'), ('emotional_arc', 'Arc'),
                    ('key_insights', 'Insights'), ('patterns', 'Patterns'),
                    ('avoidance_patterns', 'Avoidance'),
                    ('attachment_signals', 'Attachment'),
                    ('self_awareness_moments', 'Self-awareness'),
                    ('resolution', 'Resolution'),
                ]:
                    val = data.get(key)
                    if not val:
                        continue
                    if isinstance(val, list):
                        items = [i for i in val if i]
                        if items:
                            parts.append(f"{label}: {'; '.join(items)}")
                    else:
                        parts.append(f"{label}: {val}")
                summaries_list.append(" | ".join(parts))
            except Exception:
                if s.summary:
                    summaries_list.append(s.summary)

        summaries_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries_list)])
        prompt = PATTERN_PROMPT.format(summaries=summaries_text)

        llm = get_llm(temperature=0.2)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            print(f"[PATTERN MEMORY LLM ERROR] {e}")
            return

        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        try:
            parsed = json.loads(raw)
        except Exception:
            return

        # Pattern memory is user-level — stored under owner_id regardless of couple
        # Find any existing pattern for this user across any couple
        existing = db.query(Memory).filter(
            Memory.owner_id == user_id,
            Memory.memory_type == "pattern"
        ).first()

        from models.database import generate_id
        from datetime import datetime
        if existing:
            existing.content = json.dumps(parsed)
            existing.created_at = datetime.utcnow()
        else:
            db.add(Memory(
                id=generate_id(),
                couple_id=couple_id,  # keep a couple ref for DB constraint
                owner_id=user_id,
                content=json.dumps(parsed),
                memory_type="pattern"
            ))
        db.commit()
    finally:
        db.close()


# ─────────────────────────────────────────────
# COUPLE PATTERN MEMORY UPDATER
# ─────────────────────────────────────────────

COUPLE_PATTERN_PROMPT = """Analyze shared relationship support session summaries for a couple.
Extract patterns about how they interact. Only what the data shows. No sides taken.

Return ONLY valid JSON — no preamble, no markdown.

{{
  "dynamic": "one sentence — how this couple tends to interact",
  "recurring_conflicts": ["topics or situations causing recurring friction"],
  "communication_breakdowns": ["things that cause communication to break down"],
  "what_works": ["things that help them connect or move forward"],
  "avoidance_patterns": ["things either or both partners consistently avoid"],
  "attachment_dynamic": "one sentence — how their attachment styles interact",
  "progress": "one sentence — improving, stable, or harder",
  "watch_for": "one sentence — most important dynamic to watch"
}}

SHARED SESSION SUMMARIES (most recent first):
{summaries}"""


async def update_couple_pattern(couple_id: str) -> None:
    db = SessionLocal()
    try:
        past_sessions = db.query(Session).filter(
            Session.couple_id == couple_id,
            Session.session_type == "shared",
            Session.is_active == False,
            Session.summary_json != None
        ).order_by(Session.created_at.desc()).limit(10).all()

        if not past_sessions:
            return

        summaries_list = []
        for s in past_sessions:
            try:
                data = json.loads(s.summary_json)
                parts = []
                for key, label in [
                    ('main_topic', 'Topic'), ('emotional_arc', 'Arc'),
                    ('key_insights', 'Insights'), ('patterns', 'Patterns'),
                    ('avoidance_patterns', 'Avoidance'),
                    ('attachment_signals', 'Attachment'),
                    ('resolution', 'Resolution'),
                ]:
                    val = data.get(key)
                    if not val:
                        continue
                    if isinstance(val, list):
                        items = [i for i in val if i]
                        if items:
                            parts.append(f"{label}: {'; '.join(items)}")
                    else:
                        parts.append(f"{label}: {val}")
                summaries_list.append(" | ".join(parts))
            except Exception:
                if s.summary:
                    summaries_list.append(s.summary)

        summaries_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries_list)])
        prompt = COUPLE_PATTERN_PROMPT.format(summaries=summaries_text)

        llm = get_llm(temperature=0.2)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
        except Exception as e:
            print(f"[COUPLE PATTERN LLM ERROR] {e}")
            return

        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        try:
            parsed = json.loads(raw)
        except Exception:
            print(f"[COUPLE PATTERN] parse failed, skipping")
            return

        existing = db.query(Memory).filter(
            Memory.couple_id == couple_id,
            Memory.owner_id == None,
            Memory.memory_type == "couple_pattern"
        ).first()

        from models.database import generate_id
        from datetime import datetime
        if existing:
            existing.content = json.dumps(parsed)
            existing.created_at = datetime.utcnow()
        else:
            db.add(Memory(
                id=generate_id(),
                couple_id=couple_id,
                owner_id=None,
                content=json.dumps(parsed),
                memory_type="couple_pattern"
            ))
        db.commit()
        print(f"[COUPLE PATTERN] updated for couple {couple_id[:8]}")
    finally:
        db.close()