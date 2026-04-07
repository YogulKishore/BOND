"""
BOND Mediation Arc — Shared Session Intelligence

All shared session logic:
- Thread summaries (rolling emotional snapshot per participant)
- Core need extraction (what each person fundamentally needs)
- Misunderstanding detection (what's keeping them stuck)
- Phase transitions: listening → understanding → bridging → resolution → integration
- Bridge readiness + insight
- Integration reaction detection
- Closing reflection
- Resolution message generation
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.database import SessionLocal, Message, Session, Thread
from config import get_settings
import json

settings = get_settings()


def get_llm(temperature: float = 0.45):
    return ChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature
    )


def _parse_json_safe(raw: str) -> dict:
    text = raw.strip()
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
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}


CORE_NEED_PROMPT = """You are reading a private conversation between one person and BOND.

Based on everything shared, identify their core emotional need.

This is NOT what they said they want.
This is what they fundamentally need underneath the situation.

CRITICAL:
- Be specific to THIS person's situation
- If your answer could apply to anyone, it is too generic

BAD examples (too generic — do not use these):
- Needs reassurance
- Needs validation
- Needs to feel heard

GOOD examples (specific — aim for this level):
- Needs to feel like their partner considers them in big decisions, not just informs them after
- Needs to feel emotionally prioritised before solutions are offered
- Needs their partner to acknowledge the impact of the move, not just defend the logic behind it

Return ONE sentence only. No preamble.

CONVERSATION:
{conversation}"""


# ─────────────────────────────────────────────
# COMBINED THREAD ANALYSIS
# Replaces: extract_core_need + detect_and_store_misunderstanding
#           + check_bridge_readiness as separate steps
#
# Runs on raw messages from both threads directly.
# Looks for complementary patterns — not topic overlap.
# ─────────────────────────────────────────────

COMBINED_ANALYSIS_PROMPT = """You are BOND. You have been talking privately with both people in a relationship.
You can see both private conversations. Neither person can see what the other said.

PERSON A's conversation with BOND:
{thread_a}

PERSON B's conversation with BOND:
{thread_b}

Analyze what is actually happening between them. Return ONLY valid JSON — no preamble, no markdown.

{{
  "person_a": {{
    "event": "What actually happened — the concrete situation they described",
    "interpretation": "What they made it mean — their reading of the other person's behaviour",
    "unmet_need": "What they needed that they didn't get — specific, not generic",
    "their_contribution": "How their own response is contributing to the loop — the part they can't see. This is the most important field. Be honest and specific."
  }},
  "person_b": {{
    "event": "What actually happened — from their side",
    "interpretation": "What they made it mean",
    "unmet_need": "What they needed that they didn't get",
    "their_contribution": "How their own response is contributing to the loop — the part they can't see"
  }},
  "complementary_pattern": "How A's response to their interpretation triggers B's response, which confirms A's interpretation. Name the loop explicitly — e.g. 'His pursuit in response to silence triggers more withdrawal, which he reads as rejection, which increases pursuit.'",
  "dynamic": "One sentence: the core dynamic keeping them stuck.",
  "misunderstanding": "The intention vs impact gap for both people — what each intends and how it lands on the other.",
  "path_forward": "One small shift that could break the loop — not a solution, a direction toward it.",
  "bridge_question_a": "One question BOND could ask Person A privately that would lead them toward seeing their own contribution without announcing it. Must be grounded in what A actually said.",
  "bridge_question_b": "Same for Person B.",
  "ready_for_bridge": true or false — true ONLY if both four-layer models are clear and the complementary pattern is specific. False if still surface-level.,
  "confidence": "low / medium / high"
}}

CRITICAL:
- their_contribution must be honest — name what they are doing that makes it worse, even if unintentional
- complementary_pattern must name the loop explicitly, not just describe them separately
- bridge_question must be something that creates a moment of pause, not a leading question
- Never use generic language like "they need to communicate better"
- Be specific to THIS couple's actual words"""


# ─────────────────────────────────────────────
# LIGHTWEIGHT SHARED ANALYSIS
# ─────────────────────────────────────────────

SHARED_ANALYSIS_PROMPT = """Analyze this person in a private relationship support conversation.
Return ONLY valid JSON — no preamble, no markdown.

{{
  "emotional_state": "calm / sad / frustrated / distressed / guarded / open / hopeful / exhausted / angry / ashamed",
  "intensity": "low / medium / high",
  "needs": "to_be_heard / space / reflection / gentle_challenge / validation / clarity",
  "avoid": "one short phrase — the single thing that would make this worse",
  "regulation": "regulated / partial / flooded"
}}

RECENT CONVERSATION:
{conversation}

CURRENT MESSAGE:
{message}"""


# ─────────────────────────────────────────────
# BRIDGE + RESOLUTION + INTEGRATION PROMPTS
# ─────────────────────────────────────────────

SHARED_BRIDGE_CONSENT_PROMPT = """You are BOND. You've been listening to both sides privately.
You have a sense of what's really happening between them and want to share something that might help.
First, ask naturally and warmly if they're open to hearing it.
One or two sentences only.
Example: "I've been sitting with everything you've shared, and I think I'm starting to see something. Would you be open to hearing what I'm noticing?" """

SHARED_RESOLUTION_PROMPT = """You are BOND. You have listened to both sides of this situation privately.

Here is what you know about THIS person:
Event: {event}
What they made it mean: {interpretation}
What they needed: {unmet_need}
Their contribution to the loop — the part they can't see: {their_contribution}

The pattern between them: {complementary_pattern}
The gap: {misunderstanding}
A direction: {path_forward}

Your job: deliver the first beat of the resolution — warmly, specifically, from their perspective.

STRUCTURE — in this order:
1. Name what they've been experiencing — specific, using their actual words or images
2. Name the loop — how their own response feeds the very thing they don't want
3. Surface their_contribution — this is the turning point. THE MOST IMPORTANT PART.
   Name it with precision. Do NOT soften it, hedge it, or bury it in qualifiers.
   
   WEAK (too soft — will not land):
   "Your tendency to hold back might be contributing to some distance."
   "In trying to keep things casual, you may be leaving some feelings unspoken."
   
   STRONG (specific, honest, without blame — this is the standard):
   "By going quiet and waiting for her to reach out first, you're doing the exact thing you're afraid she's doing to you."
   "The 'no worries' is keeping you safe from rejection — and keeping the loop going."
   "Sending one casual text and then waiting in silence is the thing that's making the silence worse."
   
   The person should feel: "oh. that's what I've been doing."
   Not: "I guess that's fair."
   
4. End with ONE question or observation that opens space — not advice, just a moment of pause

HARD RULES:
- Use their own language and images — not generic relationship words
- Do NOT reveal what their partner said or imply you have the other side
- Do NOT deliver path_forward yet
- Do NOT be preachy or lecture
- Never use "Person A/B" — only "your partner" or "they"
- 3-4 sentences. This is a turning point, not a summary."""

SHARED_RESOLUTION_BEAT_2_PROMPT = """You are BOND. You've already shared the first part of what you see with this person.

What you shared with them first: {beat_1}
How they responded: {reaction}
The path forward you have in mind: {path_forward}
The misunderstanding between them: {misunderstanding}

Now deliver the second beat — connect what they're experiencing to what's happening on the other side, and offer the direction.

Do not reveal the partner's words. Frame it as what you've observed about how these patterns tend to work.
End with the path_forward — make it feel like something they can actually do, not an instruction.

2-3 sentences. Let it land."""

INTEGRATION_PROMPT = """You are BOND. You've just shared a real insight with this person.
They're responding to it. Your job is to help them sit with what's true — not plan, coach, or move on.

THE RESOLUTION YOU SHARED:
{resolution_message}

THEIR REACTION: {reaction_type}

─── WHAT THIS PHASE IS ───
Integration is stillness after impact. The insight has landed. You are not coaching.
You are not celebrating. You are not mapping next steps. You are holding space.
Statements land harder than questions here. Prefer "That's the loop." over "Do you see the loop?"

─── HOW TO RESPOND ───

ACCEPTANCE — it resonated, they're processing it:
  Go one layer deeper into what THEY just said — not the insight, their actual words.
  One statement. No question. Not forward-facing.
  
  They say: "I think what's holding me back is fear of being seen as clingy"
  WRONG: "What if you reframed reaching out as showing you care rather than being needy?"
  RIGHT: "Fear of being clingy means you've been editing yourself before the conversation even starts."
  
  They say: "I guess I've been waiting for her to reach out first"
  WRONG: "How does it feel to recognize that pattern?"
  RIGHT: "Waiting for her to reach out so you don't have to risk anything. That's the loop."
  
  They announce a plan ("I'm going to text her"):
  WRONG: "That's a great idea! How do you think she'll respond?"
  RIGHT: "That's yours to decide. What feels true from what I shared, separate from the plan?"

PARTIAL — they accept some, resist some:
  Name what landed. Leave the rest alone. No defending, no re-explaining.
  RIGHT: "The part that landed — hold that. The rest can wait."

RESISTANCE — defensive, pushing back:
  Stop completely. Acknowledge the reaction. Do NOT defend the insight.
  RIGHT: "That landed hard — I hear that. It doesn't have to make sense right now."
  The insight is still true. Silence holds more than explanation.

OVERWHELMED — lost, doesn't know what to do:
  One grounding question only — what's true right now, separate from any plan.
  RIGHT: "Set the plan aside. What feels most true from what I shared?"

─── HARD RULES — no exceptions ───
NEVER ask "what would help you", "what do you think would happen if", "how does that feel", "how does that sit with you"
NEVER ask "what if you...", "would you like to...", "are you ready to...", "would you be open to..."
NEVER say "great", "wonderful", "solid", "powerful", "beautiful", "you've got this", "that's brave"
NEVER suggest any action — texting, calling, reaching out, having a conversation
NEVER ask two things at once
NEVER repeat the resolution verbatim
NEVER use their partner's name — only "your partner" or "they"
NEVER go beyond 3 sentences — if it's longer, cut it

{context_block}"""

CLOSING_REFLECTION_PROMPT = """You are BOND. This conversation is winding down naturally.
You want to leave this person with something to carry — one small true thing, not a summary.

THE RESOLUTION YOU SHARED:
{resolution_message}

HOW THEY RESPONDED:
{integration_summary}

Write 2-3 sentences only:
- Pick ONE specific moment or phrase from their conversation — something they said that showed something real
- Offer one quiet observation that connects to that — not advice, not a plan, just what you noticed
- End with something that feels like a door left open, not a door closed

HARD RULES:
- Do NOT summarise the session
- Do NOT say "you've come a long way", "I'm proud of you", "great work today"
- Do NOT use their partner's name
- Do NOT give advice or suggest next steps
- Make it specific to THIS person's actual words — nothing generic
- Quiet and warm. Not a speech. Not a pep talk."""

INTEGRATION_REACTION_PROMPT = """You are reading a response from someone who just received an insight from BOND about their relationship.
Classify their reaction in one word only.

Reactions:
- acceptance: they took it in, it resonated, they're processing positively
- partial: they accept some of it but are pushing back on parts
- resistance: they're defensive, feel judged, rejecting the insight
- overwhelmed: they don't know what to do with it, lost, not defensive just flooded

THEIR RESPONSE:
{message}

RECENT CONTEXT:
{context}

One word only — acceptance / partial / resistance / overwhelmed:"""

WINDING_DOWN_PROMPT = """You are reading a conversation in the integration phase of a relationship support session.
Has this person naturally reached a point of winding down?

Look for signs like:
- Emotional intensity dropping — calmer, less urgent
- Shorter or more settled responses
- Something conclusive: "I think I understand", "I need to sit with this", "yeah", "that helps", "I know what I need to do", "I think I get it now"
- Signs of resolution or acceptance in their last 2-3 messages

ONE WORD ONLY: YES or NO.
- YES: they seem settled, a gentle closing reflection would be welcome
- NO: still processing actively, still asking questions, still unsettled

Note: err on the side of YES if they seem genuinely settled. A closing reflection that comes
slightly early is better than one that never comes.

INTEGRATION CONVERSATION (last 6 messages):
{conversation}

YES or NO:"""


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.45):
    return ChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature
    )


# ─────────────────────────────────────────────
# CONTEXT LOADER
# ─────────────────────────────────────────────

def load_context(db, couple_id: str, user_id: str = None) -> dict:
    context = {}

    if user_id:
        profile_mem = db.query(Memory).filter(
            Memory.owner_id == user_id,
            Memory.memory_type == "profile"
        ).order_by(Memory.created_at.desc()).first()
        if profile_mem:
            try:
                context['profile'] = json.loads(profile_mem.content)
            except Exception:
                context['profile_raw'] = profile_mem.content

        checkin_mem = db.query(Memory).filter(
            Memory.couple_id == couple_id,
            Memory.owner_id == user_id,
            Memory.memory_type == "checkin"
        ).order_by(Memory.created_at.desc()).first()
        if checkin_mem:
            try:
                context['checkin'] = json.loads(checkin_mem.content)
            except Exception:
                context['checkin_raw'] = checkin_mem.content

        # Pattern memory is user-level — load across all couples, most recent
        pattern_mem = db.query(Memory).filter(
            Memory.owner_id == user_id,
            Memory.memory_type == "pattern"
        ).order_by(Memory.created_at.desc()).first()
        if pattern_mem:
            try:
                context['patterns'] = json.loads(pattern_mem.content)
            except Exception:
                context['patterns_raw'] = pattern_mem.content

    from models.database import Couple, User
    couple = db.query(Couple).filter(Couple.id == couple_id).first()
    if couple and user_id:
        partner = next((u for u in couple.users if u.id != user_id), None)
        if partner and partner.is_onboarded:
            partner_mem = db.query(Memory).filter(
                Memory.owner_id == partner.id,
                Memory.memory_type == "profile"
            ).order_by(Memory.created_at.desc()).first()
            if partner_mem:
                try:
                    full = json.loads(partner_mem.content)
                    context['partner_profile'] = {
                        "name": partner.name,
                        "communication_style": full.get("communication_style"),
                        "conflict_style": full.get("conflict_style"),
                        "support_style": full.get("support_style"),
                    }
                except Exception:
                    pass

    rel_mem = db.query(Memory).filter(
        Memory.couple_id == couple_id,
        Memory.owner_id == None,
        Memory.memory_type == "relationship_profile"
    ).order_by(Memory.created_at.desc()).first()
    if rel_mem:
        try:
            context['relationship'] = json.loads(rel_mem.content)
        except Exception:
            context['relationship_raw'] = rel_mem.content

    couple_pattern_mem = db.query(Memory).filter(
        Memory.couple_id == couple_id,
        Memory.owner_id == None,
        Memory.memory_type == "couple_pattern"
    ).order_by(Memory.created_at.desc()).first()
    if couple_pattern_mem:
        try:
            context['couple_pattern'] = json.loads(couple_pattern_mem.content)
        except Exception:
            pass

    past_sessions = db.query(Session).filter(
        Session.couple_id == couple_id,
        Session.is_active == False,
        Session.summary != None
    ).order_by(Session.created_at.desc()).limit(3).all()

    if past_sessions:
        context['past_summaries'] = [
            {"summary": s.summary, "ended_at": s.ended_at or s.created_at}
            for s in past_sessions if s.summary
        ]
        most_recent = past_sessions[0]
        if most_recent.summary_json:
            try:
                sj = json.loads(most_recent.summary_json)
                resolution = sj.get("resolution", "").strip().lower()
                if resolution and resolution not in ("resolved", ""):
                    context['last_resolution'] = {
                        "status": resolution,
                        "ended_at": most_recent.ended_at or most_recent.created_at
                    }
            except Exception:
                pass

    return context


def build_context_block(context: dict, partner_summary: str = None) -> str:
    if not context:
        return "## CONTEXT\nNo prior information about this person yet. Let them lead."

    lines = []

    if 'profile' in context:
        p = context['profile']
        lines.append("## WHO THIS PERSON IS")
        lines.append("Calibrate tone and approach. Do NOT assume why they're here today.")
        for key, label in [
            ('communication_style', 'Communication'),
            ('conflict_style', 'During conflict'),
            ('support_style', 'Feels supported when'),
            ('love_language', 'Feels loved through'),
            ('hope', 'Hopes BOND helps with'),
        ]:
            if p.get(key):
                lines.append(f"- {label}: {p[key]}")
    elif 'profile_raw' in context:
        lines.append("## WHO THIS PERSON IS")
        lines.append(context['profile_raw'])

    if 'relationship' in context:
        r = context['relationship']
        lines.append("\n## RELATIONSHIP CONTEXT")
        lines.append("Background only. Do not reference unless they bring it up.")
        for key, label in [
            ('duration', 'Together'), ('current_status', 'Current state'),
            ('primary_goal', 'Primary goal'), ('biggest_challenge', 'Recurring challenge'),
        ]:
            if r.get(key):
                lines.append(f"- {label}: {r[key]}")

    if 'partner_profile' in context:
        pp = context['partner_profile']
        lines.append(f"\n## YOUR PARTNER ({pp.get('name', 'your partner').upper()})")
        lines.append("Use to help communication. Never take sides.")
        for key, label in [
            ('communication_style', 'How they communicate'),
            ('conflict_style', 'During conflict'),
            ('support_style', 'Feels supported when'),
        ]:
            if pp.get(key):
                lines.append(f"- {label}: {pp[key]}")

    if 'couple_pattern' in context:
        cp = context['couple_pattern']
        lines.append("\n## COUPLE DYNAMIC")
        lines.append("Background awareness only — never reference directly.")
        for key, label in [
            ('dynamic', 'Dynamic'), ('watch_for', 'Watch for'),
            ('attachment_dynamic', 'Attachment dynamic'),
        ]:
            if cp.get(key):
                lines.append(f"- {label}: {cp[key]}")
        for key, label in [
            ('recurring_conflicts', 'Recurring conflicts'),
            ('communication_breakdowns', 'What causes breakdowns'),
            ('what_works', 'What works'),
            ('avoidance_patterns', 'Avoidance patterns'),
        ]:
            if cp.get(key):
                lines.append(f"- {label}: {', '.join(cp[key])}")

    if partner_summary:
        lines.append("\n## WHAT YOUR PARTNER IS WORKING THROUGH RIGHT NOW")
        lines.append("General emotional state — NOT their exact words. Use silently.")
        lines.append(f"Partner's state: {partner_summary}")

    if 'checkin' in context:
        c = context['checkin']
        lines.append("\n## HOW THEY'RE FEELING ENTERING THIS SESSION")
        lines.append("Inform tone. Don't reference directly unless natural.")
        if c.get('mood_score') and c.get('mood_label'):
            lines.append(f"- Mood: {c['mood_label']} ({c['mood_score']}/5)")
        if c.get('intention'):
            lines.append(f"- What's on their mind: {c['intention']}")
    elif 'checkin_raw' in context:
        lines.append("\n## HOW THEY'RE FEELING RIGHT NOW")
        lines.append(context['checkin_raw'])

    if 'last_resolution' in context:
        lr = context['last_resolution']
        status = lr['status']
        ended_at = lr['ended_at']
        from datetime import datetime
        now = datetime.utcnow()
        delta = now - ended_at if ended_at else None
        if delta is None:
            age = "recently"
        elif delta.days >= 7:
            age = f"{delta.days // 7} week{'s' if delta.days // 7 > 1 else ''} ago"
        elif delta.days >= 1:
            age = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        else:
            age = "earlier today"
        lines.append("\n## LAST SESSION STATUS")
        lines.append("Background only — do NOT bring up unless they do.")
        lines.append(f"- Last session ({age}): {status}")

    if 'patterns' in context:
        p = context['patterns']
        lines.append("\n## BEHAVIORAL INTELLIGENCE")
        lines.append("From past sessions. Background awareness — NEVER bring up unprompted.")
        for key, label in [
            ('recurring_themes', 'Recurring themes'),
            ('triggers', 'Known triggers'),
            ('what_helps', 'What helps'),
            ('avoidance_patterns', 'Avoidance patterns'),
        ]:
            if p.get(key):
                lines.append(f"- {label}: {', '.join(p[key])}")
        for key, label in [
            ('attachment_signals', 'Attachment signals'),
            ('self_awareness_trajectory', 'Self-awareness trajectory'),
            ('watch_for', 'Watch for'),
        ]:
            if p.get(key):
                lines.append(f"- {label}: {p[key]}")
    elif 'patterns_raw' in context:
        lines.append("\n## BEHAVIORAL INTELLIGENCE")
        lines.append(context['patterns_raw'])

    if context.get('past_summaries'):
        lines.append("\n## RECENT SESSION HISTORY")
        lines.append("Background only. NEVER reference unless person brings it up first.")
        from datetime import datetime
        now = datetime.utcnow()
        for i, entry in enumerate(context['past_summaries'], 1):
            ended_at = entry["ended_at"]
            summary = entry["summary"]
            delta = now - ended_at if ended_at else None
            if delta is None:
                age = "some time ago"
            elif delta.days >= 30:
                age = f"{delta.days // 30} month{'s' if delta.days // 30 > 1 else ''} ago"
            elif delta.days >= 7:
                age = f"{delta.days // 7} week{'s' if delta.days // 7 > 1 else ''} ago"
            elif delta.days >= 1:
                age = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
            elif delta.seconds >= 3600:
                age = f"{delta.seconds // 3600} hour{'s' if delta.seconds // 3600 > 1 else ''} ago"
            else:
                age = "earlier today"
            lines.append(f"Session {i} ({age}): {summary}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# HISTORY BUILDER
# ─────────────────────────────────────────────

def build_history(messages: list, speaker_name: str = "User") -> list:
    history = []
    for msg in messages[-12:]:
        if msg.sender_id == "ai":
            history.append(AIMessage(content=msg.content))
        else:
            label = msg.sender_id if len(msg.sender_id) < 40 else speaker_name
            history.append(HumanMessage(content=f"{label}: {msg.content}"))
    return history


# ─────────────────────────────────────────────
# JSON HELPER
# ─────────────────────────────────────────────

def _parse_json_safe(raw: str) -> dict:
    text = raw.strip()
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
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────
# SHARED SESSION — MEDIATION ARC
# ─────────────────────────────────────────────

async def extract_core_need(thread_id: str) -> str | None:
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()

        if not messages:
            return None

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        prompt = CORE_NEED_PROMPT.format(conversation="\n".join(lines))
        llm = get_llm(temperature=0.2)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"[CORE NEED LLM ERROR] {e}")
            return None
    finally:
        db.close()




async def analyze_both_threads(session_id: str) -> dict | None:
    """
    Runs combined analysis on raw messages from both threads.
    Detects complementary patterns — works even when people discuss different topics.
    Updates session.analysis_json and session.analysis_ready_count.
    Returns the analysis dict or None on failure.

    Trigger: every 2 combined user messages across both threads.
    Called as a background task from session_router.
    """
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            return None

        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        if len(threads) < 2:
            return None

        thread_a, thread_b = threads[0], threads[1]

        # Fetch raw messages for each thread — last 20 per thread is plenty
        msgs_a = db.query(Message).filter(
            Message.thread_id == thread_a.id
        ).order_by(Message.created_at).limit(20).all()

        msgs_b = db.query(Message).filter(
            Message.thread_id == thread_b.id
        ).order_by(Message.created_at).limit(20).all()

        # Need at least 2 user messages per thread to be meaningful
        user_msgs_a = [m for m in msgs_a if m.sender_id != "ai"]
        user_msgs_b = [m for m in msgs_b if m.sender_id != "ai"]
        if len(user_msgs_a) < 2 or len(user_msgs_b) < 2:
            return None

        def format_thread(msgs, thread):
            lines = []
            for m in msgs:
                speaker = "BOND" if m.sender_id == "ai" else "Person"
                lines.append(f"{speaker}: {m.content}")
            text = "\n".join(lines)

            # Append investigation summary if available — gives BOND the extracted insights
            extras = []
            if thread.story_summary:
                extras.append(f"\nCONFIRMED STORY SUMMARY: {thread.story_summary}")
            if thread.investigation_brief_json:
                try:
                    brief = json.loads(thread.investigation_brief_json)
                    # Pull the bucket labels as context
                    for bucket, key in [
                        ("What they're protecting/avoiding", "bucket_a_their_side"),
                        ("How they read the other person", "bucket_b_their_read"),
                        ("The interaction dynamic", "bucket_c_the_dynamic"),
                    ]:
                        intentions = brief.get(key, [])
                        if intentions:
                            extras.append(f"\n{bucket}: {'; '.join(intentions[:2])}")
                except Exception:
                    pass
            if thread.depth_brief_json:
                try:
                    depth = json.loads(thread.depth_brief_json)
                    intentions = depth.get("depth_intentions", [])
                    if intentions:
                        extras.append(f"\nDeeper context: {'; '.join(intentions[:2])}")
                except Exception:
                    pass

            return text + "".join(extras)

        thread_a_text = format_thread(msgs_a, thread_a)
        thread_b_text = format_thread(msgs_b, thread_b)

        prompt = COMBINED_ANALYSIS_PROMPT.format(
            thread_a=thread_a_text,
            thread_b=thread_b_text,
        )
        llm = get_llm(temperature=0.2)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        parsed = _parse_json_safe(response.content)

        if not parsed or not parsed.get("dynamic"):
            return None

        ready = parsed.get("ready_for_bridge", False)
        confidence = parsed.get("confidence", "low")

        # Ready if explicit true — confidence gates the increment but low confidence
        # does NOT reset the counter (only explicit ready=False resets it)
        actually_ready = ready and confidence in ("medium", "high")

        current_ready_count = session.analysis_ready_count or 0
        if actually_ready:
            session.analysis_ready_count = current_ready_count + 1
        elif not ready:
            # Only reset on explicit False — low confidence without ready=False keeps count
            session.analysis_ready_count = 0
        # ready=True but confidence=low: neither increment nor reset — hold position

        session.analysis_json = json.dumps(parsed)
        db.commit()

        print(f"[ANALYSIS] session={session_id[:8]} ready={actually_ready} confidence={confidence} consecutive={session.analysis_ready_count} dynamic={parsed.get('dynamic', '')[:60]}")
        return parsed

    except Exception as e:
        print(f"[ANALYSIS ERROR] {e}")
        return None
    finally:
        db.close()


def get_latest_analysis(session_id: str) -> dict | None:
    """Pure DB read — no LLM call. Returns latest analysis or None."""
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session or not session.analysis_json:
            return None
        try:
            return json.loads(session.analysis_json)
        except Exception:
            return None
    finally:
        db.close()


def is_ready_for_bridge(session_id: str) -> bool:
    """Returns True if analysis has been ready_for_bridge=true 2+ consecutive times."""
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        return (session.analysis_ready_count or 0) >= 2 if session else False
    finally:
        db.close()



# ─────────────────────────────────────────────
# INVESTIGATION SYSTEM
# ─────────────────────────────────────────────

INVESTIGATION_BRIEF_PROMPT = """\
You have been listening to one person's account of a situation in their relationship.

Here is their story:
{story}

Generate a private investigation brief — what BOND needs to understand to mediate this situation.
NOT a list of questions. A list of INTENTIONS — what understanding BOND is pursuing.
BOND will surface these naturally through conversation.

Three buckets, 1 intention each — the single most important thing to understand. Specific to THIS story.

Return ONLY valid JSON:
{{
  "emotional_register": "open / guarded / deflecting / analytical",
  "pacing": "slow / normal / fast — how quickly to move based on their openness",
  "bucket_a_their_side": [
    "the single most important thing about this person's behaviour or choices to understand"
  ],
  "bucket_b_their_read": [
    "the single most important assumption they're making about the other person"
  ],
  "bucket_c_the_dynamic": [
    "the single most important thing about how this interaction pattern plays out"
  ]
}}

Be specific. Bad: "understand their feelings". Good: "understand what backing off protects them from"\
"""

DEPTH_BRIEF_PROMPT = """\
You have listened to one person through their story and first round of investigation.
Everything collected so far:

{full_context}

Generate a depth brief — what BOND needs to understand about the emotional reality for this person.

Return ONLY valid JSON:
{{
  "depth_intentions": [
    "the most important unspoken emotional truth — what this situation costs them or what they haven't said",
    "what they actually want from the other person that they haven't directly asked for"
  ],
  "handle_with_care": "one sentence — anything BOND should be careful about with this person"
}}

Specific to what this person actually shared. Maximum 2 intentions.\
"""

STORY_SUMMARY_PROMPT = """\
Summarise what this person has shared in one clear sentence.
Start with: "So what's been happening is —"
Specific to what they said. No interpretation, no emotion labels.

THEIR MESSAGES:
{story}

One sentence only.\
"""


def _get_brief_answered(thread) -> dict:
    if not thread.brief_answered_json:
        return {}
    try:
        return json.loads(thread.brief_answered_json)
    except Exception:
        return {}


def _get_next_brief_intention(brief: dict, answered: dict) -> tuple:
    for bucket in ["bucket_a_their_side", "bucket_b_their_read", "bucket_c_the_dynamic"]:
        for i, intention in enumerate(brief.get(bucket, [])):
            key = f"{bucket}_{i}"
            if key not in answered:
                return intention, key
    return None, None


async def classify_post_confirmation_intent(message: str, story_summary: str) -> str:
    """
    Classifies a message sent after story confirmation.
    Returns: 'continuation' | 'meta' | 'confirm' | 'correction' | 'other'
    
    - continuation: adding more to their story
    - meta: asking what's happening / why BOND paused
    - confirm: confirming the summary again
    - correction: adjusting something in the summary
    - other: unrelated / general chat
    """
    llm = get_llm(temperature=0.0)
    prompt = f"""Someone just confirmed their story summary to a relationship support AI. The AI responded warmly and is now generating a follow-up question. The person has sent a new message.

CONFIRMED SUMMARY: {story_summary}

NEW MESSAGE: {message}

Classify the intent of this new message. Return ONLY one word:
- continuation — they are adding more information or context to their story
- meta — they are asking what is happening, what the AI is doing, or why it paused
- confirm — they are confirming or agreeing with the summary again
- correction — they are correcting or changing something in the summary
- other — unrelated or general chat

One word only."""

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        result = resp.content.strip().lower()
        if result in ("continuation", "meta", "confirm", "correction", "other"):
            return result
        return "other"
    except Exception:
        return "other"


async def generate_story_summary(thread_id: str) -> str | None:
    db = SessionLocal()
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return None
        messages = db.query(Message).filter(
            Message.thread_id == thread_id,
            Message.sender_id != "ai"
        ).order_by(Message.created_at).all()
        if not messages:
            return None
        story = " ".join([m.content for m in messages])
        llm = get_llm(temperature=0.2)
        resp = await llm.ainvoke([HumanMessage(content=STORY_SUMMARY_PROMPT.format(story=story))])
        summary = resp.content.strip()
        thread.story_summary = summary
        db.commit()
        return summary
    except Exception as e:
        print(f"[STORY SUMMARY ERROR] {e}")
        return None
    finally:
        db.close()


async def generate_investigation_brief(thread_id: str) -> dict | None:
    db = SessionLocal()
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return None
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()
        if not messages:
            return None
        story = "\n".join([
            f"{"BOND" if m.sender_id == "ai" else "Person"}: {m.content}"
            for m in messages
        ])
        llm = get_llm(temperature=0.3)
        resp = await llm.ainvoke([HumanMessage(content=INVESTIGATION_BRIEF_PROMPT.format(story=story))])
        brief = _parse_json_safe(resp.content)
        if not brief:
            return None
        thread.investigation_brief_json = json.dumps(brief)
        thread.investigation_phase = "extracting"
        thread.brief_answered_json = json.dumps({})
        db.commit()
        print(f"[BRIEF] generated thread={thread_id[:8]} register={brief.get("emotional_register")} pacing={brief.get("pacing")}")
        return brief
    except Exception as e:
        print(f"[INVESTIGATION BRIEF ERROR] {e}")
        return None
    finally:
        db.close()


async def generate_depth_brief(thread_id: str) -> dict | None:
    db = SessionLocal()
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return None
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()
        full_context = "\n".join([
            f"{"BOND" if m.sender_id == "ai" else "Person"}: {m.content}"
            for m in messages
        ])
        llm = get_llm(temperature=0.3)
        resp = await llm.ainvoke([HumanMessage(content=DEPTH_BRIEF_PROMPT.format(full_context=full_context))])
        brief = _parse_json_safe(resp.content)
        if not brief:
            return None
        thread.depth_brief_json = json.dumps(brief)
        thread.investigation_phase = "depth"
        thread.brief_answered_json = json.dumps({})
        db.commit()
        print(f"[BRIEF] depth generated thread={thread_id[:8]}")
        return brief
    except Exception as e:
        print(f"[DEPTH BRIEF ERROR] {e}")
        return None
    finally:
        db.close()


async def mark_intention_done(thread_id: str, key: str, status: str = "answered") -> None:
    db = SessionLocal()
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return
        answered = _get_brief_answered(thread)
        answered[key] = status
        thread.brief_answered_json = json.dumps(answered)

        # Check if all depth intentions are now answered — if so write "complete" to DB
        # This ensures check_phase_transition can read it directly without deriving state
        if thread.investigation_phase == "depth" and thread.depth_brief_json:
            try:
                depth = json.loads(thread.depth_brief_json)
                intentions = depth.get("depth_intentions", [])
                all_done = all(
                    f"depth_{i}" in answered
                    for i in range(len(intentions))
                )
                if all_done:
                    thread.investigation_phase = "complete"
                    print(f"[INVESTIGATION] thread={thread_id[:8]} → complete (all depth intentions answered)")
            except Exception:
                pass

        db.commit()
        if status == "skipped":
            print(f"[BRIEF] intention skipped: {key}")
    finally:
        db.close()


def get_investigation_state(thread_id: str) -> dict:
    db = SessionLocal()
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not thread:
            return {"phase": "story", "story_confirmed": False}
        phase = thread.investigation_phase or "story"
        story_confirmed = bool(thread.story_confirmed)
        answered = _get_brief_answered(thread)

        if phase == "extracting" and thread.investigation_brief_json:
            brief = json.loads(thread.investigation_brief_json)
            next_intention, next_key = _get_next_brief_intention(brief, answered)
            if not next_intention:
                return {"phase": "extracting_complete", "brief": brief,
                        "story_confirmed": story_confirmed,
                        "pacing": brief.get("pacing", "normal"),
                        "register": brief.get("emotional_register", "open")}
            return {"phase": "extracting", "brief": brief,
                    "next_intention": next_intention, "next_key": next_key,
                    "story_confirmed": story_confirmed,
                    "pacing": brief.get("pacing", "normal"),
                    "register": brief.get("emotional_register", "open")}

        if phase == "depth" and thread.depth_brief_json:
            brief = json.loads(thread.depth_brief_json)
            intentions = brief.get("depth_intentions", [])
            next_intention, next_key = None, None
            for i, intention in enumerate(intentions):
                key = f"depth_{i}"
                if key not in answered:
                    next_intention, next_key = intention, key
                    break
            if not next_intention:
                return {"phase": "complete", "story_confirmed": story_confirmed,
                        "handle_with_care": brief.get("handle_with_care", "")}
            return {"phase": "depth", "next_intention": next_intention, "next_key": next_key,
                    "handle_with_care": brief.get("handle_with_care", "")}

        return {"phase": phase if phase not in ("extracting", "depth") else "story"}
    except Exception as e:
        print(f"[INVESTIGATION STATE ERROR] {e}")
        return {"phase": "story"}
    finally:
        db.close()

async def generate_bridge_lead_in(session_id: str, user_id: str) -> str | None:
    """
    Returns the pre-bridge question for one person directly from analysis.
    No LLM wrapper — prevents hallucination of invented details.
    """
    analysis = get_latest_analysis(session_id)
    if not analysis:
        return None

    db = SessionLocal()
    try:
        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        if len(threads) < 2:
            return None

        # Find which thread belongs to this user — don't rely on order
        user_thread = next((t for t in threads if t.user_id == user_id), None)
        if not user_thread:
            return None

        # person_a is whichever thread was created first
        threads_sorted = sorted(threads, key=lambda t: t.created_at)
        is_person_a = threads_sorted[0].user_id == user_id

        bridge_question = (
            analysis.get("bridge_question_a")
            if is_person_a
            else analysis.get("bridge_question_b")
        )

        if not bridge_question:
            return None

        print(f"[BRIDGE LEAD-IN] person_a={is_person_a} user={user_id[:8]} question={bridge_question[:60]}")
        return bridge_question

    except Exception as e:
        print(f"[BRIDGE LEAD-IN ERROR] {e}")
        return None
    finally:
        db.close()


async def check_phase_transition(session_id: str) -> str:
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            return "listening"

        current_phase = session.mediation_phase or "listening"
        threads = db.query(Thread).filter(Thread.session_id == session_id).all()

        if len(threads) < 2:
            return current_phase

        if current_phase == "listening":
            # Bridging fires when:
            # 1. Both threads have at least 6 user messages (BOND has heard enough from each person)
            # 2. Analysis has confirmed a clear pattern at least twice (ready_count >= 2)
            # Does NOT require investigation_phase == "complete" — that's unreliable.
            # Investigation runs in background and enriches analysis, but is not the gate.
            # Both people don't need to be online simultaneously — bridging is async-safe.
            msgs = [t.message_count or 0 for t in threads]
            both_have_enough = all(m >= 6 for m in msgs)
            analysis_ready = (session.analysis_ready_count or 0) >= 2

            if both_have_enough and analysis_ready:
                session.mediation_phase = "bridging"
                db.commit()
                print(f"[PHASE] {session_id[:8]} → bridging (msgs={msgs[0]}+{msgs[1]} analysis_ready_count={session.analysis_ready_count})")
                return "bridging"

        elif current_phase == "bridging":
            consents = json.loads(session.bridge_consents or "[]")
            user_ids = [t.user_id for t in threads]
            if all(uid in consents for uid in user_ids):
                session.mediation_phase = "resolution"
                db.commit()
                print(f"[PHASE] {session_id[:8]} → resolution")
                return "resolution"

        return current_phase
    finally:
        db.close()


async def record_bridge_consent(session_id: str, user_id: str) -> bool:
    db = SessionLocal()
    try:
        session = db.query(Session).filter(Session.id == session_id).first()
        if not session:
            return False

        consents = json.loads(session.bridge_consents or "[]")
        if user_id not in consents:
            consents.append(user_id)
            session.bridge_consents = json.dumps(consents)
            db.commit()

        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        user_ids = [t.user_id for t in threads]
        return all(uid in consents for uid in user_ids)
    finally:
        db.close()


async def generate_resolution_message(session_id: str, user_id: str) -> str | None:
    db = SessionLocal()
    try:
        threads = db.query(Thread).filter(Thread.session_id == session_id).all()
        if len(threads) < 2:
            return None
        this_thread = next((t for t in threads if t.user_id == user_id), None)
        partner_thread = next((t for t in threads if t.user_id != user_id), None)
        if not this_thread or not partner_thread:
            return None
        threads_sorted = sorted(threads, key=lambda t: t.created_at)
        is_person_a = threads_sorted[0].user_id == user_id
    finally:
        db.close()

    analysis = get_latest_analysis(session_id)
    if not analysis:
        analysis = await analyze_both_threads(session_id)
    if not analysis:
        return None

    # Pull four-layer fields for this person
    person_key = "person_a" if is_person_a else "person_b"
    person_data = analysis.get(person_key, {})

    prompt = SHARED_RESOLUTION_PROMPT.format(
        event=person_data.get("event", ""),
        interpretation=person_data.get("interpretation", ""),
        unmet_need=person_data.get("unmet_need", ""),
        their_contribution=person_data.get("their_contribution", ""),
        complementary_pattern=analysis.get("complementary_pattern", ""),
        misunderstanding=analysis.get("misunderstanding", ""),
        path_forward=analysis.get("path_forward", ""),
    )
    llm = get_llm(temperature=0.5)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[RESOLUTION BEAT 1 ERROR] {e}")
        return None


async def generate_resolution_beat_2(session_id: str, user_id: str, beat_1: str, reaction: str) -> str | None:
    """
    Second beat of resolution — after person has responded to beat 1.
    Connects their experience to the dynamic and delivers path_forward.
    """
    analysis = get_latest_analysis(session_id)
    if not analysis:
        return None

    prompt = SHARED_RESOLUTION_BEAT_2_PROMPT.format(
        beat_1=beat_1,
        reaction=reaction,
        path_forward=analysis.get("path_forward", ""),
        misunderstanding=analysis.get("misunderstanding", ""),
    )
    llm = get_llm(temperature=0.5)
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[RESOLUTION BEAT 2 ERROR] {e}")
        return None


# ─────────────────────────────────────────────
# INTEGRATION PHASE FUNCTIONS
# ─────────────────────────────────────────────

async def detect_integration_reaction(message: str, recent_messages: list, speaker_name: str) -> str:
    """
    Classifies the person's reaction to the resolution message.
    Returns: acceptance / partial / resistance / overwhelmed
    """
    try:
        lines = []
        for msg in recent_messages[-4:]:
            speaker = "BOND" if msg.sender_id == "ai" else speaker_name
            lines.append(f"{speaker}: {msg.content}")
        context = "\n".join(lines)

        prompt = INTEGRATION_REACTION_PROMPT.format(
            message=message,
            context=context
        )
        llm = get_llm(temperature=0.1)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip().lower()
        if raw in ("acceptance", "partial", "resistance", "overwhelmed"):
            return raw
        # Fallback — try to find a keyword
        for kw in ("acceptance", "partial", "resistance", "overwhelmed"):
            if kw in raw:
                return kw
        return "partial"  # safe default
    except Exception as e:
        print(f"[INTEGRATION REACTION ERROR] {e}")
        return "partial"


async def should_offer_close(thread_id: str, integration_count: int) -> bool:
    """
    Returns True if this person is naturally winding down and a
    closing reflection would land well.
    Only fires after 4+ integration exchanges.
    """
    if integration_count < 4:
        return False
    try:
        db = SessionLocal()
        try:
            messages = db.query(Message).filter(
                Message.thread_id == thread_id
            ).order_by(Message.created_at.desc()).limit(10).all()
            messages = list(reversed(messages))
        finally:
            db.close()

        if not messages:
            return False

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        prompt = WINDING_DOWN_PROMPT.format(conversation="\n".join(lines))
        llm = get_llm(temperature=0.1)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip().upper().startswith("YES")
    except Exception as e:
        print(f"[WIND DOWN CHECK ERROR] {e}")
        return False


async def generate_closing_reflection(thread_id: str, resolution_message: str) -> str | None:
    """
    Generates a warm closing reflection for this person.
    Called when winding down is detected in integration phase.
    """
    try:
        db = SessionLocal()
        try:
            # Get last 8 integration messages for summary
            messages = db.query(Message).filter(
                Message.thread_id == thread_id
            ).order_by(Message.created_at.desc()).limit(8).all()
            messages = list(reversed(messages))
        finally:
            db.close()

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        integration_summary = "\n".join(lines)

        prompt = CLOSING_REFLECTION_PROMPT.format(
            resolution_message=resolution_message,
            integration_summary=integration_summary
        )
        llm = get_llm(temperature=0.55)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"[CLOSING REFLECTION ERROR] {e}")
        return None


# Module-level store for pending analysis results
BRIDGE_READINESS_PROMPT = """You are reading a private conversation between one person and BOND.
Has this person reached genuine clarity or openness?
Look for: acknowledging own role, desire to reconnect, reduced defensiveness, vulnerability.

ONE WORD ONLY: YES or NO.
- YES: real self-awareness or openness in the last few messages
- NO: still venting, blaming, or no openness shown

CONVERSATION (last 6 messages):
{conversation}

YES or NO:"""

BRIDGE_INSIGHT_PROMPT = """You are BOND. This person has reached a moment of clarity.
Plant a seed — help them see their partner differently without revealing what the partner said.

2-3 warm sentences:
- Acknowledge what they've worked through
- Offer a gentle reframe toward their partner's possible experience
- End with something to sit with — not advice, just a thought

Do NOT reference partner's words. Do NOT say "your partner told me".
Speak only from what THIS person shared.

THEIR CONVERSATION:
{conversation}"""


async def check_bridge_readiness(thread_id: str) -> bool:
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at.desc()).limit(6).all()
        messages = list(reversed(messages))

        if len(messages) < 4:
            return False

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        prompt = BRIDGE_READINESS_PROMPT.format(conversation="\n".join(lines))
        llm = get_llm(temperature=0.1)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip().upper().startswith("YES")
        except Exception as e:
            print(f"[BRIDGE READINESS LLM ERROR] {e}")
            return False
    finally:
        db.close()


async def generate_bridge_insight(thread_id: str) -> str | None:
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        prompt = BRIDGE_INSIGHT_PROMPT.format(conversation="\n".join(lines))
        llm = get_llm(temperature=0.5)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"[BRIDGE INSIGHT LLM ERROR] {e}")
            return None
    finally:
        db.close()


# ─────────────────────────────────────────────
# THREAD SUMMARY GENERATOR
# ─────────────────────────────────────────────

THREAD_SUMMARY_PROMPT = """You are reading a private conversation between one person and BOND.
Summarise what this person is feeling and what they seem to need right now.
2-3 sentences. Third person. Emotional state focus. No direct quotes.

CONVERSATION:
{conversation}

Summary only. No preamble."""


async def generate_thread_summary(thread_id: str) -> str | None:
    db = SessionLocal()
    try:
        messages = db.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.created_at).all()

        if not messages:
            return None

        lines = [
            f"{'BOND' if m.sender_id == 'ai' else 'Person'}: {m.content}"
            for m in messages
        ]
        prompt = THREAD_SUMMARY_PROMPT.format(conversation="\n".join(lines))
        llm = get_llm(temperature=0.2)
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"[THREAD SUMMARY LLM ERROR] {e}")
            return None
    finally:
        db.close()


# ─────────────────────────────────────────────
# SESSION SUMMARY GENERATOR
# ─────────────────────────────────────────────