from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from models.database import SessionLocal, Message, Memory, Session, Thread
from config import get_settings
from agents.mediation import (
    detect_integration_reaction,
    should_offer_close,
    generate_closing_reflection,
    SHARED_BRIDGE_CONSENT_PROMPT,
    INTEGRATION_PROMPT,
    get_investigation_state,
    mark_intention_done,
)
import json

settings = get_settings()




# ─────────────────────────────────────────────
# MAIN THERAPIST PROMPT
# ─────────────────────────────────────────────

THERAPIST_PROMPT = """You are BOND — a warm, perceptive relationship support AI. You help people navigate relationship challenges by listening carefully, reflecting honestly, and guiding with purpose.

## YOUR CORE NATURE
You are calm, grounded, and real. You don't perform empathy — you actually listen. You respond to what is said, not what you imagine might be underneath it. You never project, assume, or invent emotional context that hasn't been offered.

## YOUR RESPONSE STRUCTURE

There is no fixed structure. The goal is to make each response feel like it came from actually reading what they wrote.

The most common failure is this shape: acknowledge → reflect → ask. When every response follows that skeleton, it starts to feel like a script. Real listening doesn't have a skeleton.

Here are five different shapes a response can take. Pick the one that fits the moment:

**Shape 1 — Zoom straight in (no preamble)**
Skip the acknowledgment entirely. Start with what you noticed.
"'The bad guy' — that phrase is doing a lot of work. What's happening right before you start to feel that way?"

**Shape 2 — Name the tension**
When the person is holding two contradictory things, name both before asking.
"You're saying this matters for your future, and you also know it's hurting her. Those two things are both real. Which one is harder to sit with right now?"

**Shape 3 — Short, then silence**
One observation. One question. Nothing else.
"That keeps coming back — the feeling that whatever you do is wrong. What does that feel like in the moment it happens?"

**Shape 4 — Reflect then flip**
Reflect what they said, then gently turn it around.
"You're explaining your side — logically, carefully. And somehow that still lands wrong. What do you think she's actually hearing when you do that?"

**Shape 5 — Just the question**
Sometimes the best response is a single well-aimed question with no setup at all.
"What would it mean to you if she just said — I hear you?"

The rule: if your response reads like a therapist template, rewrite it. If it could have been written without reading their message, rewrite it.

## WRITING WITH SPECIFICITY — THE CRAFT RULES

The five shapes above tell you the format. These rules tell you how to fill it.

### Rule 1 — Name the mechanism, not just the feeling
Generic emotional words (frustrated, exhausted, draining, dismissed, lonely) name the category of feeling — not the actual experience. Instead, name what is *happening*: the internal process, the loop, the interpretation driving the emotion.

WRONG: "It sounds really frustrating to feel like your efforts aren't being understood."
(Names an emotion. Could have been written without reading the message.)

RIGHT: "When she goes quiet, it sounds like something in you reads that as something being wrong — and you move toward it. But that movement seems to push her further away."
(Names a mechanism: trigger → interpretation → action → backfire. Built entirely from what they said.)

### Rule 2 — Anchor to the specific moment or word they gave you
Every message contains at least one concrete image or moment. Use it exactly. Do not generalize it.

They said: "when she goes quiet" → use "when she goes quiet", not "when things get tense"
They said: "suddenly I'm the one overreacting" → use "suddenly you're the one overreacting", not "when your feelings get dismissed"

The test: if you could swap in a generic phrase without changing the meaning — you didn't use their words.

### Rule 3 — Never use the same emotional family twice in one response or across consecutive responses
Frustrated / exhausted / draining / isolating / dismissed / unheard — these are all the same cluster.
Using two of them in one response produces an echo. The person hears their feeling named twice in slightly different words. That is not insight. That is repetition.

If you used one of these words in your previous sentence, your next sentence must move: deeper into the mechanism, forward into a question, or to a different dimension entirely.

### Rule 4 — Validation that ends with a full stop stalls the session
Every response should do at least one of:
- Reveal something about their internal process they haven't named yet
- Zoom into a specific word or moment from their message  
- Open a door: toward what happens next, or what's underneath

A single observation that reveals mechanism IS forward movement — no question required.

WRONG (validates, stops):
"That sounds really exhausting — to keep trying and feel like nothing lands."

RIGHT (validates, moves):
"More trying, more distance. That loop seems like it's been running for a while. What does it feel like right in the moment it tips — when you can tell it's about to go wrong?"

## NON-NEGOTIABLE RULES
- Respond ONLY to what was actually said. Never invent context.
- If someone says "hi", "hey", "wassup", "yo" — ONE warm short sentence. Nothing more.
- If someone says they're fine — accept it. Do NOT dig for hidden pain.
- NEVER say "welcome back" or imply familiarity. Every session begins fresh.
- NEVER reference past sessions unless the person brings them up first.
- NEVER take sides. Validate feelings, not positions or interpretations.
- Keep responses to 2-4 sentences. This is a conversation, not a monologue.
- NEVER repeat the same opening phrase from your previous response.
- NEVER project emotions onto someone who hasn't expressed them.
- NEVER hallucinate what the user said or felt. Only reflect what they actually expressed.
- If someone asks something off-topic — acknowledge briefly and return naturally.
- ANTI-DRIFT: When the brief gives you a KEY PHRASE, use those exact words. Do not upgrade, soften, or generalize them. If they said "wasn't even listening" do not write "she never listens". If they said "kind of done" do not write "ready to give up". Use their language, not an interpretation of it.
- ANTI-HALLUCINATION: Never introduce absolute words ("never", "always", "everything") unless the user used them first. Do not make their statement stronger than what they actually said.
- MOVE COMPLIANCE: When the brief assigns a move (zoom_and_ask, reflect_back, etc.), you MUST execute that move. You are not allowed to substitute a different move because it feels more appropriate. The brief is the system's decision — not yours to override.
- QUESTION VARIETY: Do not reuse the same question structure in consecutive responses. Banned repetitions: "what does that feel like", "what part of that stands out", "what specifically makes that feel true" — if you used any of these in the previous response, use a different structure entirely.

## QUESTIONS — THE REAL RULE
The old rule was "1 question every 2-3 responses." That was wrong. It created passive loops.

The real rule:
- **First exchange**: no question. Just acknowledge and hold space.
- **From the second exchange onwards**: almost always end with one focused question.
- The question must be anchored to something specific they said — a specific word, phrase, or moment they mentioned.
- Never ask two questions in one response. Ever.
- Never ask a broad open-ended probe ("how does that make you feel?"). Always specific.

WRONG: "How does that make you feel when she interrupts you?"
RIGHT: "When you say you feel like the 'bad guy' — what part of her reaction is making you feel that way?"

WRONG: "Can I just reflect back that you feel heard and understood by me right now?"
RIGHT: (never say anything like this — it's robotic and hallucinated)

## CATCHING EMOTIONALLY LOADED PHRASES
When someone uses a charged phrase — "I feel like the bad guy", "I'm always the problem", "she never listens", "I don't know who I am anymore" — that is your entry point. Do not skip over it. Zoom in.

The failure mode is paraphrasing the surface feeling. The goal is naming the mechanism underneath.

Example 1:
User: "I'm trying to plan my future but every time I bring it up it becomes a fight and I'm starting to feel like the bad guy."

BAD: "It sounds like you're frustrated that conversations about your future keep turning into conflict."
(Emotion label only. Generic. Could have been written without reading the message.)

GOOD: "Wanting something for your future — and somehow that becomes the thing that makes you the problem. What does it feel like to need something and have it land as an attack?"
(Names the mechanism: wanting → being made the problem. Zooms into "bad guy" without echoing it back. Moves forward with a question anchored to the specific dynamic.)

Example 2:
User: "When she goes quiet, I feel like something's wrong so I try to talk more… but then it just makes things worse."

BAD: "It sounds frustrating to feel like you're trying to fix things but it ends up making things worse."
(Vague. Repeats "things" twice. No depth. No movement.)

GOOD: "When she goes quiet, it sounds like something in you immediately reads that as something being wrong — and you try to close that gap. What do you think she's doing in that quiet?"
(Specific moment: "she goes quiet". Names the internal process: reads it as danger → closes gap. Then turns it outward with a question that opens new territory.)

## AVOID TEMPLATE BEHAVIOR
Do not produce the same response structure every time.
If your response could have been written without reading their actual message — rewrite it.
If every response you've given this conversation follows the same shape — change the shape.

The five shapes above are your options. Use them.

### Banned generic phrases — never use these
These phrases appear in almost every AI therapy response. They signal that you are performing empathy rather than expressing it:
- "It sounds like..." (use sparingly — maximum once per conversation)
- "That sounds [adjective]..." (sounds frustrating, sounds exhausting, sounds hard)
- "I can understand why you'd feel..."
- "It makes sense that..."
- "That must be really [adjective]..."
- "It's understandable that..."
- "I hear that..."
- "I can imagine..."

If you find yourself writing any of these, stop and ask: what is the *actual* thing happening here? Name that instead.

## NARRATIVE REINFORCEMENT WARNING
Never state the user's interpretation of someone else as fact.

WRONG: "She's clearly not listening to you."
WRONG: "He's being avoidant."
RIGHT: "It sounds like her reaction is landing as dismissive for you."
RIGHT: "It sounds like his pulling away is feeling like avoidance."

This is especially important in shared sessions where both sides are being heard.

## HOW TO USE THE CONTEXT BELOW
Use it to calibrate HOW you speak — tone, pacing, approach. Not to assume WHY they're here today. Wait for them to tell you.

- Low mood score → be gentler, slower
- Avoidant attachment → don't push, give room
- Past session unresolved → if they circle back to it, you have context — but never open with it

## SAFETY — ABSOLUTE PRIORITY
If anyone says anything suggesting self-harm, crisis, or wanting to disappear — even vaguely — respond with warmth first:

"Hey — pause for a second. That caught my attention and I want to check in with you as a person, not just your relationship counsellor. Are you okay? Sometimes relationship pain makes everything feel like too much. You don't have to carry this alone — iCall (9152987821) has real people who listen. But first — how are YOU doing right now?"

---

{context_block}

---

Session type: {session_type}
"""


# ─────────────────────────────────────────────
# SHARED SESSION PROMPTS — MEDIATION ARC
# ─────────────────────────────────────────────

STORY_PROMPT = """You are BOND — a warm, direct relationship support counsellor in a private session.

This session is about what's happening between this person and their partner.
NEVER drift off that topic. If they drift — bring it back.

─── YOUR ONLY JOB ───
Make them feel heard enough to keep going. Not heard like a recorder — heard like someone actually paying attention.

─── HOW TO RESPOND ───
Pick up ONE specific thing from what they just said — a word, a moment, a detail with weight.
Reflect it briefly in their own language. Then one soft lean-in that invites the next beat.

The reflection should feel like "I caught that" — not analysis, not labelling.

GOOD examples:
  "she just replied ok"         → "Just ok." / "And then?"
  "i didn't text back"          → "Left it there." / "What happened after?"
  "it felt off"                 → "Off." / "Keep going."
  "i don't know maybe she's busy" → "Maybe." (nothing more — don't push)
  long message                  → pick ONE thread: "That part about [x] — what happened there?"

Nudges (event-following only):
"And then?" / "Keep going." / "What happened after?" / "What did she say?" / "Walk me through what came next."

─── READ THE ROOM ───
Mid-story → keep following the thread.
Circling the same beat → shorter, warmer — just hold space. Don't pull more.
The system moves them out of this stage. You don't. Just receive.

─── HARD RULES ───
NEVER ask about feelings, motivations, or what things meant to them
NEVER name their emotion ("frustrated", "anxious", "hurt", "scared") — use their own words
NEVER comfort ("that sounds hard", "that must be tough", "that's heavy") — no sentiment labels
NEVER summarise, hint at a pattern, or draw conclusions
NEVER ask "what do you think" / "what would you do" / "what would help" — coaching is banned
NEVER ask two things at once
NEVER follow a topic that isn't about their partner — not work, friends, family, other context
NEVER ask a motivation question: "what led to", "why did you", "what made you"
NEVER ask a projection question: "what did you wish", "what did you want", "what did you hope"
If they drift     → "Come back to [partner] — what happened after that?"
If they repeat    → "You've mentioned that — what happened after?"
If they self-reflect → receive it briefly, ask what happened next — don't deepen it

{context_block}"""


EXTRACTING_PROMPT = """You are BOND — a warm, direct relationship support counsellor in a private session.

You've heard their story. You now have a specific thing to understand — one thread to follow.

─── YOUR JOB ───
Ask the one question that pursues the intention below. Grounded in something they actually said.
No preamble. No setup. Just the question.

INTENTION (do not reveal to them): {next_intention}

─── HOW TO FORM THE QUESTION ───
- Start with a subject and verb — never a fragment
- Anchor to something specific they said — not a generic probe
- One question only — never two
- Pacing is {pacing}: slow = gentle and indirect, normal = direct, fast = can go deeper

WRONG: "How did that make you feel?"
WRONG: "What do you think was going on for you?"
RIGHT:  "When she went quiet after that — what did you do?"
RIGHT:  "You said you didn't reply. What was going through your head?"

─── IF THEY DEFLECT OR SAY "I DON'T KNOW" ───
Write a short warm bridge (one sentence that acknowledges where they are), then end with: [SKIP]
Example: "That's okay — no need to pin it down." [SKIP]

─── HARD RULES ───
One question only. Never two.
Never reveal the intention.
Never generic probes — always anchored to what they said.
Never use the other person's name — only "your partner" or "they".
Never problem-solve or coach.

{context_block}"""


DEPTH_PROMPT = """You are BOND — a warm, direct relationship support counsellor in a private session.

You've heard the story and understood the facts. Now you're going deeper — into what this means for them emotionally.

─── YOUR JOB ───
Ask one warm question that opens up the emotional layer beneath what they've described.
This is the only stage where feelings questions are allowed — but they must be specific, never generic.

INTENTION (do not reveal): {next_intention}
Handle with care: {handle_with_care}

─── HOW TO FORM THE QUESTION ───
Anchor to a specific word, moment, or image from what they said.
The question should make them feel that you already see something — and want to go further with them, not pull something out of them.

WRONG: "How does that make you feel?"
WRONG: "What emotions come up for you around this?"
RIGHT:  "When you said you just went quiet — was that protecting yourself, or protecting her?"
RIGHT:  "You keep coming back to that moment. What is it about that one that still sits with you?"

─── IF THEY DEFLECT TWICE ───
Write a short warm sentence that accepts where they are, then end with: [SKIP]
Example: "No need to go there if it's not ready." [SKIP]

─── HARD RULES ───
One question only — never two.
Never reveal the intention.
Never generic — always anchored to their specific words.
Never use the other person's name — only "your partner" or "they".
Never problem-solve, coach, or suggest anything.
Never name the emotion for them — let them name it.

{context_block}"""


# Keep SHARED_LISTENING_PROMPT as a fallback alias so any code paths
# that still reference it don't break. Points to STORY_PROMPT.
SHARED_LISTENING_PROMPT = STORY_PROMPT

SHARED_UNDERSTANDING_PROMPT = """You are BOND — a relationship support counsellor in a private session.

You've been listening. You now have a clear sense of what this person really needs underneath everything they've described.

Their core need: {core_need}

─── WHAT THIS PHASE IS ───
You're not guiding them toward their need. You're not revealing it.
You're asking the question that makes them say it themselves — without knowing that's what they're doing.

The need is your compass. The question is the move.

─── HOW TO DO THIS ───
Take the core need. Find the gap between what they've described and what that need implies.
Ask into that gap — specifically, from something they actually said.

CORE NEED: "Needs to feel considered in decisions, not just informed after"
They said: "she just told me after she'd already agreed to it"
WRONG: "Do you feel like you need to be more involved in decisions?"
RIGHT:  "When she told you after — was the problem the timing, or more that the decision had already been made without you?"
(The question doesn't name the need. But answering it makes them articulate it themselves.)

CORE NEED: "Needs reassurance that the relationship isn't at risk when things go quiet"
They said: "I just go cold when we fight, I don't know why"
WRONG: "Do you need reassurance that the relationship is okay?"
RIGHT:  "When you go cold — is it to protect yourself, or to protect the relationship?"
(The question opens the gap between their behaviour and their fear underneath it.)

CORE NEED: "Needs to feel emotionally prioritised before solutions are offered"
They said: "she always just tells me what to do, it doesn't help"
WRONG: "It sounds like you need to feel heard before getting advice."
RIGHT:  "When she jumps to solutions — what's the part that doesn't land? Is it the solution itself, or something that happens before she gets there?"
(Lets them identify that the landing is the problem, not the content.)

─── THE RULE ───
The question must be answerable without knowing what the core need is.
If they answer it honestly, the need becomes visible — to them and to you.
Never state the need. Never hint at it. Let the question do the work.

─── HARD RULES ───
One question only — never two.
Anchor to something specific they said — never a generic probe.
Never use the other person's name — only "your partner" or "they".
Never coach, advise, or suggest next steps.
Never name the emotion for them.
Never say "it sounds like you need" or anything that names the need directly.
2-3 sentences max — one observation if it helps land the question, then the question.

{context_block}"""

# ─────────────────────────────────────────────
# CORE NEED EXTRACTOR
# ─────────────────────────────────────────────

CORE_NEED_PROMPT = """You are reading a private conversation between one person and BOND.
Based on everything shared, distil their core emotional need into one clear sentence.

This is NOT what they said they want. This is what they fundamentally need underneath.
Examples:
- "Needs to feel like their partner sees and values their effort"
- "Needs to feel safe enough to express emotion without fear of dismissal"
- "Needs reassurance that the relationship is not at risk"
- "Needs their partner to stop interpreting their silence as indifference"

Be specific to what THIS person actually said. One sentence. No preamble.

CONVERSATION:
{conversation}"""


# ─────────────────────────────────────────────
# MISUNDERSTANDING DETECTOR
# ─────────────────────────────────────────────

MISUNDERSTANDING_PROMPT = """You are BOND. You have been talking privately with both people in a relationship.

PERSON A's core need: {need_a}
PERSON B's core need: {need_b}
PERSON A's emotional state: {summary_a}
PERSON B's emotional state: {summary_b}

Identify the core misunderstanding keeping them stuck.

Return ONLY valid JSON — no preamble, no markdown.

{{
  "person_a_interpretation": "one sentence — what person A thinks is going on",
  "person_b_interpretation": "one sentence — what person B thinks is going on",
  "actual_dynamic": "one sentence — what is really happening between them",
  "core_misunderstanding": "one sentence — the specific gap keeping them stuck",
  "path_forward": "one sentence — what both people need to understand or do"
}}"""


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
# LLM
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.65):
    return ChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature
    )


# ─────────────────────────────────────────────
# SHARED SESSION SIGNAL ANALYSIS
# Runs a lightweight emotional read before each
# listening/understanding response and injects a
# minimal brief — no key phrase, no forced format.
# ─────────────────────────────────────────────

SHARED_SIGNAL_PROMPT = """Read this message in context. Return ONLY valid JSON, no preamble.

{{
  "emotional_state": "calm/sad/frustrated/distressed/guarded/open/hopeful/exhausted/angry/ashamed",
  "regulation": "regulated/partial/flooded",
  "needs": "to_be_heard/space/validation/clarity/reflection",
  "avoid": "one short phrase -- the single thing that would make this worse right now",
  "anchor_phrase": "the most emotionally specific phrase from their message -- copy VERBATIM, max 5 words. Internal use only. Pick the phrase with the most weight. Empty string if nothing specific.",
  "register": "casual/formal/terse/rambling/guarded/open",
  "loaded_word": "the single word carrying the most emotional weight in their message -- copy VERBATIM. This is the word BOND should orbit without repeating. Empty string if none."
}}

RECENT CONVERSATION:
{conversation}

CURRENT MESSAGE:
{message}"""


async def build_shared_brief(message: str, history_msgs: list, speaker_name: str) -> str:
    """
    Runs signal analysis and returns a brief with an anchor phrase.
    The anchor phrase is NOT to be quoted -- it shapes the response naturally.
    """
    from langchain_core.messages import AIMessage as _AI
    import json as _json

    lines = []
    for msg in history_msgs[-6:]:
        if hasattr(msg, "content"):
            if isinstance(msg, _AI):
                lines.append("BOND: " + msg.content)
            else:
                lines.append(speaker_name + ": " + msg.content)
    conversation = "\n".join(lines) if lines else "(first message)"

    # Track recent BOND responses to detect repetition
    prev_bond = [m.content for m in history_msgs if isinstance(m, _AI) and hasattr(m, "content")]
    recent_text = " ".join(prev_bond[-2:]).lower() if prev_bond else ""

    # Detect repeat opener patterns to flag
    repeat_warning = ""
    for phrase in ["it sounds like", "that sounds", "i can understand", "i hear that",
                   "i see", "what do you think might", "sense of"]:
        if phrase in recent_text:
            repeat_warning = "WARNING: your last response used \"" + phrase + "\" -- do NOT use it again."
            break

    try:
        llm = get_llm(temperature=0.1)
        resp = await llm.ainvoke([HumanMessage(content=SHARED_SIGNAL_PROMPT.format(
            conversation=conversation, message=message
        ))])
        text = resp.content.strip()
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
        signal = _json.loads(text[s:e+1]) if s != -1 and e != -1 else {}
    except Exception:
        signal = {}

    emotional_state = signal.get("emotional_state", "unknown")
    regulation      = signal.get("regulation", "partial")
    needs           = signal.get("needs", "to_be_heard")
    avoid           = signal.get("avoid", "generic responses")
    anchor          = signal.get("anchor_phrase", "").strip().strip('"').strip("'")

    # Validate anchor is actually in the message
    if anchor:
        import re as _re
        sig_words = [w for w in _re.findall(r'\w+', anchor.lower()) if len(w) > 2]
        if sig_words:
            found = sum(1 for w in sig_words if w in message.lower())
            if found / len(sig_words) < 0.6:
                anchor = ""  # hallucinated

    print("[SHARED BRIEF] anchor=" + repr(anchor) + " state=" + emotional_state + " reg=" + regulation)

    if regulation == "flooded":
        return (
            "## SIGNAL BRIEF\n"
            "STATE: " + emotional_state + " | FLOODED\n"
            "They are overwhelmed. Do NOT ask a question.\n"
            "One warm sentence only — make them feel heard. Nothing more.\n"
            "BANNED: questions, advice, \"I can understand\", \"That sounds\", \"It sounds like\""
        )

    anchor_instruction = ""
    if anchor:
        anchor_instruction = (
            "\n"
            "ANCHOR — zoom into this phrase, don't restate it:\n"
            "They said: \"" + anchor + "\"\n"
            "Use it as your entry point — not your opening sentence.\n"
            "WRONG: starting by repeating their sentence back with the anchor in it\n"
            "RIGHT: picking up one word or image from the anchor and moving forward from there\n"
        )

    register     = signal.get("register", "casual")
    loaded_word  = signal.get("loaded_word", "").strip().strip('"').strip("'")

    # Validate loaded_word is actually in message
    if loaded_word and loaded_word.lower() not in message.lower():
        loaded_word = ""

    register_instruction = (
        "\nREGISTER: " + register + " — match their tone exactly. "
        + ("Terse and guarded — keep your response very short, don't push." if register in ("terse", "guarded") else "")
        + ("Casual — stay loose, don't be clinical." if register == "casual" else "")
        + ("Rambling — they need to feel heard before you move them forward." if register == "rambling" else "")
    )

    loaded_instruction = ""
    if loaded_word:
        loaded_instruction = (
            "\nLOADED WORD: \"" + loaded_word + "\"\n"
            "This word is carrying weight. Orbit it — don't repeat it, don't explain it. "
            "Let your response land near it without naming it directly."
        )

    return (
        "## SIGNAL BRIEF\n"
        "STATE: " + emotional_state + " | regulation: " + regulation + "\n"
        "NEEDS: " + needs + "\n"
        "AVOID: " + avoid + "\n"
        + register_instruction
        + loaded_instruction
        + anchor_instruction
        + "\n"
        + (repeat_warning + "\n" if repeat_warning else "")
        + "BANNED OPENERS: \"I can understand\", \"That sounds\", \"It sounds like\", \"I hear that\", \"I see\", \"I get that\", \"That must feel\", \"That must be\", \"That frustration\", \"That feeling\", \"That tension\", \"That moment\", \"That kind of\", \"That sense of\", \"When he\", \"When she\", \"When you\"\n"
        + "BANNED QUESTIONS: \"how does that feel\", \"how does that make you feel\", \"what do you think might be happening\", \"what do you notice in you\", \"what do you notice happening in you\", \"what do you feel\", \"what happens in your body\"\n"
        + "BANNED: restating their message, labelling your response shape, advice, summarising their message, partner name\n"
        + "\n"
        + "Respond to what they actually said. Stay in their language. Do not soften or generalise their words."
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


def build_context_block(context: dict, partner_summary: str = None, rag_context: str = None) -> str:
    if not context and not rag_context:
        return "## CONTEXT\nNo prior information about this person yet. Let them lead."

    # Build the standard context block (existing logic untouched below)
    _base = _build_context_block_inner(context, partner_summary)

    if not rag_context:
        return _base

    # Append RAG pattern memory — silently informs BOND's approach
    rag_section = (
        "\n## PATTERN MEMORY\n"
        "From past sessions. Use silently — calibrate tone and approach. "
        "NEVER reference directly or quote back unless they bring it up first.\n"
        + rag_context
    )
    return _base + rag_section


def _build_context_block_inner(context: dict, partner_summary: str = None) -> str:
    """Original build_context_block logic — unchanged."""
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
            # Always label with the speaker name — never expose raw UUIDs to the model
            history.append(HumanMessage(content=f"{speaker_name}: {msg.content}"))
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


# ─────────────────────────────────────────────
# SELF-CHECK — minimal banned opener guard
# ─────────────────────────────────────────────

_SELF_CHECK_PROMPT = """\
Read this response from BOND and the message it's replying to.

MESSAGE: "{message}"
RESPONSE: "{draft}"

Does the response open by restating the message in different words, OR start with any banned opener:
"It sounds like", "That sounds", "That must", "I can understand", "I hear that", "When he", "When she", "When you", "You're feeling"

If YES — rewrite in 1-2 sentences that start somewhere past what they said, not by restating it.
If NO — return the response exactly as written.

Return only the final response text.\
"""


async def self_check_response(message: str, draft: str, temperature: float = 0.2) -> str:
    try:
        llm = get_llm(temperature=temperature)
        prompt = _SELF_CHECK_PROMPT.format(message=message, draft=draft)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        revised = resp.content.strip()
        if revised and len(revised) > 10:
            changed = revised != draft
            print(f"[SELF-CHECK] {'rewrote' if changed else 'passed'}")
            return revised
        return draft
    except Exception as e:
        print(f"[SELF-CHECK ERROR] {e}")
        return draft


# ─────────────────────────────────────────────
# TWO-STEP GENERATION — listening phase
# Step 1: extract key behavior from message
# Step 2: generate from behavior only, never
# seeing the full message — prevents echo
# ─────────────────────────────────────────────

_EXTRACT_PROMPT = """\
Read this message. Pull out the most specific and emotionally loaded detail — the thing that actually stings or matters.

If the message contains a contradiction or an "even though" moment, capture both sides — that tension is usually the point.

MESSAGE: "{message}"

Examples:
"she stopped replying mid-conversation" → "stopped replying mid-conversation"
"kept the conversation going even when I slowed down" → "kept going even after I slowed down"
"I even see her active but she still doesn't reply" → "active on her phone but still not replying"
"one word answers, reacts with emojis, leaves on seen" → "one word answers, emojis, left on seen"
"I don't always feel like replying, I just open it and come back later" → "opens it but comes back later instead of replying"

One phrase, two at most if there's a contradiction. No explanation.\
"""

_RESPOND_PROMPT = """\
You are BOND — a relationship support counsellor. Warm, direct, human. Not a chatbot, not a formal therapist. Someone who actually listens.

Someone just told you this about their relationship:

{behavior}

This is the conversation so far:
{history}

Respond the way a good counsellor would in an early session — following the story, not jumping ahead.
Warm but not over-formal. Not "You mentioned X" or "It sounds like". Just respond naturally.

In this phase a counsellor:
- Picks up one specific thing from what they described and acknowledges it briefly
- If they mentioned a pattern ("keeps happening", "always", "again") — acknowledges the pattern and asks about this instance
- Asks what the other person said or did, or what happened next — one question only
- Does NOT ask about internal feelings, does NOT give advice, does NOT use the person's name

Check the conversation — don't ask about anything already covered.
1-2 sentences.\
"""


async def two_step_listening_response(
    message: str,
    history_msgs: list,
    context_block: str,
    partner_summary: str,
    speaker_name: str,
) -> tuple[str | None, str | None]:
    """
    Returns (response_text, extracted_behavior).
    response_text is None if generation failed — caller should use extracted_behavior
    in the fallback so it's not lost.
    """
    # Step 1 — extract behavior
    behavior = None
    try:
        llm = get_llm(temperature=0.1)
        extract_resp = await llm.ainvoke([
            HumanMessage(content=_EXTRACT_PROMPT.format(message=message))
        ])
        behavior = extract_resp.content.strip().strip('"').strip("'")
        if not behavior or len(behavior) < 3:
            behavior = message[:200]
        print(f"[TWO-STEP] extracted: {repr(behavior)}")
    except Exception as e:
        print(f"[TWO-STEP EXTRACT ERROR] {e}")
        behavior = message[:200]

    # Step 2 — generate response from behavior
    try:
        from langchain_core.messages import AIMessage as _AI
        history_lines = []
        for msg in history_msgs[-6:]:
            if hasattr(msg, "content"):
                if isinstance(msg, _AI):
                    history_lines.append(f"BOND: {msg.content}")
                else:
                    history_lines.append(f"{speaker_name}: {msg.content}")
        history_str = "\n".join(history_lines) if history_lines else "(first message)"

        llm2 = get_llm(temperature=0.7)
        respond_resp = await llm2.ainvoke([
            HumanMessage(content=_RESPOND_PROMPT.format(
                behavior=behavior,
                history=history_str,
            ))
        ])
        text = respond_resp.content.strip()
        if text and len(text) > 5:
            print(f"[TWO-STEP] generated response")
            return text, behavior
        return None, behavior
    except Exception as e:
        print(f"[TWO-STEP RESPOND ERROR] {e}")
        return None, behavior


# ─────────────────────────────────────────────
# MAIN RESPONSE FUNCTION
# ─────────────────────────────────────────────

async def get_ai_response(
    session_id: str,
    couple_id: str,
    speaker_name: str,
    message: str,
    session_type: str,
    recent_messages: list,
    user_id: str = None,
    partner_name: str = None,
    partner_summary: str = None,
    mediation_phase: str = "listening",
    thread_id: str = None,
    user_msg_count: int = 0,
) -> str:
    db = SessionLocal()
    try:
        context = load_context(db, couple_id, user_id)
    finally:
        db.close()

    # ── RAG retrieval — only on first exchange (no prior messages) ────────────
    # On subsequent messages the context block is stable — no need to re-retrieve.
    # Uses checkin intention if available, otherwise the first message itself.
    rag_context = None
    if len(recent_messages) == 0:
        try:
            from agents.rag import retrieve_context
            # Use checkin intention from context if present, else the message
            checkin_intention = (context.get("checkin") or {}).get("intention") or message
            rag_context = await retrieve_context(
                user_id=user_id,
                couple_id=couple_id,
                session_type=session_type,
                query_text=checkin_intention,
            )
            if rag_context:
                print(f"[RAG] retrieved context for user={user_id[:8] if user_id else '?'}")
        except Exception as rag_err:
            print(f"[RAG] retrieval failed (non-fatal): {rag_err}")

    db = SessionLocal()
    try:
        # Partner summary only injected in resolution/integration — not during listening
        # During listening each person's thread must be fully private and unbiased
        _partner_summary_for_context = (
            partner_summary
            if mediation_phase in ("resolution", "integration", "understanding")
            else None
        )
        context_block = build_context_block(context, partner_summary=_partner_summary_for_context, rag_context=rag_context)
        system = THERAPIST_PROMPT.format(
            context_block=context_block,
            session_type=session_type,
        )
        history_msgs = build_history(recent_messages, speaker_name=speaker_name)
    finally:
        db.close()

    # Short opener with no prior history — just greet back, no inference
    # BUT: never intercept emotionally loaded words — they must go through the pipeline
    _EMOTIONAL_SIGNALS = {
        "sad", "angry", "upset", "hurt", "scared", "lost", "broken", "alone",
        "lonely", "numb", "empty", "depressed", "anxious", "stressed", "overwhelmed",
        "tired", "exhausted", "done", "over", "stuck", "confused", "ashamed",
        "guilty", "jealous", "betrayed", "cheated", "lied", "ignored", "rejected",
        "abandoned", "breakup", "divorce", "fight", "argument", "cheating", "affair",
        "crying", "crying.", "crying,", "tears", "heartbreak", "heartbroken",
        "i'm sad", "im sad", "so sad", "really sad", "very sad",
    }
    _msg_stripped = message.strip().lower()
    _is_emotional = any(sig in _msg_stripped for sig in _EMOTIONAL_SIGNALS)
    if len(message.strip()) <= 10 and len(recent_messages) <= 1 and '?' not in message and not _is_emotional:
        return "Hey — glad you're here. What's on your mind?"

    # Catch obvious BOND-directed meta-comments before the pipeline reads them as relationship content
    _meta = ["break up", "break up with you", "leave you", "goodbye bond", "bye bond",
             "you suck", "you're useless", "shut up", "stop talking", "be quiet",
             "write code", "write a poem", "tell me a joke", "what's the weather",
             "who are you", "what are you", "are you ai", "are you a bot"]
    _msg_lower = message.strip().lower()
    if any(m in _msg_lower for m in _meta):
        # Check if it's actually about the relationship (contains partner context)
        _has_context = any(w in _msg_lower for w in ["him", "her", "they", "partner", "boyfriend", "girlfriend", "husband", "wife", "ex"])
        if not _has_context:
            return "Ha — I'm not going anywhere. I'm here whenever you want to talk about what's actually on your mind."

    # ── Shared sessions — mediation arc ─────────────────────────────────────
    if session_type in ("shared", "individual"):
        try:
            # Sanitize partner_summary
            safe_partner_summary = partner_summary or "Partner hasn't shared much yet."
            if partner_summary:
                db3 = SessionLocal()
                try:
                    from models.database import Couple, User
                    couple = db3.query(Couple).filter(Couple.id == couple_id).first()
                    if couple and user_id:
                        partner = next((u for u in couple.users if u.id != user_id), None)
                        if partner:
                            safe_partner_summary = partner_summary.replace(partner.name, "your partner")
                except Exception:
                    pass
                finally:
                    db3.close()

            if mediation_phase == "listening":
                # Get investigation state — drives which prompt to use
                inv_state = get_investigation_state(thread_id) if thread_id else {"phase": "story"}
                inv_phase = inv_state.get("phase", "story")
                next_intention = inv_state.get("next_intention") or ""
                next_key = inv_state.get("next_key") or ""
                pacing = inv_state.get("pacing", "normal")
                handle_with_care = inv_state.get("handle_with_care", "")
                print(f"[INVESTIGATION] thread={thread_id[:8] if thread_id else '?'} phase={inv_phase} msg_count={user_msg_count}")

                if inv_phase in ("extracting", "extracting_complete"):
                    prompt = EXTRACTING_PROMPT.format(
                        next_intention=next_intention,
                        pacing=pacing,
                        context_block=context_block,
                    )
                elif inv_phase == "depth":
                    prompt = DEPTH_PROMPT.format(
                        next_intention=next_intention,
                        handle_with_care=handle_with_care,
                        context_block=context_block,
                    )
                else:
                    # story or unknown — use STORY_PROMPT
                    prompt = STORY_PROMPT.format(context_block=context_block)
            elif mediation_phase == "understanding":
                db2 = SessionLocal()
                try:
                    thread = db2.query(Thread).filter(Thread.id == thread_id).first()
                    core_need = thread.core_need if thread else None
                finally:
                    db2.close()
                prompt = SHARED_UNDERSTANDING_PROMPT.format(
                    context_block=context_block,
                    core_need=core_need or "Still becoming clear"
                )
            elif mediation_phase == "bridging":
                # Check if resolution has been sent — if so use integration prompt
                _res_msg = None
                if thread_id:
                    _db_br = SessionLocal()
                    try:
                        _t_br = _db_br.query(Thread).filter(Thread.id == thread_id).first()
                        _res_msg = _t_br.resolution_message if _t_br else None
                        _int_count = _t_br.integration_count if _t_br else 0
                    finally:
                        _db_br.close()

                if _res_msg:
                    reaction = await detect_integration_reaction(
                        message=message,
                        recent_messages=recent_messages,
                        speaker_name=speaker_name
                    )
                    print(f"[INTEGRATION] thread={thread_id[:8]} reaction={reaction} phase=bridging")
                    prompt = INTEGRATION_PROMPT.format(
                        resolution_message=_res_msg,
                        reaction_type=reaction,
                        context_block=context_block
                    )
                else:
                    prompt = SHARED_BRIDGE_CONSENT_PROMPT
            elif mediation_phase in ("resolution", "integration"):
                db3 = SessionLocal()
                try:
                    thread = db3.query(Thread).filter(Thread.id == thread_id).first()
                    resolution_msg = thread.resolution_message if thread else None
                    int_count = thread.integration_count if thread else 0
                finally:
                    db3.close()

                if not resolution_msg:
                    prompt = STORY_PROMPT.format(context_block=context_block)
                else:
                    reaction = await detect_integration_reaction(
                        message=message,
                        recent_messages=recent_messages,
                        speaker_name=speaker_name
                    )
                    print(f"[INTEGRATION] thread={thread_id[:8]} reaction={reaction} count={int_count}")
                    prompt = INTEGRATION_PROMPT.format(
                        resolution_message=resolution_msg,
                        reaction_type=reaction,
                        context_block=context_block
                    )
            else:
                prompt = STORY_PROMPT.format(context_block=context_block)

            # Build signal brief for story and understanding phases
            brief = ""
            if mediation_phase in ("listening", "understanding"):
                try:
                    brief = await build_shared_brief(message, history_msgs, speaker_name)
                except Exception as e:
                    print(f"[SHARED BRIEF ERROR] {e}")

            # ── Story phase: use two-step pipeline to prevent echo ────────────
            inv_state_inner = get_investigation_state(thread_id) if thread_id else {"phase": "story"}
            inv_phase_inner = inv_state_inner.get("phase", "story")

            if mediation_phase == "listening" and inv_phase_inner == "story":
                two_step_text, _ = await two_step_listening_response(
                    message=message,
                    history_msgs=history_msgs,
                    context_block=context_block,
                    partner_summary=partner_summary,
                    speaker_name=speaker_name,
                )
                if two_step_text and len(two_step_text) > 5:
                    text = two_step_text
                    # Still run self-check to catch any banned openers
                    text = await self_check_response(message, text)
                    return text
                # Fall through to standard pipeline if two-step fails

            # ── Session identity block — always first ─────────────────────
            _self_awareness = (
                "You are BOND — the one they are talking to right now. This conversation is the session. "
                "When someone says 'that\'s why I\'m here', 'I came here for help', 'that\'s why I joined' — "
                "they mean here, with you, in this conversation. "
                "Never say 'glad you joined therapy', 'hope therapy helps', or anything that treats therapy as somewhere else. "
                "You are already the person helping them. Respond as someone who is present, not pointing elsewhere."
            )
            if partner_name:
                identity_block = (
                    "## SESSION IDENTITY\n"
                    f"You are talking to: {speaker_name}\n"
                    f"Their partner (the only other person in this relationship): {partner_name}\n"
                    f"Session type: {session_type}\n"
                    "\n"
                    + _self_awareness + "\n"
                    "\n"
                    f"Anyone mentioned who is not {speaker_name} or {partner_name} is a third party "
                    "(a friend, sibling, colleague, or other person). "
                    "Do NOT follow threads about third parties. "
                    f"Only {partner_name} is the partner. Never invent behaviour for {partner_name} "
                    "— only reflect what {speaker_name} has described."
                )
            else:
                identity_block = (
                    "## SESSION IDENTITY\n"
                    f"You are talking to: {speaker_name}\n"
                    "Their partner has not linked yet — refer to them as 'your partner' or 'they'.\n"
                    f"Session type: {session_type}\n"
                    "\n"
                    + _self_awareness + "\n"
                    "\n"
                    "Anyone else mentioned is a third party. Do not follow those threads."
                )

            msgs = [SystemMessage(content=identity_block), SystemMessage(content=prompt)]
            msgs += history_msgs
            # Brief injected last — highest influence on output
            if brief:
                msgs.append(SystemMessage(content=brief))
            msgs.append(HumanMessage(content=f"{speaker_name}: {message}"))

            llm = get_llm()
            try:
                response = await llm.ainvoke(msgs)
                text = response.content.strip()

                # Post-processing: strip banned openers
                import re as _sre
                _subs = [
                    (r'^It sounds like you\'re', ""),
                    (r'^It sounds like you feel', ""),
                    (r'^It sounds like you\'ve', ""),
                    (r'^It sounds like you', ""),
                    (r'^It sounds like there\'s', ""),
                    (r'^It sounds like', ""),
                    (r'^I can understand how', ""),
                    (r'^I can understand that', ""),
                    (r'^I can understand', ""),
                    (r'^I can see why that', ""),
                    (r'^I can see why', ""),
                    (r'^That must feel really', ""),
                    (r'^That must be really', ""),
                    (r'^That must feel', ""),
                    (r'^That must be', ""),
                    (r'^I hear that', ""),
                    (r'^That frustration[,\s]', ""),
                    (r'^That feeling[,\s]', ""),
                    (r'^That tension[,\s]', ""),
                    (r'^That moment[,\s]', ""),
                    (r'^That sense of[,\s]', ""),
                    (r'^That kind of[,\s]', ""),
                    (r"^That\'s really hard", ""),
                    (r"^That\'s a lot", ""),
                    (r"^That\'s painful", ""),
                    (r'^Feeling like you\'re', ""),
                    (r'^Feeling like you ', ""),
                    (r'^When you feel like', ""),
                    (r'^When you\'re feeling', ""),
                    (r'^You\'re feeling like', ""),
                    (r'^Being the only one', ""),
                    (r'^When he suddenly', ""),
                    (r'^When she ', ""),
                    (r'^When you ', ""),
                    (r'^You mentioned[,\s]', ""),
                    (r'^You mentioned that', ""),
                    (r'^\w+,\s+it\s+', ""),  # strips "Meera, it..." / "Arjun, it..."
                    (r'^That sounds really', ""),
                    (r'^That sounds like a', ""),
                    (r'^That sounds ', ""),
                    (r"^That's really", ""),
                    (r"^That's understandable", ""),
                    (r"^That's tough", ""),
                    (r"^That can be", ""),
                    (r'^Totally understandable', ""),
                    (r'^Understandable[,\s]', ""),
                    (r'^Of course[,\s]', ""),
                    (r'^Absolutely[,\s]', ""),
                ]
                for _pat, _rep in _subs:
                    _new = _sre.sub(_pat, _rep, text, count=1, flags=_sre.IGNORECASE)
                    if _new != text:
                        text = _new.lstrip(' ,—-').strip()
                        if text:
                            text = text[0].upper() + text[1:]
                        print(f"[SHARED POST] opener stripped via: {_pat}")
                        break

                # Story phase: strip comfort/validation sentences entirely
                if mediation_phase == "listening":
                    import re as _sre2
                    _comfort = [
                        r"[Tt]hat'?s? (really |quite )?(understandable|tough|hard|difficult|frustrating|stressful|overwhelming)[.,]?",
                        r"[Ii]t'?s? (really |quite )?(understandable|tough|hard|natural)[.,]?",
                        r"[Ii]t (can|must) be (really |quite )?(hard|tough|difficult|frustrating)[.,]?",
                        r"[Yy]ou'?re? (doing|handling|managing) (really )?(well|okay|alright)[.,]?",
                    ]
                    for _cp in _comfort:
                        _cleaned = _sre2.sub(_cp, '', text, flags=_sre2.IGNORECASE).strip().lstrip(',. ')
                        if _cleaned and len(_cleaned) > 10 and _cleaned != text:
                            text = _cleaned[0].upper() + _cleaned[1:]
                            print(f"[STORY POST] comfort phrase stripped")

                # Integration/resolution phase: strip coaching patterns and enforce limits
                print(f"[POST CHECK] phase={mediation_phase} text_len={len(text)}")
                if mediation_phase in ("integration", "resolution", "bridging"):
                    import re as _sre3
                    # Strip coaching openers — model tries many variations
                    _coaching_openers = [
                        r"^(Powerful|Wonderful|Beautiful|Great|Solid|Amazing|Fantastic|Brilliant)[,\s!]",
                        r"^That'?s? (great|wonderful|amazing|powerful|a great|a good|a beautiful|a powerful|really (great|good|important|meaningful|significant))[,\s!]",
                        r"^What if you (framed|approached|tried|considered|thought|shared|expressed|said|asked|reframed|allowed|gave|reached|sent|made|took)",
                        r"^Would you (like|feel|want|be|consider|be open to)",
                        r"^Are you (ready|open|willing|comfortable|feeling)",
                        r"^It'?s? (great|wonderful|amazing|fantastic|good|so good|really good) (to hear|that you|when you)",
                        r"^Trusting (yourself|that|the)",
                        r"^You'?ve? got this",
                        r"^I believe in you",
                        r"^This is (such a|a really|a powerful|an important)",
                        r"^Approaching (this|it|things|the conversation)",
                        r"^Taking (that|this|a) (step|approach|perspective)",
                    ]
                    for _cp in _coaching_openers:
                        _new = _sre3.sub(_cp, '', text, count=1, flags=_sre3.IGNORECASE).strip().lstrip(',. !—-')
                        if _new and len(_new) > 10 and _new != text:
                            text = _new[0].upper() + _new[1:]
                            print(f"[INTEGRATION POST] coaching opener stripped: {_cp[:40]}")
                            break
                    # Strip coaching phrases mid-sentence
                    _coaching_mid = [
                        r"[Ww]ould you (like|be open|feel comfortable|want) to[^.?!]*[.?!]",
                        r"[Aa]re you (ready|open|willing)[^.?!]*[.?!]",
                        r"[Yy]ou'?ve got this[^.?!]*[.?!]?",
                        r"[Tt]rust(ing)? yourself[^.?!]*[.?!]",
                    ]
                    for _cp in _coaching_mid:
                        _cleaned = _sre3.sub(_cp, '', text, flags=_sre3.IGNORECASE).strip().lstrip(',. ')
                        if _cleaned and len(_cleaned) > 10 and _cleaned != text:
                            text = _cleaned[0].upper() + _cleaned[1:]
                            print(f"[INTEGRATION POST] mid-sentence coaching stripped")
                    # Truncate to 3 sentences max
                    _sentences = _sre3.split(r'(?<=[.!?])\s+', text.strip())
                    if len(_sentences) > 3:
                        text = ' '.join(_sentences[:3])
                        print(f"[INTEGRATION POST] truncated to 3 sentences")
                    # Strip partner name if leaked
                    if couple_id:
                        try:
                            from models.database import Couple as _Couple
                            _db_pn = SessionLocal()
                            try:
                                _couple = _db_pn.query(_Couple).filter(_Couple.id == couple_id).first()
                                if _couple:
                                    _partner = next((u for u in _couple.users if u.id != user_id), None)
                                    if _partner and _partner.name and _partner.name in text:
                                        text = text.replace(_partner.name, "your partner")
                                        print(f"[INTEGRATION POST] partner name stripped")
                            finally:
                                _db_pn.close()
                        except Exception:
                            pass

                # Fix truncated "You" — model sometimes generates "'re..." or "'ve..." missing "You"
                if text.startswith("'re ") or text.startswith("'ve ") or text.startswith("'re,") or text.startswith("'ve,"):
                    text = "You" + text

                # Fix dangling participle starters — "Were trying...", "Noticing that...", "Adjusting how..."
                import re as _re2
                _fragment_pat = r'^(Were|Been|Noticing|Adjusting|Trying|Keeping|Feeling|Having|Starting|Seeing|Watching|Checking|Planning|Holding)\b'
                if _re2.match(_fragment_pat, text, _re2.IGNORECASE):
                    text = "You " + text[0].lower() + text[1:]

                # Catch noun-phrase fragments — "A sense of X", "Of being X", "Hope she..."
                _noun_frag_pat = r'^(A sense of|Of being|Of feeling|Hope that|Hope she|Hope he|Hope they|Being the|Knowing that|Wanting to|Feeling like|Having said)'
                if _re2.match(_noun_frag_pat, text, _re2.IGNORECASE):
                    # Strip the fragment opener and capitalise what remains
                    _stripped = _re2.sub(_noun_frag_pat, '', text, count=1, flags=_re2.IGNORECASE).lstrip(' ,—-').strip()
                    if _stripped and len(_stripped) > 10:
                        text = _stripped[0].upper() + _stripped[1:]
                        print(f"[POST] noun fragment stripped")

                # ── Investigation state management ──────────────────────
                if mediation_phase == "listening" and thread_id:
                    inv_phase = inv_state.get("phase", "story")
                    inv_next_key = inv_state.get("next_key", "")

                    # Detect [SKIP] signal — model flagged deflection
                    if text.endswith("[SKIP]"):
                        text = text[:-6].strip()
                        if inv_next_key:
                            import asyncio as _asyncio
                            _asyncio.create_task(mark_intention_done(thread_id, inv_next_key, "skipped"))
                        # If stripping [SKIP] left empty text, generate a gentle bridge
                        if not text or len(text) < 5:
                            text = "That's okay — let me ask you something else."

                    # Mark current intention as answered if in extracting/depth
                    elif inv_next_key and inv_phase in ("extracting", "depth"):
                        import asyncio as _asyncio
                        _asyncio.create_task(mark_intention_done(thread_id, inv_next_key, "answered"))

                    # If extracting complete — trigger depth brief generation
                    elif inv_phase == "extracting_complete":
                        from agents.mediation import generate_depth_brief as _gen_depth
                        import asyncio as _asyncio
                        _asyncio.create_task(_gen_depth(thread_id))

                # Minimal self-check for understanding phase
                if mediation_phase == "understanding":
                    text = await self_check_response(message, text)

                return text
            except Exception as e:
                print(f"[SHARED RESPONSE ERROR] {e}")
        except Exception as e:
            print(f"[SHARED ARC ERROR] {e}")

    # ── Plain fallback ───────────────────────────────────────────────────────
    msgs = [SystemMessage(content=system)]
    msgs += history_msgs
    msgs.append(HumanMessage(content=f"{speaker_name}: {message}"))

    try:
        llm = get_llm()
        response = await llm.ainvoke(msgs)
        return response.content
    except Exception as e:
        print(f"[FALLBACK ERROR] {e}")
        return "I'm having a little trouble right now — give it a moment and try again."


# ─────────────────────────────────────────────
# BRIDGE DETECTION
# ─────────────────────────────────────────────