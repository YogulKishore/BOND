"""
test_rag.py — BOND RAG diagnostic script

Run from your project root:
    python test_rag.py

What it checks:
  1. ChromaDB is accessible and collections exist
  2. What documents are stored (per user and per couple)
  3. Embedding works (calls OpenAI)
  4. Smart query transform works
  5. Full retrieve_context() returns something useful
  6. What BOND would actually see in its context block

You'll need at least one completed + ended session for data to exist.
"""

import asyncio
import os
import sys
import json

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ── Resolve the bond_rag path the same way rag.py does ────────────────────────
_RAG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bond_rag"))

DIVIDER = "─" * 60


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─────────────────────────────────────────────
# STEP 1 — ChromaDB inspection (no LLM needed)
# ─────────────────────────────────────────────

def inspect_chromadb():
    section("STEP 1 — ChromaDB collections")

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    print(f"bond_rag path: {_RAG_PATH}")
    if not os.path.exists(_RAG_PATH):
        print("❌  bond_rag directory does not exist — embedding has never run")
        return []

    client = chromadb.PersistentClient(
        path=_RAG_PATH,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    collections = client.list_collections()
    if not collections:
        print("❌  No collections found — embed_session may have failed silently")
        return []

    print(f"✓  Found {len(collections)} collection(s):\n")
    for col in collections:
        c = client.get_collection(col.name)
        count = c.count()
        print(f"  [{col.name}]  →  {count} document(s)")

    return [c.name for c in collections]


# ─────────────────────────────────────────────
# STEP 2 — Show stored documents per collection
# ─────────────────────────────────────────────

def inspect_documents(collection_names: list):
    section("STEP 2 — Stored documents")

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=_RAG_PATH,
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    for name in collection_names:
        c = client.get_collection(name)
        results = c.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        print(f"\n── {name} ──")
        if not docs:
            print("  (empty)")
            continue

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            doc_type = meta.get("doc_type", "unknown")
            session_n = meta.get("session_number", "?")
            session_date = meta.get("session_date", "")[:10]
            print(f"\n  [{i+1}] type={doc_type}  session={session_n}  date={session_date}")
            # Print first 300 chars of the document
            preview = doc[:300].replace("\n", " ")
            print(f"  {preview}{'...' if len(doc) > 300 else ''}")


# ─────────────────────────────────────────────
# STEP 3 — Test embedding (calls OpenAI)
# ─────────────────────────────────────────────

async def test_embedding():
    section("STEP 3 — Embedding API")
    try:
        from agents.rag import embed
        vec = await embed("she goes quiet when we fight and I don't know what to do")
        print(f"✓  Embedding returned {len(vec)}-dimensional vector")
        print(f"   First 5 values: {[round(v, 4) for v in vec[:5]]}")
    except Exception as e:
        print(f"❌  Embedding failed: {e}")


# ─────────────────────────────────────────────
# STEP 4 — Test smart query transform
# ─────────────────────────────────────────────

async def test_smart_query():
    section("STEP 4 — Smart query transform")

    from agents.rag import _SMART_QUERY_PROMPT, get_llm
    from langchain_core.messages import HumanMessage

    test_cases = [
        ("feeling anxious", "she just stops talking when we fight"),
        ("overwhelmed", "he never listens when I try to explain how I feel"),
        ("", "I keep apologising even when it's not my fault"),
    ]

    llm = get_llm(temperature=0.1)
    for checkin, message in test_cases:
        try:
            prompt = _SMART_QUERY_PROMPT.format(
                checkin=checkin or message,
                message=message
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            print(f"\n  Input:  \"{message}\"")
            print(f"  Query:  \"{resp.content.strip()}\"")
        except Exception as e:
            print(f"❌  Smart query failed: {e}")


# ─────────────────────────────────────────────
# STEP 5 — Full retrieve_context() per user
# ─────────────────────────────────────────────

async def test_retrieval(collection_names: list):
    section("STEP 5 — Full retrieval (what BOND actually sees)")

    from agents.rag import retrieve_context
    from models.database import SessionLocal, Session, Thread
    from models.database import User, Couple

    db = SessionLocal()
    try:
        # Find all users that have personal collections
        personal_colls = [n for n in collection_names if n.startswith("personal_")]
        couple_colls   = [n for n in collection_names if n.startswith("couple_")]

        if not personal_colls:
            print("No personal collections found — no retrieval to test")
            return

        # Get users from DB
        users = db.query(User).all()
        couples = db.query(Couple).all()

        user_map  = {u.id[:32]: u for u in users}
        couple_map = {c.id[:32]: c for c in couples}

        for coll_name in personal_colls:
            prefix = coll_name[len("personal_"):]
            user = user_map.get(prefix)
            if not user:
                print(f"\n  [{coll_name}] — user not found in DB (deleted?)")
                continue

            # Find a couple for this user
            couple = user.couples[0] if user.couples else None
            if not couple:
                print(f"\n  [{user.name}] — no couple found, skipping")
                continue

            session_type = "individual"
            if any(cn == f"couple_{couple.id[:32]}" for cn in couple_colls):
                session_type = "shared"

            print(f"\n  Testing retrieval for: {user.name} (couple: {couple.id[:8]}...)")
            print(f"  Session type: {session_type}")

            try:
                result = await retrieve_context(
                    user_id=user.id,
                    couple_id=couple.id,
                    session_type=session_type,
                    query_text="feeling disconnected, she goes quiet when I try to talk",
                )
                if result:
                    print(f"\n  ✓  Retrieved context ({len(result)} chars):\n")
                    # Print with indentation
                    for line in result.split("\n"):
                        print(f"    {line}")
                else:
                    print("  ❌  retrieve_context returned None")
                    print("      Possible reasons:")
                    print("      - No documents above cosine distance threshold (0.65)")
                    print("      - Embedding failed during retrieval")
                    print("      - Collection exists but has no matching vectors")
            except Exception as e:
                print(f"  ❌  Retrieval error: {e}")

        # Test couple collection if exists
        for coll_name in couple_colls:
            prefix = coll_name[len("couple_"):]
            couple = couple_map.get(prefix)
            if couple:
                print(f"\n  Couple collection found: {couple.id[:8]}... ({coll_name})")
                print("  (couple_dynamic docs are retrieved automatically in shared sessions)")
    finally:
        db.close()


# ─────────────────────────────────────────────
# STEP 6 — Simulate what happens at session start
# ─────────────────────────────────────────────

async def test_context_injection(collection_names: list):
    section("STEP 6 — Context block injection simulation")

    from models.database import SessionLocal, User, Couple
    from agents.rag import retrieve_context

    personal_colls = [n for n in collection_names if n.startswith("personal_")]
    if not personal_colls:
        print("No personal collections — skipping")
        return

    db = SessionLocal()
    try:
        users = db.query(User).all()
        user_map = {u.id[:32]: u for u in users}

        for coll_name in personal_colls[:1]:  # just test first user
            prefix = coll_name[len("personal_"):]
            user = user_map.get(prefix)
            if not user or not user.couples:
                continue

            couple = user.couples[0]
            print(f"  Simulating first message from: {user.name}")

            rag_result = await retrieve_context(
                user_id=user.id,
                couple_id=couple.id,
                session_type="individual",
                query_text="I don't know how to talk to her anymore",
            )

            if rag_result:
                print("\n  ✓  This is what gets appended to BOND's context block:\n")
                print("  ## PATTERN MEMORY")
                print("  From past sessions. Use silently — calibrate tone and approach.")
                print("  NEVER reference directly or quote back unless they bring it up first.")
                for line in rag_result.split("\n"):
                    print(f"  {line}")
            else:
                print("  ❌  No RAG context retrieved — BOND would have no long-term memory")
                print("      This means the session starts fresh with no pattern awareness")
    finally:
        db.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def main():
    print("\n" + "═" * 60)
    print("  BOND RAG Diagnostic")
    print("═" * 60)

    # Step 1 — no LLM, just filesystem + ChromaDB
    collection_names = inspect_chromadb()

    if not collection_names:
        print("\n⛔  No collections found. Nothing to test.")
        print("    Make sure you've:")
        print("    1. Run at least one full session")
        print("    2. Ended the session via the end button (not just closing the tab)")
        print("    3. Waited a few seconds for the background task to complete")
        return

    # Step 2 — show what's stored
    inspect_documents(collection_names)

    # Step 3-6 — requires LLM calls
    print(f"\n{'─'*60}")
    print("  Steps 3-6 require OpenAI API calls.")
    answer = input("  Run them? (y/n): ").strip().lower()
    if answer != 'y':
        print("  Skipping LLM steps. Run again and press y to test retrieval.")
        return

    await test_embedding()
    await test_smart_query()
    await test_retrieval(collection_names)
    await test_context_injection(collection_names)

    section("DONE")
    print("  If all steps showed ✓, RAG is working end to end.")
    print("  If retrieve_context returned None, the most likely cause is")
    print("  the cosine distance threshold (0.65) — the stored docs aren't")
    print("  semantically close enough to the test queries.")
    print("  Try a query that matches what was actually discussed in the session.\n")


if __name__ == "__main__":
    asyncio.run(main())
