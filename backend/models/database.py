from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, ForeignKey, Table, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid

DATABASE_URL = "sqlite:///./bond.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_id():
    return str(uuid.uuid4())

user_couple = Table('user_couple', Base.metadata,
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('couple_id', String, ForeignKey('couples.id'), primary_key=True),
)

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_id)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    is_onboarded = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    couples = relationship("Couple", secondary=user_couple, back_populates="users")

class Couple(Base):
    __tablename__ = "couples"
    id = Column(String, primary_key=True, default=generate_id)
    invite_code = Column(String, unique=True, nullable=False)
    label = Column(String, default="My relationship")
    is_relationship_profiled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    users = relationship("User", secondary=user_couple, back_populates="couples")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=generate_id)
    couple_id = Column(String, ForeignKey("couples.id"), nullable=False)
    session_type = Column(String, nullable=False)        # individual / shared / async
    initiated_by = Column(String, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    summary = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)

    # Mediation arc phase for shared sessions
    # listening → understanding → bridging → resolution → integration
    mediation_phase = Column(String, default="listening", nullable=True)

    # Tracks which users have consented to receiving the shared insight
    # Stored as JSON array of user_ids e.g. '["uuid1", "uuid2"]'
    bridge_consents = Column(Text, nullable=True)

    # Latest combined analysis of both threads.
    # Updated every 2 combined messages. Stores the full analysis JSON.
    # Replaces misunderstandings_json + separate core_need extractions.
    analysis_json = Column(Text, nullable=True)

    # How many consecutive times ready_for_bridge=true has been returned.
    # Bridge only fires when this reaches 2 — prevents false positives.
    analysis_ready_count = Column(Integer, default=0)

    # Sequential session number for this couple — used in RAG metadata
    # so BOND can say "session 3" in retrieved context.
    session_number = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    messages = relationship("Message", back_populates="session")
    threads = relationship("Thread", back_populates="session")

class Thread(Base):
    """
    A private conversation channel between one user and BOND within a shared session.
    Each shared session has one Thread per participant.

    Fields:
      session_id     — the shared session this thread belongs to
      user_id        — the user this thread belongs to
      thread_summary — rolling LLM-generated summary of this person's emotional state,
                       updated every 3 user messages. Fed to BOND when responding to
                       the OTHER participant — never their raw messages.
      core_need      — distilled single sentence: what this person fundamentally needs.
                       Extracted when enough signal exists. Used in Phase 2+.
      message_count  — running count of user messages in this thread.
                       Used to trigger phase transitions without re-querying.
    """
    __tablename__ = "threads"
    id = Column(String, primary_key=True, default=generate_id)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    thread_summary = Column(Text, nullable=True)
    core_need = Column(Text, nullable=True)              # distilled need — used in Phase 2+
    message_count = Column(Integer, default=0)           # running user message count
    resolution_message = Column(Text, nullable=True)     # the resolution BOND sent — referenced in integration
    integration_count = Column(Integer, default=0)       # messages exchanged in integration phase

    # Investigation system
    story_summary = Column(Text, nullable=True)          # BOND's confirmed summary of the story
    story_confirmed = Column(Boolean, default=False)     # True once user confirms the summary
    investigation_brief_json = Column(Text, nullable=True)  # Phase 2 brief — intentions to pursue
    depth_brief_json = Column(Text, nullable=True)       # Phase 3 brief — deeper intentions
    investigation_phase = Column(String, default='story') # story / extracting / depth / complete
    brief_answered_json = Column(Text, nullable=True)    # which brief items are answered/skipped

    # Async delivery — queued content waiting for next connect
    bridge_pending = Column(Text, nullable=True)          # bridge lead-in question to deliver on next connect
    resolution_pending = Column(Text, nullable=True)      # resolution message to deliver on next connect
    bridge_consented = Column(Boolean, default=False)     # True once this person has consented
    resolution_beat2_sent = Column(Boolean, default=False)  # True once beat-2 has been sent to this thread

    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="threads")
    messages = relationship("Message", back_populates="thread")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=generate_id)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    thread_id = Column(String, ForeignKey("threads.id"), nullable=True)
    sender_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    is_private = Column(Boolean, default=False)

    # Stores the pipeline analysis result for this message as JSON.
    # Keys: emotional_state, intensity, trajectory, defensiveness,
    #       emotional_regulation, self_awareness, needs, patterns_activating
    # Enables longitudinal signal tracking across sessions.
    analysis_json = Column(Text, nullable=True)

    # Stores pipeline decisions for AI messages: move, stage, key_phrase,
    #   constraints_fired, regen_happened, post_processed, shared_signal
    pipeline_json = Column(Text, nullable=True)

    sentiment_score = Column(String, nullable=True)      # kept for backward compat
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="messages")
    thread = relationship("Thread", back_populates="messages")

class Memory(Base):
    __tablename__ = "memories"
    id = Column(String, primary_key=True, default=generate_id)
    couple_id = Column(String, ForeignKey("couples.id"), nullable=False)
    owner_id = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String, nullable=False)
    # profile / relationship_profile / checkin / pattern / couple_pattern
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

    # Safe migration: add new columns to existing DB if they don't exist
    from sqlalchemy import inspect, text
    inspector = inspect(engine)

    with engine.connect() as conn:
        # sessions table
        session_cols = [c['name'] for c in inspector.get_columns('sessions')]
        if 'mediation_phase' not in session_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN mediation_phase TEXT DEFAULT 'listening'"))
            print("[DB] Added sessions.mediation_phase")
        if 'bridge_consents' not in session_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN bridge_consents TEXT"))
            print("[DB] Added sessions.bridge_consents")
        if 'analysis_json' not in session_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN analysis_json TEXT"))
            print("[DB] Added sessions.analysis_json")
        if 'analysis_ready_count' not in session_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN analysis_ready_count INTEGER DEFAULT 0"))
            print("[DB] Added sessions.analysis_ready_count")
        if 'session_number' not in session_cols:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN session_number INTEGER DEFAULT 1"))
            print("[DB] Added sessions.session_number")

        # threads table
        thread_cols = [c['name'] for c in inspector.get_columns('threads')]
        if 'core_need' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN core_need TEXT"))
            print("[DB] Added threads.core_need")
        if 'message_count' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN message_count INTEGER DEFAULT 0"))
            print("[DB] Added threads.message_count")
        if 'resolution_message' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN resolution_message TEXT"))
            print("[DB] Added threads.resolution_message")
        if 'integration_count' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN integration_count INTEGER DEFAULT 0"))
            print("[DB] Added threads.integration_count")
        if 'story_summary' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN story_summary TEXT"))
            print("[DB] Added threads.story_summary")
        if 'story_confirmed' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN story_confirmed BOOLEAN DEFAULT 0"))
            print("[DB] Added threads.story_confirmed")
        if 'investigation_brief_json' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN investigation_brief_json TEXT"))
            print("[DB] Added threads.investigation_brief_json")
        if 'depth_brief_json' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN depth_brief_json TEXT"))
            print("[DB] Added threads.depth_brief_json")
        if 'investigation_phase' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN investigation_phase TEXT DEFAULT 'story'"))
            print("[DB] Added threads.investigation_phase")
        if 'brief_answered_json' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN brief_answered_json TEXT"))
            print("[DB] Added threads.brief_answered_json")
        if 'bridge_pending' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN bridge_pending TEXT"))
            print("[DB] Added threads.bridge_pending")
        if 'resolution_pending' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN resolution_pending TEXT"))
            print("[DB] Added threads.resolution_pending")
        if 'bridge_consented' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN bridge_consented BOOLEAN DEFAULT 0"))
            print("[DB] Added threads.bridge_consented")
        if 'resolution_beat2_sent' not in thread_cols:
            conn.execute(text("ALTER TABLE threads ADD COLUMN resolution_beat2_sent BOOLEAN DEFAULT 0"))
            print("[DB] Added threads.resolution_beat2_sent")

        # messages table
        message_cols = [c['name'] for c in inspector.get_columns('messages')]
        if 'analysis_json' not in message_cols:
            conn.execute(text("ALTER TABLE messages ADD COLUMN analysis_json TEXT"))
            print("[DB] Added messages.analysis_json")
        if 'pipeline_json' not in message_cols:
            conn.execute(text("ALTER TABLE messages ADD COLUMN pipeline_json TEXT"))
            print("[DB] Added messages.pipeline_json")

        conn.commit()

if __name__ == "__main__":
    init_db()
    print("Database initialised")