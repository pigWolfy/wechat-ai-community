"""Database models and session management for user conversation tracking."""

import time
from contextlib import asynccontextmanager

from sqlalchemy import Column, Integer, String, Text, Float, create_engine, Index
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


class UserSession(Base):
    """Track user analysis sessions and conversation history."""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    open_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        Index("idx_openid_time", "open_id", "created_at"),
    )


class UserProfile(Base):
    """Accumulated user profile — grows smarter with every interaction.

    Each row is one "fact" extracted from conversations. Facts are categorized
    by dimension (e.g. "user_info", "target_info", "relationship", "risk")
    so they can be selectively loaded into the AI analysis context.
    """
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    open_id = Column(String(64), nullable=False, index=True)
    dimension = Column(String(32), nullable=False)   # category of this fact
    key = Column(String(64), nullable=False)          # e.g. "age", "hobby", "red_flag"
    value = Column(Text, nullable=False)              # the extracted fact
    confidence = Column(String(16), default="medium") # low / medium / high
    source_summary = Column(Text, default="")         # which conversation produced this
    updated_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        Index("idx_profile_openid_dim", "open_id", "dimension"),
    )


class PendingTask(Base):
    """Track async analysis tasks (for retry mechanism)."""
    __tablename__ = "pending_tasks"

    msg_id = Column(String(64), primary_key=True)
    open_id = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="processing")  # processing / done
    result = Column(Text, default="")
    created_at = Column(Float, nullable=False, default=time.time)


# Async engine
_engine = create_async_engine(settings.database_url, echo=False)
_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create tables if they don't exist."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_db():
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def save_message(open_id: str, role: str, content: str):
    """Save a conversation message."""
    async with get_db() as db:
        msg = UserSession(open_id=open_id, role=role, content=content, created_at=time.time())
        db.add(msg)


async def get_recent_history(open_id: str, limit: int = 10) -> list[dict]:
    """Get recent conversation history for context."""
    from sqlalchemy import select

    async with get_db() as db:
        stmt = (
            select(UserSession)
            .where(UserSession.open_id == open_id)
            .order_by(UserSession.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

    # Return in chronological order
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]


async def set_pending_task(msg_id: str, open_id: str):
    """Mark a message as being processed."""
    async with get_db() as db:
        task = PendingTask(msg_id=msg_id, open_id=open_id, status="processing", created_at=time.time())
        db.add(task)


async def complete_pending_task(msg_id: str, result: str):
    """Mark a task as completed with its result."""
    from sqlalchemy import update

    async with get_db() as db:
        stmt = (
            update(PendingTask)
            .where(PendingTask.msg_id == msg_id)
            .values(status="done", result=result)
        )
        await db.execute(stmt)


async def get_pending_task(msg_id: str) -> PendingTask | None:
    """Get a pending task by message ID."""
    from sqlalchemy import select

    async with get_db() as db:
        stmt = select(PendingTask).where(PendingTask.msg_id == msg_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# User Profile operations
# ---------------------------------------------------------------------------

async def get_user_profile(open_id: str) -> list[dict]:
    """Get all profile facts for a user, grouped by dimension."""
    from sqlalchemy import select

    async with get_db() as db:
        stmt = (
            select(UserProfile)
            .where(UserProfile.open_id == open_id)
            .order_by(UserProfile.dimension, UserProfile.updated_at.desc())
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "dimension": r.dimension,
            "key": r.key,
            "value": r.value,
            "confidence": r.confidence,
        }
        for r in rows
    ]


async def upsert_profile_facts(open_id: str, facts: list[dict]):
    """Insert or update profile facts for a user.

    Each fact dict: {"dimension": str, "key": str, "value": str, "confidence": str}
    If a fact with the same (open_id, dimension, key) exists, update it.
    """
    from sqlalchemy import select

    async with get_db() as db:
        for fact in facts:
            stmt = (
                select(UserProfile)
                .where(
                    UserProfile.open_id == open_id,
                    UserProfile.dimension == fact["dimension"],
                    UserProfile.key == fact["key"],
                )
            )
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.value = fact["value"]
                existing.confidence = fact.get("confidence", "medium")
                existing.updated_at = time.time()
            else:
                db.add(UserProfile(
                    open_id=open_id,
                    dimension=fact["dimension"],
                    key=fact["key"],
                    value=fact["value"],
                    confidence=fact.get("confidence", "medium"),
                    updated_at=time.time(),
                ))


def format_profile_for_prompt(profile_facts: list[dict]) -> str:
    """Format profile facts into a readable string for AI context injection."""
    if not profile_facts:
        return ""

    grouped: dict[str, list[str]] = {}
    dim_labels = {
        "user_info": "📋 用户自身信息",
        "target_info": "👤 对方信息",
        "relationship": "💑 关系状态",
        "communication": "💬 沟通模式",
        "risk": "🚩 已识别风险",
        "progress": "📈 关系进展",
        "advice_history": "📝 已给建议",
    }

    for fact in profile_facts:
        dim = fact["dimension"]
        label = dim_labels.get(dim, dim)
        if label not in grouped:
            grouped[label] = []
        conf_mark = {"high": "✓", "medium": "~", "low": "?"}.get(fact["confidence"], "")
        grouped[label].append(f"  • {fact['key']}: {fact['value']} [{conf_mark}]")

    parts = []
    for label, items in grouped.items():
        parts.append(f"{label}")
        parts.extend(items)
    return "\n".join(parts)
