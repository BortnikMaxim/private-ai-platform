from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings


class Base(DeclarativeBase):
    pass


def create_engine(database_url: str | None = None) -> AsyncEngine:
    return create_async_engine(
        database_url or settings.database_url,
        pool_pre_ping=True,
    )


# Module level engine for tooling that runs outside the FastAPI lifespan
# (Alembic, backend.scripts.*). The application itself uses the engine built in
# the lifespan from its own Settings.
engine: AsyncEngine = create_engine()

SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Request scoped session.

    Any exception escaping the handler rolls the transaction back before the
    connection returns to the pool.
    """
    session_factory = getattr(request.app.state, "session_factory", None) or SessionLocal

    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
