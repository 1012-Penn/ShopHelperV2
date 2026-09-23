"""Engine, schema initialization, and session helpers."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base


def make_engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_tables(engine: Engine) -> None:
    from app.db import models  # noqa: F401 - importing registers mapped tables

    Base.metadata.create_all(engine)
