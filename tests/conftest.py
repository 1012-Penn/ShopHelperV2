import pytest

from app.db.session import create_tables, make_engine, make_session_factory


@pytest.fixture

def db_session_factory():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    create_tables(engine)
    session_factory = make_session_factory(engine)
    yield session_factory
    engine.dispose()


@pytest.fixture

def db_session(db_session_factory):
    with db_session_factory() as session:
        yield session
