import pytest

from app.db.base import Base
from app.db.session import create_tables, make_engine, make_session_factory


@pytest.fixture

def db_session():
    engine = make_engine("sqlite+pysqlite:///:memory:")
    create_tables(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        yield session
    engine.dispose()
