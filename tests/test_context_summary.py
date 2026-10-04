import threading

from workflow_helpers import make_workflow

from app.db.models import Conversation, Message
from app.services.context.logging import ContextLog
from app.services.context.repository import ContextRepository
from app.services.context.summary import SummaryBatch, SummaryWorker


def seed(service):
    with service.session_factory.begin() as s:
        s.add(Conversation(conversation_id="c", user_id="u"))
        s.flush()
        s.add_all(
            [
                Message(conversation_id="c", role="user", content="订单12345想退货"),
                Message(conversation_id="c", role="assistant", content="需要核实"),
            ]
        )
    return ContextRepository(service.session_factory)


def test_worker_does_not_wait_and_commits_fixed_snapshot(tmp_path):
    service = make_workflow(tmp_path)
    repo = seed(service)
    entered, release = threading.Event(), threading.Event()

    def summarize(batch):
        entered.set()
        assert release.wait(5)
        return "用户订单12345想退货，资格未核实。"

    worker = SummaryWorker(repo, summarize, ContextLog(tmp_path / "app.log"))
    batch = SummaryBatch("c", None, 1, 2, "订单12345想退货", "")
    assert worker.submit(batch)
    assert entered.wait(2)
    assert not worker.submit(batch)
    repo.advance_layer1("c", 3)
    release.set()
    worker.close()
    meta, parts = repo.read("c")
    assert meta.summary_upto_msg_id == 2
    assert meta.layer1_from_msg_id == 3
    assert len(parts) == 1
    assert not repo.append(batch, "重复段")
    assert len(repo.read("c")[1]) == 1
    log = (tmp_path / "app.log").read_text()
    assert (
        "summary start" in log and "summary done 第1段" in log and "summary skip" in log
    )
    service.close()


def test_failure_does_not_advance(tmp_path):
    service = make_workflow(tmp_path)
    repo = seed(service)

    def fail(batch):
        raise RuntimeError("provider down")

    worker = SummaryWorker(repo, fail, ContextLog(tmp_path / "app.log"))
    worker.submit(SummaryBatch("c", None, 1, 2, "订单12345", ""))
    worker.close()
    assert repo.read("c")[0].summary_upto_msg_id is None
    assert repo.read("c")[1] == []
    assert "summary fail" in (tmp_path / "app.log").read_text()
    service.close()


def test_migration_is_repeatable(tmp_path):
    from sqlalchemy import inspect, text

    from app.db.session import create_tables, make_engine

    engine = make_engine(f"sqlite:///{tmp_path}/legacy.db")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE conversations (conversation_id VARCHAR(128) PRIMARY KEY, user_id VARCHAR(128), status VARCHAR(32), created_at DATETIME)"
            )
        )
    create_tables(engine)
    create_tables(engine)
    assert {"summary_upto_msg_id", "layer1_from_msg_id"} <= {
        c["name"] for c in inspect(engine).get_columns("conversations")
    }
    assert "conversation_summaries" in inspect(engine).get_table_names()
    engine.dispose()
