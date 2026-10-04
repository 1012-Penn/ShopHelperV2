from langchain_core.messages import AIMessage
from sqlalchemy import select
from workflow_helpers import ScriptedModel, make_workflow

from app.db.models import Message
from app.schemas import ChatRequest


def test_greeting_logs_history_and_keeps_checkpoint(tmp_path):
    service = make_workflow(tmp_path)
    for i in range(24):
        assert (
            list(
                service.stream_events(
                    ChatRequest(conversation_id="c", user_id="u", message="你好")
                )
            )[-1]["event"]
            == "done"
        )
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert len(state["messages"]) == 48
    assert all(m.id.startswith("sql-") for m in state["messages"])
    log = service.context.log.path.read_text()
    assert log.count('"event": "history_ctx"') == 24
    assert "层1 降级" not in log and "summary trigger" not in log
    service.close()


def test_model_order_and_tools_not_in_message_table(tmp_path):
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_logistics", "id": "t", "args": {"order_id": "1001"}}
        ],
    )
    model = ScriptedModel("物流", [call])
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", user_id="u", message="订单1001物流")
        )
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert any(m.type == "tool" for m in state["messages"])
    with service.session_factory() as s:
        rows = list(s.scalars(select(Message)))
        assert [r.role for r in rows] == ["user", "assistant"]
    agent_calls = [ms for ms in model.calls if "七类" not in ms[0].content]
    assert all(sum(m.type == "system" for m in ms) == 1 for ms in agent_calls)
    assert all("本轮数据" not in ms[0].content for ms in agent_calls)
    assert agent_calls[0][1].content == "订单1001物流"
    assert "早期会话摘要" in agent_calls[0][2].content
    log = service.context.log.path.read_text()
    assert "model_ctx" in log and "history_ctx" in log
    service.close()


def test_app_startup_rejects_inadequate_window(monkeypatch):
    import pytest
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("MODEL_CONTEXT_WINDOW", "4000")
    with pytest.raises(ValueError, match="上下文预算不足"), TestClient(create_app()):
        pass
