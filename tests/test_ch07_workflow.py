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
    agent_calls = [
        ms
        for ms in model.calls
        if "本轮意图选择节点" not in ms[0].content
        and "本轮问题独立化节点" not in ms[0].content
    ]
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


def test_formal_resolver_uses_summary_and_window_instead_of_full_checkpoint(tmp_path):
    from langchain_core.messages import HumanMessage

    model = ScriptedModel()
    service = make_workflow(tmp_path, model)
    state = {
        "question": "最开始那个订单呢？",
        "conversation_id": "c",
        "messages": [HumanMessage(content="已经离窗的长原文" * 1000)],
        "context_history": [HumanMessage(content="最近这一轮")],
        "context_summary": "最早订单778899，用户称未收到货。",
        "previous_order": None,
        "tokens": 0,
        "usage": [],
    }
    result = service._refer(state)
    rendered = str(model.calls[-1])
    assert "已经离窗的长原文" not in rendered
    assert "778899" in rendered and "最近这一轮" in rendered
    assert result["reference_resolved"] is True
    service.close()


def test_legacy_pending_order_checkpoint_resumes_without_new_user_row(
    tmp_path, monkeypatch
):
    from workflow_helpers import EvidenceRetriever

    evidence = [
        {
            "chunk_id": 1,
            "source_key": "policy:return",
            "answer": "演示退货政策",
            "score": 0.9,
            "category": "退换货与退款 / 退货条件",
            "content_type": "policy",
        }
    ]
    service = make_workflow(
        tmp_path, ScriptedModel("退款退货"), EvidenceRetriever(evidence)
    )
    prepare = service.context.prepare
    monkeypatch.setattr(
        service.context,
        "prepare",
        lambda *args: {
            "context_history": [],
            "context_summary": "",
            "context_omitted_summaries": 0,
        },
    )
    offered = list(
        service.stream_events(
            ChatRequest(conversation_id="legacy", message="耳机能退吗？")
        )
    )
    offer = next(e["data"] for e in offered if e["event"] == "order_choices")
    config = {"configurable": {"thread_id": "legacy"}}
    assert "current_message_id" not in service.graph.get_state(config).values
    monkeypatch.setattr(service.context, "prepare", prepare)
    events = list(
        service.stream_events(
            ChatRequest(
                conversation_id="legacy",
                selected_order_id="DEMO-1001",
                selection_message_id=offer["message_id"],
                request_id=offer["request_id"],
            )
        )
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state(config).values
    assert state["current_message_id"] is not None
    assert state["question"] == "耳机能退吗？"
    with service.session_factory() as session:
        users = list(session.scalars(select(Message).where(Message.role == "user")))
        assert len(users) == 1 and users[0].content == "耳机能退吗？"
    service.close()
