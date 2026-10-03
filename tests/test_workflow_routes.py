import json

import pytest
from sqlalchemy import func, select
from workflow_helpers import EvidenceRetriever, ScriptedModel, make_workflow

from app.db.models import LowConfidenceQuestion, Ticket
from app.schemas import ChatRequest


@pytest.mark.parametrize(
    "intent,route",
    [
        ("物流", "business"),
        ("订单", "business"),
        ("售后", "business"),
        ("商品咨询", "knowledge"),
        ("退款退货", "knowledge"),
        ("投诉", "complaint"),
        ("闲聊", "chitchat"),
    ],
)
def test_seven_intents_map_to_four_fixed_routes(tmp_path, intent, route):
    service = make_workflow(tmp_path, ScriptedModel(intent))
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", message="一个需要分类的问题")
        )
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["route"] == route
    assert ("retrieve" in state["path"]) is (route == "knowledge")
    service.close()


def test_common_greeting_does_not_construct_or_call_model(tmp_path):
    service = make_workflow(tmp_path)
    service.model_factory = lambda: pytest.fail("greeting must be zero-model")
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="你好！"))
    )
    assert events[-1]["event"] == "done"
    assert "订单" in "".join(
        e["data"]["content"] for e in events if e["event"] == "token"
    )
    service.close()


def test_complaint_has_two_suggestions_without_any_ticket(tmp_path):
    model = ScriptedModel("投诉")
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="我要投诉"))
    )
    actions = next(e["data"] for e in events if e["event"] == "actions")
    assert [a["type"] for a in actions["items"]] == ["handoff", "create_ticket"]
    assert len(model.calls) == 1
    with service.session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0
    service.close()


def test_weak_policy_never_enters_agent_and_records_raw_question(tmp_path):
    service = make_workflow(tmp_path, ScriptedModel("退款退货"))
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="退货政策"))
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["gate_passed"] is False
    assert state["decisions"] == 0
    assert state["path"] == [
        "refer",
        "classify",
        "route",
        "retrieve",
        "gate",
        "fixed",
        "log",
    ]
    with service.session_factory() as s:
        row = s.scalar(select(LowConfidenceQuestion))
        assert row.raw_question == "退货政策"
        assert row.source == "retrieval_low_conf"
    service.close()


def test_retrieval_outage_is_error_not_weak_evidence(tmp_path):
    service = make_workflow(
        tmp_path,
        ScriptedModel("商品咨询"),
        EvidenceRetriever(error=RuntimeError("offline")),
    )
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="商品参数"))
    )
    assert events[-1]["event"] == "error"
    assert not any(e["event"] == "done" for e in events)
    with service.session_factory() as s:
        assert s.scalar(select(func.count()).select_from(LowConfidenceQuestion)) == 0
    records = [json.loads(line) for line in service.log_path.read_text().splitlines()]
    assert records[-1]["status"] == "error"
    assert records[-1]["path"][-1] == "retrieve"
    service.close()


def test_sqlite_checkpoint_survives_reopen_and_owner_checked_before_checkpoint(
    tmp_path,
):
    service = make_workflow(tmp_path)
    list(service.stream_events(ChatRequest(conversation_id="c", message="你好")))
    service.close()
    reopened = make_workflow(tmp_path)
    list(reopened.stream_events(ChatRequest(conversation_id="c", message="谢谢")))
    state = reopened.graph.get_state({"configurable": {"thread_id": "c"}}).values
    users = [m.content for m in state["messages"] if m.type == "human"]
    assert users == ["你好", "谢谢"]
    unauthorized = list(
        reopened.stream_events(
            ChatRequest(conversation_id="c", user_id="other", message="读取历史")
        )
    )
    assert unauthorized[-1]["event"] == "error"
    reopened.close()


def test_nonfinite_or_blank_evidence_cannot_open_gate(tmp_path):
    service = make_workflow(
        tmp_path,
        ScriptedModel("商品咨询"),
        EvidenceRetriever(
            [
                {"n": 1, "score": float("nan"), "answer": "虚假高分"},
                {"n": 2, "score": 1, "answer": " "},
            ]
        ),
    )
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="商品参数"))
    )
    assert events[-1]["event"] == "done"
    assert (
        service.graph.get_state({"configurable": {"thread_id": "c"}}).values[
            "gate_passed"
        ]
        is False
    )
    service.close()


def test_invalid_intent_json_never_enters_agent(tmp_path):
    model = ScriptedModel("不合法类别")
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="分类这个问题"))
    )
    assert events[-1]["event"] == "done"
    assert len(model.calls) == 1
    assert events[-1]["data"]["stop_reason"] == "invalid_intent"
    service.close()
