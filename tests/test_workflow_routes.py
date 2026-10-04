import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import func, select
from workflow_helpers import EvidenceRetriever, ScriptedModel, make_workflow

from app.db.models import LowConfidenceQuestion, Ticket
from app.schemas import ChatRequest


@pytest.mark.parametrize(
    "intent,route",
    [
        ("物流", "business"),
        ("订单", "business"),
        ("售后", "high_risk"),
        ("商品咨询", "knowledge"),
        ("退款退货", "high_risk"),
        ("投诉", "complaint"),
        ("闲聊", "chitchat"),
    ],
)
def test_seven_intents_map_to_four_fixed_routes(tmp_path, intent, route):
    service = make_workflow(tmp_path, ScriptedModel(intent))
    message = "一个需要分类的问题 DEMO-1001" if route == "high_risk" else "一个需要分类的问题"
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", message=message)
        )
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["route"] == route
    assert ("retrieve" in state["path"]) is (route == "knowledge")
    assert ("order_check" in state["path"]) is (route == "high_risk")
    service.close()


def test_common_greeting_uses_the_intent_prompt(tmp_path):
    model = ScriptedModel("闲聊")
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="你好！"))
    )
    assert events[-1]["event"] == "done"
    assert "订单" in "".join(
        e["data"]["content"] for e in events if e["event"] == "token"
    )
    assert len(model.calls) == 2
    service.close()


def test_low_confidence_intent_routes_to_other_fixed_fallback(tmp_path):
    service = make_workflow(tmp_path, ScriptedModel("退款退货", confidence=0.2))
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", message="这个问题有点奇怪")
        )
    )
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert events[-1]["data"]["route"] == "other"
    assert state["intent"] == "其他"
    assert state["intent_confidence"] == 0.2
    assert state["path"] == ["refer", "classify", "route", "fixed", "log"]
    service.close()


def test_complaint_has_two_suggestions_without_any_ticket(tmp_path):
    model = ScriptedModel("投诉")
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="我要投诉"))
    )
    actions = next(e["data"] for e in events if e["event"] == "actions")
    assert [a["type"] for a in actions["items"]] == ["handoff", "create_ticket"]
    assert len(model.calls) == 2
    with service.session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0
    service.close()


def test_weak_policy_never_enters_agent_and_records_raw_question(tmp_path):
    service = make_workflow(tmp_path, ScriptedModel("退款退货"))
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单 DEMO-1001 退货政策"))
    )
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["gate_passed"] is False
    assert state["decisions"] == 0
    assert state["path"] == [
        "refer",
        "classify",
        "route",
        "order_check",
        "expand_policy",
        "retrieve_policy",
        "policy_gate",
        "fixed",
        "log",
    ]
    with service.session_factory() as s:
        row = s.scalar(select(LowConfidenceQuestion))
        assert row.raw_question == "订单 DEMO-1001 退货政策"
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
    assert len(model.calls) == 2
    assert events[-1]["data"]["stop_reason"] == "invalid_intent"
    service.close()


def test_refund_without_order_pauses_then_bound_selection_resumes_policy_flow(tmp_path):
    evidence = [
        {
            "chunk_id": 1,
            "source_key": "policy:returns",
            "answer": "演示政策依据。",
            "score": 0.9,
            "category": "退换货与退款 / 退货条件",
            "content_type": "policy",
        }
    ]
    retriever = EvidenceRetriever(evidence)
    service = make_workflow(tmp_path, ScriptedModel("退款退货"), retriever)
    offered = list(
        service.stream_events(
            ChatRequest(conversation_id="refund-c", message="这个耳机能退吗？")
        )
    )
    order_event = next(event for event in offered if event["event"] == "order_choices")
    assert order_event["data"]["conversation_id"] == "refund-c"
    assert order_event["data"]["message_id"] > 0
    assert order_event["data"]["request_id"]
    assert order_event["data"]["choices"]
    assert offered[-1]["event"] != "done"
    paused = service.graph.get_state({"configurable": {"thread_id": "refund-c"}})
    assert paused.next == ("await_order",)

    forged = list(
        service.stream_events(
            ChatRequest(
                conversation_id="refund-c",
                selected_order_id="DEMO-1002",
                selection_message_id=order_event["data"]["message_id"],
                request_id="forged-request",
            )
        )
    )
    assert forged[-1]["event"] == "error"

    resumed = list(
        service.stream_events(
            ChatRequest(
                conversation_id="refund-c",
                selected_order_id="DEMO-1001",
                selection_message_id=order_event["data"]["message_id"],
                request_id=order_event["data"]["request_id"],
            )
        )
    )
    assert resumed[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "refund-c"}}).values
    assert state["order"]["order_id"] == "DEMO-1001"
    assert state["gate_passed"] is True
    assert state["path"][-7:] == [
        "await_order", "expand_policy", "retrieve_policy", "policy_gate",
        "agent", "agent_answer", "log",
    ]
    assert retriever.queries
    assert len(retriever.queries[-1]) == 3
    assert len([m for m in state["messages"] if m.type == "human"]) == 1
    repeated = list(
        service.stream_events(
            ChatRequest(
                conversation_id="refund-c",
                selected_order_id="DEMO-1001",
                selection_message_id=order_event["data"]["message_id"],
                request_id=order_event["data"]["request_id"],
            )
        )
    )
    assert repeated[-1]["event"] == "error"
    service.close()


def test_multi_turn_reference_and_intents_route_each_turn_independently(tmp_path):
    class ConversationModel(ScriptedModel):
        def invoke(self, messages):
            self.calls.append(messages)
            prompt = messages[0].content
            if "本轮问题独立化节点" in prompt:
                raw = messages[-1].content.split("本轮原问题：\n", 1)[-1]
                if raw == "它能退吗？":
                    question = "订单 DEMO-1001 的云朵降噪耳机能否退货？"
                    return AIMessage(
                        content=json.dumps(
                            {"question": question, "reference_resolved": True},
                            ensure_ascii=False,
                        )
                    )
                if raw == "那物流到哪了？":
                    return AIMessage(
                        content=json.dumps(
                            {"question": "订单 DEMO-1001 的物流到哪了？", "reference_resolved": True},
                            ensure_ascii=False,
                        )
                    )
                return AIMessage(
                    content=json.dumps(
                        {"question": raw, "reference_resolved": True}, ensure_ascii=False
                    )
                )
            if "本轮意图选择节点" in prompt:
                query = messages[-1].content
                intent = "退款退货" if "退" in query else "物流"
                return AIMessage(
                    content=json.dumps({"intent": intent, "confidence": 0.96}, ensure_ascii=False)
                )
            if "检索查询扩写节点" in prompt:
                return AIMessage(content='{"queries":["退货条件和期限"]}')
            return super().invoke(messages)

    evidence = [{
        "chunk_id": 1, "source_key": "policy:return", "score": 0.9,
        "answer": "演示退货政策。", "category": "退换货与退款 / 退货条件",
        "content_type": "policy",
    }]
    model = ConversationModel()
    service = make_workflow(tmp_path, model, EvidenceRetriever(evidence))
    turns = [
        ("订单 DEMO-1001 的物流到哪了？", "物流", True),
        ("它能退吗？", "退款退货", True),
        ("那物流到哪了？", "物流", True),
    ]
    states = []
    for index, (message, intent, resolved) in enumerate(turns):
        events = list(service.stream_events(ChatRequest(conversation_id="cycle", message=message)))
        assert events[-1]["event"] == "done"
        state = service.graph.get_state({"configurable": {"thread_id": "cycle"}}).values
        assert state["intent"] == intent
        assert state["reference_resolved"] is resolved
        states.append((state["resolved_question"], state["path"]))
    assert states[1][0] == "订单 DEMO-1001 的云朵降噪耳机能否退货？"
    assert "retrieve_policy" in states[1][1]
    assert "retrieve_policy" not in states[2][1]
    final_state = service.graph.get_state({"configurable": {"thread_id": "cycle"}}).values
    assert [message.content for message in final_state["agent_messages"] if message.type == "human"] == [
        "那物流到哪了？"
    ]
    service.close()


def test_unresolved_reference_cannot_turn_a_resolver_guess_into_an_order_id(tmp_path):
    class GuessingResolver(ScriptedModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if "本轮问题独立化节点" in messages[0].content:
                return AIMessage(
                    content=json.dumps(
                        {
                            "question": "订单 DEMO-1001 的商品能否退款？",
                            "reference_resolved": False,
                        },
                        ensure_ascii=False,
                    )
                )
            if "本轮意图选择节点" in messages[0].content:
                return AIMessage(content='{"intent":"退款退货","confidence":0.95}')
            return AIMessage(content='{"queries":["退款条件"]}')

    retriever = EvidenceRetriever(
        [{
            "chunk_id": 1, "source_key": "policy:return", "score": 0.9,
            "answer": "policy", "category": "退换货与退款 / 退货条件",
            "content_type": "policy",
        }]
    )
    service = make_workflow(tmp_path, GuessingResolver(), retriever)
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="guess", message="它能退吗？")
        )
    )
    assert any(event["event"] == "order_choices" for event in events)
    assert not retriever.queries
    state = service.graph.get_state({"configurable": {"thread_id": "guess"}}).values
    assert state["reference_resolved"] is False
    assert state["resolved_question"] == "它能退吗？"
    assert state["order"] is None
    service.close()


def test_order_selection_uses_current_explicit_id_or_unique_history_only(tmp_path):
    service = make_workflow(tmp_path, ScriptedModel("退款退货"))
    history = [
        HumanMessage(content="查订单 DEMO-1001 的物流"),
        HumanMessage(content="再查订单 DEMO-1002 的物流"),
    ]
    ambiguous = service._order_check({
        "question": "这个能退吗？",
        "resolved_question": "订单 DEMO-1001 的商品能退吗？",
        "reference_resolved": True,
        "messages": history,
        "user_id": "demo-user",
    })
    assert ambiguous["order"] is None

    explicit = service._order_check({
        "question": "订单 DEMO-1001 能退吗？",
        "resolved_question": "订单 DEMO-1002 能退吗？",
        "reference_resolved": True,
        "messages": history,
        "user_id": "demo-user",
    })
    assert explicit["order"]["order_id"] == "DEMO-1001"
    assert explicit["resolved_question"] == "订单 DEMO-1001 能退吗？"
    assert explicit["reference_resolved"] is False
    service.close()
