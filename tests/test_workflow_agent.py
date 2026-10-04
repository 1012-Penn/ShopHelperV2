import json
from threading import Event

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import func, select
from workflow_helpers import EvidenceRetriever, ScriptedModel, make_workflow

from app.db.models import Ticket
from app.schemas import ChatRequest
from app.services.workflow.policy import Limits


def call(name, id="id", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": id, "args": args}])


def state(service, conversation_id="c"):
    return service.graph.get_state(
        {"configurable": {"thread_id": conversation_id}}
    ).values


def test_simple_logistics_converges_with_one_real_existing_tool(tmp_path):
    model = ScriptedModel("物流", [call("query_logistics", order_id="1001")])
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", message="订单1001物流到哪了")
        )
    )
    assert events[-1]["event"] == "done"
    assert state(service)["tool_calls"] == 1
    assert [t["name"] for t in state(service)["tool_trace"]] == ["query_logistics"]
    assert "retrieve" not in state(service)["path"]
    assert set(model.bound_tools) == {"query_order", "query_product", "query_logistics"}
    service.close()


def test_order_then_logistics_react_uses_observations(tmp_path):
    model = ScriptedModel(
        "订单",
        [
            call("query_order", "o", order_id="1001"),
            call("query_logistics", "l", order_id="1001"),
        ],
    )
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(
            ChatRequest(conversation_id="c", message="先查订单1001再查物流")
        )
    )
    assert events[-1]["event"] == "done"
    assert state(service)["tool_calls"] == 2
    assert state(service)["decisions"] == 3
    messages = state(service)["messages"]
    assert [m.tool_call_id for m in messages if m.type == "tool"] == ["o", "l"]
    assert "模拟数据" in next(m.content for m in messages if m.type == "tool")
    service.close()


def test_knowledge_gate_passes_evidence_to_agent_then_resets_on_business(tmp_path):
    evidence = [
        {
            "n": 1,
            "score": 0.9,
            "answer": "七天内可申请退货",
            "chunk_id": 7,
            "source_key": "doc:policy:1",
            "section_path": ["退货"],
            "category": "退换货与退款 / 退货条件",
            "content_type": "policy",
        }
    ]
    model = ScriptedModel("退款退货", chunks=["可申请退货[1]"])
    service = make_workflow(tmp_path, model, EvidenceRetriever(evidence))
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单 DEMO-1001 退货政策"))
    )
    assert state(service)["path"][:7] == [
        "refer",
        "classify",
        "route",
        "order_check",
        "expand_policy",
        "retrieve_policy",
        "policy_gate",
    ]
    assert "七天内可申请退货" in str(model.calls[-1])
    assert (
        next(e["data"]["items"] for e in events if e["event"] == "citations")[0][
            "chunk_id"
        ]
        == 7
    )
    model.intent = "物流"
    list(
        service.stream_events(
            ChatRequest(conversation_id="c", message="现在查订单1001物流")
        )
    )
    assert state(service)["evidence"] == []
    assert state(service)["actions"] == []
    service.close()


def test_missing_information_asks_user_without_tool(tmp_path):
    model = ScriptedModel(
        "物流",
        [AIMessage(content='{"next":"clarify","missing":"订单号","actions":[]}')],
        chunks=["请提供订单号"],
    )
    service = make_workflow(tmp_path, model)
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="查物流"))
    )
    assert state(service)["tool_calls"] == 0
    assert "订单号" in "".join(
        e["data"]["content"] for e in events if e["event"] == "token"
    )
    service.close()


def test_forged_write_tool_never_creates_ticket_and_pairs_error_observation(tmp_path):
    model = ScriptedModel(
        "售后", [call("create_ticket", description="帮我建单", ticket_type="售后")]
    )
    service = make_workflow(
        tmp_path,
        model,
        EvidenceRetriever([{
            "chunk_id": 1, "source_key": "policy:after-sale", "score": 0.9,
            "answer": "演示售后条款。", "category": "退换货与退款 / 售后",
            "content_type": "policy",
        }]),
    )
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单 DEMO-1001 售后问题"))
    )
    assert events[-1]["event"] == "done"
    assert state(service)["stop_reason"] == "forbidden_tool"
    assert len([m for m in state(service)["messages"] if m.type == "tool"]) == 1
    with service.session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticket)) == 0
    service.close()


def test_high_risk_agent_cannot_query_another_order_after_policy_gate(tmp_path):
    model = ScriptedModel(
        "退款退货",
        [call("query_order", "other-order", order_id="DEMO-1002")],
    )
    policy = EvidenceRetriever([{
        "chunk_id": 1, "source_key": "policy:return", "score": 0.9,
        "answer": "演示退货条款。", "category": "退换货与退款 / 退货条件",
        "content_type": "policy",
    }])
    service = make_workflow(tmp_path, model, policy)
    events = list(service.stream_events(ChatRequest(
        conversation_id="c", message="订单 DEMO-1001 能退吗？"
    )))
    assert events[-1]["event"] == "done"
    state = service.graph.get_state({"configurable": {"thread_id": "c"}}).values
    assert state["route"] == "high_risk"
    assert state["stop_reason"] == "forbidden_tool"
    assert state["tool_trace"][0]["name"] == "query_order"
    assert model.bound_tools == []
    service.close()


def test_repeated_read_only_call_is_stopped(tmp_path):
    model = ScriptedModel(
        "订单",
        [
            call("query_order", "1", order_id="1001"),
            call("query_order", "2", order_id="1001"),
        ],
    )
    service = make_workflow(tmp_path, model)
    list(service.stream_events(ChatRequest(conversation_id="c", message="查订单1001")))
    assert state(service)["stop_reason"] == "repeated_tool"
    assert state(service)["tool_calls"] == 1
    service.close()


def test_decision_limit_stops_with_tool_calls_paired(tmp_path):
    model = ScriptedModel(
        "订单",
        [
            call("query_order", "1", order_id="1001"),
            call("query_order", "2", order_id="1002"),
        ],
    )
    service = make_workflow(tmp_path, model, limits=Limits(max_decisions=1))
    list(service.stream_events(ChatRequest(conversation_id="c", message="查订单1001")))
    assert state(service)["decisions"] == 1
    assert state(service)["stop_reason"] == "decision_limit"
    assert len([m for m in state(service)["messages"] if m.type == "tool"]) == 1
    service.close()


def test_token_budget_prevents_agent_model_call(tmp_path):
    model = ScriptedModel("物流")
    service = make_workflow(tmp_path, model, limits=Limits(max_tokens=1400))
    list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单1001物流"))
    )
    assert state(service)["stop_reason"] == "token_budget"
    assert state(service)["decisions"] == 0
    service.close()


def test_final_stream_is_not_buffered_until_complete(tmp_path):
    release = Event()

    class Streaming(ScriptedModel):
        def stream(self, messages):
            yield AIMessageChunk(content="正在")
            if not release.wait(3):
                raise RuntimeError("consumer never received first chunk")
            yield AIMessageChunk(content="查询结果")

    service = make_workflow(tmp_path, Streaming("订单"))
    events = service.stream_events(
        ChatRequest(conversation_id="c", message="订单1001状态")
    )
    first = next(e for e in events if e["event"] == "token")
    assert first["data"]["content"] == "正在"
    release.set()
    assert list(events)[-1]["event"] == "done"
    service.close()


def test_midstream_exception_is_error_without_success_done(tmp_path):
    class Broken(ScriptedModel):
        def stream(self, messages):
            yield AIMessageChunk(content="部分回复")
            raise RuntimeError("provider dropped connection")

    service = make_workflow(tmp_path, Broken("订单"))
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单1001状态"))
    )
    assert events[-1]["event"] == "error"
    assert not any(e["event"] == "done" for e in events)
    service.close()


def test_invalid_completion_json_stops_before_streaming_model_answer(tmp_path):
    class NoAnswer(ScriptedModel):
        def stream(self, messages):
            pytest.fail("invalid protocol must not stream answer")

    service = make_workflow(
        tmp_path, NoAnswer("订单", [AIMessage(content="随便编个答案")])
    )
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="订单1001状态"))
    )
    assert events[-1]["event"] == "done"
    assert state(service)["stop_reason"] == "invalid_agent_signal"
    service.close()


def test_tool_status_includes_running_and_completed(tmp_path):
    service = make_workflow(
        tmp_path, ScriptedModel("订单", [call("query_order", order_id="1001")])
    )
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="查订单1001"))
    )
    statuses = [e["data"]["status"] for e in events if e["event"] == "tool_status"]
    assert statuses == ["running", "done"]
    service.close()


def test_unexpected_executor_failure_is_observation_not_orphaned_call(tmp_path):
    service = make_workflow(
        tmp_path, ScriptedModel("订单", [call("query_order", order_id="1001")])
    )

    class BrokenRunner:
        def run(self, *args):
            raise RuntimeError("executor stopped unexpectedly")

        def close(self):
            pass

    service.runner_factory = lambda tools: BrokenRunner()
    events = list(
        service.stream_events(ChatRequest(conversation_id="c", message="查订单1001"))
    )
    assert events[-1]["event"] == "done"
    assert len([m for m in state(service)["messages"] if m.type == "tool"]) == 1
    assert state(service)["tool_trace"][0]["is_error"] is True
    service.close()


def test_error_log_preserves_completed_decision_and_tool_usage(tmp_path):
    class BrokenAnswer(ScriptedModel):
        def stream(self, messages):
            raise RuntimeError("provider unavailable")
            yield

    service = make_workflow(
        tmp_path, BrokenAnswer("订单", [call("query_order", order_id="1001")])
    )
    list(service.stream_events(ChatRequest(conversation_id="c", message="查订单1001")))
    record = json.loads(service.log_path.read_text().splitlines()[-1])
    assert record["decisions"] == 2
    assert record["tool_calls"] == 1
    assert record["tokens"] > 0
    service.close()


@pytest.mark.parametrize("boundary", ["persist", "runner_factory", "runner_close"])
def test_failed_tool_node_does_not_poison_followup_protocol(tmp_path, boundary):
    model = ScriptedModel("订单", [call("query_order", "orphan", order_id="1001")])
    service = make_workflow(tmp_path, model)
    original_save, original_factory = (
        service.store.save_tool_pair,
        service.runner_factory,
    )

    def fail(*args):
        raise RuntimeError("transient failure")

    if boundary == "persist":
        service.store.save_tool_pair = fail
    elif boundary == "runner_factory":
        service.runner_factory = fail
    else:

        def factory(tools):
            runner = original_factory(tools)
            runner.close = fail
            return runner

        service.runner_factory = factory
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="c", message="查订单1001")
            )
        )
        assert events[-1]["event"] == "error"
        service.store.save_tool_pair, service.runner_factory = (
            original_save,
            original_factory,
        )
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="c", message="再查询订单1001")
            )
        )
        assert events[-1]["event"] == "done"
        for messages in model.calls:
            pending = set()
            for message in messages:
                if message.type == "ai":
                    pending.update(c["id"] for c in message.tool_calls)
                elif message.type == "tool":
                    pending.discard(message.tool_call_id)
                elif message.type == "human":
                    assert not pending, "follow-up sent before tool observations"
            assert not pending, "model received orphaned tool request"
    finally:
        service.close()


@pytest.mark.parametrize("actual_usage", [False, True])
def test_final_stream_over_budget_is_error_without_success_save(tmp_path, actual_usage):
    class OverBudget(ScriptedModel):
        def invoke(self, messages):
            response = super().invoke(messages)
            response.usage_metadata = {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            }
            return response

        def stream(self, messages):
            if actual_usage:
                yield AIMessageChunk(
                    content="部分回复",
                    usage_metadata={
                        "input_tokens": 5000,
                        "output_tokens": 10,
                        "total_tokens": 5010,
                    },
                )
            else:
                yield AIMessageChunk(content="查询结果" * 1000)

    service = make_workflow(
        tmp_path, OverBudget("订单"), limits=Limits(max_tokens=5000)
    )
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="c", message="订单1001状态")
            )
        )
        assert events[-1]["event"] == "error"
        assert not any(e["event"] == "done" for e in events)
        assert state(service).get("message_id") is None
        records = [
            json.loads(line)
            for line in (tmp_path / "workflow.jsonl").read_text().splitlines()
        ]
        assert records[-1]["tokens"] > 5000
    finally:
        service.close()
