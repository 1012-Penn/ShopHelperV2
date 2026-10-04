import json
import time
from pathlib import Path

import pytest
from langchain.tools import tool
from sqlalchemy import select

from app.db.models import Conversation, Ticket
from app.services.knowledge.retriever import FAQHit
from app.tools.business import build_tools
from app.tools.definitions import ExecutionContext, ToolDefinition
from app.tools.registry import ToolRegistry, ToolRunner


class FixtureFAQRetriever:
    def search(self, query):
        if "退货" in query or "退换货" in query:
            return [
                FAQHit(
                    question="退换货条件",
                    answer="退换货资格和期限以购买时适用的政策与订单详情为准。",
                    category="退换货",
                    score=0.91,
                )
            ]
        if "邮费" in query or "运费" in query:
            return [
                FAQHit(
                    question="运费计算",
                    answer="请在结算页填写收货地址后查看当前订单运费。",
                    category="配送",
                    score=0.89,
                )
            ]
        return []


@pytest.fixture

def tools(db_session_factory):
    return build_tools(db_session_factory, "demo-tools", FixtureFAQRetriever())


@pytest.fixture

def tools_by_name(db_session_factory):
    with db_session_factory.begin() as session:
        session.add(Conversation(conversation_id="demo-tools", user_id="demo-user", status="open"))
    return {
        business_tool.name: business_tool
        for business_tool in build_tools(db_session_factory, "demo-tools", FixtureFAQRetriever())
    }


@pytest.fixture

def runner(tools):
    return ToolRunner(ToolRegistry(tools), timeout_seconds=0.2, max_retries=2)


@pytest.fixture

def runner_with_counted_order():
    calls = {"query_order": 0}

    @tool("query_order")
    def counted_order(order_id: str) -> str:
        """Count execution of a validated order lookup."""
        calls["query_order"] += 1
        return order_id

    return ToolRunner(ToolRegistry([counted_order]), timeout_seconds=0.2, max_retries=2), calls


@pytest.fixture

def runner_with_flaky_and_slow_tools():
    calls = {"flaky": 0, "always_fail": 0, "slow": 0, "business_error": 0}

    @tool
    def flaky() -> str:
        """Succeed after one transient failure."""
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise ConnectionError("temporary network issue")
        return "ready"

    def always_fail(args, context):
        """Always time out to exercise the retry limit."""
        calls["always_fail"] += 1
        raise TimeoutError("temporary network issue")

    @tool
    def business_error() -> str:
        """Fail with a non-transient error that must not be retried."""
        calls["business_error"] += 1
        raise RuntimeError("invalid business state")

    @tool
    def slow() -> str:
        """Finish only after the configured tool timeout."""
        calls["slow"] += 1
        time.sleep(0.05)
        return "late"

    return (
        ToolRunner(ToolRegistry([flaky]), timeout_seconds=0.2, max_retries=2),
        ToolRunner(ToolRegistry([ToolDefinition(
            "always_fail", "Always time out", {
                "type": "object", "properties": {}, "additionalProperties": False,
            }, "builtin", always_fail,
        )]), timeout_seconds=0.2, max_retries=2),
        ToolRunner(ToolRegistry([slow]), timeout_seconds=0.001, max_retries=0),
        ToolRunner(ToolRegistry([business_error]), timeout_seconds=0.2, max_retries=2),
        calls,
    )


def test_registry_contains_only_four_builtin_business_tools(tools):
    assert {business_tool.name for business_tool in tools} == {
        "query_order", "query_product", "query_faq", "create_ticket"
    }


def test_demo_data_tools_return_labeled_sample_data(tools_by_name):
    assert "模拟" in tools_by_name["query_order"].invoke({"order_id": "1001"})
    assert "模拟" in tools_by_name["query_product"].invoke({"product_query": "耳机"})


def test_create_ticket_uses_bound_conversation(db_session, tools_by_name):
    result = json.loads(tools_by_name["create_ticket"].invoke({"description": "商品故障", "ticket_type": "退货"}))
    ticket = db_session.scalar(select(Ticket).where(Ticket.ticket_no == result["ticket_no"]))
    assert ticket.conversation_id == "demo-tools"


def test_query_faq_preserves_name_input_and_json_contract(tools):
    registry = ToolRegistry(tools)
    faq = registry.get("query_faq")
    assert set(faq.input_schema["properties"]) == {"query"}
    matched = json.loads(faq.invoke({"query": "邮费是多少"}))
    missed = json.loads(faq.invoke({"query": "怎样给家里的猫梳毛"}))
    assert matched["matched"] is True
    assert matched["items"] == [
        {
            "question": "运费计算",
            "answer": "请在结算页填写收货地址后查看当前订单运费。",
            "category": "配送",
        }
    ]
    assert missed == {"matched": False, "message": "FAQ 未命中关键词：怎样给家里的猫梳毛"}


def test_faq_evaluation_cases_match_labels_and_answer_phrases(tools):
    faq = ToolRegistry(tools).get("query_faq")
    cases = json.loads((Path(__file__).parent / "fixtures" / "faq_cases.json").read_text())

    for case in cases:
        result = json.loads(faq.invoke({"query": case["query"]}))
        assert bool(result["matched"]) is (case["expected"] == "hit"), case["query"]
        if case["expected"] == "hit":
            assert case["answer_contains"] in result["items"][0]["answer"]


def test_invalid_tool_arguments_are_rejected_without_retry(runner_with_counted_order):
    runner, calls = runner_with_counted_order
    result = runner.run("query_order", {"unexpected": True}, "call-invalid")
    assert result.status == "validation_blocked"
    assert result.error_kind == "invalid_arguments"
    assert result.retry_count == 0
    assert calls["query_order"] == 0


def test_runner_rejects_unknown_tool_without_execution(runner):
    result = runner.run("drop_database", {}, "call-x")
    assert result.status == "permission_denied"
    assert result.error_kind == "permission_denied"
    assert result.source == "unknown"


def test_runner_retries_transient_failures_but_not_business_errors(runner_with_flaky_and_slow_tools):
    transient, always_fail, slow, business_error, calls = runner_with_flaky_and_slow_tools
    recovered = transient.run("flaky", {}, "call-1")
    assert recovered.is_error is False
    assert recovered.retry_count == 1
    assert calls["flaky"] == 2
    exhausted = always_fail.run("always_fail", {}, "call-exhausted")
    assert exhausted.is_error is True
    assert exhausted.retry_count == 2
    assert calls["always_fail"] == 3
    failed = business_error.run("business_error", {}, "call-business-error")
    assert failed.is_error is True
    assert failed.retry_count == 0
    assert calls["business_error"] == 1
    timed_out = slow.run("slow", {}, "call-2")
    assert timed_out.is_error is True
    assert timed_out.status == "timeout"
    assert timed_out.retry_count == 0
    assert "Traceback" not in timed_out.content


def test_timed_out_ticket_does_not_wait_retry_or_repeat_same_call():
    created = {"count": 0}

    @tool("create_ticket")
    def delayed_ticket(description: str, ticket_type: str) -> str:
        """Simulate a ticket commit that finishes after its timeout."""
        time.sleep(0.2)
        created["count"] += 1
        return "ticket-created"

    runner = ToolRunner(ToolRegistry([delayed_ticket]), timeout_seconds=0.03, max_retries=2)
    context = ExecutionContext("conv-ticket", "demo-user", True, "call-ticket")
    started = time.monotonic()
    result = runner.run("create_ticket", {"description": "商品故障", "ticket_type": "售后"}, "call-ticket", context)

    assert time.monotonic() - started < 0.055
    assert result.is_error is True
    assert result.status == "timeout"
    assert result.retry_count == 0
    duplicate = runner.run("create_ticket", {"description": "商品故障", "ticket_type": "售后"}, "call-ticket", context)
    assert duplicate.is_error is True
    assert duplicate.status == "permission_denied"
    time.sleep(0.22)
    assert created["count"] == 1
    runner.close()



def test_confirmed_ticket_timeout_returns_bounded_unknown_outcome():
    @tool("create_ticket")
    def stalled_ticket(description: str, ticket_type: str) -> str:
        """Simulate a ticket write blocked beyond the configured timeout."""
        time.sleep(0.2)
        return "ticket-created"

    runner = ToolRunner(ToolRegistry([stalled_ticket]), timeout_seconds=0.005, max_retries=2)
    context = ExecutionContext("conv-stalled", "demo-user", True, "call-stalled")

    started = time.monotonic()
    result = runner.run("create_ticket", {"description": "商品故障", "ticket_type": "售后"}, "call-stalled", context)

    assert time.monotonic() - started < 0.1
    assert result.is_error is True
    assert result.status == "timeout"
    assert result.retry_count == 0
    assert "结果暂未确认" in result.content
    assert "请勿重复提交" in result.content
