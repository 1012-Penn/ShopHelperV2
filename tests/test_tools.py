import json
import time
from pathlib import Path

import pytest
from langchain.tools import tool
from sqlalchemy import select

from app.db.models import Conversation, FAQ, Ticket
from app.tools.business import build_tools
from app.tools.registry import ToolInputError, ToolRegistry, ToolRunner, UnknownToolError
from scripts.seed import seed_faq


@pytest.fixture

def tools(db_session_factory):
    return build_tools(db_session_factory, "demo-tools")


@pytest.fixture

def tools_by_name(db_session_factory):
    with db_session_factory.begin() as session:
        session.add(Conversation(conversation_id="demo-tools", user_id="demo-user", status="open"))
    return {business_tool.name: business_tool for business_tool in build_tools(db_session_factory, "demo-tools")}


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
    calls = {"flaky": 0, "always_fail": 0, "slow": 0}

    @tool
    def flaky() -> str:
        """Succeed after one transient failure."""
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise RuntimeError("temporary")
        return "ready"

    @tool
    def always_fail() -> str:
        """Always fail to exercise the retry limit."""
        calls["always_fail"] += 1
        raise RuntimeError("temporary")

    @tool
    def slow() -> str:
        """Finish only after the configured tool timeout."""
        calls["slow"] += 1
        time.sleep(0.05)
        return "late"

    return (
        ToolRunner(ToolRegistry([flaky]), timeout_seconds=0.2, max_retries=2),
        ToolRunner(ToolRegistry([always_fail]), timeout_seconds=0.2, max_retries=2),
        ToolRunner(ToolRegistry([slow]), timeout_seconds=0.001, max_retries=0),
        calls,
    )


def test_registry_contains_only_five_business_tools(tools):
    assert {business_tool.name for business_tool in tools} == {
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket"
    }


def test_demo_data_tools_return_labeled_sample_data(tools_by_name):
    assert "模拟" in tools_by_name["query_order"].invoke({"order_id": "1001"})
    assert "模拟" in tools_by_name["query_product"].invoke({"product_query": "耳机"})
    assert "模拟" in tools_by_name["query_logistics"].invoke({"order_id": "1001"})


def test_create_ticket_uses_bound_conversation(db_session, tools_by_name):
    result = json.loads(tools_by_name["create_ticket"].invoke({"description": "商品故障", "ticket_type": "退货"}))
    ticket = db_session.scalar(select(Ticket).where(Ticket.ticket_no == result["ticket_no"]))
    assert ticket.conversation_id == "demo-tools"


def test_query_faq_uses_literal_like_and_misses_postage(db_session, tools):
    seed_faq(db_session)
    registry = ToolRegistry(tools)
    matched = json.loads(registry.get("query_faq").invoke({"query": "退货政策是什么"}))
    missed = json.loads(registry.get("query_faq").invoke({"query": "邮费是多少"}))
    assert matched["matched"] is True
    assert matched["items"][0]["answer"] == "商品签收后 7 天内可申请退货，商品需保持完好。"
    assert missed == {"matched": False, "message": "FAQ 未命中关键词：邮费是多少"}


def test_faq_evaluation_cases_match_labels(db_session, tools):
    seed_faq(db_session)
    faq = ToolRegistry(tools).get("query_faq")
    cases = json.loads((Path(__file__).parent / "fixtures" / "faq_cases.json").read_text())

    for case in cases:
        result = json.loads(faq.invoke({"query": case["query"]}))
        assert bool(result["matched"]) is (case["expected"] == "hit"), case["query"]


def test_invalid_tool_arguments_are_rejected_without_retry(runner_with_counted_order):
    runner, calls = runner_with_counted_order
    with pytest.raises(ToolInputError):
        runner.run("query_order", {"unexpected": True}, "call-invalid")
    assert calls["query_order"] == 0


def test_runner_rejects_unknown_tool_without_execution(runner):
    with pytest.raises(UnknownToolError):
        runner.run("drop_database", {}, "call-x")


def test_runner_retries_timeout_and_returns_safe_error(runner_with_flaky_and_slow_tools):
    transient, always_fail, slow, calls = runner_with_flaky_and_slow_tools
    assert transient.run("flaky", {}, "call-1").is_error is False
    assert calls["flaky"] == 2
    exhausted = always_fail.run("always_fail", {}, "call-exhausted")
    assert exhausted.is_error is True
    assert calls["always_fail"] == 3
    timed_out = slow.run("slow", {}, "call-2")
    assert timed_out.is_error is True
    assert "Traceback" not in timed_out.content
