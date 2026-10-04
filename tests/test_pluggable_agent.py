import pytest
from langchain_core.messages import AIMessage

from app.schemas import ChatRequest
from app.tools.definitions import ToolDefinition
from tests.workflow_helpers import ScriptedModel, make_workflow


def test_registered_tool_is_used_without_agent_core_change(tmp_path):
    model = ScriptedModel(
        intent="订单",
        decisions=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_shop_hours",
                        "args": {},
                        "id": "hours-call",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    )
    service = make_workflow(tmp_path, model=model)
    try:
        calls = []
        service.tool_registry.register(
            ToolDefinition(
                "query_shop_hours",
                "查询门店营业时间",
                {"type": "object", "properties": {}},
                "builtin",
                lambda args, context: (
                    calls.append(context.conversation_id) or {"hours": "09:00—18:00"}
                ),
            )
        )
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="hours", message="查门店营业时间")
            )
        )
        assert calls == ["hours"]
        assert "query_shop_hours" in model.bound_tools
        assert any(e["event"] == "done" for e in events)
    finally:
        service.close()


def test_after_sale_status_query_uses_mcp_business_route(tmp_path):
    model = ScriptedModel(
        intent="售后",
        decisions=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_warranty",
                        "args": {"order_id": "DEMO-1001"},
                        "id": "w",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    )
    service = make_workflow(tmp_path, model=model)
    try:
        events = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="warranty", message="查订单DEMO-1001是否在保"
                )
            )
        )
        assert not any(e["event"] == "order_choices" for e in events)
        assert any(
            e["event"] == "tool_status" and e["data"]["tool_name"] == "query_warranty"
            for e in events
        )
    finally:
        service.close()


def test_lazy_retriever_forwards_policy_filters():
    from app.services.workflow.runtime import LazyRetriever

    seen = []

    class Retriever:
        def retrieve(self, question, category=None, **filters):
            seen.append((question, category, filters))
            return [], {}

    lazy = LazyRetriever(Retriever)
    assert lazy.retrieve(
        "退货政策", category_prefixes=["退换货"], content_types=["policy"]
    ) == ([], {})
    assert seen == [
        (
            "退货政策",
            None,
            {"category_prefixes": ["退换货"], "content_types": ["policy"]},
        )
    ]


def test_ticket_intent_does_not_split_tool_protocol(tmp_path):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    service = make_workflow(tmp_path)
    try:
        service.context.model_messages = lambda state, prompt: [
            SystemMessage(content=prompt),
            HumanMessage(content="帮我建工单"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_order",
                        "args": {"order_id": "DEMO-1001"},
                        "id": "query",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="{}", tool_call_id="query"),
        ]
        from app.services.workflow.agent import AgentNodes

        messages = AgentNodes(service).messages(
            {
                "ticket_intent": {"description": "无法充电"},
                "tool_observations": [ToolMessage(content="{}", tool_call_id="query")],
            },
            "测试",
        )
        call_index = next(
            i
            for i, m in enumerate(messages)
            if isinstance(m, AIMessage) and m.tool_calls
        )
        assert isinstance(messages[call_index + 1], ToolMessage)
    finally:
        service.close()


def test_invalid_json_tool_call_is_returned_and_audited(tmp_path):
    from langchain_core.messages import ToolMessage
    from sqlalchemy import select

    from app.db.models import ToolAuditLog

    model = ScriptedModel(
        intent="订单",
        decisions=[
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "query_order",
                        "args": '{"order_id":',
                        "id": "bad-json",
                        "type": "invalid_tool_call",
                    }
                ],
            )
        ],
    )
    service = make_workflow(tmp_path, model=model)
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="invalid-json", message="查询订单DEMO-1001")
            )
        )
        with service.session_factory() as session:
            audit = session.scalar(
                select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "bad-json")
            )
            assert audit is not None
            assert audit.status == "validation_blocked"
            assert audit.arguments == '{"order_id":'
        assert any(
            isinstance(m, ToolMessage)
            and m.tool_call_id == "bad-json"
            and m.status == "error"
            for call in model.calls
            for m in call
        )
        assert any(e["event"] == "done" for e in events)
    finally:
        service.close()


@pytest.mark.parametrize(
    "question",
    [
        "订单DEMO-1001已经过保修期，退货行不行？",
        "订单DEMO-1001不在保修期了，退货要满足什么要求？",
    ],
)
def test_compound_expired_warranty_and_return_eligibility_keeps_high_risk_route(
    tmp_path, question
):
    service = make_workflow(tmp_path)
    try:
        assert (
            service._route({"question": question, "intent": "退款退货"})["route"]
            == "high_risk"
        )
    finally:
        service.close()


@pytest.mark.parametrize(
    "question",
    [
        "查订单DEMO-1001是否在保",
        "订单DEMO-1001的保修状态是什么？",
        "保修什么时候到期？",
        "订单DEMO-1001保修期还有多久？",
        "查询订单 DEMO-1001 的商品是否还在保修期内",
        "帮我查询订单 DEMO-1003 的退货处理进度",
        "订单 DEMO-1002 的保修什么时候到期？",
        "查退货处理进度",
        "订单DEMO-1002退货处理进度怎么样？",
    ],
)
def test_complete_pure_after_sale_status_queries_keep_business_route(
    tmp_path, question
):
    service = make_workflow(tmp_path)
    try:
        assert (
            service._route({"question": question, "intent": "售后"})["route"]
            == "business"
        )
    finally:
        service.close()


def test_hot_tool_catalog_over_budget_stops_before_model_call(tmp_path):
    service = make_workflow(tmp_path)
    try:
        for index in range(15):
            service.tool_registry.register(
                ToolDefinition(
                    f"query_large_{index}",
                    "查询用途" * 20000,
                    {"type": "object", "properties": {}},
                    "builtin",
                    lambda args, context: {},
                )
            )
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="huge-catalog", message="查询订单DEMO-1001")
            )
        )
        assert not any(e["event"] == "error" for e in events)
        assert any(e["event"] == "done" for e in events)
    finally:
        service.close()


def test_hot_tool_schema_over_budget_stops_before_decision(tmp_path):
    model = ScriptedModel(
        intent="订单", decisions=[AIMessage(content='{"next":"answer","actions":[]}')]
    )
    service = make_workflow(tmp_path, model=model)
    try:
        service.tool_registry.register(
            ToolDefinition(
                "query_huge_schema",
                "巨型参数查询",
                {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string", "description": "参数说明" * 40000}
                    },
                },
                "builtin",
                lambda args, context: {},
            )
        )
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="huge-schema", message="查询订单DEMO-1001")
            )
        )
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["stop_reason"] == "token_budget"
        assert len(model.decisions) == 1
        assert not any(e["event"] == "tool_status" for e in events)
    finally:
        service.close()
