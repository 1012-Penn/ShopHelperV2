from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import func, select

from app.db.models import Ticket
from app.schemas import ChatRequest
from tests.workflow_helpers import ScriptedModel, make_workflow


def ticket_model():
    return ScriptedModel(
        intent="售后",
        decisions=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {"description": "耳机无法充电", "ticket_type": "售后"},
                        "id": "ticket-call",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        chunks=["工单已处理"],
    )


def test_confirmation_requires_preview_and_one_execution(tmp_path):
    service = make_workflow(tmp_path, model=ticket_model())
    try:
        events = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="ticket", message="帮我建个工单，耳机无法充电"
                )
            )
        )
        previews = [e["data"] for e in events if e["event"] == "ticket_preview"]
        assert len(previews) == 1
        with service.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket)) == 0
        from app.schemas import TicketConfirmationRequest

        req = TicketConfirmationRequest(
            conversation_id="ticket",
            request_id=previews[0]["request_id"],
            tool_call_id="ticket-call",
            approve=True,
        )
        resumed = list(service.resume_ticket_events(req))
        assert any(e["event"] == "done" for e in resumed)
        with service.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket)) == 1
            ticket = s.scalar(select(Ticket))
            assert ticket.ticket_no in "".join(
                e["data"].get("content", "") for e in resumed if e["event"] == "token"
            )
        again = list(service.resume_ticket_events(req))
        assert any(e["event"] == "error" for e in again)
        with service.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket)) == 1
    finally:
        service.close()


def test_cancel_and_foreign_confirmation(tmp_path):
    from app.db.models import ToolAuditLog
    from app.schemas import TicketConfirmationRequest

    service = make_workflow(tmp_path, model=ticket_model())
    try:
        list(
            service.stream_events(
                ChatRequest(
                    conversation_id="cancel", message="帮我建个工单，耳机无法充电"
                )
            )
        )
        preview = service.pending_ticket("cancel", "demo-user")
        foreign = TicketConfirmationRequest(
            conversation_id="cancel",
            user_id="other",
            request_id=preview["request_id"],
            tool_call_id="ticket-call",
            approve=True,
        )
        assert list(service.resume_ticket_events(foreign))[-1]["event"] == "error"
        cancelled = TicketConfirmationRequest(
            conversation_id="cancel",
            request_id=preview["request_id"],
            tool_call_id="ticket-call",
            approve=False,
        )
        list(service.resume_ticket_events(cancelled))
        with service.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket)) == 0
            row = s.scalar(
                select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "ticket-call")
            )
            assert row.status == "permission_denied"
            assert "取消" in row.result_summary
    finally:
        service.close()


def test_ticket_request_persists_across_clarification(tmp_path):
    from app.db.models import TicketIntent

    model = ScriptedModel(
        intent="售后",
        decisions=[
            AIMessage(content='{"next":"clarify","missing":"问题描述","actions":[]}')
        ],
        chunks=["请说明问题"],
    )
    service = make_workflow(tmp_path, model=model)
    try:
        list(
            service.stream_events(
                ChatRequest(conversation_id="clarify", message="帮我建个工单")
            )
        )
        with service.session_factory() as s:
            assert s.get(TicketIntent, "clarify").status == "requested"
        model.decisions = ticket_model().decisions
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="clarify", message="耳机无法充电")
            )
        )
        assert any(e["event"] == "ticket_preview" for e in events)
    finally:
        service.close()


def test_unsolicited_ticket_is_rejected(tmp_path):
    from app.db.models import ToolAuditLog

    model = ticket_model()
    model.intent = "订单"
    service = make_workflow(tmp_path, model=model)
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="no", message="请查耳机订单")
            )
        )
        assert not any(e["event"] == "ticket_preview" for e in events)
        with service.session_factory() as s:
            assert s.scalar(select(ToolAuditLog)).status == "permission_denied"
            assert s.scalar(select(func.count()).select_from(Ticket)) == 0
    finally:
        service.close()


def test_read_before_ticket_is_not_replayed_after_resume(tmp_path):
    from app.schemas import TicketConfirmationRequest
    from app.tools.definitions import ToolDefinition

    model = ticket_model()
    model.decisions[0] = AIMessage(
        content="",
        tool_calls=[
            {"name": "test_read", "args": {}, "id": "read-first", "type": "tool_call"},
            {
                "name": "create_ticket",
                "args": {"description": "耳机坏了", "ticket_type": "售后"},
                "id": "ticket-call",
                "type": "tool_call",
            },
        ],
    )
    service = make_workflow(tmp_path, model=model)
    calls = []
    try:
        service.tool_registry.register(
            ToolDefinition(
                "test_read",
                "只读计数",
                {"type": "object", "properties": {}},
                "builtin",
                lambda a, c: calls.append(1) or {"found": True},
            )
        )
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="batch", message="帮我建工单，耳机坏了")
            )
        )
        preview = next(e["data"] for e in events if e["event"] == "ticket_preview")
        assert calls == [1]
        blocked = list(
            service.stream_events(
                ChatRequest(conversation_id="batch", message="另一件事")
            )
        )
        assert blocked[-1]["event"] == "error"
        assert (
            service.pending_ticket("batch", "demo-user")["request_id"]
            == preview["request_id"]
        )
        list(
            service.resume_ticket_events(
                TicketConfirmationRequest(
                    conversation_id="batch",
                    request_id=preview["request_id"],
                    tool_call_id="ticket-call",
                    approve=False,
                )
            )
        )
        assert calls == [1]
        messages = service.graph.get_state(
            {"configurable": {"thread_id": "batch"}}
        ).values["messages"]
        assert {m.tool_call_id for m in messages if isinstance(m, ToolMessage)} == {
            "read-first",
            "ticket-call",
        }
    finally:
        service.close()


def test_ticket_preview_is_not_published_until_checkpoint_can_resume(tmp_path):
    service = make_workflow(tmp_path, model=ticket_model())
    events = service.stream_events(
        ChatRequest(
            conversation_id="preview-close",
            message="帮我建个工单，耳机无法充电",
        )
    )
    try:
        preview = None
        for event in events:
            if event["event"] == "ticket_preview":
                preview = event["data"]
                break
        assert preview is not None
        # Model a browser disconnect as soon as it has received the card.
        events.close()
        snapshot = service.graph.get_state(
            {"configurable": {"thread_id": "preview-close"}}
        )
        assert "await_ticket" in snapshot.next
        pending = service.pending_ticket("preview-close", "demo-user")
        assert pending["request_id"] == preview["request_id"]
    finally:
        events.close()
        service.close()


@pytest.mark.parametrize("cancellation", ["不需要工单了，耳机现在好了", "不建工单了，耳机现在好了", "不提交工单了", "别建单了", "工单取消吧"])
def test_negating_prior_ticket_request_clears_persisted_intent(tmp_path, cancellation):
    from app.db.models import TicketIntent

    model = ScriptedModel(
        intent="售后",
        decisions=[
            AIMessage(content='{"next":"clarify","missing":"问题描述","actions":[]}'),
            AIMessage(content='{"next":"answer","missing":"","actions":[]}'),
        ],
        chunks=["我会继续为你处理。"],
    )
    service = make_workflow(tmp_path, model=model)
    try:
        list(
            service.stream_events(
                ChatRequest(
                    conversation_id="ticket-revoked",
                    message="帮我建工单",
                )
            )
        )
        with service.session_factory() as session:
            assert session.get(TicketIntent, "ticket-revoked").status == "requested"

        list(
            service.stream_events(
                ChatRequest(
                    conversation_id="ticket-revoked",
                    message=cancellation,
                )
            )
        )

        snapshot = service.graph.get_state(
            {"configurable": {"thread_id": "ticket-revoked"}}
        )
        assert snapshot.values["ticket_intent"] is None
        with service.session_factory() as session:
            assert session.get(TicketIntent, "ticket-revoked").status == "cancelled"
    finally:
        service.close()


def test_intent_conservative_negation_and_quote():
    from app.services.workflow.ticket_confirmation import explicit_ticket_request

    assert explicit_ticket_request("帮我建个工单")
    for text in [
        "不要帮我建工单",
        "他告诉我“帮我建个工单”",
        "怎么创建工单",
        "可以帮我创建工单吗",
        "我要查询工单",
    ]:
        assert not explicit_ticket_request(text), text


def test_concurrent_confirmation_of_same_preview_creates_one_ticket(tmp_path):
    from app.schemas import TicketConfirmationRequest

    service = make_workflow(tmp_path, model=ticket_model())
    try:
        events = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="ticket-concurrent",
                    message="帮我建个工单，耳机无法充电",
                )
            )
        )
        preview = next(e["data"] for e in events if e["event"] == "ticket_preview")
        request = TicketConfirmationRequest(
            conversation_id="ticket-concurrent",
            request_id=preview["request_id"],
            tool_call_id=preview["tool_call_id"],
            approve=True,
        )

        def confirm():
            return list(service.resume_ticket_events(request))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _index: confirm(), range(2)))

        endings = [result[-1]["event"] for result in results]
        assert sorted(endings) == ["done", "error"]
        with service.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Ticket)) == 1
    finally:
        service.close()


def test_requested_ticket_intent_survives_service_restart_during_clarification(
    tmp_path,
):
    from app.db.models import TicketIntent

    clarification_model = ScriptedModel(
        intent="售后",
        decisions=[
            AIMessage(content='{"next":"clarify","missing":"问题描述","actions":[]}')
        ],
        chunks=["请说明问题"],
    )
    first_service = make_workflow(tmp_path, model=clarification_model)
    list(
        first_service.stream_events(
            ChatRequest(conversation_id="restart-clarify", message="帮我建个工单")
        )
    )
    with first_service.session_factory() as session:
        assert session.get(TicketIntent, "restart-clarify").status == "requested"
    first_service.close()

    restarted_service = make_workflow(tmp_path, model=ticket_model())
    try:
        events = list(
            restarted_service.stream_events(
                ChatRequest(
                    conversation_id="restart-clarify", message="耳机无法充电"
                )
            )
        )
        assert any(event["event"] == "ticket_preview" for event in events)
        with restarted_service.session_factory() as session:
            assert session.get(TicketIntent, "restart-clarify").status == "requested"
        with restarted_service.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Ticket)) == 0
    finally:
        restarted_service.close()


def test_order_selection_payload_cannot_resume_ticket_preview(tmp_path):
    service = make_workflow(tmp_path, model=ticket_model())
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="cross-ticket", message="帮我建个工单")
            )
        )
        preview = next(event["data"] for event in events if event["event"] == "ticket_preview")
        attempted = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="cross-ticket",
                    selected_order_id="DEMO-1001",
                    selection_message_id=preview["message_id"],
                    request_id=preview["request_id"],
                )
            )
        )
        assert attempted[-1]["event"] == "error"
        pending = service.pending_ticket("cross-ticket", "demo-user")
        assert pending["request_id"] == preview["request_id"]
        with service.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Ticket)) == 0
    finally:
        service.close()


def test_ticket_confirmation_payload_cannot_resume_order_selection(tmp_path):
    from app.schemas import TicketConfirmationRequest
    from tests.workflow_helpers import EvidenceRetriever

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
    service = make_workflow(
        tmp_path, ScriptedModel("退款退货"), EvidenceRetriever(evidence)
    )
    try:
        events = list(
            service.stream_events(
                ChatRequest(conversation_id="cross-order", message="这个耳机能退吗？")
            )
        )
        offer = next(event["data"] for event in events if event["event"] == "order_choices")
        attempted = list(
            service.resume_ticket_events(
                TicketConfirmationRequest(
                    conversation_id="cross-order",
                    request_id=offer["request_id"],
                    tool_call_id="ticket-call",
                    approve=True,
                )
            )
        )
        assert attempted[-1]["event"] == "error"
        snapshot = service.graph.get_state(
            {"configurable": {"thread_id": "cross-order"}}
        )
        assert "await_order" in snapshot.next
        assert "await_ticket" not in snapshot.next
        with service.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Ticket)) == 0
    finally:
        service.close()


def test_complaint_button_and_separate_explicit_chat_each_create_one_ticket(tmp_path):
    from app.schemas import TicketConfirmationRequest

    model = ticket_model()
    model.intent = "投诉"
    service = make_workflow(tmp_path, model=model)
    try:
        complaint_events = list(
            service.stream_events(
                ChatRequest(conversation_id="complaint-button", message="我要投诉")
            )
        )
        actions = next(
            event["data"] for event in complaint_events if event["event"] == "actions"
        )
        assert "create_ticket" in [item["type"] for item in actions["items"]]
        outcome = service.ticket_actions.submit(
            "complaint-button", "demo-user", actions["message_id"]
        )
        assert outcome["status"] == "created"

        explicit_events = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="explicit-ticket-chat",
                    message="帮我建个工单，耳机无法充电",
                )
            )
        )
        preview = next(
            event["data"]
            for event in explicit_events
            if event["event"] == "ticket_preview"
        )
        confirmed = list(
            service.resume_ticket_events(
                TicketConfirmationRequest(
                    conversation_id="explicit-ticket-chat",
                    request_id=preview["request_id"],
                    tool_call_id=preview["tool_call_id"],
                    approve=True,
                )
            )
        )
        assert confirmed[-1]["event"] == "done"
        with service.session_factory() as session:
            assert session.scalar(
                select(func.count())
                .select_from(Ticket)
                .where(Ticket.conversation_id == "complaint-button")
            ) == 1
            assert session.scalar(
                select(func.count())
                .select_from(Ticket)
                .where(Ticket.conversation_id == "explicit-ticket-chat")
            ) == 1
            assert session.scalar(select(func.count()).select_from(Ticket)) == 2
    finally:
        service.close()
