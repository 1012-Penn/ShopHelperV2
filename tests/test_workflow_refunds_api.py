import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.models import Conversation, Message
from app.main import create_app
from workflow_helpers import make_workflow


REASONS = ["商品质量问题", "错发/漏发", "不想要/不合适", "其他"]


def seed_offer(workflow, *, conversation_id="refund-conversation", order_id="DEMO-1001",
               request_id="refund-request-1", offered=True, user_id="demo-user"):
    with workflow.session_factory.begin() as session:
        session.add(Conversation(conversation_id=conversation_id, user_id="demo-user", status="open"))
        session.flush()
        row = Message(
            conversation_id=conversation_id,
            role="assistant",
            content="演示订单退款申请表",
            actions={
                "items": ["refund_form"] if offered else [],
                "refund_form": {"order_id": order_id, "request_id": request_id, "request_type": "退货"},
            },
        )
        session.add(row)
        session.flush()
        return row.id


def request_payload(message_id, *, reason="商品质量问题", order_id="DEMO-1001",
                    request_id="refund-request-1", request_type="退货"):
    return {
        "conversation_id": "refund-conversation",
        "user_id": "demo-user",
        "message_id": message_id,
        "request_id": request_id,
        "order_id": order_id,
        "request_type": request_type,
        "reason": reason,
    }


@pytest.mark.parametrize("reason", REASONS)
def test_refund_application_records_fixed_reason_once(tmp_path, reason):
    workflow = make_workflow(tmp_path)
    message_id = seed_offer(workflow)
    with TestClient(create_app(chat_service=workflow)) as client:
        payload = request_payload(message_id, reason=reason)

        first = client.post("/api/v1/refund-applications", json=payload)
        repeated = client.post("/api/v1/refund-applications", json=payload)

    assert first.status_code == 200
    assert first.json() == repeated.json()
    assert first.json()["status"] == "recorded"
    assert "演示" in first.json()["message"]
    with workflow.session_factory() as session:
        from app.db.models import RefundApplication

        assert session.scalar(select(func.count()).select_from(RefundApplication)) == 1
        saved = session.get(RefundApplication, payload["request_id"])
        assert saved.reason == reason
        assert saved.order_id == payload["order_id"]


def test_refund_application_rejects_unoffered_form_and_wrong_order(tmp_path):
    workflow = make_workflow(tmp_path)
    message_id = seed_offer(workflow, offered=False)
    with TestClient(create_app(chat_service=workflow)) as client:
        no_offer = client.post(
            "/api/v1/refund-applications", json=request_payload(message_id)
        )
        wrong_order = client.post(
            "/api/v1/refund-applications",
            json=request_payload(message_id, order_id="DEMO-1002"),
        )

    assert no_offer.status_code == 403
    assert wrong_order.status_code == 403


def test_refund_application_checks_conversation_owner(tmp_path):
    workflow = make_workflow(tmp_path)
    message_id = seed_offer(workflow)
    with TestClient(create_app(chat_service=workflow)) as client:
        response = client.post(
            "/api/v1/refund-applications",
            json={**request_payload(message_id), "user_id": "intruder"},
        )

    assert response.status_code == 403


def test_refund_application_rejects_invalid_reason_or_demo_order(tmp_path):
    workflow = make_workflow(tmp_path)
    message_id = seed_offer(workflow)
    with TestClient(create_app(chat_service=workflow)) as client:
        invalid_reason = client.post(
            "/api/v1/refund-applications",
            json=request_payload(message_id, reason="其他原因（自由输入）"),
        )
        invalid_order = client.post(
            "/api/v1/refund-applications",
            json=request_payload(message_id, order_id="NOT-A-DEMO-ORDER"),
        )

    assert invalid_reason.status_code == 422
    assert invalid_order.status_code == 422


def test_refund_request_id_cannot_be_reused_for_conflicting_content(tmp_path):
    workflow = make_workflow(tmp_path)
    message_id = seed_offer(workflow)
    with TestClient(create_app(chat_service=workflow)) as client:
        payload = request_payload(message_id)
        assert client.post("/api/v1/refund-applications", json=payload).status_code == 200
        conflict = client.post(
            "/api/v1/refund-applications",
            json={**payload, "reason": "其他"},
        )

    assert conflict.status_code == 409
    with workflow.session_factory() as session:
        from app.db.models import RefundApplication

        assert session.scalar(select(func.count()).select_from(RefundApplication)) == 1
