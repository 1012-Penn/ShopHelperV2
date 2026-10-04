import json

from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from workflow_helpers import ScriptedModel, make_workflow

from app.config import Settings
from app.db.models import Ticket
from app.main import create_app


def test_tickets_api_requires_suggestion_and_correct_owner(tmp_path):
    workflow = make_workflow(tmp_path, ScriptedModel("投诉"))
    with TestClient(create_app(chat_service=workflow)) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={"conversation_id": "api", "message": "我要投诉"},
        )
        assert response.status_code == 200
        done = next(
            json.loads(block.split("data: ", 1)[1])
            for block in response.text.split("\n\n")
            if block.startswith("event: done")
        )
        request = {
            "conversation_id": "api",
            "user_id": "demo-user",
            "message_id": done["message_id"],
        }
        denied = client.post("/api/v1/tickets", json={**request, "user_id": "intruder"})
        assert denied.status_code == 403
        first = client.post("/api/v1/tickets", json=request)
        again = client.post("/api/v1/tickets", json=request)
        assert first.status_code == 200
        assert first.json() == again.json()
        assert first.json()["status"] == "created"
        with workflow.session_factory() as s:
            assert s.scalar(select(func.count()).select_from(Ticket)) == 1


def test_lifespan_closes_checkpoint_after_last_request(tmp_path):
    workflow = make_workflow(tmp_path)
    with TestClient(create_app(chat_service=workflow)) as client:
        assert client.get("/health").status_code == 200
    import sqlite3

    try:
        workflow.checkpointer.conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return
    assert False, "checkpoint connection must be closed"


def test_default_factory_classifies_greeting_with_the_intent_model(
    tmp_path, monkeypatch
):
    from app.services.workflow import runtime
    from workflow_helpers import ScriptedModel

    settings = Settings(
        "test", "unused", "https://provider.invalid", f"sqlite:///{tmp_path}/runtime.db"
    )
    model = ScriptedModel("闲聊")
    monkeypatch.setattr(runtime, "ChatOpenAI", lambda **kwargs: model)
    workflow = runtime.build_workflow_service(
        settings,
        {
            "WORKFLOW_CHECKPOINT_PATH": str(tmp_path / "saver.sqlite"),
            "WORKFLOW_LOG_PATH": str(tmp_path / "log.jsonl"),
            "QUALITY_ENABLED": "true",
        },
    )
    with TestClient(create_app(chat_service=workflow)) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"conversation_id": "g", "message": "你好"}
        )
        assert "event: done" in response.text
        assert "event: error" not in response.text
        assert len(model.calls) == 2


def test_blank_message_and_invalid_ticket_id_rejected(tmp_path):
    workflow = make_workflow(tmp_path)
    with TestClient(create_app(chat_service=workflow)) as client:
        assert (
            client.post(
                "/api/v1/chat/stream", json={"conversation_id": "c", "message": "   "}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/tickets", json={"conversation_id": "c", "message_id": 0}
            ).status_code
            == 422
        )


def test_order_resume_payload_requires_all_bound_selection_ids():
    import pytest
    from pydantic import ValidationError

    from app.schemas import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(
            conversation_id="c",
            selected_order_id="DEMO-1001",
            selection_message_id=17,
        )
    resumed = ChatRequest(
        conversation_id="c",
        selected_order_id="DEMO-1001",
        selection_message_id=17,
        request_id="order-choice-1",
    )
    assert resumed.message == ""


def test_primary_agent_refund_form_event_submits_local_demo_application(tmp_path):
    from workflow_helpers import EvidenceRetriever

    policy = [{
        "chunk_id": 17, "source_key": "policy:return", "score": 0.95,
        "answer": "订单需符合演示政策。", "category": "退换货与退款 / 退货条件",
        "content_type": "policy",
    }]
    model = ScriptedModel(
        "退款退货",
        [AIMessage(content='{"next":"answer","actions":["refund_form"]}')],
    )
    workflow = make_workflow(tmp_path, model, EvidenceRetriever(policy))
    with TestClient(create_app(chat_service=workflow)) as client:
        streamed = client.post(
            "/api/v1/chat/stream",
            json={"conversation_id": "refund-api", "message": "订单 DEMO-1001 能退吗？"},
        )
        assert streamed.status_code == 200
        event_data = {
            block.splitlines()[0].removeprefix("event: "): json.loads(
                block.split("data: ", 1)[1]
            )
            for block in streamed.text.split("\n\n")
            if block.startswith("event: ") and "data: " in block
        }
        form = event_data["refund_form"]
        payload = {
            "conversation_id": "refund-api",
            "user_id": "demo-user",
            "message_id": form["message_id"],
            "request_id": form["request_id"],
            "order_id": form["order_id"],
            "request_type": form["request_type"],
            "reason": "不想要/不合适",
        }
        response = client.post("/api/v1/refund-applications", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"
        assert "未执行真实退款" in response.json()["message"]
