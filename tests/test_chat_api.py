import json

import pytest
from fastapi.testclient import TestClient


class FakeChatService:
    def __init__(self, chunks=None, error_after=None):
        self.chunks = chunks or ["您好", "，请提供订单号。"]
        self.error_after = error_after
        self.received = None

    async def stream(self, request):
        self.received = request
        for index, chunk in enumerate(self.chunks):
            if self.error_after == index:
                raise RuntimeError("upstream failed")
            yield chunk
        if self.error_after == len(self.chunks):
            raise RuntimeError("upstream failed")


@pytest.fixture
def app_and_service():
    from app.main import app, get_chat_service

    service = FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: service
    yield app, service
    app.dependency_overrides.clear()


def _events(body):
    blocks = [block for block in body.strip().split("\n\n") if block]
    return [
        {
            "event": block.split("\n", 1)[0].removeprefix("event: "),
            "data": json.loads(block.split("\ndata: ", 1)[1]),
        }
        for block in blocks
    ]


def test_chat_stream_returns_token_and_done_events_and_forwards_history(
    app_and_service,
):
    app, service = app_and_service
    payload = {
        "conversation_id": "c1",
        "message": "我的订单还没收到",
        "history": [{"role": "user", "content": "我想查物流"}],
    }

    response = TestClient(app).post("/api/v1/chat/stream", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _events(response.text) == [
        {"event": "token", "data": {"content": "您好"}},
        {"event": "token", "data": {"content": "，请提供订单号。"}},
        {"event": "done", "data": {"conversation_id": "c1"}},
    ]
    assert service.received.history[0].content == "我想查物流"


def test_chat_stream_json_encodes_special_text(app_and_service):
    app, service = app_and_service
    service.chunks = ['第一行\n"第二行" 中文']

    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c2", "message": "继续", "history": []},
    )

    assert _events(response.text)[0] == {
        "event": "token",
        "data": {"content": '第一行\n"第二行" 中文'},
    }


def test_chat_stream_emits_error_without_done_after_upstream_failure(
    app_and_service,
):
    app, service = app_and_service
    service.error_after = 1

    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c3", "message": "继续", "history": []},
    )

    events = _events(response.text)
    assert events[0] == {"event": "token", "data": {"content": "您好"}}
    assert events[-1]["event"] == "error"
    assert all(event["event"] != "done" for event in events)


def test_chat_stream_rejects_blank_message(app_and_service):
    app, _ = app_and_service

    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c4", "message": "", "history": []},
    )

    assert response.status_code == 422
