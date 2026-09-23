import json

from fastapi.testclient import TestClient

from app.main import create_app


def parse_sse(payload: str) -> list[tuple[str, dict]]:
    events = []
    for frame in payload.strip().split("\n\n"):
        lines = frame.splitlines()
        event_name = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = json.loads(next(line.removeprefix("data: ") for line in lines if line.startswith("data: ")))
        events.append((event_name, data))
    return events


class FakeChatService:
    def stream_events(self, request):
        assert request.conversation_id == "api-1"
        assert request.user_id == "demo-user"
        yield {"event": "tool_status", "data": {"tool_name": "query_faq", "status": "running"}}
        yield {"event": "token", "data": {"content": "退货政策"}}
        yield {"event": "done", "data": {"conversation_id": request.conversation_id}}


class FailingChatService:
    def stream_events(self, _request):
        yield {"event": "token", "data": {"content": "部分"}}
        raise RuntimeError("private model endpoint detail")


def test_chat_stream_returns_tool_status_tokens_and_done():
    client = TestClient(create_app(FakeChatService()))
    response = client.post("/api/v1/chat/stream", json={"conversation_id": "api-1", "message": "查退货政策"})

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert parse_sse(response.text) == [
        ("tool_status", {"tool_name": "query_faq", "status": "running"}),
        ("token", {"content": "退货政策"}),
        ("done", {"conversation_id": "api-1"}),
    ]


def test_chat_stream_turns_service_failure_into_safe_error_event():
    client = TestClient(create_app(FailingChatService()))
    response = client.post("/api/v1/chat/stream", json={"conversation_id": "api-2", "message": "查商品"})
    events = parse_sse(response.text)

    assert [event[0] for event in events] == ["token", "error"]
    assert "private model endpoint detail" not in response.text
    assert "Traceback" not in response.text
