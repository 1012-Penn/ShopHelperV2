import json
import re

from fastapi.testclient import TestClient

from app.main import create_app, get_after_sale_service


class FakeChatService:
    def __init__(self, events):
        self.events = events
        self.received = None

    def stream_events(self, request):
        self.received = request
        return iter(self.events)


def _events(body):
    blocks = [block for block in body.strip().split("\n\n") if block]
    return [
        {
            "event": block.split("\n", 1)[0].removeprefix("event: "),
            "data": json.loads(block.split("\ndata: ", 1)[1]),
        }
        for block in blocks
    ]


def test_chat_stream_preserves_token_and_done_events():
    service = FakeChatService(
        [
            {"event": "token", "data": {"content": "您好"}},
            {"event": "done", "data": {"conversation_id": "c1"}},
        ]
    )
    app = create_app(chat_service=service)

    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c1", "message": "我想查物流"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _events(response.text) == [
        {"event": "token", "data": {"content": "您好"}},
        {"event": "done", "data": {"conversation_id": "c1"}},
    ]
    assert service.received.user_id == "demo-user"


def test_chat_stream_json_encodes_special_text():
    content = '第一行\n"第二行" 中文'
    app = create_app(
        chat_service=FakeChatService([{"event": "token", "data": {"content": content}}])
    )

    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c2", "message": "继续"},
    )

    assert _events(response.text)[0] == {"event": "token", "data": {"content": content}}


def test_chat_stream_rejects_blank_message():
    app = create_app(chat_service=FakeChatService([]))
    response = TestClient(app).post(
        "/api/v1/chat/stream",
        json={"conversation_id": "c4", "message": ""},
    )
    assert response.status_code == 422


def test_root_serves_built_react_workbench_and_assets():
    app = create_app(chat_service=FakeChatService([]))
    client = TestClient(app)

    page = client.get("/")

    assert page.status_code == 200
    assert "喵助理 · 智能客服工作台" in page.text
    asset_path = re.search(r'src="(/assets/[^\"]+\.js)"', page.text).group(1)
    asset = client.get(asset_path)
    assert asset.status_code == 200
    assert "多轮" in asset.text


def test_after_sale_extract_returns_fixed_json_fields():
    class FakeAfterSaleService:
        def extract(self, text):
            assert text == "订单123456耳机坏了，想换货并尽快补发"
            return {
                "order_id": "123456",
                "request_type": "换货",
                "expected_solution": "尽快补发",
            }

    app = create_app(chat_service=FakeChatService([]))
    app.dependency_overrides[get_after_sale_service] = lambda: FakeAfterSaleService()
    response = TestClient(app).post(
        "/api/v1/after-sale/extract",
        json={"text": "订单123456耳机坏了，想换货并尽快补发"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "order_id": "123456",
        "request_type": "换货",
        "expected_solution": "尽快补发",
    }
