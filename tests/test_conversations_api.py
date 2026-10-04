from fastapi.testclient import TestClient
from workflow_helpers import make_workflow

from app.main import create_app
from app.schemas import ChatRequest


def test_list_and_complete_original_history_with_ownership(tmp_path):
    service = make_workflow(tmp_path)
    for cid, text in [("old", "你好"), ("new", "谢谢")]:
        list(
            service.stream_events(
                ChatRequest(conversation_id=cid, user_id="u", message=text)
            )
        )
    client = TestClient(create_app(service))
    items = client.get("/api/conversations", params={"user_id": "u"}).json()["items"]
    assert len(items) == 2 and items[0]["conversation_id"] == "new"
    assert items[1]["preview"] == "你好" and not items[1]["has_summary"]
    assert (
        client.get("/api/conversations", params={"user_id": "other"}).json()["items"]
        == []
    )
    response = client.get("/api/conversations/old/messages", params={"user_id": "u"})
    assert response.status_code == 200
    assert [m["role"] for m in response.json()["items"]] == ["user", "assistant"]
    assert response.json()["items"][0]["content"] == "你好"
    assert (
        client.get(
            "/api/conversations/old/messages", params={"user_id": "other"}
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/conversations/missing/messages", params={"user_id": "u"}
        ).status_code
        == 404
    )
    service.close()
