from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain.tools import tool
from sqlalchemy import select

from app.db.models import Message
from app.schemas import ChatRequest
from app.services.chat import ChatService
from app.tools.registry import ToolRegistry, ToolRunner


class FakeBoundModel:
    def __init__(self, model):
        self.model = model

    def stream(self, messages):
        self.model.first_call_messages = messages
        if self.model.first_error:
            raise RuntimeError("private provider detail")
        return iter(self.model.first_chunks)


class FakeModel:
    def __init__(self, first_chunks, final_chunks=(), first_error=False):
        self.first_chunks = first_chunks
        self.final_chunks = final_chunks
        self.first_error = first_error
        self.bound_tools = []
        self.second_call_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return FakeBoundModel(self)

    def stream(self, messages):
        self.second_call_messages = messages
        return iter(self.final_chunks)


def make_tool_chunks(*calls):
    return [AIMessageChunk(content="", tool_calls=list(calls))]


def make_service(db_session_factory, fake_model, tool_runner=None):
    def runner_factory(tools):
        return tool_runner or ToolRunner(ToolRegistry(tools), timeout_seconds=0.2, max_retries=0)

    return ChatService(db_session_factory, lambda: fake_model, runner_factory)


def test_tool_round_persists_request_result_and_streams_final_tokens(db_session_factory):
    model = FakeModel(
        make_tool_chunks({"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call-1"}),
        [AIMessageChunk(content="物流在运输中。")],
    )
    service = make_service(db_session_factory, model)

    events = list(service.stream_events(ChatRequest(conversation_id="demo-1", message="订单 1001 的物流到哪了")))

    assert [event["event"] for event in events] == ["tool_status", "token", "done"]
    assert events[0]["data"]["tool_name"] == "query_logistics"
    with db_session_factory() as session:
        messages = session.scalars(select(Message).order_by(Message.id)).all()
    assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0]["id"] == messages[2].tool_call_id == "call-1"
    assert messages[3].content == "物流在运输中。"
    assert all(tool.name in {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"} for tool in model.bound_tools)


def test_multiple_tool_calls_are_rejected_without_execution(db_session_factory):
    model = FakeModel(make_tool_chunks(
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "call-1"},
        {"name": "query_product", "args": {"product_query": "耳机"}, "id": "call-2"},
    ))
    runners = []

    class SpyRunner:
        def __init__(self, tools):
            self.registry = ToolRegistry(tools)
            self.calls = []

        def run(self, name, args, tool_call_id):
            self.calls.append((name, args, tool_call_id))
            raise AssertionError("multiple tool calls must not execute")

    def runner_factory(tools):
        runner = SpyRunner(tools)
        runners.append(runner)
        return runner

    service = ChatService(db_session_factory, lambda: model, runner_factory)
    events = list(service.stream_events(ChatRequest(conversation_id="demo-2", message="查订单和商品")))

    assert events[-1]["event"] == "error"
    assert runners[0].calls == []
    with db_session_factory() as session:
        assert [message.role for message in session.scalars(select(Message).order_by(Message.id))] == ["user"]


def test_no_tool_answer_streams_and_persists(db_session_factory):
    model = FakeModel([AIMessageChunk(content="你好"), AIMessageChunk(content="。")])
    events = list(make_service(db_session_factory, model).stream_events(ChatRequest(conversation_id="demo-3", message="你好")))

    assert [event["event"] for event in events] == ["token", "token", "done"]
    with db_session_factory() as session:
        assistant = session.scalar(select(Message).where(Message.role == "assistant"))
    assert assistant.content == "你好。"


def test_tool_error_is_replayed_as_matching_tool_message(db_session_factory):
    @tool("query_logistics")
    def failed_logistics(order_id: str) -> str:
        """Always fail the mocked logistics lookup."""
        raise RuntimeError("private database detail")

    error_runner = ToolRunner(ToolRegistry([failed_logistics]), timeout_seconds=0.2, max_retries=0)
    model = FakeModel(
        make_tool_chunks({"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call-error"}),
        [AIMessageChunk(content="暂时无法查询物流。")],
    )
    service = make_service(db_session_factory, model, error_runner)

    events = list(service.stream_events(ChatRequest(conversation_id="demo-4", message="查物流")))

    assert events[-1]["event"] == "done"
    tool_message = next(message for message in model.second_call_messages if isinstance(message, ToolMessage))
    assert tool_message.tool_call_id == "call-error"
    assert "private database detail" not in str(tool_message.content)


def test_model_failure_emits_safe_error_and_keeps_user_message(db_session_factory):
    model = FakeModel([], first_error=True)
    events = list(make_service(db_session_factory, model).stream_events(ChatRequest(conversation_id="demo-5", message="查商品")))

    assert events[-1] == {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}}
    with db_session_factory() as session:
        user = session.scalar(select(Message).where(Message.role == "user"))
    assert user.content == "查商品"
