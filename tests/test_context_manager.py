from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from workflow_helpers import make_workflow

from app.db.models import Conversation
from app.services.context.budget import ContextBudget
from app.services.context.manager import ContextManager


def turns(n):
    result = []
    for i in range(n):
        result.extend(
            [
                HumanMessage(
                    content=("用户订单1001要求退货。" + "补充细节。" * 35),
                    id=f"sql-{i * 2 + 1}",
                ),
                AIMessage(content="客服回复。" * 200, id=f"sql-{i * 2 + 2}"),
            ]
        )
    return result


def manager(tmp_path, budget):
    service = make_workflow(tmp_path)
    with service.session_factory.begin() as s:
        s.add(Conversation(conversation_id="c", user_id="u"))
    return service, ContextManager(
        service.session_factory,
        lambda: None,
        budget,
        tmp_path / "app.log",
        summarize=lambda b: "用户订单1001要求退货。",
    )


def test_default_keeps_original_twenty_turns(tmp_path):
    service, m = manager(tmp_path, ContextBudget())
    history = turns(20)
    result = m.prepare("c", history, 41)
    assert result["context_history"] == history
    assert not m.repository.read("c")[1]
    assert m.repository.read("c")[0].layer1_from_msg_id == 1
    m.close()
    service.close()


def test_three_layers_preserve_original_and_order(tmp_path):
    b = ContextBudget(window=18000, steps=3)
    service, m = manager(tmp_path, b)
    history = turns(7)
    history.insert(
        1,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t", "name": "query_order", "args": {"order_id": "1001"}}
            ],
        ),
    )
    history.insert(2, ToolMessage(content="巨大工具数据" * 300, tool_call_id="t"))
    originals = [x.model_dump() for x in history]
    result = m.prepare("c", history, 15)
    assert [x.model_dump() for x in history] == originals
    assert m.repository.read("c")[0].layer1_from_msg_id > 1
    assert result["context_history"][0].content == history[0].content
    assert any("工具结果" in x.content for x in result["context_history"])
    state = {
        **result,
        "conversation_id": "c",
        "question": "最开始订单呢",
        "resolved_question": "最开始订单呢",
        "current_message_id": 15,
        "messages": history + [HumanMessage(content="最开始订单呢", id="sql-15")],
        "evidence": [{"answer": "证据"}],
    }
    messages = m.model_messages(state, "固定人设")
    assert messages[0].content == "固定人设"
    assert messages[-2].content == "最开始订单呢"
    assert "证据" in messages[-1].content
    assert sum(x.type == "system" for x in messages) == 1
    m.close()
    service.close()


def test_tool_peak_caps_input_copy_but_keeps_full_checkpoint(tmp_path):
    service, m = manager(tmp_path, ContextBudget())
    raw = "工具订单详情。" * 2000
    full = [
        HumanMessage(content="订单1001", id="sql-1"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t", "name": "query_order", "args": {"order_id": "1001"}}
            ],
        ),
        ToolMessage(content=raw, tool_call_id="t"),
    ]
    state = {
        "messages": full,
        "current_message_id": 1,
        "conversation_id": "c",
        "question": "订单1001",
        "evidence": [],
    }
    messages = m.model_messages(state, "固定人设")
    assert m.budget.count(messages[-1].content) <= m.budget.tool_result
    assert full[-1].content == raw
    m.close()
    service.close()


def test_upgrade_maps_legacy_checkpoint_user_ids_without_mutating(tmp_path):
    from app.db.models import Message

    service, m = manager(tmp_path, ContextBudget())
    with service.session_factory.begin() as s:
        user = Message(conversation_id="c", role="user", content="最早订单779900想换货")
        answer = Message(conversation_id="c", role="assistant", content="还需核实")
        s.add_all([user, answer])
        s.flush()
        user_id, answer_id = user.id, answer.id
    full = [
        HumanMessage(content="最早订单779900想换货", id="legacy-uuid"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "t", "name": "query_order", "args": {"order_id": "779900"}}
            ],
        ),
        ToolMessage(content="旧工具结果", tool_call_id="t"),
        AIMessage(content="还需核实", id=f"sql-{answer_id}"),
    ]
    result = m.prepare("c", full, answer_id + 1)
    assert result["context_history"][0].id == f"sql-{user_id}"
    assert len(result["context_history"]) == 4
    assert full[0].id == "legacy-uuid"
    m.close()
    service.close()


def test_giant_tool_downgrades_to_bounded_summary_batch(tmp_path):
    import threading

    service, m = manager(tmp_path, ContextBudget(window=18000, steps=3))
    captured = []
    done = threading.Event()

    def summarize(batch):
        captured.append(batch)
        assert (
            m.budget.count(batch.text)
            < m.budget.window - m.budget.summary - m.budget.output
        )
        assert "工具结果" in batch.text
        done.set()
        return "用户订单1001要求退货，尚未解决。"

    m.worker.summarize = summarize
    history = turns(14)
    history.insert(
        1,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "huge", "name": "query_order", "args": {"order_id": "1001"}}
            ],
        ),
    )
    history.insert(2, ToolMessage(content="巨型工具观察。" * 4000, tool_call_id="huge"))
    m.prepare("c", history, 29)
    m.close()
    assert done.is_set()
    assert m.repository.read("c")[0].summary_upto_msg_id is not None
    assert len(history[2].content) > 20000
    service.close()


def test_summary_backlog_is_split_at_complete_turn_boundary(tmp_path):
    service, m = manager(tmp_path, ContextBudget(window=18000, steps=3))
    batches = []
    m.worker.summarize = lambda batch: batches.append(batch) or "订单1001仍待核实。"
    history = turns(120)
    m.prepare("c", history, 241)
    m.close()
    assert len(batches) == 1
    batch = batches[0]
    assert m.budget.count(batch.text) + m.budget.summary + 350 + m.budget.safety < 18000
    assert batch.upto_msg_id < batch.layer1_from_msg_id - 1
    assert batch.upto_msg_id % 2 == 0
    assert m.repository.read("c")[0].summary_upto_msg_id == batch.upto_msg_id
    service.close()
