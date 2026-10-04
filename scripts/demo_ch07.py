"""Reproducible 25-turn mechanism demonstration, no external APIs required."""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from threading import Event

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import func, select

from app.db.models import ConversationSummary, Message
from app.db.session import create_tables, make_engine, make_session_factory
from app.schemas import ChatRequest
from app.services.context.budget import ContextBudget
from app.services.workflow.graph import WorkflowService
from app.services.workflow.policy import Limits
from app.tools.registry import ToolRegistry, ToolRunner


class DemoModel:
    def bind(self, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        if "七类" in messages[0].content:
            return AIMessage(content='{"intent":"订单"}')
        if "指代消解器" in messages[0].content:
            background = "\n".join(str(m.content) for m in messages)
            return AIMessage(
                content=json.dumps(
                    {
                        "question": "订单778899未收到货后来如何处理"
                        if "778899" in background
                        else "需要补充订单号"
                    },
                    ensure_ascii=False,
                )
            )
        return AIMessage(content='{"next":"answer","actions":[]}')

    def stream(self, messages):
        current = next(
            m.content
            for m in reversed(messages)
            if m.type == "human" and "背景数据" not in m.content
        )
        if "最开始" in current:
            visible = "\n".join(str(m.content) for m in messages)
            yield AIMessageChunk(
                content="订单778899，之前的诉求是未收到货，仍需核实处理进展。"
                if "778899" in visible
                else "请补充订单号。"
            )
        else:
            yield AIMessageChunk(
                content="已记录本轮订单核查问题，尚未查询，不能确认处理进展。"
                + ("回复说明保留原文。" * 75)
            )


class NoKnowledge:
    def retrieve(self, *args):
        return [], {}


def run_demo(output_dir, demo=False, live=False):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "business.db").exists():
        raise ValueError("demo output directory must be fresh")
    engine = make_engine(f"sqlite:///{root}/business.db")
    create_tables(engine)
    factory = make_session_factory(engine)
    saver = SqliteSaver(
        sqlite3.connect(str(root / "checkpoint.db"), check_same_thread=False)
    )
    if live:
        from langchain_openai import ChatOpenAI

        from app.config import Settings

        settings = Settings.from_env()

        def model_factory():
            return ChatOpenAI(
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                temperature=0,
                timeout=45,
                max_retries=0,
                stream_usage=True,
            )
    else:
        model = DemoModel()
        model_factory = lambda: model
    budget = ContextBudget(window=18000, steps=3) if demo else ContextBudget()
    release, started, completed = Event(), Event(), Event()

    def summarize(batch):
        started.set()
        if not release.wait(5):
            raise TimeoutError("demo did not release summary")
        if live:
            from app.services.context.summary import ModelSummarizer

            result = ModelSummarizer(model_factory, budget)(batch)
            completed.set()
            return result
        # Deterministic fixture verifies transport, not the quality of an LLM summary.
        result = (
            "订单778899未收到货，用户要求核实处理进展。"
            if "778899" in batch.text
            else "用户继续核查订单，进展尚未确认。"
        )
        completed.set()
        return result

    service = WorkflowService(
        factory,
        model_factory,
        lambda tools: ToolRunner(ToolRegistry(tools), 1, 0),
        NoKnowledge(),
        saver,
        root / "workflow.jsonl",
        limits=Limits(
            max_decisions=budget.steps,
            max_tokens=1000000,
            max_output_tokens=budget.output,
        ),
        context_budget=budget,
        context_log_path=root / "app.log",
        summarizer=summarize,
    )
    blocked_response_finished = False
    try:
        for i in range(24):
            question = (
                (
                    "订单778899一直未收到货，我要求核实处理进展。"
                    if i == 0
                    else f"第{i + 1}轮继续核查当前订单的处理情况。"
                )
                + "我在此补充收货和沟通细节，客服尚未核实处理结果，请保持这些内容为用户陈述。"
                * 13
            )
            events = list(
                service.stream_events(
                    ChatRequest(
                        conversation_id="demo", user_id="demo-user", message=question
                    )
                )
            )
            assert events[-1]["event"] == "done", events[-1]
            if started.is_set() and not release.is_set():
                blocked_response_finished = not completed.is_set()
                release.set()
            if release.is_set():
                # Allow the immutable task to commit; production replies do not wait here.
                deadline = time.monotonic() + (55 if live else 5)
                while service.context.worker.active and time.monotonic() < deadline:
                    time.sleep(0.01)
        final_events = list(
            service.stream_events(
                ChatRequest(
                    conversation_id="demo",
                    user_id="demo-user",
                    message="最开始那个订单后来怎么说",
                )
            )
        )
        assert final_events[-1]["event"] == "done", final_events[-1]
        answer = "".join(
            e["data"]["content"] for e in final_events if e["event"] == "token"
        )
        state = service.graph.get_state({"configurable": {"thread_id": "demo"}}).values
        rows = [
            json.loads(line) for line in (root / "app.log").read_text().splitlines()
        ]
        with factory() as s:
            originals = s.scalar(select(func.count()).select_from(Message))
            segments = s.scalar(select(func.count()).select_from(ConversationSummary))
        if live and ("778899" not in answer or "未收到" not in answer):
            raise AssertionError(
                "live follow-up did not preserve earliest order and request: " + answer
            )
        result = {
            "mode": ("live-" if live else "fixture-") + ("demo" if demo else "default"),
            "rounds": 25,
            "history_budget": budget.history_tokens,
            "layer1_budget": budget.layer1_tokens,
            "layer2_budget": budget.layer2_tokens,
            "downgrades": sum(r["event"].startswith("层1 降级") for r in rows),
            "summaries": segments,
            "original_messages": originals,
            "checkpoint_messages": len(state["messages"]),
            "max_input_tokens": max(
                r["tokens"] for r in rows if r["event"] == "model_ctx"
            ),
            "reply_finished_while_summary_blocked": blocked_response_finished,
            "last_answer": answer,
            "log": str(root / "app.log"),
        }
        (root / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        return result
    finally:
        release.set()
        service.close()
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use configured real chat model for all nodes and summaries",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = (
        args.output_dir
        or f".runtime/ch07/{'demo' if args.demo else 'default'}-{time.time_ns()}"
    )
    print(
        json.dumps(run_demo(root, args.demo, args.live), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
