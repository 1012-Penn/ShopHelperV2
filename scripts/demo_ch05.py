"""Trace the five chapter acceptance examples through the real graph."""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEMO_CASES = [
    ("policy", "退货政策是什么"),
    ("logistics", "订单 1001 的物流到哪了"),
    ("complaint", "我要投诉"),
    ("chitchat", "你好"),
    ("complex", "先查订单1001状态，再查询它的物流轨迹"),
]


class FixtureModel:
    """Only for an explicitly labelled synthetic command demonstration."""

    def __init__(self, case):
        self.case, self.step = case, 0

    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        usage = {"input_tokens": 120, "output_tokens": 40, "total_tokens": 160}
        if "七类" in messages[0].content:
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": {
                            "policy": "退款退货",
                            "logistics": "物流",
                            "complaint": "投诉",
                            "chitchat": "闲聊",
                            "complex": "订单",
                        }[self.case]
                    },
                    ensure_ascii=False,
                ),
                usage_metadata=usage,
            )
        tool = None
        if self.case == "logistics" and self.step == 0:
            tool = "query_logistics"
        elif self.case == "complex" and self.step < 2:
            tool = ["query_order", "query_logistics"][self.step]
        self.step += 1
        if tool:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool,
                        "id": f"call-{self.step}",
                        "args": {"order_id": "1001"},
                    }
                ],
                usage_metadata=usage,
            )
        return AIMessage(content='{"next":"answer","actions":[]}', usage_metadata=usage)

    def stream(self, messages):
        from langchain_core.messages import AIMessageChunk

        for content in (
            ["根据当轮政策，", "七天内可申请退货[1]。"]
            if self.case == "policy"
            else ["已完成查询，", "具体状态请参考工具提供的模拟数据。"]
        ):
            yield AIMessageChunk(content=content)


class FixtureRetriever:
    def retrieve(self, question, category=None):
        return [
            {
                "n": 1,
                "chunk_id": 1,
                "score": 0.99,
                "answer": "七天内可申请退货，以订单和实际政策为准。",
                "source_key": "fixture",
                "section_path": ["演示政策"],
            }
        ], {"strategy": "hybrid_rerank", "synthetic": True}


def run_demo(live=False, output_dir=None):
    out = Path(
        output_dir
        or (
            "evaluation/ch05/runs/demo-"
            + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid4().hex[:6]
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    if live:
        from app.services.workflow.runtime import build_workflow_service

        service = build_workflow_service()
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        from app.db.session import create_tables, make_engine, make_session_factory
        from app.services.workflow.graph import WorkflowService
        from app.tools.registry import ToolRegistry, ToolRunner

        engine = make_engine(f"sqlite:///{out}/business.db")
        create_tables(engine)
        service = WorkflowService(
            make_session_factory(engine),
            lambda: FixtureModel("policy"),
            lambda tools: ToolRunner(ToolRegistry(tools), 1, 0),
            FixtureRetriever(),
            SqliteSaver(
                sqlite3.connect(
                    str(out / "checkpoints.sqlite"), check_same_thread=False
                )
            ),
            out / "workflow.jsonl",
        )
        service._closers.append(engine.dispose)
    from app.schemas import ChatRequest

    rows = []
    try:
        for name, question in DEMO_CASES:
            if not live:
                model = FixtureModel(name)
                service.model_factory = lambda model=model: model
            cid = "ch05-demo-" + uuid4().hex[:12]
            events = list(
                service.stream_events(
                    ChatRequest(conversation_id=cid, message=question)
                )
            )
            snapshot = service.graph.get_state(
                {"configurable": {"thread_id": cid}}
            ).values
            rows.append(
                {
                    "case": name,
                    "question": question,
                    "conversation_id": cid,
                    "answer": "".join(
                        e["data"]["content"] for e in events if e["event"] == "token"
                    ),
                    "completed": bool(events and events[-1]["event"] == "done"),
                    **{
                        k: snapshot.get(k)
                        for k in (
                            "intent",
                            "route",
                            "path",
                            "gate_passed",
                            "gate_score",
                            "decisions",
                            "tool_calls",
                            "tokens",
                            "usage",
                            "tool_trace",
                            "stop_reason",
                            "message_id",
                        )
                    },
                    "actions": [e["data"] for e in events if e["event"] == "actions"],
                    "errors": [e["data"] for e in events if e["event"] == "error"],
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False))
    finally:
        service.close()
    checks = acceptance_checks(rows)
    report = {
        "mode": "live" if live else "fixture",
        "synthetic": not live,
        "checks": checks,
        "accepted": all(checks.values()),
        "cases": rows,
    }
    (out / "demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "demo.md").write_text(
        f"# ch05 graph demonstration ({report['mode']})\n\n"
        "Tools for orders/logistics remain chapter-2 simulated business data. Complaint actions are suggestions only.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in checks.items())
        + "\n\n"
        + "\n\n".join(
            f"## {r['case']}\n\n{r['question']}\n\n{r['answer']}\n\nPath: {' → '.join(r['path'] or [])}"
            for r in rows
        )
        + "\n"
    )
    return report


def acceptance_checks(rows):
    return {
        "agent_answers_completed": all(
            r["stop_reason"] == "answer"
            for r in rows
            if r["route"] == "business" or r["gate_passed"]
        ),
        "all_completed": all(r["completed"] for r in rows),
        "policy_forced_retrieval": "retrieve" in (rows[0]["path"] or []),
        "logistics_agent_tool": rows[1]["tool_calls"] == 1
        and "retrieve" not in (rows[1]["path"] or []),
        "complaint_two_suggestions": rows[2]["decisions"] == 0
        and bool(rows[2]["actions"])
        and [a["type"] for a in rows[2]["actions"][0]["items"]]
        == ["handoff", "create_ticket"],
        "chitchat_zero_model": rows[3]["tokens"] == 0,
        "complex_multiple_tools": (rows[4]["tool_calls"] or 0) > 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report = run_demo(args.live, args.output_dir)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "accepted": report["accepted"],
                "checks": report["checks"],
            },
            ensure_ascii=False,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
