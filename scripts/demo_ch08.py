"""Run the chapter 8 tool mechanisms against isolated databases and local MCP servers.

This fixture demonstrates wiring and safety mechanisms. Its ScriptedModel does
not measure or claim real model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import func, select

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Keep direct script execution bound to this checkout, even with stale PYTHONPATH.
sys.path.insert(0, str(REPOSITORY_ROOT))
USER_ID = "ch08-demo-user"


class ScriptedModel:
    """Minimal deterministic model boundary; the workflow itself stays real."""

    def __init__(self) -> None:
        self.decisions: list[AIMessage] = []
        self.bound_tools: list[str] = []

    def bind(self, **_kwargs):
        return self

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        prompt = str(messages[0].content)
        if "本轮问题独立化节点" in prompt:
            question = str(messages[-1].content).split("本轮原问题：\n", 1)[-1]
            return AIMessage(
                content=json.dumps(
                    {"question": question, "reference_resolved": True},
                    ensure_ascii=False,
                )
            )
        if "本轮意图选择节点" in prompt:
            current = next(
                (
                    str(item.content)
                    for item in reversed(messages)
                    if getattr(item, "type", None) == "human"
                    and not str(item.content).startswith(
                        ("可用工具用途清单", "早期会话事实摘要")
                    )
                ),
                "",
            )
            question = current.split("本轮原问题：\n", 1)[-1]
            intent = (
                "售后"
                if any(word in question for word in ("工单", "保修", "退货进度"))
                else "物流"
            )
            return AIMessage(
                content=json.dumps(
                    {"intent": intent, "confidence": 0.99}, ensure_ascii=False
                )
            )
        if self.decisions:
            return self.decisions.pop(0)
        return AIMessage(content='{"next":"answer","missing":"","actions":[]}')

    def stream(self, messages):
        observations = [item for item in messages if isinstance(item, ToolMessage)]
        text = (
            "演示模式：查询已运行，返回内容为模拟数据。"
            if observations
            else "演示模式：已完成。"
        )
        yield AIMessageChunk(content=text)

    def call_tool(self, name: str, args: dict[str, Any], call_id: str) -> None:
        self.decisions = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": call_id, "type": "tool_call"}
                ],
            )
        ]


class NoKnowledge:
    def retrieve(self, *_args, **_kwargs):
        return [], {"strategy": "fixture_no_knowledge"}


class MCPProcess:
    def __init__(self, module: str, port: int, extra_logistics_tool: bool = False):
        self.module = module
        self.port = port
        self.extra_logistics_tool = extra_logistics_tool
        self.process: subprocess.Popen | None = None
        self.initial_pid: int | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> None:
        env = os.environ.copy()
        env["MCP_HOST"] = "127.0.0.1"
        env["MCP_PORT"] = str(self.port)
        if self.extra_logistics_tool:
            env["LOGISTICS_EXTRA_TOOL"] = "query_eta"
        else:
            env.pop("LOGISTICS_EXTRA_TOOL", None)
        self.process = subprocess.Popen(
            [sys.executable, "-m", self.module],
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.initial_pid is None:
            self.initial_pid = self.process.pid

    def wait_ready(self, timeout_seconds: float = 12) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(f"MCP server process failed to start: {self.module}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError(f"MCP server did not become ready: {self.url}")

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        self.process = None

    def restart_with_query_eta(self) -> None:
        self.stop()
        self.extra_logistics_tool = True
        self.start()
        self.wait_ready()

    def as_report(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "url": self.url,
            "port": self.port,
            "pid_before_restart": self.initial_pid,
            "pid_after_restart": self.process.pid if self.process else None,
        }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ticket_count(factory) -> int:
    from app.db.models import Ticket

    with factory() as session:
        return int(session.scalar(select(func.count()).select_from(Ticket)) or 0)


def _audit_rows(factory) -> list[dict[str, Any]]:
    from app.db.models import ToolAuditLog

    with factory() as session:
        rows = list(session.scalars(select(ToolAuditLog).order_by(ToolAuditLog.id)))
    return [
        {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "tool_call_id": row.tool_call_id,
            "tool_name": row.tool_name,
            "source": row.source,
            "status": row.status,
            "retry_count": row.retry_count,
            "duration_ms": row.duration_ms,
        }
        for row in rows
    ]


def _tool_observations(service, conversation_id: str) -> list[dict[str, Any]]:
    snapshot = service.graph.get_state({"configurable": {"thread_id": conversation_id}})
    observations = []
    for message in snapshot.values.get("messages", []):
        if isinstance(message, ToolMessage):
            try:
                content: Any = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                content = message.content
            observations.append(
                {
                    "tool_call_id": message.tool_call_id,
                    "content": content,
                    "is_error": bool(message.status == "error"),
                }
            )
    return observations


def _run_tool_turn(
    service,
    model,
    conversation_id: str,
    message: str,
    name: str,
    args: dict[str, Any],
    call_id: str,
) -> list[dict[str, Any]]:
    from app.schemas import ChatRequest

    model.call_tool(name, args, call_id)
    events = list(
        service.stream_events(
            ChatRequest(
                conversation_id=conversation_id,
                user_id=USER_ID,
                message=message,
            )
        )
    )
    if any(event.get("event") == "error" for event in events):
        raise RuntimeError(f"fixture workflow failed for {name}: {events[-1]}")
    return events


def _await_preview(
    service, model, conversation_id: str, message: str, call_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = _run_tool_turn(
        service,
        model,
        conversation_id,
        message,
        "create_ticket",
        {"description": "耳机左耳没有声音", "ticket_type": "售后"},
        call_id,
    )
    previews = [
        event["data"] for event in events if event.get("event") == "ticket_preview"
    ]
    if len(previews) != 1:
        raise AssertionError(
            f"expected exactly one ticket preview, got {len(previews)}; events={events}"
        )
    return previews[0], events


def _probe_retries(service) -> dict[str, Any]:
    from app.tools.audit import AuditWriter
    from app.tools.definitions import ExecutionContext, ToolDefinition
    from app.tools.engine import ToolEngine
    from app.tools.registry import ToolRegistry

    calls = {"read": 0, "write": 0}

    def read_timeout(_args, _context):
        calls["read"] += 1
        raise TimeoutError("fixture read timeout")

    def write_timeout(_args, _context):
        calls["write"] += 1
        raise TimeoutError("fixture write timeout")

    schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    audit = AuditWriter(service.session_factory)
    read_engine = ToolEngine(
        ToolRegistry(
            [
                ToolDefinition(
                    "timeout_read", "模拟只读超时", schema, "builtin", read_timeout
                )
            ]
        ),
        timeout_seconds=1,
        max_retries=2,
        audit=audit,
    )
    write_engine = ToolEngine(
        ToolRegistry(
            [
                ToolDefinition(
                    "create_ticket", "模拟写超时", schema, "builtin", write_timeout
                )
            ]
        ),
        timeout_seconds=1,
        max_retries=2,
        audit=audit,
        session_factory=service.session_factory,
    )
    try:
        read_result = read_engine.run(
            "timeout_read",
            {},
            "ch08-read-timeout",
            ExecutionContext("retry-read", USER_ID),
        )
        write_result = write_engine.run(
            "create_ticket",
            {},
            "ch08-write-timeout",
            ExecutionContext("retry-write", USER_ID, True, "ch08-write-timeout"),
        )
    finally:
        read_engine.close()
        write_engine.close()
    if calls != {"read": 3, "write": 1}:
        raise AssertionError(f"unexpected retry counts: {calls}")
    if read_result.retry_count != 2 or read_result.status != "timeout":
        raise AssertionError(f"unexpected read timeout result: {read_result}")
    if write_result.retry_count != 0 or write_result.status != "timeout":
        raise AssertionError(f"unexpected write timeout result: {write_result}")
    return {
        "read_timeout": {
            "calls": calls["read"],
            "retry_count": read_result.retry_count,
        },
        "write_timeout": {
            "calls": calls["write"],
            "retry_count": write_result.retry_count,
        },
    }


def run_fixture(output_dir: Path) -> dict[str, Any]:
    from app.db.session import create_tables, make_engine, make_session_factory
    from app.schemas import TicketConfirmationRequest
    from app.services.workflow.graph import WorkflowService
    from app.tools.definitions import ToolDefinition
    from app.tools.mcp import MCPDiscovery
    from app.tools.policy import ToolPolicy

    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "business.sqlite"
    checkpoint_path = output_dir / "checkpoint.sqlite"
    if database_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"demo output directory must be fresh: {output_dir}")

    logistics = MCPProcess("mcp_servers.logistics", _free_port())
    after_sale = MCPProcess("mcp_servers.after_sale", _free_port())
    service = None
    discovery = None
    try:
        logistics.start()
        after_sale.start()
        logistics.wait_ready()
        after_sale.wait_ready()

        db_engine = make_engine(f"sqlite:///{database_path}")
        create_tables(db_engine)
        factory = make_session_factory(db_engine)
        conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        discovery = MCPDiscovery(
            {
                "logistics": {"transport": "http", "url": logistics.url},
                "after_sale": {"transport": "http", "url": after_sale.url},
            },
            timeout_seconds=3,
        )
        policy_path = output_dir / "tool-permissions.json"
        policy_path.write_text(
            json.dumps(
                {
                    "mcp": {
                        "logistics": {"query_logistics": "read"},
                        "after_sale": {
                            "query_warranty": "read",
                            "query_return_progress": "read",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        policy = ToolPolicy(policy_path)
        model = ScriptedModel()
        service = WorkflowService(
            factory,
            lambda: model,
            lambda _tools: None,
            NoKnowledge(),
            SqliteSaver(conn),
            output_dir / "workflow.jsonl",
            tool_policy=policy,
            discovery=discovery,
            tool_timeout_seconds=1,
            tool_max_retries=2,
        )
        registry = service.tool_registry
        registry_identity = id(service.tool_registry)
        discovery.refresh(service.tool_registry)
        required_initial = {
            "query_logistics",
            "query_warranty",
            "query_return_progress",
        }
        if not required_initial <= service.tool_registry.names:
            raise AssertionError(
                f"initial MCP discovery missing tools: {service.tool_registry.names}"
            )

        local_results = []
        service.tool_registry.register(
            ToolDefinition(
                "query_shop_hours_ch08",
                "查询演示门店营业时间。",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "builtin",
                lambda _args, _context: {"hours": "09:00—18:00", "source": "模拟数据"},
            )
        )
        local_events = _run_tool_turn(
            service,
            model,
            "demo-local-tool",
            "查询门店今天营业时间",
            "query_shop_hours_ch08",
            {},
            "local-call",
        )
        local_results.append(
            any(event.get("event") == "tool_status" for event in local_events)
        )

        _run_tool_turn(
            service,
            model,
            "demo-logistics",
            "查询订单 DEMO-1001 物流状态",
            "query_logistics",
            {"order_id": "DEMO-1001"},
            "logistics-call",
        )
        _run_tool_turn(
            service,
            model,
            "demo-warranty",
            "查询订单 DEMO-1001 保修状态",
            "query_warranty",
            {"order_id": "DEMO-1001"},
            "warranty-call",
        )
        _run_tool_turn(
            service,
            model,
            "demo-return",
            "查询订单 DEMO-1002 退货进度",
            "query_return_progress",
            {"order_id": "DEMO-1002"},
            "return-call",
        )

        logistics.restart_with_query_eta()
        discovery.refresh(service.tool_registry)
        if "query_eta" not in service.tool_registry.names:
            raise AssertionError(
                "query_eta was not discovered after restarting logistics"
            )
        denied = service.tool_engine.run(
            "query_eta",
            {"order_id": "DEMO-1001"},
            "eta-denied",
        )
        if denied.status != "permission_denied":
            raise AssertionError(
                "new MCP tool was not denied before local authorization"
            )
        current_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        current_policy["mcp"]["logistics"]["query_eta"] = "read"
        policy_path.write_text(json.dumps(current_policy), encoding="utf-8")
        service_reused = id(service.tool_registry) == registry_identity
        eta_events = _run_tool_turn(
            service,
            model,
            "demo-eta",
            "查询订单 DEMO-1001 物流预计到达时间",
            "query_eta",
            {"order_id": "DEMO-1001"},
            "eta-call",
        )
        mcp_query_results = {
            "logistics": _tool_observations(service, "demo-logistics"),
            "warranty": _tool_observations(service, "demo-warranty"),
            "return_progress": _tool_observations(service, "demo-return"),
            "eta": _tool_observations(service, "demo-eta"),
        }
        if not all(mcp_query_results.values()):
            raise AssertionError(
                f"one or more real MCP query results are missing: {mcp_query_results}"
            )

        before_ticket = _ticket_count(factory)
        confirm_preview, confirm_preview_events = _await_preview(
            service,
            model,
            "demo-confirm",
            "请帮我创建人工工单，问题描述是耳机左耳没有声音",
            "confirm-call",
        )
        if _ticket_count(factory) != before_ticket:
            raise AssertionError("ticket preview wrote a ticket before confirmation")
        confirmed = list(
            service.resume_ticket_events(
                TicketConfirmationRequest(
                    conversation_id="demo-confirm",
                    user_id=USER_ID,
                    request_id=confirm_preview["request_id"],
                    tool_call_id=confirm_preview["tool_call_id"],
                    approve=True,
                )
            )
        )
        count_after_confirm = _ticket_count(factory)

        cancel_preview, cancel_preview_events = _await_preview(
            service,
            model,
            "demo-cancel",
            "请帮我创建人工工单，问题描述是耳机左耳没有声音",
            "cancel-call",
        )
        cancelled = list(
            service.resume_ticket_events(
                TicketConfirmationRequest(
                    conversation_id="demo-cancel",
                    user_id=USER_ID,
                    request_id=cancel_preview["request_id"],
                    tool_call_id=cancel_preview["tool_call_id"],
                    approve=False,
                )
            )
        )
        count_after_cancel = _ticket_count(factory)
        retry_counts = _probe_retries(service)
        audit_rows = _audit_rows(factory)

        tool_rows = [
            row
            for row in audit_rows
            if row["tool_name"]
            in required_initial | {"query_eta", "query_shop_hours_ch08"}
        ]
        if not {
            "query_logistics",
            "query_warranty",
            "query_return_progress",
            "query_eta",
            "query_shop_hours_ch08",
        } <= {row["tool_name"] for row in tool_rows}:
            raise AssertionError("audit is missing one or more queried tools")
        if not any(
            row["tool_call_id"] == confirm_preview["tool_call_id"]
            and row["status"] == "success"
            for row in audit_rows
        ):
            raise AssertionError("confirmed ticket success was not audited")
        if not any(
            row["tool_call_id"] == cancel_preview["tool_call_id"]
            and row["status"] == "permission_denied"
            for row in audit_rows
        ):
            raise AssertionError(
                "ticket cancellation permission denial was not audited"
            )
        if not any(event.get("event") == "done" for event in confirmed):
            raise AssertionError("ticket confirmation did not finish")
        if not any(event.get("event") == "done" for event in cancelled):
            raise AssertionError("ticket cancellation did not finish")
        if (
            count_after_confirm != before_ticket + 1
            or count_after_cancel != count_after_confirm
        ):
            raise AssertionError("ticket counts do not match confirmation/cancellation")

        mechanisms = [
            {"name": "dynamic_local_registration", "passed": local_results[0]},
            {"name": "real_mcp_logistics_and_after_sale", "passed": True},
            {
                "name": "single_server_restart_and_hot_authorization",
                "passed": service_reused and bool(eta_events),
            },
            {"name": "ticket_preview_confirmation_and_cancellation", "passed": True},
            {
                "name": "transient_read_retry_and_write_no_retry",
                "passed": retry_counts["read_timeout"]["calls"] == 3
                and retry_counts["write_timeout"]["calls"] == 1,
            },
        ]
        if not all(item["passed"] for item in mechanisms):
            raise AssertionError(f"one or more fixture mechanisms failed: {mechanisms}")
        result = {
            "mode": "fixture",
            "synthetic": True,
            "quality_evaluation": "not_evaluated",
            "fixture_notice": "ScriptedModel 只验证接线与执行机制，不代表真实模型质量评估。",
            "output_dir": str(output_dir),
            "service_reused_after_logistics_restart": service_reused,
            "mcp_processes": [logistics.as_report(), after_sale.as_report()],
            "ticket_confirm_events": [event["event"] for event in confirmed],
            "ticket_cancel_events": [event["event"] for event in cancelled],
            "ticket_preview_events": {
                "confirm": [
                    event["event"]
                    for event in confirm_preview_events
                    if event.get("event") == "ticket_preview"
                ],
                "cancel": [
                    event["event"]
                    for event in cancel_preview_events
                    if event.get("event") == "ticket_preview"
                ],
            },
            "tickets": {
                "after_confirm": count_after_confirm,
                "after_cancel": count_after_cancel,
            },
            "mcp_query_results": mcp_query_results,
            "attempts": retry_counts,
            "mechanisms": mechanisms,
            "audit_rows": audit_rows,
        }
        (output_dir / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    finally:
        if service is not None:
            service.close()
        elif discovery is not None:
            discovery.close()
        logistics.stop()
        after_sale.stop()
        if "db_engine" in locals():
            db_engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="store_true", help="run deterministic mechanism checks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / ".runtime/ch08/demo",
        help="output directory (default: .runtime/ch08/demo)",
    )
    args = parser.parse_args(argv)
    if not args.fixture:
        parser.error("this demo currently requires --fixture")
    output = args.output.resolve()
    if (output / "business.sqlite").exists() or (output / "checkpoint.sqlite").exists():
        output = output / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    report = run_fixture(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
