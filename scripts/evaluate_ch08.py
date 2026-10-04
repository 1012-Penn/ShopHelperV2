"""Evaluate the chapter-eight tool workflow with real model calls or wiring-only fixture mode."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CASES_PATH = REPOSITORY_ROOT / "evaluation/ch08/cases.json"
USER_ID = "ch08-evaluation-user"


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation cases must be a non-empty JSON list")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise TypeError("each case requires a string id")
        if case["id"] in ids:
            raise ValueError(f"duplicate evaluation case id: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case.get("steps"), list) or not case["steps"]:
            raise ValueError(f"case {case['id']} requires steps")
        for step in case["steps"]:
            if not isinstance(step, dict) or step.get("kind") not in {
                "message",
                "confirmation",
            }:
                raise ValueError(f"case {case['id']} has an invalid step")
            if step["kind"] == "message" and not isinstance(
                step.get("message"), str
            ):
                raise ValueError(f"case {case['id']} message step requires message")
            if step["kind"] == "confirmation" and type(step.get("approve")) is not bool:
                raise ValueError(f"case {case['id']} confirmation requires bool approve")
    return cases


def select_cases(cases: list[dict[str, Any]], case_ids: list[str] | None) -> list[dict[str, Any]]:
    if not case_ids:
        return cases
    by_id = {case["id"]: case for case in cases}
    unknown = [case_id for case_id in case_ids if case_id not in by_id]
    if unknown:
        raise ValueError(f"unknown evaluation case id(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in dict.fromkeys(case_ids)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    errors = sum(row.get("status") == "error" for row in rows)
    completed = sum(row.get("status") == "completed" for row in rows)
    check_names = sorted(
        {name for row in rows for name in (row.get("checks") or {})}
    )
    checks = {
        name: {
            "passed": sum(bool((row.get("checks") or {}).get(name)) for row in rows),
            "denominator": sum(name in (row.get("checks") or {}) for row in rows),
        }
        for name in check_names
    }
    return {
        "case_count": total,
        "completed": completed,
        "errors": errors,
        "checks": checks,
    }


def _fixture_plan(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe requests the runner would issue, without inventing model outcomes."""
    rows = []
    for case in cases:
        steps = []
        for step in case["steps"]:
            if step["kind"] == "message":
                planned = {
                    "kind": "message",
                    "request": {
                        "conversation_id": f"ch08-{case['id']}",
                        "user_id": USER_ID,
                        "message": step["message"],
                    },
                    "expectations": step.get("expect", {}),
                }
            else:
                planned = {
                    "kind": "confirmation",
                    "request_fields": [
                        "conversation_id",
                        "user_id",
                        "request_id",
                        "tool_call_id",
                        "approve",
                    ],
                    "approve": step["approve"],
                    "preview_binding": "requires a real pending ticket preview",
                    "expectations": step.get("expect", {}),
                }
            steps.append(planned)
        rows.append(
            {
                "id": case["id"],
                "area": case["area"],
                "execution": "not_run",
                "setup_tool_names": [tool["name"] for tool in case.get("setup_tools", [])],
                "steps": steps,
            }
        )
    return rows


def evaluate_fixture(
    output_dir: Path | str | None = None, case_ids: list[str] | None = None
) -> dict[str, Any]:
    """Check labelled case and request-plan wiring only; never score model quality."""
    cases = select_cases(load_cases(), case_ids)
    rows = _fixture_plan(cases)
    report = {
        "mode": "fixture",
        "synthetic": True,
        "quality_metrics": None,
        "model": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "case_count": len(rows),
            "completed": 0,
            "errors": 0,
            "checks": {},
            "executed": 0,
        },
        "cases": rows,
    }
    out = Path(output_dir) if output_dir else _default_output("fixture")
    report["output_dir"] = str(out)
    _write_report(out, report)
    return report


def _default_output(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return REPOSITORY_ROOT / ".runtime/ch08/runs" / f"{mode}-{stamp}-{uuid.uuid4().hex[:6]}"


def _new_live_service(root: Path):
    from app.config import Settings
    from app.services.quality.runtime import config
    from app.services.workflow.runtime import build_workflow_service

    settings = Settings.from_env()
    database_path = root / "business.sqlite"
    checkpoint_path = root / "checkpoints.sqlite"
    if database_path.exists() or checkpoint_path.exists():
        raise FileExistsError("live evaluation storage already exists; choose a fresh output directory")
    settings = replace(settings, database_url=f"sqlite:///{database_path}")
    values = config()
    values.update(
        {
            "WORKFLOW_CHECKPOINT_PATH": str(checkpoint_path),
            "WORKFLOW_LOG_PATH": str(root / "workflow.jsonl"),
            "CONTEXT_LOG_PATH": str(root / "context.jsonl"),
            "QUALITY_ENABLED": "false",
        }
    )
    # Preserve the user's currently configured MCP endpoints and tool permissions.
    return build_workflow_service(settings=settings, values=values), settings


def _ticket_count(service) -> int:
    from sqlalchemy import func, select

    from app.db.models import Ticket

    with service.session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(Ticket)) or 0)


def _audit_rows(service, conversation_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from app.db.models import ToolAuditLog

    with service.session_factory() as session:
        records = list(
            session.scalars(
                select(ToolAuditLog)
                .where(
                    ToolAuditLog.conversation_id == conversation_id,
                    ToolAuditLog.id > after_id,
                )
                .order_by(ToolAuditLog.id)
            )
        )
        return [
            {
                "id": record.id,
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "status": record.status,
                "retry_count": record.retry_count,
                "duration_ms": record.duration_ms,
                "source": record.source,
            }
            for record in records
        ]


def _run_events(events) -> dict[str, Any]:
    collected = []
    for event in events:
        # Copy only the observable contract into the report; never include provider secrets.
        data = event.get("data") if isinstance(event, dict) else None
        collected.append(
            {
                "event": event.get("event", "message"),
                "data": data if isinstance(data, dict) else {},
            }
        )
    return {
        "events": collected,
        "answer": "".join(
            item["data"].get("content", "")
            for item in collected
            if item["event"] == "token"
            and isinstance(item["data"].get("content"), str)
        ),
        "tool_names": [
            item["data"].get("tool_name")
            for item in collected
            if item["event"] == "tool_status"
            and isinstance(item["data"].get("tool_name"), str)
        ],
        "previews": [
            item["data"] for item in collected if item["event"] == "ticket_preview"
        ],
        "has_error_event": any(item["event"] == "error" for item in collected),
        "error_messages": [
            item["data"]["message"]
            for item in collected
            if item["event"] == "error"
            and isinstance(item["data"].get("message"), str)
        ],
        "workflow_nodes": [
            item["data"]["node"]
            for item in collected
            if item["event"] == "workflow_status"
            and isinstance(item["data"].get("node"), str)
        ],
    }


def _workflow_trace(path: Path | str | None, conversation_id: str) -> dict[str, Any]:
    if not path:
        return {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("conversation_id") != conversation_id:
            continue
        return {
            key: record.get(key)
            for key in (
                "path",
                "intent",
                "route",
                "stop_reason",
                "status",
                "error_type",
                "decisions",
                "tool_calls",
            )
        }
    return {}


def _check_step(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    names = actual["tool_names"]
    answer = actual["answer"]
    previews = actual["previews"]
    if "tool_calls" in expected:
        checks["tool_calls"] = all(name in names for name in expected["tool_calls"])
    if "preview" in expected:
        checks["preview"] = bool(previews) is expected["preview"]
    if expected.get("preview_any"):
        preview_text = " ".join(
            str(preview.get(key, ""))
            for preview in previews
            for key in ("ticket_type", "description")
        )
        checks["preview_content"] = any(
            text in preview_text for text in expected["preview_any"]
        )
    if expected.get("answer_any"):
        checks["answer_content"] = any(text in answer for text in expected["answer_any"])
    if expected.get("forbidden_any"):
        checks["forbidden_content_absent"] = not any(
            text in answer for text in expected["forbidden_any"]
        )
    if expected.get("forbidden_tools"):
        checks["forbidden_tools_absent"] = not any(
            name in names for name in expected["forbidden_tools"]
        )
    if "invalid_arguments" in expected:
        checks["invalid_arguments_audited"] = actual.get("invalid_arguments", False)
    if "forbidden_tool" in expected:
        checks["forbidden_tool_denied"] = actual.get("forbidden_tool", False)
    if "ticket_count_delta" in expected:
        checks["ticket_count_delta"] = (
            actual.get("ticket_count_delta") == expected["ticket_count_delta"]
        )
    if "audit_status" in expected:
        checks["audit_status"] = expected["audit_status"] in actual.get(
            "audit_statuses", []
        )
    if actual.get("has_error_event"):
        checks["no_sse_error"] = False
    return checks


def _expected_check_names(expected: dict[str, Any]) -> list[str]:
    mapping = {
        "tool_calls": "tool_calls",
        "preview": "preview",
        "preview_any": "preview_content",
        "answer_any": "answer_content",
        "forbidden_any": "forbidden_content_absent",
        "forbidden_tools": "forbidden_tools_absent",
        "invalid_arguments": "invalid_arguments_audited",
        "forbidden_tool": "forbidden_tool_denied",
        "ticket_count_delta": "ticket_count_delta",
        "audit_status": "audit_status",
    }
    return [mapping[key] for key in expected if key in mapping]


def _register_setup_tools(service, case: dict[str, Any]):
    """Install case-scoped real registry entries for live model boundary checks."""
    setup_tools = case.get("setup_tools", [])
    if not setup_tools:
        return lambda: None

    from app.tools.definitions import ToolDefinition

    policy = getattr(service, "tool_policy", None)
    original_policy_path = getattr(policy, "path", None)
    registry = service.tool_registry
    sources = {item["source"] for item in setup_tools}
    previous_definitions = {
        source: [definition for definition in registry.definitions if definition.source == source]
        for source in sources
    }
    policy_dir = tempfile.TemporaryDirectory(prefix="ch08-eval-policy-")
    try:
        permissions = {"mcp": {}}
        if original_policy_path is not None:
            try:
                current = json.loads(Path(original_policy_path).read_text(encoding="utf-8"))
                if isinstance(current, dict) and isinstance(current.get("mcp"), dict):
                    permissions["mcp"].update(current["mcp"])
            except (OSError, ValueError, TypeError):
                pass
        for item in setup_tools:
            source = item["source"]
            if source.startswith("mcp:"):
                server_name = source.removeprefix("mcp:")
                permissions["mcp"].setdefault(server_name, {})[item["name"]] = "read"
        policy_path = Path(policy_dir.name) / "tool_permissions.json"
        policy_path.write_text(json.dumps(permissions), encoding="utf-8")
        if policy is not None:
            policy.path = policy_path
        for item in setup_tools:
            result = item.get("result", {})
            service.tool_registry.register(
                ToolDefinition(
                    name=item["name"],
                    description=item["description"],
                    input_schema=item["input_schema"],
                    source=item["source"],
                    handler=lambda _args, _context, value=result: value,
                )
            )
    except Exception:
        for source, definitions in previous_definitions.items():
            registry.replace_source(source, definitions)
        if policy is not None:
            policy.path = original_policy_path
        policy_dir.cleanup()
        raise

    def cleanup():
        for source, definitions in previous_definitions.items():
            registry.replace_source(source, definitions)
        if policy is not None:
            policy.path = original_policy_path
        policy_dir.cleanup()

    return cleanup


def _evaluate_case(service, case: dict[str, Any], case_index: int) -> dict[str, Any]:
    conversation_id = f"ch08-eval-{case_index}-{uuid.uuid4().hex[:8]}"
    case_start_tickets = _ticket_count(service)
    cleanup_setup = _register_setup_tools(service, case)
    try:
        return _evaluate_case_steps(
            service,
            case,
            conversation_id,
            case_start_tickets,
            0,
        )
    finally:
        cleanup_setup()


def _evaluate_case_steps(
    service,
    case: dict[str, Any],
    conversation_id: str,
    case_start_tickets: int,
    last_audit_id: int,
) -> dict[str, Any]:
    from app.schemas import ChatRequest, TicketConfirmationRequest

    all_steps = []
    all_checks: dict[str, bool] = {}
    errors = []
    last_preview = None
    for index, step in enumerate(case["steps"], start=1):
        before_tickets = _ticket_count(service)
        try:
            if step["kind"] == "message":
                events = service.stream_events(
                    ChatRequest(
                        conversation_id=conversation_id,
                        user_id=USER_ID,
                        message=step["message"],
                    )
                )
            else:
                pending = service.pending_ticket(conversation_id, USER_ID)
                if not isinstance(pending, dict):
                    raise RuntimeError("pending ticket preview unavailable")
                last_preview = pending
                request_id = pending.get("request_id")
                tool_call_id = pending.get("tool_call_id")
                if not isinstance(request_id, str) or not isinstance(tool_call_id, str):
                    raise RuntimeError("pending preview is missing confirmation identifiers")
                events = service.resume_ticket_events(
                    TicketConfirmationRequest(
                        conversation_id=conversation_id,
                        user_id=USER_ID,
                        request_id=request_id,
                        tool_call_id=tool_call_id,
                        approve=step["approve"],
                    )
                )
            actual = _run_events(events)
            actual["workflow_trace"] = _workflow_trace(
                getattr(getattr(service, "logger", None), "path", None),
                conversation_id,
            )
            after_tickets = _ticket_count(service)
            audits = _audit_rows(service, conversation_id, last_audit_id)
            if audits:
                last_audit_id = max(item["id"] for item in audits)
            actual["ticket_count_delta"] = after_tickets - before_tickets
            actual["audit_statuses"] = [item["status"] for item in audits]
            actual["invalid_arguments"] = "validation_blocked" in actual["audit_statuses"]
            actual["forbidden_tool"] = "permission_denied" in actual["audit_statuses"]
            if actual["previews"]:
                last_preview = actual["previews"][-1]
            checks = _check_step(step.get("expect", {}), actual)
            all_checks.update({f"step_{index}_{key}": value for key, value in checks.items()})
            all_steps.append(
                {
                    "kind": step["kind"],
                    "checks": checks,
                    "actual": {
                        "answer": actual["answer"],
                        "tool_names": actual["tool_names"],
                        "preview": last_preview,
                        "ticket_count_delta": actual["ticket_count_delta"],
                        "audit_statuses": actual["audit_statuses"],
                        "audits": audits,
                        "has_error_event": actual["has_error_event"],
                        "error_messages": actual["error_messages"],
                        "workflow_nodes": actual["workflow_nodes"],
                        "workflow_trace": actual["workflow_trace"],
                    },
                }
            )
            if actual["has_error_event"]:
                errors.extend(actual["error_messages"] or ["sse_error"])
            trace_error = actual["workflow_trace"].get("error_type")
            if trace_error:
                errors.append(f"workflow:{trace_error}")
        except Exception as error:  # noqa: BLE001 — every labelled case stays in the denominator
            errors.append(type(error).__name__)
            failed_checks = {
                key: False
                for key in _expected_check_names(step.get("expect", {}))
            }
            all_steps.append(
                {
                    "kind": step["kind"],
                    "checks": failed_checks,
                    "error": type(error).__name__,
                }
            )
            for key in failed_checks:
                all_checks[f"step_{index}_{key}"] = False
    final_tickets = _ticket_count(service)
    all_checks["case_completed"] = not errors
    return {
        "id": case["id"],
        "conversation_id": conversation_id,
        "area": case["area"],
        "label": case["label"],
        "status": "error" if errors else "completed",
        "error": ", ".join(errors) if errors else None,
        "checks": all_checks,
        "passed": bool(all_checks) and all(all_checks.values()) and not errors,
        "ticket_count_before": case_start_tickets,
        "ticket_count_after": final_tickets,
        "ticket_count_delta": final_tickets - case_start_tickets,
        "steps": all_steps,
    }


def evaluate_live(
    output_dir: Path | str | None = None, case_ids: list[str] | None = None
) -> dict[str, Any]:
    cases = select_cases(load_cases(), case_ids)
    out = Path(output_dir) if output_dir else _default_output("live")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("live evaluation output directory must be empty")
    service, settings = _new_live_service(out / "runtime")
    try:
        rows = [_evaluate_case(service, case, index) for index, case in enumerate(cases, 1)]
        summary = summarize(rows)
        passed = sum(bool(row.get("passed")) for row in rows)
        ticket_deltas = [row.get("ticket_count_delta", 0) for row in rows]
        previews_by_call = {}
        for row in rows:
            for step in row.get("steps", []):
                preview = step.get("actual", {}).get("preview")
                if not preview:
                    continue
                key = preview.get("tool_call_id") or preview.get("request_id") or row["id"]
                previews_by_call[key] = preview
        previews = list(previews_by_call.values())
        summary["cases_with_effective_ticket_description"] = sum(
            any(
                isinstance((preview := step.get("actual", {}).get("preview")), dict)
                and isinstance(preview.get("description"), str)
                and bool(preview["description"].strip())
                for step in row.get("steps", [])
            )
            for row in rows
        )
        summary["ticket_preview_count"] = len(previews)
        summary["cases_with_ticket_created"] = sum(delta > 0 for delta in ticket_deltas)
        summary["tickets_created_total"] = sum(max(0, delta) for delta in ticket_deltas)
        summary["passed"] = passed
        summary["pass_rate"] = passed / len(rows) if rows else 0.0
        report = {
            "mode": "live",
            "synthetic": False,
            "quality_metrics": summary,
            "model": settings.model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out),
            "storage": {
                "database": str(out / "runtime/business.sqlite"),
                "checkpoints": str(out / "runtime/checkpoints.sqlite"),
            },
            "summary": summary,
            "cases": rows,
        }
        _write_report(out, report)
        return report
    finally:
        service.close()


def _write_report(out: Path, report: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    if report["mode"] == "fixture":
        headline = (
            f"Fixture wiring only: {summary['case_count']} labelled cases; "
            "zero cases executed; no quality score."
        )
    else:
        headline = (
            f"Live model: {summary['passed']}/{summary['case_count']} cases passed; "
            f"{summary['errors']} case errors included in the denominator."
        )
    lines = [
        "# ch08 tool workflow evaluation",
        "",
        f"Mode: {report['mode']}. Model: {report.get('model') or 'none'}.",
        "",
        headline,
        "",
        "Fixture mode 仅验证接线，不代表真实模型质量。",
        "",
        "| Case | Area | Status | Passed checks | Errors | Tickets |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in report["cases"]:
        checks = row.get("checks") or {}
        score = f"{sum(bool(value) for value in checks.values())}/{len(checks)}" if checks else "—"
        lines.append(
            f"| {row['id']} | {row.get('area', '')} | {row.get('status', row.get('execution'))} "
            f"| {score} | {row.get('error') or ''} | {row.get('ticket_count_delta', '—')} |"
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fixture", action="store_true", help="validate case/request wiring only")
    modes.add_argument("--live", action="store_true", help="call the configured real model and tools")
    parser.add_argument("--output-dir", help="fresh directory for reports and isolated live SQLite/checkpoints")
    parser.add_argument("--case", action="append", dest="case_ids", help="run only this labelled case id; repeat to select multiple")
    args = parser.parse_args(argv)
    report = (
        evaluate_live(args.output_dir, args.case_ids)
        if args.live
        else evaluate_fixture(args.output_dir, args.case_ids)
    )
    print(
        json.dumps(
            {"mode": report["mode"], **report["summary"], "output": report["output_dir"]},
            ensure_ascii=False,
        )
    )
    return 0 if report["mode"] == "fixture" or report["summary"].get("passed") == report["summary"]["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
