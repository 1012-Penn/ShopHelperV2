import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from sqlalchemy import select

from app.db.session import create_tables, make_engine, make_session_factory
from app.tools.definitions import ExecutionContext, ToolDefinition
from app.tools.registry import ToolRegistry


@pytest.fixture
def rig(tmp_path):
    from app.tools.audit import AuditWriter
    from app.tools.engine import ToolEngine

    db = make_engine(f"sqlite:///{tmp_path}/audit.db")
    create_tables(db)
    sessions = make_session_factory(db)
    registry = ToolRegistry([])
    engine = ToolEngine(
        registry,
        timeout_seconds=0.01,
        max_retries=2,
        audit=AuditWriter(sessions),
        session_factory=sessions,
    )
    yield registry, engine, sessions
    engine.close()
    db.dispose()


def add(registry, handler, name="lookup", source="builtin", schema=None):
    registry.register(
        ToolDefinition(
            name,
            "测试工具",
            schema
            or {"type": "object", "properties": {}, "additionalProperties": False},
            source,
            handler,
        )
    )


@pytest.mark.parametrize("args", [{}, {"n": "2"}, {"n": 0}, {"n": 5}, {"n": True}])
def test_strict_validation_returns_observation_and_audit(rig, args):
    from app.db.models import ToolAuditLog

    registry, engine, sessions = rig
    calls = []
    add(
        registry,
        lambda a, c: calls.append(a),
        schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 3}},
            "required": ["n"],
        },
    )
    result = engine.run("lookup", args, "bad", ExecutionContext("conv", "user"))
    assert result.status == "validation_blocked"
    assert result.error_kind == "invalid_arguments"
    assert not calls
    with sessions() as s:
        audit = s.scalar(select(ToolAuditLog))
        assert audit.status == "validation_blocked" and audit.arguments == args


def test_read_timeout_retries_but_write_timeout_never_retries(rig):
    from app.db.models import ToolAuditLog

    registry, engine, sessions = rig
    calls = []

    def slow(a, c):
        calls.append(c)
        time.sleep(0.1)
        return {"status": "open"}

    add(registry, slow)
    add(registry, slow, "create_ticket")
    read = engine.run("lookup", {}, "read", ExecutionContext("conv", "user"))
    write = engine.run(
        "create_ticket", {}, "write", ExecutionContext("conv", "user", True, "write")
    )
    assert read.status == "timeout" and read.retry_count == 2
    assert write.status == "timeout" and write.retry_count == 0
    again = engine.run(
        "create_ticket", {}, "write", ExecutionContext("conv", "user", True, "write")
    )
    assert again.is_error and len(calls) == 4
    with sessions() as s:
        rows = list(s.scalars(select(ToolAuditLog)))
        assert len(rows) == 2
        assert all(r.duration_ms > 0 for r in rows)


def test_empty_business_and_nontransient_error_no_retry(rig):
    registry, engine, _ = rig
    calls = []
    add(
        registry,
        lambda a, c: calls.append(1) or {"found": False, "message": "没有订单"},
    )
    result = engine.run("lookup", {}, "empty", ExecutionContext())
    assert result.error_kind == "not_found" and len(calls) == 1

    def fail(a, c):
        calls.append(1)
        raise ValueError("业务失败")

    add(registry, fail, "bad")
    result = engine.run("bad", {}, "fail", ExecutionContext())
    assert result.status == "failed" and result.retry_count == 0 and len(calls) == 2


def test_write_requires_matching_confirmation(rig):
    registry, engine, _ = rig
    calls = []
    add(registry, lambda a, c: calls.append(1) or {}, "create_ticket")
    for ctx in [
        ExecutionContext(),
        ExecutionContext("c", "u", True),
        ExecutionContext("c", "u", True, "other"),
    ]:
        assert engine.run("create_ticket", {}, "id", ctx).status == "permission_denied"
    assert not calls


def test_external_unknown_denied_and_audit_failure_does_not_block(rig):
    registry, engine, _ = rig
    add(
        registry,
        lambda a, c: {
            "status": "open",
            "ticket_no": "TKT-测试",
            "internal_debug": "secret",
        },
    )
    engine.audit.record = lambda *a, **k: (_ for _ in ()).throw(OSError("db down"))
    result = engine.run("lookup", {}, "ok", ExecutionContext())
    assert result.status == "success" and "\\u" not in result.content
    assert json.loads(result.content)["status"] == "待处理"
    add(registry, lambda a, c: {}, "evil", "mcp:logistics")
    assert (
        engine.run("evil", {}, "evil", ExecutionContext()).status == "permission_denied"
    )


@pytest.mark.parametrize("replacement_source", ["mcp:trusted", "mcp:unapproved"])
def test_discovery_cannot_swap_the_validated_executor(rig, tmp_path, replacement_source):
    from app.db.models import ToolAuditLog
    from app.tools.policy import ToolPolicy

    registry, engine, sessions = rig
    policy_path = tmp_path / "permissions.json"
    policy_path.write_text('{"mcp":{"trusted":{"lookup":"read"}}}')
    permission_checked, refreshed = Event(), Event()

    class PausingPolicy(ToolPolicy):
        def permission(self, source, name):
            permission = super().permission(source, name)
            permission_checked.set()
            assert refreshed.wait(5), "Discovery refresh never completed"
            return permission

    engine.policy = PausingPolicy(policy_path)
    calls = []
    add(registry, lambda a, c: calls.append("original") or {}, source="mcp:trusted")
    with ThreadPoolExecutor(max_workers=1) as pool:
        invocation = pool.submit(
            engine.run, "lookup", {}, "refresh-race", ExecutionContext("conv", "user")
        )
        try:
            assert permission_checked.wait(5), "Invocation never checked permission"
            registry.replace_source("mcp:trusted", [])
            registry.replace_source(replacement_source, [ToolDefinition(
                "lookup", "Replacement with different arguments",
                {"type": "object", "required": ["new_required_argument"]},
                replacement_source,
                lambda a, c: calls.append("replacement") or {},
            )])
        finally:
            refreshed.set()
        result = invocation.result(timeout=5)

    assert calls == ["original"]
    assert result.status == "success" and result.source == "mcp:trusted"
    with sessions() as session:
        audit = session.scalar(select(ToolAuditLog))
        assert audit.source == "mcp:trusted" and audit.status == "success"


def test_permission_revoked_after_graph_preflight_blocks_execution(rig, tmp_path):
    from app.db.models import ToolAuditLog
    from app.tools.policy import ToolPolicy

    registry, engine, sessions = rig
    policy_path = tmp_path / "permissions.json"
    policy_path.write_text('{"mcp":{"trusted":{"lookup":"read"}}}')
    engine.policy = ToolPolicy(policy_path)
    calls = []
    add(registry, lambda a, c: calls.append(1) or {}, source="mcp:trusted")
    context = ExecutionContext("conv", "user")
    assert engine.preflight("lookup", {}, "revoked", context) is None

    policy_path.write_text('{"mcp":{}}')
    result = engine.run("lookup", {}, "revoked", context)

    assert result.status == "permission_denied" and not calls
    with sessions() as session:
        assert session.scalar(select(ToolAuditLog)).status == "permission_denied"
