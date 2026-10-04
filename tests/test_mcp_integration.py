"""End-to-end checks for separately hosted MCP servers over Streamable HTTP."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.tools.mcp import MCPDiscovery, MCPToolError
from app.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    def __init__(self, module: str, port: int, extra_tool: bool = False):
        self.module = module
        self.port = port
        self.extra_tool = extra_tool
        self.process: subprocess.Popen[str] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self) -> None:
        env = os.environ.copy()
        env["MCP_HOST"] = "127.0.0.1"
        env["MCP_PORT"] = str(self.port)
        if self.extra_tool:
            env["LOGISTICS_EXTRA_TOOL"] = "query_eta"
        else:
            env.pop("LOGISTICS_EXTRA_TOOL", None)
        self.process = subprocess.Popen(
            [sys.executable, "-m", self.module],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=4)
            self.process = None

    def restart(self, *, extra_tool: bool = False) -> None:
        self.stop()
        self.extra_tool = extra_tool
        self.start()


@pytest.fixture
def mcp_processes() -> Iterator[tuple[ServerProcess, ServerProcess]]:
    logistics = ServerProcess("mcp_servers.logistics", _free_port())
    after_sale = ServerProcess("mcp_servers.after_sale", _free_port())
    logistics.start()
    after_sale.start()
    try:
        yield logistics, after_sale
    finally:
        logistics.stop()
        after_sale.stop()


@pytest.fixture
def discovery(mcp_processes) -> Iterator[MCPDiscovery]:
    logistics, after_sale = mcp_processes
    instance = MCPDiscovery(
        {
            "logistics": {"transport": "http", "url": logistics.url},
            "after_sale": {"transport": "http", "url": after_sale.url},
        },
        timeout_seconds=5,
    )
    try:
        yield instance
    finally:
        instance.close()


def _refresh_until_tools(discovery: MCPDiscovery, registry: ToolRegistry) -> None:
    deadline = time.monotonic() + 12
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            discovery.refresh(registry)
            if {"query_logistics", "query_warranty"} <= registry.names:
                return
        except Exception as exc:  # noqa: BLE001 - startup readiness can race the HTTP bind
            last_error = exc
        time.sleep(0.1)
    if last_error:
        raise last_error
    raise AssertionError("MCP servers did not publish their tools before timeout")


def _call(definition, args):
    return asyncio.run(definition.handler(args, None))


def test_discovers_and_calls_tools_from_two_real_http_servers(discovery):
    registry = ToolRegistry()

    _refresh_until_tools(discovery, registry)

    assert registry.get("query_logistics").source == "mcp:logistics"
    assert registry.get("query_warranty").source == "mcp:after_sale"
    assert registry.get("query_return_progress").source == "mcp:after_sale"
    logistics_result = _call(registry.get("query_logistics"), {"order_id": "A-100"})
    warranty_result = _call(registry.get("query_warranty"), {"order_id": "A-100"})
    assert "模拟数据" in str(logistics_result)
    assert "模拟数据" in str(warranty_result)


def test_restarting_only_logistics_discovers_new_tool_without_losing_after_sale(mcp_processes, discovery):
    logistics, _ = mcp_processes
    registry = ToolRegistry()
    _refresh_until_tools(discovery, registry)
    registry_identity = id(registry)

    logistics.restart(extra_tool=True)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and "query_eta" not in registry.names:
        discovery.refresh(registry)
        time.sleep(0.1)

    assert registry.get("query_eta").source == "mcp:logistics"
    after_sale_definition = registry.get("query_warranty")
    assert id(registry) == registry_identity
    assert after_sale_definition.source == "mcp:after_sale"
    assert "query_return_progress" in {item.name for item in registry.definitions}
    assert "模拟数据" in str(_call(after_sale_definition, {"order_id": "A-100"}))
    assert "模拟数据" in str(_call(registry.get("query_eta"), {"order_id": "A-100"}))


def test_is_error_result_is_normalized_instead_of_returned_as_success(discovery):
    registry = ToolRegistry()
    _refresh_until_tools(discovery, registry)

    with pytest.raises(MCPToolError, match="物流查询失败"):
        _call(registry.get("query_logistics"), {"order_id": "IS_ERROR"})


def test_failed_server_refresh_removes_only_that_servers_tools(mcp_processes, discovery):
    logistics, _ = mcp_processes
    registry = ToolRegistry()
    _refresh_until_tools(discovery, registry)

    logistics.stop()
    discovery.refresh(registry)

    assert "query_logistics" not in registry.names
    assert registry.get("query_warranty").source == "mcp:after_sale"
