"""Bounded, synchronous discovery bridge for Streamable HTTP MCP servers."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient

from .definitions import ToolDefinition


class MCPToolError(RuntimeError):
    """An MCP response had ``isError=True`` and must not be treated as data."""


class _AsyncBridge:
    """One owned asyncio loop for synchronous MCP discovery calls."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._serve, name="mcp-async-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(3):
            raise RuntimeError("MCP async loop did not start")

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def run(self, awaitable, timeout_seconds: float):
        if self._closed or self._loop is None:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise RuntimeError("MCP async bridge is closed")
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError("MCP operation timed out") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=3)


def _json_schema(tool) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return tool.get_input_schema().model_json_schema()


def _normalize_result(result):
    """Drop adapter wrappers while preferring MCP structuredContent."""
    content, artifact = result if isinstance(result, tuple) and len(result) == 2 else (result, None)
    structured = getattr(artifact, "structured_content", None)
    if structured is not None:
        return structured
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if len(texts) == 1:
            text = texts[0]
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return text
        if texts:
            return "\n".join(texts)
    if hasattr(content, "model_dump"):
        return content.model_dump()
    return content


class MCPDiscovery:
    """Refresh MCP tool definitions in a registry, one atomic source at a time."""

    def __init__(self, connections: dict[str, dict[str, Any]], timeout_seconds: float = 5):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.connections = dict(connections)
        self.timeout_seconds = timeout_seconds
        self.last_errors: dict[str, str] = {}
        self._client = MultiServerMCPClient(self.connections, handle_tool_errors=False)
        self._bridge = _AsyncBridge()
        self._lock = threading.RLock()

    async def _tools_for(self, server_name: str):
        return await asyncio.wait_for(
            self._client.get_tools(server_name=server_name),
            timeout=self.timeout_seconds,
        )

    async def _invoke(self, tool, args: dict[str, Any]):
        try:
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=self.timeout_seconds)
        except ToolException as error:
            raise MCPToolError(str(error)) from error
        return _normalize_result(result)

    def refresh(self, registry) -> None:
        """Discover each configured server and replace only its tool subset."""
        with self._lock:
            for server_name in self.connections:
                source = f"mcp:{server_name}"
                try:
                    tools = self._bridge.run(
                        self._tools_for(server_name), self.timeout_seconds + 0.25
                    )
                    definitions = []
                    for tool in tools:
                        async def invoke(args, bound_tool=tool):
                            return await self._invoke(bound_tool, args)

                        definitions.append(ToolDefinition(
                            name=tool.name,
                            description=tool.description or "MCP tool",
                            input_schema=_json_schema(tool),
                            source=source,
                            handler=lambda args, context, call=invoke: call(args),
                        ))
                    registry.replace_source(source, definitions)
                    self.last_errors.pop(server_name, None)
                except Exception as error:  # noqa: BLE001 - isolate failures to this MCP server
                    # A server outage or invalid/duplicate declaration withdraws
                    # only that server's old definitions.
                    registry.replace_source(source, [])
                    self.last_errors[server_name] = str(error)

    def close(self) -> None:
        """Stop the owned event-loop thread after application shutdown."""
        self._bridge.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
