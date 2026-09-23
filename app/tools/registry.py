"""Tool registration, input validation, and bounded execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import ValidationError


class UnknownToolError(ValueError):
    """Raised when the model requests a tool not registered by the app."""


class ToolInputError(ValueError):
    """Raised when tool arguments do not match that tool's schema."""


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    tool_call_id: str
    content: str
    is_error: bool


class ToolRegistry:
    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique")

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise UnknownToolError(f"Unknown tool: {name}") from error

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)


class ToolRunner:
    def __init__(self, registry: ToolRegistry, timeout_seconds: float, max_retries: int) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mewhelp-tool")

    def run(self, name: str, args: dict, tool_call_id: str) -> ToolResult:
        tool = self.registry.get(name)
        try:
            validated = tool.get_input_schema().model_validate(args)
        except ValidationError as error:
            raise ToolInputError(f"Invalid arguments for {name}") from error

        safe_error = "工具暂时无法完成，请稍后再试。"
        for attempt in range(self.max_retries + 1):
            future = self._executor.submit(tool.invoke, validated.model_dump())
            try:
                content = future.result(timeout=self.timeout_seconds)
                return ToolResult(name, tool_call_id, str(content), False)
            except TimeoutError:
                future.cancel()
                safe_error = "工具执行超时，请稍后再试。"
                break
            except Exception:
                if attempt == self.max_retries:
                    break

        return ToolResult(name, tool_call_id, safe_error, True)
