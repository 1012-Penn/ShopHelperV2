"""Trusted application tool metadata and execution context."""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import StructuredTool


@dataclass(frozen=True)
class ExecutionContext:
    conversation_id: str = ""
    user_id: str = ""
    ticket_requested: bool = False
    confirmed_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    source: str
    handler: Callable
    formatter: Callable | None = None

    def as_tool(self):
        # The model receives a schema; execution always goes through ToolEngine.
        return StructuredTool.from_function(
            func=lambda **args: self.handler(args, ExecutionContext()),
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
            infer_schema=False,
        )

    def invoke(self, args):
        return self.handler(args, ExecutionContext())

    def get_input_schema(self):
        return self.as_tool().get_input_schema()


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    tool_call_id: str
    content: str
    is_error: bool
    status: str = "success"
    error_kind: str | None = None
    retry_count: int = 0
    duration_ms: float = 0
    source: str = "builtin"
    error_detail: str | None = None
