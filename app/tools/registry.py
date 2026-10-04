"""Atomic tool catalog; compatibility runner delegates to the same engine."""

from threading import RLock

from jsonschema import validators
from langchain_core.tools import BaseTool

from .definitions import ToolDefinition, ToolResult

__all__ = ["ToolRegistry", "ToolResult", "ToolRunner", "UnknownToolError"]


class UnknownToolError(ValueError):
    pass


class ToolInputError(ValueError):
    pass


class ToolRegistry:
    def __init__(self, tools=()):
        self._lock = RLock()
        self._tools = {}
        for tool in tools:
            self.register(tool)

    def register(self, definition):
        if isinstance(definition, BaseTool):
            tool = definition
            schema = (
                tool.args_schema
                if isinstance(tool.args_schema, dict)
                else tool.get_input_schema().model_json_schema()
            )
            definition = ToolDefinition(
                tool.name,
                tool.description,
                schema,
                "builtin",
                lambda args, context, t=tool: t.invoke(args),
            )
        self._validate(definition)
        with self._lock:
            if definition.name in self._tools:
                raise ValueError("Tool names must be unique: " + definition.name)
            self._tools[definition.name] = definition

    @staticmethod
    def _validate(definition):
        if (
            not isinstance(definition, ToolDefinition)
            or not definition.name
            or not definition.description.strip()
            or not callable(definition.handler)
        ):
            raise ValueError("Tool name, description, schema and handler are required")
        if (
            not isinstance(definition.input_schema, dict)
            or definition.input_schema.get("type") != "object"
        ):
            raise ValueError("Tool schema must define an object")
        try:
            validators.validator_for(definition.input_schema).check_schema(
                definition.input_schema
            )
        except Exception as error:
            raise ValueError("Invalid tool JSON schema") from error
        if definition.source != "builtin" and not definition.source.startswith("mcp:"):
            raise ValueError("Invalid tool source")

    def replace_source(self, source, definitions):
        if source == "builtin":
            raise ValueError("Cannot replace built-in tools from discovery")
        fresh = {}
        for definition in definitions:
            self._validate(definition)
            if definition.source != source or definition.name in fresh:
                raise ValueError("Source mismatch or duplicate tool")
            fresh[definition.name] = definition
        with self._lock:
            rest = {n: t for n, t in self._tools.items() if t.source != source}
            if rest.keys() & fresh.keys():
                raise ValueError("Tool name conflict across sources")
            self._tools = {**rest, **fresh}

    def get(self, name):
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as error:
                raise UnknownToolError("Unknown tool: " + str(name)) from error

    @property
    def definitions(self):
        with self._lock:
            return list(self._tools.values())

    @property
    def tools(self):
        return [t.as_tool() for t in self.definitions]

    @property
    def names(self):
        with self._lock:
            return frozenset(self._tools)


class ToolRunner:
    def __init__(self, registry, timeout_seconds, max_retries, **kwargs):
        from .engine import ToolEngine

        self.registry = registry
        self.engine = ToolEngine(registry, timeout_seconds, max_retries, **kwargs)

    def run(self, name, args, tool_call_id, context=None):
        return self.engine.run(name, args, tool_call_id, context)

    def close(self):
        self.engine.close()
