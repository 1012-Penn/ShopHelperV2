import pytest

from app.tools.registry import ToolRegistry


def definition(name="query_eta", source="mcp:logistics"):
    from app.tools.definitions import ToolDefinition

    return ToolDefinition(
        name,
        "查询预计送达",
        {"type": "object", "properties": {}},
        source,
        lambda args, context: {"eta": "明天"},
    )


def test_register_and_source_replacement():
    registry = ToolRegistry([])
    registry.register(definition("hours", "builtin"))
    registry.replace_source("mcp:logistics", [definition()])
    assert registry.names == {"hours", "query_eta"}
    registry.replace_source("mcp:logistics", [])
    assert registry.names == {"hours"}


def test_conflicting_source_does_not_overwrite():
    registry = ToolRegistry([])
    registry.register(definition("create_ticket", "builtin"))
    with pytest.raises(ValueError):
        registry.replace_source("mcp:logistics", [definition("create_ticket")])
    assert registry.get("create_ticket").source == "builtin"


def test_invalid_schema_is_not_registered():
    registry = ToolRegistry([])
    with pytest.raises(ValueError):
        registry.register(
            definition().__class__(
                "bad", "", {"type": "nope"}, "builtin", lambda a, c: {}
            )
        )
    assert not registry.names


def test_hot_permission_and_corruption_fail_closed(tmp_path):
    from app.tools.policy import ToolPolicy

    path = tmp_path / "policy.json"
    path.write_text('{"mcp":{}}')
    policy = ToolPolicy(path)
    assert policy.permission("mcp:logistics", "query_eta") == "deny"
    path.write_text('{"mcp":{"logistics":{"query_eta":"read"}}}')
    assert policy.permission("mcp:logistics", "query_eta") == "read"
    path.write_text('{"mcp":{"logistics":{"query_eta":"write"}}}')
    assert policy.permission("mcp:logistics", "query_eta") == "deny"
    path.write_text("{bad")
    assert policy.permission("mcp:logistics", "query_eta") == "deny"
