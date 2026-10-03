import json


def test_demo_order_catalog_is_stable_and_unknown_orders_do_not_leak_random_data():
    from importlib.util import find_spec

    assert find_spec("app.services.workflow.demo_orders") is not None
    from app.services.workflow.demo_orders import get_demo_order, list_demo_orders

    first = list_demo_orders("demo-user")
    second = list_demo_orders("demo-user")

    assert first == second
    assert len(first) >= 2
    assert all(order["source"] == "模拟数据" for order in first)
    assert get_demo_order(first[0]["order_id"], "demo-user") == first[0]
    assert get_demo_order("NOT-A-DEMO-ORDER", "demo-user") is None


def test_demo_order_tool_returns_the_selected_order_record():
    from importlib.util import find_spec

    assert find_spec("app.services.workflow.demo_orders") is not None
    from app.services.workflow.demo_orders import list_demo_orders
    from app.tools.business import build_tools

    order = list_demo_orders("demo-user")[0]
    query_order = next(
        candidate for candidate in build_tools(None, "demo-conversation")
        if candidate.name == "query_order"
    )

    result = json.loads(query_order.invoke({"order_id": order["order_id"]}))

    assert result["order_id"] == order["order_id"]
    assert result["product"] == order["product"]
    assert result["source"] == "模拟数据"
