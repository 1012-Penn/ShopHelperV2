"""Stable simulated order data used by the workflow demo."""

from copy import deepcopy


_DEMO_ORDERS = (
    {
        "order_id": "DEMO-1001",
        "product": "云朵降噪耳机",
        "ordered_at": "2026-09-18",
        "status": "已签收",
        "total": 399.0,
        "source": "模拟数据",
    },
    {
        "order_id": "DEMO-1002",
        "product": "陶瓷马克杯（白色）",
        "ordered_at": "2026-09-25",
        "status": "已签收",
        "total": 59.0,
        "source": "模拟数据",
    },
    {
        "order_id": "DEMO-1003",
        "product": "轻量通勤双肩包",
        "ordered_at": "2026-10-01",
        "status": "运输中",
        "total": 229.0,
        "source": "模拟数据",
    },
    {
        "order_id": "DEMO-1004",
        "product": "云朵降噪耳机（蓝色）",
        "ordered_at": "2026-09-29",
        "status": "已签收",
        "total": 429.0,
        "source": "模拟数据",
    },
)


def list_demo_orders(user_id: str) -> list[dict]:
    """Return display-safe, stable demo orders for the current demo identity."""
    if not user_id:
        return []
    return deepcopy(list(_DEMO_ORDERS))


def get_demo_order(order_id: str, user_id: str) -> dict | None:
    if not user_id:
        return None
    for order in _DEMO_ORDERS:
        if order["order_id"] == order_id:
            return deepcopy(order)
    return None
