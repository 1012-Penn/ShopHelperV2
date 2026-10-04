"""Project tool outputs to useful fields and render internal status codes."""

import json

STATUS = {
    "open": "待处理",
    "closed": "已关闭",
    "shipped": "已发货",
    "delivered": "已签收",
    "in_transit": "运输中",
    "out_for_delivery": "派送中",
    "processing": "处理中",
    "approved": "已通过",
    "rejected": "未通过",
    "in_warranty": "在保",
    "expired": "已过保",
    "pending": "待处理",
    "returned": "已退回",
    "refunded": "已退款",
}
PROJECTIONS = {
    "query_order": {
        "source",
        "order_id",
        "status",
        "total",
        "found",
        "message",
        "product_name",
        "amount",
        "paid_at",
        "purchased_at",
    },
    "query_product": {
        "source",
        "product_query",
        "name",
        "price",
        "in_stock",
        "found",
        "message",
    },
    "create_ticket": {"ticket_no", "status", "message"},
}


def decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def humanize(value):
    if isinstance(value, dict):
        return {
            k: STATUS.get(v, v)
            if k in {"status", "warranty_status", "return_status"}
            and isinstance(v, str)
            else humanize(v)
            for k, v in value.items()
            if not k.startswith("internal_")
        }
    if isinstance(value, list):
        return [humanize(v) for v in value]
    return value


def format_result(definition, value):
    data = decode(value)
    if definition.formatter:
        data = definition.formatter(data)
    elif definition.name in PROJECTIONS and isinstance(data, dict):
        data = {k: v for k, v in data.items() if k in PROJECTIONS[definition.name]}
    data = humanize(data)
    return (
        json.dumps(data, ensure_ascii=False, default=str)
        if not isinstance(data, str)
        else data
    )
