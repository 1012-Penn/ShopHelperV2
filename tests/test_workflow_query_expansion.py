import json

import pytest
from langchain_core.messages import HumanMessage

from app.services.workflow import query_expansion


FALLBACK = "订单 DEMO-1001 的商品能否退货？"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"queries":["订单退货条件","退货期限","商品状态例外"]}',
         [FALLBACK, "订单退货条件", "退货期限", "商品状态例外"]),
        ('{"queries":["退款进度","退款进度","  ","" ,"新查询"]}',
         [FALLBACK, "退款进度", "新查询"]),
        ('{"queries":[]}', [FALLBACK]),
        ('{"queries":["退货资格",null]}', [FALLBACK]),
        ('{"queries":["退货资格"],"queries":["退货期限"]}', [FALLBACK]),
        ('not json', [FALLBACK]),
        ('```json\n{"queries":["退款资格"]}\n```', [FALLBACK]),
        ('{"queries":["售后资格"],"intent":"售后"}', [FALLBACK]),
    ],
)
def test_parse_queries_requires_only_queries_array_and_always_keeps_fallback(
    content, expected
):
    assert query_expansion.parse_queries(content, FALLBACK) == expected


def test_parse_queries_limits_to_four_and_discards_overlong_items():
    content = json.dumps(
        {"queries": ["条件一", "条件二", "条件三", "条件四", "条件五", "超长" * 130]},
        ensure_ascii=False,
    )
    assert query_expansion.parse_queries(content, FALLBACK) == [
        FALLBACK,
        "条件一",
        "条件二",
        "条件三",
    ]


def test_parse_queries_preserves_the_canonical_fallback_text_verbatim():
    assert query_expansion.parse_queries('{"queries":[]}', "  已完整的问题？  ") == [
        "  已完整的问题？  "
    ]


class ModelReply:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.content})()


def test_expansion_is_limited_to_refund_and_after_sale_with_order_context():
    model = ModelReply('{"queries":["退货资格","拆封商品退货例外"]}')
    order = {"order_id": "DEMO-1001", "product": "云朵耳机", "status": "已签收"}

    queries = query_expansion.expand_policy_queries(
        model, FALLBACK, order, "退款退货"
    )

    assert queries == [FALLBACK, "退货资格", "拆封商品退货例外"]
    assert isinstance(model.messages[-1], HumanMessage)
    assert "DEMO-1001" in model.messages[-1].content
    assert "queries" in model.messages[0].content
    with pytest.raises(ValueError):
        query_expansion.expand_policy_queries(model, FALLBACK, order, "物流")
