def test_extract_returns_fixed_after_sale_fields():
    from app.services.after_sale import AfterSaleService

    class FakeStructured:
        def invoke(self, messages):
            return {
                "order_id": "123456",
                "request_type": "换货",
                "expected_solution": "尽快补发",
            }

    result = AfterSaleService(FakeStructured()).extract(
        "订单123456耳机坏了，想换货并尽快补发"
    )
    assert result.order_id == "123456"
    assert result.request_type == "换货"
    assert result.expected_solution == "尽快补发"


def test_extract_preserves_missing_fields_as_none():
    from app.services.after_sale import AfterSaleService

    class FakeStructured:
        def invoke(self, messages):
            return {"order_id": None, "request_type": "物流", "expected_solution": None}

    result = AfterSaleService(FakeStructured()).extract("快递到哪里了")
    assert result.order_id is None
    assert result.request_type == "物流"
    assert result.expected_solution is None
