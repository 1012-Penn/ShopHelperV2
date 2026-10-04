import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.services.workflow import intents


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"intent":"退款退货","confidence":0.91}', ("退款退货", 0.91)),
        ('{"intent":"其他","confidence":0}', ("其他", 0.0)),
        ('{"intent":"物流","confidence":1}', ("物流", 1.0)),
    ],
)
def test_parse_intent_accepts_only_named_intent_and_confidence(content, expected):
    assert intents.parse_intent(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '```json\n{"intent":"物流","confidence":0.8}\n```',
        '{"intent":"物流","confidence":0.8,"extra":1}',
        '{"intent":"物流"}',
        '{"intent":"业务咨询","confidence":0.8}',
        '{"intent":"物流","confidence":true}',
        '{"intent":"物流","confidence":"0.8"}',
        '{"intent":"物流","confidence":NaN}',
        '{"intent":"物流","confidence":-0.1}',
        '{"intent":"物流","confidence":1.1}',
        '{"intent":"物流","confidence":0.8,"confidence":0.9}',
        '[]',
    ],
)
def test_parse_intent_rejects_invalid_json_shapes_and_confidences(content):
    with pytest.raises(ValueError):
        intents.parse_intent(content)


class ModelReply:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": self.content})()


def test_resolve_question_preserves_a_complete_question_exactly():
    original = "  请问，我的保温杯能放洗碗机吗？  "
    model = ModelReply(
        json.dumps(
            {"question": original, "reference_resolved": True}, ensure_ascii=False
        )
    )

    question, resolved = intents.resolve_question(model, [], original)

    assert question == original
    assert resolved is True
    assert model.messages[0].content == intents.REFERENCE_PROMPT


def test_unresolved_but_classifiable_referent_is_preserved_without_guessing():
    original = "它能退吗？"
    model = ModelReply(
        json.dumps(
            {"question": original, "reference_resolved": False}, ensure_ascii=False
        )
    )

    assert intents.resolve_question(model, [], original) == (original, False)


def test_reference_context_omits_tool_messages_and_keeps_user_assistant_history():
    history = [
        HumanMessage(content="这款云朵耳机续航多久？"),
        ToolMessage(content="内部工具数据", tool_call_id="tool-1"),
        HumanMessage(content="它能退吗？"),
    ]
    model = ModelReply(
        '{"question":"云朵降噪耳机能退吗？","reference_resolved":true}'
    )

    question, resolved = intents.resolve_question(model, history, "它能退吗？")

    assert (question, resolved) == ("云朵降噪耳机能退吗？", True)
    rendered = str(model.messages)
    assert "内部工具数据" not in rendered
    assert "这款云朵耳机续航多久？" in rendered
    assert "它能退吗？" in rendered


def test_reference_parser_rejects_extra_keys_and_non_boolean_resolution():
    for content in (
        '{"question":"它能退吗？","reference_resolved":false,"intent":"退款退货"}',
        '{"question":"它能退吗？","reference_resolved":"false"}',
        '{"question":"它能退吗？","reference_resolved":0}',
        '{"question":"它能退吗？","reference_resolved":false,"reference_resolved":true}',
        '{"question":"某商品能退吗？","reference_resolved":false}',
    ):
        model = ModelReply(content)
        with pytest.raises(ValueError):
            intents.resolve_question(model, [], "它能退吗？")


def test_intent_prompt_has_exact_output_contract_boundary_examples_and_other():
    assert '{"intent":"退款退货","confidence":0.93}' in intents.INTENT_PROMPT
    assert "confidence" in intents.INTENT_PROMPT
    for intent in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"):
        assert intent in intents.INTENT_PROMPT
    for boundary in ("边界示例", "已提交售后", "投诉", "拿不准"):
        assert boundary in intents.INTENT_PROMPT
