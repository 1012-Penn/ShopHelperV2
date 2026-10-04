import pytest

from app.services.context.budget import ContextBudget, token_count


def test_demo_budget():
    b = ContextBudget.from_env(
        {
            "MODEL_CONTEXT_WINDOW": "18000",
            "MAX_OUTPUT_TOKENS": "2000",
            "MAX_USER_INPUT_TOKENS": "2000",
            "MAX_AGENT_STEPS": "3",
            "TOOL_RESULT_MAX_TOKENS": "1200",
            "RERANK_TOP_K": "5",
        }
    )
    assert (b.history_tokens, b.layer1_tokens, b.layer2_tokens) == (5650, 3954, 1695)


def test_default_retains_twenty_steady_turns():
    b = ContextBudget()
    assert b.layer1_tokens >= 20 * b.steady_turn_tokens


def test_inadequate_and_invalid_config():
    with pytest.raises(ValueError, match="上下文预算不足"):
        ContextBudget(window=4000)
    with pytest.raises(ValueError):
        ContextBudget.from_env({"MAX_AGENT_STEPS": "0"})


def test_token_calibration_is_shared():
    assert token_count("中文") == 2
    assert token_count("a" * 8) == 2
    assert token_count("中文", chinese_ratio=2) == 4


def test_legacy_agent_limits_and_canonical_precedence():
    old=ContextBudget.from_env({'AGENT_MAX_OUTPUT_TOKENS':'512','AGENT_MAX_DECISIONS':'2'})
    assert old.output==512 and old.steps==2
    new=ContextBudget.from_env({'AGENT_MAX_OUTPUT_TOKENS':'512','MAX_OUTPUT_TOKENS':'2000','AGENT_MAX_DECISIONS':'2','MAX_AGENT_STEPS':'3'})
    assert new.output==2000 and new.steps==3
