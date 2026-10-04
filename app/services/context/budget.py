"""One calibrated estimator and window-derived budgets (not cumulative API cost)."""

import json
import math
import re
from dataclasses import dataclass


def token_count(value, chinese_ratio=1.0):
    if not isinstance(value, str):
        if isinstance(value, (list, tuple)):
            return sum(token_count(v, chinese_ratio) + 6 for v in value)
        if hasattr(value, "content"):
            return token_count(value.content, chinese_ratio) + token_count(
                getattr(value, "tool_calls", []), chinese_ratio
            )
        value = json.dumps(value, ensure_ascii=False, default=str)
    chinese = len(re.findall(r"[\u3400-\u9fff]", value))
    return math.ceil(chinese * chinese_ratio + (len(value) - chinese) / 4)


@dataclass(frozen=True)
class ContextBudget:
    window: int = 128000
    output: int = 2000
    user_input: int = 2000
    steps: int = 4
    tool_result: int = 1200
    top_k: int = 5
    system_tools: int = 2200
    evidence_per_item: int = 150
    summary: int = 1000
    safety: int = 500
    target_turns: int = 40
    steady_turn_tokens: int = 2000
    decision_overhead: int = 100
    chinese_ratio: float = 1.0
    assistant_chars: int = 60

    def __post_init__(self):
        for key, val in vars(self).items():
            if key == "chinese_ratio":
                if not math.isfinite(val) or val <= 0:
                    raise ValueError("CHINESE_TOKEN_RATIO must be positive")
            elif type(val) is not int or val < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.layer1_tokens < self.steady_turn_tokens:
            raise ValueError("上下文预算不足：层1连一轮稳态对话都装不下")

    @property
    def peak(self):
        return self.user_input + self.steps * (
            self.tool_result + self.decision_overhead
        )

    @property
    def history_tokens(self):
        available = (
            self.window
            - self.system_tools
            - self.top_k * self.evidence_per_item
            - self.summary
            - self.output
            - self.safety
            - self.peak
        )
        return min(self.target_turns * self.steady_turn_tokens, available)

    @property
    def layer1_tokens(self):
        # A single boundary envelope token is intentionally not assigned to a layer.
        return self.history_tokens * 7 // 10 - 1

    @property
    def layer2_tokens(self):
        return self.history_tokens * 3 // 10

    def count(self, value):
        return token_count(value, self.chinese_ratio)

    @classmethod
    def from_env(cls, values):
        values = dict(values)
        for canonical, legacy in [
            ("MAX_OUTPUT_TOKENS", "AGENT_MAX_OUTPUT_TOKENS"),
            ("MAX_AGENT_STEPS", "AGENT_MAX_DECISIONS"),
        ]:
            if not values.get(canonical) and values.get(legacy):
                values[canonical] = values[legacy]
        names = {
            "window": "MODEL_CONTEXT_WINDOW",
            "output": "MAX_OUTPUT_TOKENS",
            "user_input": "MAX_USER_INPUT_TOKENS",
            "steps": "MAX_AGENT_STEPS",
            "tool_result": "TOOL_RESULT_MAX_TOKENS",
            "top_k": "RERANK_TOP_K",
            "system_tools": "CONTEXT_SYSTEM_TOOLS_TOKENS",
            "evidence_per_item": "CONTEXT_EVIDENCE_ITEM_TOKENS",
            "summary": "CONTEXT_SUMMARY_TOKENS",
            "safety": "CONTEXT_SAFETY_TOKENS",
            "target_turns": "CONTEXT_TARGET_TURNS",
            "steady_turn_tokens": "CONTEXT_STEADY_TURN_TOKENS",
            "decision_overhead": "CONTEXT_DECISION_TOKENS",
            "assistant_chars": "CONTEXT_ASSISTANT_CHARS",
            "chinese_ratio": "CHINESE_TOKEN_RATIO",
        }
        return cls(
            **{
                k: (float(values[v]) if k == "chinese_ratio" else int(values[v]))
                for k, v in names.items()
                if values.get(v) is not None
            }
        )
