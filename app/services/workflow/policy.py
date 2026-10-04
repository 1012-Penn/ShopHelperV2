"""Budgets shared by the plain loop and graph agent; no orchestration dependency."""

import json
from dataclasses import dataclass


class BudgetExceeded(ValueError):
    def __init__(self, reason, usage_update=None):
        super().__init__(reason)
        self.usage_update = usage_update or {}


@dataclass(frozen=True)
class Limits:
    max_decisions: int = 4
    max_tool_calls: int = 6
    max_output_tokens: int = 512
    max_tokens: int = 12000

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("agent limits must be positive integers")


def estimate_tokens(value):
    # UTF-8 byte count is deliberately conservative for providers without usage.
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return len(value.encode("utf-8")) + 16


def output_allowance(messages, used, limits, tools=None, token_counter=estimate_tokens):
    needed = token_counter(messages) + (token_counter(tools) if tools else 0)
    available = limits.max_tokens - used - needed
    if available < 1:
        raise BudgetExceeded("token_budget")
    return min(limits.max_output_tokens, available), needed


def measured_usage(response, input_estimate, output_estimate):
    usage = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage_metadata", None)
    )
    total = (usage or {}).get("total_tokens")
    if type(total) is int and total > 0:
        return total, False
    return input_estimate + output_estimate, True


STOP_TEXT = "本轮查询已达到处理上限，请补充信息或联系人工客服核实。"

DEFAULT_LIMITS = Limits()
