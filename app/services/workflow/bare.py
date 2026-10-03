"""An agent is an ordinary LLM/tool/observation loop, without an agent framework."""

import json
import logging
from uuid import uuid4

from .policy import (
    DEFAULT_LIMITS,
    STOP_TEXT,
    BudgetExceeded,
    estimate_tokens,
    measured_usage,
    output_allowance,
)


def run_bare_agent(messages, call_model, run_tool, limits=DEFAULT_LIMITS):
    messages = list(messages)
    decisions = tool_count = used = 0
    estimated = False
    seen = set()
    reason, answer = "decision_limit", STOP_TEXT
    for _ in range(limits.max_decisions):
        try:
            allowance, input_estimate = output_allowance(messages, used, limits)
        except BudgetExceeded:
            reason = "token_budget"
            break
        response = call_model(messages, allowance)
        decisions += 1
        count, approximate = measured_usage(
            response, input_estimate, estimate_tokens(response)
        )
        used += count
        estimated |= approximate
        calls = response.get("tool_calls") or []
        if not calls:
            answer = response.get("content") or "请补充问题信息。"
            reason = "answer"
            break
        normalized = [{**call, "id": call.get("id") or str(uuid4())} for call in calls]
        messages.append({"role": "assistant", "content": "", "tool_calls": normalized})
        stopped = False
        for call in normalized:
            signature = json.dumps(
                [call.get("name"), call.get("args")], ensure_ascii=False, sort_keys=True
            )
            if signature in seen:
                reason, content, stopped = "repeated_tool", "停止：重复工具请求。", True
            elif tool_count >= limits.max_tool_calls:
                reason, content, stopped = (
                    "tool_limit",
                    "停止：工具次数达到上限。",
                    True,
                )
            elif used >= limits.max_tokens:
                reason, content, stopped = (
                    "token_budget",
                    "停止：token 预算耗尽。",
                    True,
                )
            elif stopped:
                content = "本轮已停止，没有执行该工具。"
            else:
                seen.add(signature)
                tool_count += 1
                try:
                    content = run_tool(call["name"], call.get("args") or {}, call["id"])
                except Exception as error:  # noqa: BLE001 - isolate external provider/tool failures
                    logging.getLogger(__name__).warning(
                        "Bare tool failed (%s)", type(error).__name__
                    )
                    content = "工具请求无效或执行失败。"
            messages.append(
                {"role": "tool", "content": str(content), "tool_call_id": call["id"]}
            )
        if stopped:
            break
    return {
        "answer": answer,
        "stop_reason": reason,
        "decisions": decisions,
        "tool_calls": tool_count,
        "tokens": used,
        "estimated": estimated,
        "messages": messages,
    }
