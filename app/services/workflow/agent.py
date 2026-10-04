"""ReAct decisions and observations, then an unbound real streaming answer."""

import json
import logging
import time
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer

from app.tools.business import build_tools
from app.tools.registry import ToolInputError, UnknownToolError

from .policy import (
    STOP_TEXT,
    BudgetExceeded,
    estimate_tokens,
    measured_usage,
    output_allowance,
)
from .prompts import ANSWER_PROMPT, DECISION_PROMPT

READ_ONLY = frozenset({"query_order", "query_product", "query_logistics"})
HIGH_RISK_PROMPT = """本次是高风险退款/售后判断：订单已校验且已有强制政策证据。不得查询其他订单、商品或物流，只判断本轮诉求与这笔订单是否符合所给政策；不确定时澄清。
仅当 intent 为退款退货、你判断该订单符合申请条件且用户需要继续办理时，在最终 JSON actions 中加入 refund_form；系统会显示固定原因的演示表单。证据不足、不符合或需要澄清时不加；售后意图不提供 refund_form。"""


class AgentNodes:
    def __init__(self, service):
        self.service = service

    def tools(self, state):
        if state.get("route") == "high_risk":
            return []
        return [
            t
            for t in build_tools(self.service.session_factory, state["conversation_id"])
            if t.name in READ_ONLY
        ]

    def messages(self, state, prompt):
        context = {
            "original_question": state["question"],
            "resolved_question": state.get("resolved_question", state["question"]),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "validated_order": state.get("order"),
            "evidence": state["evidence"],
            "mode": state.get("agent_next"),
            "missing": state.get("missing", ""),
        }
        if state.get("route") == "high_risk":
            prompt += "\n" + HIGH_RISK_PROMPT
        return [
            SystemMessage(
                content=prompt
                + "\n本轮数据："
                + json.dumps(context, ensure_ascii=False)
            ),
            *state.get("agent_messages", state["messages"]),
        ]

    def decide(self, state):
        if state["decisions"] >= self.service.limits.max_decisions:
            return {"agent_next": "stop", "stop_reason": "decision_limit"}
        messages = self.messages(state, DECISION_PROMPT)
        tools = self.tools(state)
        schemas = [tool.get_input_schema().model_json_schema() for tool in tools]
        try:
            allowance, input_estimate = output_allowance(
                messages, state["tokens"], self.service.limits, schemas
            )
        except BudgetExceeded:
            return {"agent_next": "stop", "stop_reason": "token_budget"}
        model = self.service.model_factory()
        if tools:
            model = model.bind_tools(tools)
        response = model.bind(max_tokens=allowance).invoke(messages)
        count, approximate = measured_usage(
            response,
            input_estimate,
            estimate_tokens(str(response.content) + str(response.tool_calls)),
        )
        update = {
            "decisions": state["decisions"] + 1,
            "tokens": state["tokens"] + count,
            "usage": [
                *state["usage"],
                {"stage": "agent_decide", "tokens": count, "estimated": approximate},
            ],
        }
        if response.tool_calls:
            calls = [
                {**call, "id": call.get("id") or str(uuid4())}
                for call in response.tool_calls
            ]
            update.update(
                pending={"calls": calls},
                agent_next="tools",
            )
            return update
        try:
            signal = json.loads(response.content)
            if not isinstance(signal, dict) or signal.get("next") not in {
                "answer",
                "clarify",
            }:
                raise ValueError("invalid completion signal")
            actions = signal.get("actions", [])
            allowed_actions = {"handoff", "create_ticket"}
            if (
                state.get("route") == "high_risk"
                and state.get("intent") == "退款退货"
                and state.get("order")
                and state.get("gate_passed")
            ):
                allowed_actions.add("refund_form")
            if not isinstance(actions, list) or any(a not in allowed_actions for a in actions):
                raise ValueError("invalid actions")
            if "refund_form" in actions and signal["next"] != "answer":
                raise ValueError("refund form requires a final answer")
            missing = signal.get("missing", "")
            if not isinstance(missing, str):
                raise TypeError("invalid missing information")
            update.update(
                agent_next=signal["next"],
                missing=missing,
                actions=list(dict.fromkeys(actions)),
                pending=None,
            )
        except (ValueError, TypeError):
            update.update(
                agent_next="stop", stop_reason="invalid_agent_signal", pending=None
            )
        return update

    def execute(self, state):
        runner = self.service.runner_factory(self.tools(state))
        observations, trace = [], list(state["tool_trace"])
        seen, count = list(state["seen_calls"]), state["tool_calls"]
        reason, actions = "", list(state["actions"])
        pending = state["pending"]["calls"]
        try:
            for call in pending:
                signature = json.dumps(
                    [call["name"], call.get("args")], ensure_ascii=False, sort_keys=True
                )
                content, is_error = "本轮已停止，没有执行该工具。", True
                started = time.monotonic()
                if reason:
                    pass
                elif state.get("route") == "high_risk" or call["name"] not in READ_ONLY:
                    reason = "forbidden_tool"
                    content = "该工具不能由自主Agent执行，只能建议用户通过按钮选择。"
                    if call["name"] == "create_ticket":
                        actions = ["create_ticket"]
                elif signature in seen:
                    reason, content = "repeated_tool", "停止重复工具请求。"
                elif count >= self.service.limits.max_tool_calls:
                    reason, content = "tool_limit", "工具次数达到上限。"
                elif state["tokens"] >= self.service.limits.max_tokens:
                    reason, content = "token_budget", "token预算耗尽。"
                else:
                    seen.append(signature)
                    count += 1
                    get_stream_writer()(
                        {
                            "event": "tool_status",
                            "data": {
                                "tool_name": call["name"],
                                "status": "running",
                                "step": count,
                            },
                        }
                    )
                    try:
                        result = runner.run(
                            call["name"], call.get("args") or {}, call["id"]
                        )
                        content, is_error = result.content, result.is_error
                    except (ToolInputError, UnknownToolError):
                        content = "工具请求无效，请补充必要信息。"
                    except Exception as error:  # noqa: BLE001 - isolate external provider/tool failures
                        logging.getLogger(__name__).warning(
                            "Tool failed (%s)", type(error).__name__
                        )
                        content = "工具暂时无法完成，请稍后再试。"
                observations.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=call["id"],
                        status="error" if is_error else "success",
                    )
                )
                trace.append(
                    {
                        "name": call["name"],
                        "call_id": call["id"],
                        "is_error": is_error,
                        "seconds": time.monotonic() - started,
                    }
                )
                get_stream_writer()(
                    {
                        "event": "tool_status",
                        "data": {
                            "tool_name": call["name"],
                            "status": "failed" if is_error else "done",
                            "step": count,
                        },
                    }
                )
        finally:
            runner.close()
        # Commit the assistant request and every observation together, never an orphan.
        self.service.store.save_tool_pair(
            state["conversation_id"],
            AIMessage(content="", tool_calls=pending),
            observations,
        )
        turn_messages = [AIMessage(content="", tool_calls=pending), *observations]
        return {
            "messages": turn_messages,
            "agent_messages": [*state.get("agent_messages", []), *turn_messages],
            "tool_calls": count,
            "seen_calls": seen,
            "pending": None,
            "tool_trace": trace,
            "actions": actions,
            "agent_next": "stop" if reason else "decide",
            **({"stop_reason": reason} if reason else {}),
        }

    def answer(self, state):
        if state["agent_next"] == "stop":
            get_stream_writer()({"event": "token", "data": {"content": STOP_TEXT}})
            return {"answer": STOP_TEXT}
        messages = self.messages(state, ANSWER_PROMPT)
        try:
            allowance, input_estimate = output_allowance(
                messages, state["tokens"], self.service.limits
            )
        except BudgetExceeded:
            get_stream_writer()({"event": "token", "data": {"content": STOP_TEXT}})
            return {"answer": STOP_TEXT, "stop_reason": "token_budget"}
        parts, full = [], None
        for chunk in (
            self.service.model_factory().bind(max_tokens=allowance).stream(messages)
        ):
            full = chunk if full is None else full + chunk
            content = chunk.content
            if not isinstance(content, str):
                content = "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            if content:
                parts.append(content)
                get_stream_writer()({"event": "token", "data": {"content": content}})
        answer = "".join(parts)
        if not answer.strip():
            raise ValueError("empty model answer")
        count, approximate = measured_usage(
            full, input_estimate, estimate_tokens(answer)
        )
        usage = [
            *state["usage"],
            {"stage": "agent_answer", "tokens": count, "estimated": approximate},
        ]
        if state["tokens"] + count > self.service.limits.max_tokens:
            raise BudgetExceeded(
                "token_budget", {"tokens": state["tokens"] + count, "usage": usage}
            )
        return {
            "answer": answer,
            "tokens": state["tokens"] + count,
            "usage": usage,
            "stop_reason": "clarify" if state["agent_next"] == "clarify" else "answer",
        }
