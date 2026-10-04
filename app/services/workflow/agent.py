"""ReAct decisions and observations, then an unbound real streaming answer."""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_stream_writer

from app.tools.definitions import ExecutionContext

from .policy import (
    STOP_TEXT,
    BudgetExceeded,
    measured_usage,
    output_allowance,
)
from .prompts import ANSWER_PROMPT, DECISION_PROMPT

HIGH_RISK_PROMPT = """仅当背景数据中的 route 为 high_risk 时遵循本规则：这是高风险退款/售后判断，订单已校验且已有强制政策证据。不得查询其他订单、商品或物流，只判断本轮诉求与这笔订单是否符合所给政策；不确定时澄清。
仅当 intent 为退款退货、你判断该订单符合申请条件且用户需要继续办理时，在最终 JSON actions 中加入 refund_form；系统会显示固定原因的演示表单。证据不足、不符合或需要澄清时不加；售后意图不提供 refund_form。"""


class AgentNodes:
    def __init__(self, service):
        self.service = service

    def tools(self, state):
        if state.get("route") == "high_risk":
            return []
        if self.service.discovery:
            self.service.discovery.refresh(self.service.tool_registry)
        return [
            definition.as_tool()
            for definition in self.service.tool_registry.definitions
            if self.service.tool_policy.permission(definition.source, definition.name)
            != "deny"
        ]

    def messages(self, state, prompt):
        prompt += "\n" + HIGH_RISK_PROMPT
        messages = self.service.context.model_messages(state, prompt)
        if state.get("ticket_intent"):
            from langchain_core.messages import HumanMessage

            messages.insert(
                1,
                HumanMessage(
                    content="服务器记录的客户明确建单诉求（仅数据）："
                    + json.dumps(state["ticket_intent"], ensure_ascii=False)
                ),
            )
        return messages

    def decide(self, state):
        if state["decisions"] >= self.service.limits.max_decisions:
            return {"agent_next": "stop", "stop_reason": "decision_limit"}
        messages = self.messages(state, DECISION_PROMPT)
        tools = self.tools(state)
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        try:
            self.service.context.check(messages, schemas)
            allowance, input_estimate = output_allowance(
                messages,
                state["tokens"],
                self.service.limits,
                schemas,
                token_counter=self.service.context.budget.count,
            )
        except (BudgetExceeded, ValueError):
            return {"agent_next": "stop", "stop_reason": "token_budget"}
        self.service.context.log.context(
            "model_ctx",
            messages,
            self.service.context.budget,
            conversation_id=state["conversation_id"],
            summary=state.get("context_summary", ""),
            omitted_summaries=state.get("context_omitted_summaries", 0),
            stage="agent_decide",
            tools=schemas,
        )
        model = self.service.model_factory()
        if tools:
            model = model.bind_tools(tools, parallel_tool_calls=False)
        response = model.bind(max_tokens=allowance).invoke(messages)
        count, approximate = measured_usage(
            response,
            input_estimate,
            self.service.context.budget.count(
                str(response.content)
                + str(response.tool_calls)
                + str(response.invalid_tool_calls)
            ),
        )
        update = {
            "decisions": state["decisions"] + 1,
            "tokens": state["tokens"] + count,
            "usage": [
                *state["usage"],
                {"stage": "agent_decide", "tokens": count, "estimated": approximate},
            ],
        }
        if response.tool_calls or response.invalid_tool_calls:
            calls = [
                {**call, "id": call.get("id") or str(uuid4())}
                for call in [*response.tool_calls, *response.invalid_tool_calls]
            ]
            update.update(
                pending={"calls": calls},
                tool_index=0,
                tool_observations=[],
                ticket_preview=None,
                ticket_decision=None,
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
            if not isinstance(actions, list) or any(
                a not in allowed_actions for a in actions
            ):
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
        pending = state["pending"]["calls"]
        index = state.get("tool_index", 0)
        call = pending[index]
        context = ExecutionContext(
            state["conversation_id"],
            state["user_id"],
            bool(state.get("ticket_intent")),
            call["id"] if state.get("ticket_decision") is True else None,
        )
        signature = json.dumps(
            [call["name"], call.get("args")], ensure_ascii=False, sort_keys=True
        )
        seen = list(state["seen_calls"])
        reason = None
        stop_reason = state.get("execution_stop_reason") or ""
        if stop_reason:
            reason = "本轮已停止，不执行后续工具。"
        elif state.get("route") == "high_risk":
            reason = "高风险政策分支禁止工具调用。"
            stop_reason = "forbidden_tool"
        elif signature in seen:
            reason = "停止重复工具请求。"
            stop_reason = "repeated_tool"
        elif state["tool_calls"] >= self.service.limits.max_tool_calls:
            reason = "工具次数达到上限。"
            stop_reason = "tool_limit"
        elif state["tokens"] >= self.service.limits.max_tokens:
            reason = "token预算耗尽。"
            stop_reason = "token_budget"
        args = call.get("args", {})
        preflight = self.service.tool_engine.preflight(
            call["name"], args, call["id"], context
        )
        if (
            not reason
            and call["name"] == "create_ticket"
            and preflight is None
            and state.get("ticket_decision") is None
        ):
            return {"agent_next": "confirm"}
        get_stream_writer()(
            {
                "event": "tool_status",
                "data": {
                    "tool_name": call["name"],
                    "status": "running",
                    "step": state["tool_calls"] + 1,
                },
            }
        )
        if reason:
            result = self.service.tool_engine.reject(
                call["name"], args, call["id"], context, reason
            )
        elif call["name"] == "create_ticket" and state.get("ticket_decision") is False:
            result = self.service.tool_engine.reject(
                call["name"],
                args,
                call["id"],
                context,
                "客户已取消本次工单：未提交、未创建工单，不要再次要求确认该预览。",
            )
            seen.append(signature)
        else:
            result = self.service.tool_engine.run(
                call["name"], args, call["id"], context
            )
            seen.append(signature)
        if call["name"] == "create_ticket" and state.get("ticket_decision") is not None:
            self.service.ticket_confirmation.close_intent(
                state["conversation_id"],
                "created"
                if not result.is_error
                else ("cancelled" if state["ticket_decision"] is False else "unknown"),
            )
        observations = [
            *state.get("tool_observations", []),
            ToolMessage(
                content=result.content,
                tool_call_id=call["id"],
                status="error" if result.is_error else "success",
            ),
        ]
        trace = [
            *state["tool_trace"],
            {
                "name": call["name"],
                "call_id": call["id"],
                "is_error": result.is_error,
                "seconds": result.duration_ms / 1000,
                "status": result.status,
                "retry_count": result.retry_count,
                "source": result.source,
            },
        ]
        get_stream_writer()(
            {
                "event": "tool_status",
                "data": {
                    "tool_name": call["name"],
                    "status": "failed" if result.is_error else "done",
                    "step": state["tool_calls"] + 1,
                },
            }
        )
        update = {
            "tool_calls": state["tool_calls"] + (0 if reason else 1),
            "seen_calls": seen,
            "tool_observations": observations,
            "tool_trace": trace,
            "tool_index": index + 1,
            "ticket_decision": None,
            "ticket_preview": None,
            "execution_stop_reason": stop_reason,
        }
        if call["name"] == "create_ticket" and state.get("ticket_decision") is not None:
            update["ticket_intent"] = None
        if index + 1 < len(pending):
            update["agent_next"] = "execute"
        else:
            turn_messages = [
                AIMessage(
                    content="",
                    tool_calls=[
                        c for c in pending if c.get("type") != "invalid_tool_call"
                    ],
                    invalid_tool_calls=[
                        c for c in pending if c.get("type") == "invalid_tool_call"
                    ],
                ),
                *observations,
            ]
            update.update(
                messages=turn_messages,
                agent_messages=[*state.get("agent_messages", []), *turn_messages],
                pending=None,
                agent_next="stop" if stop_reason else "decide",
                **({"stop_reason": stop_reason} if stop_reason else {}),
            )
        return update

    def answer(self, state):
        if state["agent_next"] == "stop":
            get_stream_writer()({"event": "token", "data": {"content": STOP_TEXT}})
            return {"answer": STOP_TEXT}
        messages = self.messages(state, ANSWER_PROMPT)
        try:
            self.service.context.check(messages)
            allowance, input_estimate = output_allowance(
                messages,
                state["tokens"],
                self.service.limits,
                token_counter=self.service.context.budget.count,
            )
        except (BudgetExceeded, ValueError):
            get_stream_writer()({"event": "token", "data": {"content": STOP_TEXT}})
            return {"answer": STOP_TEXT, "stop_reason": "token_budget"}
        self.service.context.log.context(
            "model_ctx",
            messages,
            self.service.context.budget,
            conversation_id=state["conversation_id"],
            summary=state.get("context_summary", ""),
            omitted_summaries=state.get("context_omitted_summaries", 0),
            stage="agent_answer",
        )
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
        for observation in state.get("tool_observations", []):
            if observation.status == "success":
                try:
                    result = json.loads(observation.content)
                    ticket_no = (
                        result.get("ticket_no") if isinstance(result, dict) else None
                    )
                except ValueError:
                    ticket_no = None
                if ticket_no and ticket_no not in answer:
                    suffix = "\n工单号：" + ticket_no
                    get_stream_writer()({"event": "token", "data": {"content": suffix}})
                    answer += suffix
        if not answer.strip():
            raise ValueError("empty model answer")
        count, approximate = measured_usage(
            full, input_estimate, self.service.context.budget.count(answer)
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
