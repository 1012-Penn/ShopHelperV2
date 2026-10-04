"""Fixed routing and evidence gate around a checkpointed agent subflow."""

import json
import logging
import math
import re
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.services.context.manager import ContextManager
from app.services.quality.generation import REFUSAL
from app.services.quality.ledger import QualityLedger

from .actions import TicketActions
from .agent import AgentNodes
from .intents import (
    CHATTER_TEXT,
    COMPLAINT_TEXT,
    INTENT_PROMPT,
    INVALID_INTENT_TEXT,
    REFER_PROMPT,
    ROUTES,
    is_greeting,
    parse_intent,
)
from .logging import WorkflowLog
from .policy import (
    DEFAULT_LIMITS,
    BudgetExceeded,
    measured_usage,
    output_allowance,
)
from .state import WorkflowState
from .storage import ConversationLocks, ConversationStore


class WorkflowService:
    def __init__(
        self,
        session_factory,
        model_factory,
        runner_factory,
        retriever,
        checkpointer,
        log_path,
        limits=DEFAULT_LIMITS,
        min_score=0.05,
        strategy="hybrid_rerank",
        context_budget=None,
        context_log_path=None,
        summarizer=None,
    ):
        if not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("knowledge threshold must be finite in [0,1]")
        self.session_factory, self.model_factory = session_factory, model_factory
        self.runner_factory, self.retriever, self.checkpointer = (
            runner_factory,
            retriever,
            checkpointer,
        )
        self.limits, self.min_score, self.strategy = limits, min_score, strategy
        self.store, self.locks = ConversationStore(session_factory), ConversationLocks()
        self.ticket_actions = TicketActions(session_factory, runner_factory, self.locks)
        self.ledger = QualityLedger(session_factory)
        self.log_path = Path(log_path)
        self.logger = WorkflowLog(log_path)
        self.context = ContextManager(
            session_factory,
            model_factory,
            context_budget,
            context_log_path or self.log_path.parent / "app.log",
            summarize=summarizer,
        )
        self._closers = []
        self.graph = self._build_graph()

    def _tracked(self, name, node):
        def wrapped(state):
            path = [*state.get("path", []), name]
            get_stream_writer()(
                {
                    "event": "workflow_status",
                    "data": {"node": name, "step": state.get("decisions", 0)},
                }
            )
            return {**node({**state, "path": path}), "path": path}

        return wrapped

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        agent = AgentNodes(self)
        nodes = {
            "refer": self._refer,
            "classify": self._classify,
            "route": self._route,
            "retrieve": self._retrieve,
            "gate": self._gate,
            "fixed": self._fixed,
            "agent": agent.decide,
            "tools": agent.execute,
            "agent_answer": agent.answer,
            "log": self._log,
        }
        for name, node in nodes.items():
            builder.add_node(name, self._tracked(name, node))
        builder.add_edge(START, "refer")
        builder.add_edge("refer", "classify")
        builder.add_edge("classify", "route")
        builder.add_conditional_edges(
            "route",
            lambda s: s["route"] or "error",
            {
                "knowledge": "retrieve",
                "business": "agent",
                "complaint": "fixed",
                "chitchat": "fixed",
                "error": "fixed",
            },
        )
        builder.add_edge("retrieve", "gate")
        builder.add_conditional_edges(
            "gate", lambda s: s["gate_passed"], {True: "agent", False: "fixed"}
        )
        builder.add_conditional_edges(
            "agent",
            lambda s: s["agent_next"],
            {
                "tools": "tools",
                "answer": "agent_answer",
                "clarify": "agent_answer",
                "stop": "agent_answer",
            },
        )
        builder.add_conditional_edges(
            "tools",
            lambda s: s["agent_next"],
            {"decide": "agent", "stop": "agent_answer"},
        )
        builder.add_edge("agent_answer", "log")
        builder.add_edge("fixed", "log")
        builder.add_edge("log", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _refer(self, state):
        question = state["question"]
        if not (
            state.get("context_history") or state.get("context_summary")
        ) or not re.search(r"那个|这个|它|最开始|之前|刚才|前面|后来", question):
            return {"resolved_question": question}
        prompt = REFER_PROMPT
        messages = self.context.history_messages(state, prompt)
        estimate = self.context.check(messages)
        self.context.log.context(
            "refer_ctx",
            messages,
            self.context.budget,
            conversation_id=state["conversation_id"],
        )
        response = (
            self.model_factory()
            .bind(max_tokens=min(400, self.context.budget.output))
            .invoke(messages)
        )
        count, approximate = measured_usage(
            response, estimate, self.context.budget.count(response.content)
        )
        try:
            resolved = json.loads(response.content).get("question", question)
            if not isinstance(resolved, str) or not resolved.strip():
                resolved = question
        except (ValueError, TypeError, AttributeError):
            resolved = question
        return {
            "resolved_question": resolved,
            "tokens": state["tokens"] + count,
            "usage": [
                *state["usage"],
                {"stage": "refer", "tokens": count, "estimated": approximate},
            ],
        }

    def _classify(self, state):
        if is_greeting(state["question"]):
            return {"intent": "闲聊"}
        messages = self.context.history_messages(
            {**state, "question": state["resolved_question"]}, INTENT_PROMPT
        )
        self.context.check(messages)
        self.context.log.context(
            "classify_ctx",
            messages,
            self.context.budget,
            conversation_id=state["conversation_id"],
        )
        try:
            allowance, estimated_input = output_allowance(
                messages,
                state["tokens"],
                self.limits,
                token_counter=self.context.budget.count,
            )
        except BudgetExceeded:
            return {"intent": None, "stop_reason": "token_budget"}
        response = (
            self.model_factory()
            .bind(max_tokens=allowance, response_format={"type": "json_object"})
            .invoke(messages)
        )
        count, estimated = measured_usage(
            response, estimated_input, self.context.budget.count(response.content)
        )
        update = {
            "tokens": state["tokens"] + count,
            "usage": [
                *state["usage"],
                {"stage": "classify", "tokens": count, "estimated": estimated},
            ],
        }
        try:
            update["intent"] = parse_intent(response.content)
        except (ValueError, TypeError):
            update.update(intent=None, stop_reason="invalid_intent")
        return update

    def _route(self, state):
        return {"route": ROUTES.get(state.get("intent"))}

    def _retrieve(self, state):
        evidence, trace = self.retriever.retrieve(
            state["resolved_question"], state.get("category")
        )
        selected = []
        evidence_budget = (
            self.context.budget.top_k * self.context.budget.evidence_per_item
        )
        for item in evidence[: self.context.budget.top_k]:
            if self.context.budget.count([*selected, item]) <= evidence_budget:
                selected.append(item)
        return {
            "evidence": selected,
            "retrieval_trace": {
                **trace,
                "context_evidence_tokens": self.context.budget.count(selected),
            },
        }

    def _gate(self, state):
        evidence = [
            e
            for e in state["evidence"]
            if isinstance(e.get("answer"), str)
            and e["answer"].strip()
            and isinstance(e.get("score"), (int, float))
            and math.isfinite(e["score"])
        ]
        score = max((e["score"] for e in evidence), default=None)
        passed = score is not None and score >= self.min_score
        if not passed:
            self.ledger.add_low_confidence(
                state["conversation_id"],
                state["question"],
                "retrieval_low_conf",
                f"strategy={state['retrieval_trace'].get('strategy', self.strategy)}; score={score}; threshold={self.min_score}",
            )
        return {
            "evidence": evidence,
            "gate_passed": passed,
            "gate_score": score,
            **({} if passed else {"stop_reason": "retrieval_low_conf"}),
        }

    def _fixed(self, state):
        if state["route"] == "complaint":
            answer, actions, reason = (
                COMPLAINT_TEXT,
                ["handoff", "create_ticket"],
                "complaint",
            )
        elif state["route"] == "chitchat":
            answer, actions, reason = CHATTER_TEXT, [], "chitchat"
        elif state["route"] == "knowledge":
            answer, actions, reason = REFUSAL, [], state["stop_reason"]
        else:
            answer, actions, reason = INVALID_INTENT_TEXT, [], state["stop_reason"]
        get_stream_writer()({"event": "token", "data": {"content": answer}})
        return {"answer": answer, "actions": actions, "stop_reason": reason}

    def _log(self, state):
        mid = self.store.save_answer(
            state["conversation_id"],
            state["answer"],
            state["evidence"],
            state["actions"],
            state["question"],
        )
        state = {**state, "message_id": mid}
        self.logger.write(state)
        if state["evidence"]:
            get_stream_writer()(
                {
                    "event": "citations",
                    "data": {"items": state["evidence"], "message_id": mid},
                }
            )
        if state["actions"]:
            labels = {"handoff": "转人工", "create_ticket": "建工单"}
            get_stream_writer()(
                {
                    "event": "actions",
                    "data": {
                        "message_id": mid,
                        "items": [
                            {"type": a, "label": labels[a]} for a in state["actions"]
                        ],
                    },
                }
            )
        return {
            "message_id": mid,
            "messages": [AIMessage(content=state["answer"], id=f"sql-{mid}")],
        }

    def stream_events(self, request):
        config = {
            "configurable": {"thread_id": request.conversation_id},
            "recursion_limit": 40,
        }
        observed_path = []
        owner_checked = False
        initial = {
            "run_id": str(uuid4()),
            "conversation_id": request.conversation_id,
            "user_id": request.user_id,
            "question": request.message,
            "resolved_question": request.message,
            "category": request.category,
            "intent": None,
            "route": None,
            "path": [],
            "evidence": [],
            "retrieval_trace": {},
            "gate_passed": False,
            "gate_score": None,
            "decisions": 0,
            "tool_calls": 0,
            "tokens": 0,
            "usage": [],
            "seen_calls": [],
            "tool_trace": [],
            "pending": None,
            "agent_next": "",
            "missing": "",
            "actions": [],
            "answer": "",
            "stop_reason": "",
            "message_id": None,
        }
        with self.locks.hold(request.conversation_id):
            try:
                self.store.check_owner(request.conversation_id, request.user_id)
                owner_checked = True
                previous = self.graph.get_state(config).values
                if (
                    self.context.budget.count(request.message)
                    > self.context.budget.user_input
                ):
                    yield {
                        "event": "error",
                        "data": {
                            "message": "当前输入超过配置的上下文预算，请缩短后重试。"
                        },
                    }
                    return
                history, current_id = self.store.prepare(request)
                complete_history = previous.get("messages", history)
                initial.update(
                    self.context.prepare(
                        request.conversation_id, complete_history, current_id
                    )
                )
                history_messages = self.context.history_messages(initial, INTENT_PROMPT)
                self.context.log.context(
                    "history_ctx",
                    history_messages,
                    self.context.budget,
                    conversation_id=request.conversation_id,
                    summary=initial["context_summary"],
                    omitted_summaries=initial["context_omitted_summaries"],
                )
                initial["messages"] = [
                    *([] if previous else history),
                    HumanMessage(content=request.message, id=f"sql-{current_id}"),
                ]
                for event in self.graph.stream(
                    initial, config=config, stream_mode="custom"
                ):
                    if event.get("event") == "workflow_status":
                        observed_path.append(event["data"]["node"])
                    yield event
                final = self.graph.get_state(config).values
                yield {
                    "event": "done",
                    "data": {
                        "conversation_id": request.conversation_id,
                        "message_id": final["message_id"],
                        "route": final["route"],
                        "stop_reason": final["stop_reason"],
                    },
                }
            except Exception as error:  # noqa: BLE001 - isolate external provider/tool failures
                try:
                    latest = (
                        self.graph.get_state(config).values if owner_checked else {}
                    )
                    if latest.get("run_id") != initial["run_id"]:
                        latest = {}
                    self.logger.write(
                        {
                            **initial,
                            **latest,
                            **getattr(error, "usage_update", {}),
                            "path": observed_path,
                            "stop_reason": "service_error",
                        },
                        status="error",
                        error_type=type(error).__name__,
                    )
                except Exception as log_error:  # noqa: BLE001 - preserve original workflow failure
                    logging.getLogger(__name__).error(
                        "Workflow log failed (%s)", type(log_error).__name__
                    )
                yield {
                    "event": "error",
                    "data": {"message": "暂时无法处理，请稍后再试。"},
                }

    def close(self):
        self.context.close()
        conn = getattr(self.checkpointer, "conn", None)
        if conn:
            conn.close()
        for closer in self._closers:
            closer()
        self._closers.clear()
