"""Fixed routing and evidence gate around a checkpointed agent subflow."""

import logging
import math
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.services.quality.generation import REFUSAL
from app.services.quality.ledger import QualityLedger

from .actions import TicketActions
from .agent import AgentNodes
from .intents import (
    CHATTER_TEXT,
    COMPLAINT_TEXT,
    INTENT_PROMPT,
    INVALID_INTENT_TEXT,
    ROUTES,
    is_greeting,
    parse_intent,
)
from .logging import WorkflowLog
from .policy import (
    DEFAULT_LIMITS,
    BudgetExceeded,
    estimate_tokens,
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
            "refer": lambda s: {"resolved_question": s["question"]},
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

    def _classify(self, state):
        if is_greeting(state["question"]):
            return {"intent": "闲聊"}
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=state["resolved_question"]),
        ]
        try:
            allowance, estimated_input = output_allowance(
                messages, state["tokens"], self.limits
            )
        except BudgetExceeded:
            return {"intent": None, "stop_reason": "token_budget"}
        response = self.model_factory().bind(max_tokens=allowance).invoke(messages)
        count, estimated = measured_usage(
            response, estimated_input, estimate_tokens(response.content)
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
        return {"evidence": evidence, "retrieval_trace": trace}

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
                history = self.store.prepare(request)
                initial["messages"] = [
                    *([] if previous else history),
                    HumanMessage(content=request.message),
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
        conn = getattr(self.checkpointer, "conn", None)
        if conn:
            conn.close()
        for closer in self._closers:
            closer()
        self._closers.clear()
