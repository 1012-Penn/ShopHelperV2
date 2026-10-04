"""Fixed routing and evidence gate around a checkpointed agent subflow."""

import logging
import math
import re
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from app.db.models import Conversation, Message
from app.services.context.manager import ContextManager
from app.services.quality.generation import REFUSAL
from app.services.quality.ledger import QualityLedger

from .actions import TicketActions
from .agent import AgentNodes
from .demo_orders import get_demo_order, list_demo_orders
from .intents import (
    CHATTER_TEXT,
    COMPLAINT_TEXT,
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_PROMPT,
    INVALID_INTENT_TEXT,
    REFER_PROMPT,
    ROUTES,
    parse_intent,
    parse_reference_resolution,
)
from .logging import WorkflowLog
from .policy import (
    DEFAULT_LIMITS,
    BudgetExceeded,
    measured_usage,
    output_allowance,
)
from .query_expansion import expand_policy_queries
from .retrieval import EvidenceAdapter, retrieve_policy_queries
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
            "order_check": self._order_check,
            "offer_orders": self._offer_orders,
            "await_order": self._await_order,
            "expand_policy": self._expand_policy,
            "retrieve_policy": self._retrieve_policy,
            "policy_gate": self._policy_gate,
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
                "high_risk": "order_check",
                "business": "agent",
                "complaint": "fixed",
                "chitchat": "fixed",
                "other": "fixed",
                "error": "fixed",
            },
        )
        builder.add_conditional_edges(
            "order_check",
            lambda s: "offer" if s.get("order") is None else "expand",
            {"offer": "offer_orders", "expand": "expand_policy"},
        )
        builder.add_edge("offer_orders", "await_order")
        builder.add_edge("await_order", "expand_policy")
        builder.add_edge("expand_policy", "retrieve_policy")
        builder.add_edge("retrieve_policy", "policy_gate")
        builder.add_conditional_edges(
            "policy_gate", lambda s: s["gate_passed"], {True: "agent", False: "fixed"}
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
            intent, confidence = parse_intent(response.content)
            update["intent_confidence"] = confidence
            if confidence < INTENT_CONFIDENCE_THRESHOLD:
                intent = "其他"
                update["stop_reason"] = "low_intent_confidence"
            elif intent == "其他":
                update["stop_reason"] = "other_intent"
            update["intent"] = intent
        except (ValueError, TypeError):
            update.update(intent=None, stop_reason="invalid_intent")
        return update

    def _refer(self, state):
        question = state["question"]
        history = list(state.get("context_history", []))
        if state.get("previous_order"):
            selected = state["previous_order"]
            history.append(
                AIMessage(
                    content=(
                        "本会话中用户已通过订单选择器选定订单 "
                        f"{selected['order_id']}（{selected['product']}）。"
                    )
                )
            )
        messages = self.context.history_messages(
            {**state, "context_history": history}, REFER_PROMPT
        )
        # Preserve the formal resolver's current-question delimiter without
        # passing complete checkpoint history to the model.
        if state.get("context_summary"):
            messages[-1] = HumanMessage(
                content=messages[-1].content + "\n本轮原问题：\n" + question
            )
        else:
            messages[-1] = HumanMessage(content="本轮原问题：\n" + question)
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
            resolved, reference_resolved = parse_reference_resolution(
                response.content, question
            )
        except (ValueError, TypeError):
            resolved, reference_resolved = question, False
        return {
            "resolved_question": resolved,
            "reference_resolved": reference_resolved,
            "tokens": state["tokens"] + count,
            "usage": [
                *state["usage"],
                {"stage": "refer", "tokens": count, "estimated": approximate},
            ],
        }

    def _route(self, state):
        return {"route": ROUTES.get(state.get("intent"))}

    def _order_check(self, state):
        current_ids = {
            match.upper()
            for match in re.findall(
                r"DEMO-\d{4}", state["question"], flags=re.IGNORECASE
            )
        }
        historical_ids = {
            match.upper()
            for message in state.get("messages", [])
            if isinstance(message, HumanMessage)
            for match in re.findall(r"DEMO-\d{4}", message.content, flags=re.IGNORECASE)
        }
        # An ID written in this turn is authoritative. Otherwise allow history
        # only when there is exactly one candidate; never let the resolver pick
        # between several previously mentioned orders by itself.
        order_id = next(iter(current_ids)) if len(current_ids) == 1 else None
        resolver_conflict = False
        if order_id is not None:
            resolved_ids = {
                match.upper()
                for match in re.findall(
                    r"DEMO-\d{4}", state["resolved_question"], flags=re.IGNORECASE
                )
            }
            resolver_conflict = bool(resolved_ids and resolved_ids != {order_id})
        if not current_ids and state.get("reference_resolved"):
            if state.get("previous_order"):
                historical_ids.add(state["previous_order"]["order_id"])
            if len(historical_ids) == 1:
                candidate = next(iter(historical_ids))
                resolved_ids = {
                    match.upper()
                    for match in re.findall(
                        r"DEMO-\d{4}", state["resolved_question"], flags=re.IGNORECASE
                    )
                }
                if not resolved_ids or resolved_ids == {candidate}:
                    order_id = candidate
        order = (
            get_demo_order(order_id, state["user_id"]) if order_id is not None else None
        )
        result = {"order": order}
        if resolver_conflict:
            # Keep the user's explicit order authoritative in every downstream
            # prompt/query too; never mix it with a resolver-invented order.
            result.update(
                resolved_question=state["question"],
                reference_resolved=False,
            )
        return result

    def _offer_orders(self, state):
        choices = list_demo_orders(state["user_id"])
        request_id = str(uuid4())
        content = "请先选择要查询政策的订单（演示数据）："
        message_id = self.store.save_answer(
            state["conversation_id"],
            content,
            [],
            ["order_choices"],
            state["question"],
            offer={
                "order_choice": {
                    "request_id": request_id,
                    "order_ids": [order["order_id"] for order in choices],
                }
            },
        )
        get_stream_writer()(
            {
                "event": "order_choices",
                "data": {
                    "conversation_id": state["conversation_id"],
                    "message_id": message_id,
                    "request_id": request_id,
                    "content": content,
                    "choices": choices,
                },
            }
        )
        return {
            "order_choices": choices,
            "selection_request_id": request_id,
            "selection_message_id": message_id,
        }

    def _await_order(self, state):
        # Keep interrupt first: LangGraph replays this node on resume.
        resumed = interrupt(
            {
                "request_id": state["selection_request_id"],
                "message_id": state["selection_message_id"],
                "order_ids": [item["order_id"] for item in state["order_choices"]],
            }
        )
        order_id = resumed.get("order_id") if isinstance(resumed, dict) else resumed
        order = get_demo_order(order_id, state["user_id"])
        if order is None or order_id not in {
            item["order_id"] for item in state["order_choices"]
        }:
            raise ValueError("invalid demo order selection")
        update = {"order": order}
        if not state.get("current_message_id"):
            # Upgrade an existing chapter-six paused checkpoint in place. Do
            # not create a new user row when its order selector is resumed.
            with self.session_factory() as session:
                current = session.scalar(
                    select(Message)
                    .where(
                        Message.conversation_id == state["conversation_id"],
                        Message.role == "user",
                        Message.id < state["selection_message_id"],
                    )
                    .order_by(Message.id.desc())
                    .limit(1)
                )
                if current is None:
                    raise ValueError("pending workflow has no original user message")
                current_id = current.id
            complete = list(state.get("messages", []))
            current_index = next(
                (
                    i
                    for i in range(len(complete) - 1, -1, -1)
                    if isinstance(complete[i], HumanMessage)
                ),
                len(complete),
            )
            update.update(
                self.context.prepare(
                    state["conversation_id"], complete[:current_index], current_id
                )
            )
            self.context.log.context(
                "history_ctx",
                self.context.history_messages({**state, **update}, INTENT_PROMPT),
                self.context.budget,
                conversation_id=state["conversation_id"],
                summary=update["context_summary"],
                omitted_summaries=update["context_omitted_summaries"],
            )
        return update

    def _expand_policy(self, state):
        try:
            queries = expand_policy_queries(
                self.model_factory(),
                state["resolved_question"],
                state["order"],
                state["intent"],
            )
        except Exception as error:  # noqa: BLE001 - preserve canonical-query fallback
            logging.getLogger(__name__).warning(
                "Policy query expansion failed (%s)", type(error).__name__
            )
            queries = [state["resolved_question"]]
        return {"policy_queries": queries}

    def _retrieve_policy(self, state):
        adapter = EvidenceAdapter(self.retriever, self.strategy)
        evidence, trace = retrieve_policy_queries(
            adapter, state["policy_queries"], state["intent"]
        )
        return {"evidence": evidence, "retrieval_trace": trace}

    def _policy_gate(self, state):
        result = self._gate(state)
        if not result["gate_passed"]:
            result["stop_reason"] = "policy_missing"
        return result

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
        elif state["route"] in {"knowledge", "high_risk"}:
            answer, actions, reason = REFUSAL, [], state["stop_reason"]
        else:
            answer, actions, reason = INVALID_INTENT_TEXT, [], state["stop_reason"]
        get_stream_writer()({"event": "token", "data": {"content": answer}})
        return {"answer": answer, "actions": actions, "stop_reason": reason}

    def _log(self, state):
        offer = None
        if (
            "refund_form" in state["actions"]
            and state.get("route") == "high_risk"
            and state.get("intent") == "退款退货"
            and state.get("order")
        ):
            request_id = str(uuid4())
            request_type = (
                "退货"
                if re.search(r"退货|退回", state["resolved_question"])
                else "退款"
            )
            offer = {
                "refund_form": {
                    "request_id": request_id,
                    "order_id": state["order"]["order_id"],
                    "request_type": request_type,
                }
            }
        mid = self.store.save_answer(
            state["conversation_id"],
            state["answer"],
            state["evidence"],
            state["actions"],
            state["question"],
            offer=offer,
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
            labels = {
                "handoff": "转人工",
                "create_ticket": "建工单",
                "refund_form": "提交演示申请",
            }
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
        if offer:
            get_stream_writer()(
                {
                    "event": "refund_form",
                    "data": {"message_id": mid, **offer["refund_form"]},
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
            "order": None,
            "order_choices": [],
            "selection_request_id": None,
            "selection_message_id": None,
            "policy_queries": [],
            "agent_messages": [],
        }
        with self.locks.hold(request.conversation_id):
            try:
                self.store.check_owner(request.conversation_id, request.user_id)
                owner_checked = True
                snapshot = self.graph.get_state(config)
                if request.selected_order_id is not None:
                    values = snapshot.values
                    if (
                        "await_order" not in snapshot.next
                        or values.get("selection_message_id")
                        != request.selection_message_id
                        or values.get("selection_request_id") != request.request_id
                    ):
                        raise ValueError("order selection is stale or already used")
                    with self.session_factory() as session:
                        offered = session.get(Message, request.selection_message_id)
                        metadata = offered.actions or {} if offered else {}
                        order_offer = metadata.get("order_choice") or {}
                        conversation = session.get(
                            Conversation, request.conversation_id
                        )
                        if (
                            conversation is None
                            or conversation.user_id != request.user_id
                            or offered is None
                            or offered.role != "assistant"
                            or offered.conversation_id != request.conversation_id
                            or "order_choices" not in (metadata.get("items") or [])
                            or order_offer.get("request_id") != request.request_id
                            or request.selected_order_id
                            not in order_offer.get("order_ids", [])
                        ):
                            raise ValueError("invalid or foreign order selection")
                    if (
                        get_demo_order(request.selected_order_id, request.user_id)
                        is None
                    ):
                        raise ValueError("unknown demo order selection")
                    graph_input = Command(
                        resume={"order_id": request.selected_order_id}
                    )
                else:
                    if "await_order" in snapshot.next:
                        raise ValueError(
                            "conversation has a pending workflow selection"
                        )
                    initial["previous_order"] = snapshot.values.get("order")
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
                    initial.update(
                        self.context.prepare(
                            request.conversation_id,
                            snapshot.values.get("messages", history),
                            current_id,
                        )
                    )
                    history_messages = self.context.history_messages(
                        initial, INTENT_PROMPT
                    )
                    self.context.log.context(
                        "history_ctx",
                        history_messages,
                        self.context.budget,
                        conversation_id=request.conversation_id,
                        summary=initial["context_summary"],
                        omitted_summaries=initial["context_omitted_summaries"],
                    )
                    initial["messages"] = [
                        *([] if snapshot.values else history),
                        HumanMessage(content=request.message, id=f"sql-{current_id}"),
                    ]
                    initial["agent_messages"] = [HumanMessage(content=request.message)]
                    graph_input = initial
                for event in self.graph.stream(
                    graph_input, config=config, stream_mode="custom"
                ):
                    if event.get("event") == "workflow_status":
                        observed_path.append(event["data"]["node"])
                    yield event
                final = self.graph.get_state(config).values
                if self.graph.get_state(config).next:
                    return
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
