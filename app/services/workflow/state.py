"""One persistent State spanning the fixed workflow and agent loop."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_messages: list[BaseMessage]
    conversation_id: str
    user_id: str
    run_id: str
    question: str
    resolved_question: str
    category: str | None
    intent: str | None
    intent_confidence: float | None
    reference_resolved: bool
    route: str | None
    path: list[str]
    evidence: list[dict]
    retrieval_trace: dict
    gate_passed: bool
    gate_score: float | None
    decisions: int
    tool_calls: int
    tokens: int
    usage: list[dict]
    seen_calls: list[str]
    tool_trace: list[dict]
    pending: dict | None
    agent_next: str
    missing: str
    actions: list[str]
    answer: str
    stop_reason: str
    message_id: int | None
    order: dict | None
    previous_order: dict | None
    order_choices: list[dict]
    selection_request_id: str | None
    selection_message_id: int | None
    policy_queries: list[str]
