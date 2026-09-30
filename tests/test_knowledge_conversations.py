from sqlalchemy import func, select

from app.db.models import Conversation, KnowledgeChunk, KnowledgeExtractionCursor, KnowledgeQAStaging, Message
from app.services.knowledge.conversations import (
    ConversationKnowledgeExtractor,
    ExtractedCandidate,
    ExtractionBatch,
    LangChainConversationModel,
)
from app.services.knowledge.privacy import redact_customer_data
from app.services.knowledge.repository import ConversationTurn, KnowledgeRepository


def _message(session, conversation_id, role, content, *, tool_calls=None, tool_call_id=None):
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(message)
    session.flush()
    return message


def _conversation(session, conversation_id):
    session.add(Conversation(conversation_id=conversation_id, user_id="customer", status="closed"))
    session.flush()


def test_redaction_masks_mobile_email_identity_order_and_ticket_ids():
    text = "电话13800138000 或 +86 13900139000 或 +8613700137000 邮箱user@example.com 身份证11010519491231002X 订单号：ORD-2026-9988 工单号T1001"

    redacted = redact_customer_data(text)

    assert "13800138000" not in redacted
    assert "13900139000" not in redacted
    assert "13700137000" not in redacted
    assert "user@example.com" not in redacted
    assert "11010519491231002X" not in redacted
    assert "ORD-2026-9988" not in redacted
    assert "T1001" not in redacted
    assert "[PHONE]" in redacted and "[EMAIL]" in redacted and "[ID_CARD]" in redacted


def test_repository_pairs_user_with_final_assistant_after_tool_cycle(db_session_factory):
    with db_session_factory.begin() as session:
        _conversation(session, "tool-turn")
        user = _message(session, "tool-turn", "user", "邮费是多少？")
        _message(session, "tool-turn", "assistant", "", tool_calls=[{"name": "query_order"}])
        _message(session, "tool-turn", "tool", "内部订单号ORD-4444", tool_call_id="call-1")
        answer = _message(session, "tool-turn", "assistant", "费用以结算页显示为准。")

    turns, cursor = KnowledgeRepository(db_session_factory).load_turns_after(0, 20)

    assert turns == [ConversationTurn(user.id, answer.id, "邮费是多少？", "费用以结算页显示为准。")]
    assert cursor == answer.id


def test_consecutive_users_are_not_mispaired_and_empty_answers_are_skipped(db_session_factory):
    with db_session_factory.begin() as session:
        _conversation(session, "consecutive")
        _message(session, "consecutive", "user", "旧问题")
        current_user = _message(session, "consecutive", "user", "当前问题")
        answer = _message(session, "consecutive", "assistant", "这是当前问题的回答。")

    turns, cursor = KnowledgeRepository(db_session_factory).load_turns_after(0, 20)

    assert turns == [ConversationTurn(current_user.id, answer.id, "当前问题", "这是当前问题的回答。")]
    assert cursor == answer.id


def test_empty_assistant_answers_are_skipped(db_session_factory):
    with db_session_factory.begin() as session:
        _conversation(session, "empty-answer")
        _message(session, "empty-answer", "user", "问题")
        _message(session, "empty-answer", "assistant", "   ")
        current_user = _message(session, "empty-answer", "user", "后续问题")
        answer = _message(session, "empty-answer", "assistant", "后续问题的回答。")

    turns, cursor = KnowledgeRepository(db_session_factory).load_turns_after(0, 20)

    assert turns == [ConversationTurn(current_user.id, answer.id, "后续问题", "后续问题的回答。")]
    assert cursor == answer.id


class FakeExtractionModel:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates or []
        self.error = error
        self.last_input = []
        self.calls = 0

    def extract_pairs(self, turns):
        self.calls += 1
        self.last_input = [turn.user_text + " " + turn.assistant_text for turn in turns]
        if self.error:
            raise self.error
        return self.candidates


def _messages_with_pii(db_session_factory):
    with db_session_factory.begin() as session:
        _conversation(session, "extract-pii")
        user = _message(session, "extract-pii", "user", "电话13800138000，邮箱user@example.com，邮费是多少？")
        _message(session, "extract-pii", "assistant", "", tool_calls=[{"name": "query_order"}])
        _message(session, "extract-pii", "tool", "内部订单ORD-7777", tool_call_id="call-1")
        answer = _message(
            session,
            "extract-pii",
            "assistant",
            "费用以结算页为准。请勿联系user@example.com，订单ORD-8899。",
        )
    return user, answer


def test_extractor_sends_redacted_turns_and_keeps_source_messages_unchanged(db_session_factory):
    user, answer = _messages_with_pii(db_session_factory)
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=0,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。订单ORD-5555。",
    )])
    repository = KnowledgeRepository(db_session_factory)
    summary = ConversationKnowledgeExtractor(model, repository).run(batch_size=20)

    assert summary.staged_pairs == 1 and summary.inserted == 1
    assert summary.skipped_tool_messages == 2
    assert all(raw not in " ".join(model.last_input) for raw in ("13800138000", "user@example.com", "ORD-8899"))
    with db_session_factory() as session:
        assert session.get(Message, user.id).content.startswith("电话13800138000")
        assert "user@example.com" in session.get(Message, answer.id).content
        staged = list(session.scalars(select(KnowledgeQAStaging)))
        assert len(staged) == 1 and staged[0].promoted_at is not None
        assert all(raw not in staged[0].answer for raw in ("user@example.com", "ORD-5555"))
        cursor = session.get(KnowledgeExtractionCursor, "conversation_knowledge")
        assert cursor.last_message_id == answer.id


def test_model_failure_does_not_advance_checkpoint_or_create_staging(db_session_factory):
    _messages_with_pii(db_session_factory)
    repository = KnowledgeRepository(db_session_factory)
    model = FakeExtractionModel(error=RuntimeError("provider failed"))

    try:
        ConversationKnowledgeExtractor(model, repository).run(batch_size=20)
    except RuntimeError:
        pass
    else:
        assert False, "provider failure should abort this extraction batch"

    with db_session_factory() as session:
        assert session.get(KnowledgeExtractionCursor, "conversation_knowledge") is None
        assert session.scalar(select(func.count()).select_from(KnowledgeQAStaging)) == 0


def test_exact_duplicate_across_cron_batches_promotes_only_once(db_session_factory):
    with db_session_factory.begin() as session:
        _conversation(session, "batch-one")
        _message(session, "batch-one", "user", "邮费是多少？")
        _message(session, "batch-one", "assistant", "费用以结算页为准。")
        _conversation(session, "batch-two")
        _message(session, "batch-two", "user", "运费多少钱？")
        _message(session, "batch-two", "assistant", "费用以结算页为准。")
    repository = KnowledgeRepository(db_session_factory)
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=0,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。",
    )])
    extractor = ConversationKnowledgeExtractor(model, repository)

    first = extractor.run(batch_size=1)
    second = extractor.run(batch_size=1)

    assert first.inserted == 1
    assert second.deduped == 1 and second.inserted == 0
    with db_session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.content_type == "conversation_qa")
        ) == 1


def test_staged_rows_are_promoted_on_retry_after_promotion_failure(db_session_factory, monkeypatch):
    user, answer = _messages_with_pii(db_session_factory)
    repository = KnowledgeRepository(db_session_factory)
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=0,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。",
    )])
    extractor = ConversationKnowledgeExtractor(model, repository)
    original_promote = repository.promote_staged
    should_fail = True

    def fail_once():
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("interrupted before promotion")
        return original_promote()

    monkeypatch.setattr(repository, "promote_staged", fail_once)
    try:
        extractor.run(batch_size=20)
    except RuntimeError:
        pass
    else:
        assert False, "promotion failure should be visible to the scheduler"
    recovered = extractor.run(batch_size=20)

    assert recovered.inserted == 1
    with db_session_factory() as session:
        assert session.get(KnowledgeExtractionCursor, "conversation_knowledge").last_message_id == answer.id
        assert session.scalar(
            select(func.count()).select_from(KnowledgeQAStaging).where(KnowledgeQAStaging.promoted_at.is_(None))
        ) == 0


def test_empty_extraction_still_advances_checkpoint_for_completed_turn(db_session_factory):
    user, answer = _messages_with_pii(db_session_factory)
    repository = KnowledgeRepository(db_session_factory)
    model = FakeExtractionModel([])

    summary = ConversationKnowledgeExtractor(model, repository).run(batch_size=20)
    repeated = ConversationKnowledgeExtractor(model, repository).run(batch_size=20)

    assert summary.staged_pairs == 0 and summary.last_message_id == answer.id
    assert repeated.messages_read == 0 and model.calls == 1
    assert user.id < answer.id


def test_invalid_turn_index_fails_without_advancing_checkpoint(db_session_factory):
    _messages_with_pii(db_session_factory)
    repository = KnowledgeRepository(db_session_factory)
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=3,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。",
    )])

    try:
        ConversationKnowledgeExtractor(model, repository).run(batch_size=20)
    except ValueError:
        pass
    else:
        assert False, "out-of-range source indexes must be rejected"

    assert repository.extraction_cursor() == 0


def test_messages_read_counts_rows_not_message_id_gaps(db_session_factory):
    _messages_with_pii(db_session_factory)
    with db_session_factory.begin() as session:
        tool_message = session.scalar(select(Message).where(Message.role == "tool"))
        session.delete(tool_message)
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=0,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。",
    )])

    summary = ConversationKnowledgeExtractor(model, KnowledgeRepository(db_session_factory)).run(batch_size=20)

    assert summary.messages_read == 3


def test_candidate_is_deduplicated_against_existing_knowledge(db_session_factory):
    repository = KnowledgeRepository(db_session_factory)
    from app.services.knowledge.content import ChunkDraft

    repository.upsert_drafts(
        [
            ChunkDraft(
                source_key="faq:already-known",
                category="配送",
                questions=["邮费是多少？"],
                answer="费用以结算页为准。",
                chapter_path=["配送"],
                content_type="product_faq",
                is_critical=False,
            )
        ]
    )
    with db_session_factory.begin() as session:
        _conversation(session, "already-known")
        _message(session, "already-known", "user", "郵費是多少？")
        _message(session, "already-known", "assistant", "费用以结算页为准。")
    model = FakeExtractionModel([ExtractedCandidate(
        source_turn_index=0,
        category="配送",
        questions=["邮费是多少？"],
        answer="费用以结算页为准。",
    )])

    summary = ConversationKnowledgeExtractor(model, repository).run(batch_size=10)

    assert summary.deduped == 1 and summary.inserted == 0


def test_langchain_adapter_uses_documented_structured_output_and_source_indexes(monkeypatch):
    import json
    import langchain_openai

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def with_structured_output(self, schema, *, method):
            self.schema = schema
            self.method = method
            return self

        def invoke(self, messages):
            self.messages = messages
            return ExtractionBatch(
                candidates=[
                    ExtractedCandidate(
                        source_turn_index=0,
                        category="配送",
                        questions=["邮费是多少？"],
                        answer="费用以结算页为准。",
                    )
                ]
            )

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    model = LangChainConversationModel("chat-model", "test-key", "https://chat.example/v1")
    turns = [ConversationTurn(10, 11, "邮费是多少？", "费用以结算页为准。")]

    candidates = model.extract_pairs(turns)

    assert model._structured_model.kwargs == {
        "model": "chat-model",
        "api_key": "test-key",
        "base_url": "https://chat.example/v1",
        "temperature": 0,
    }
    assert model._structured_model.schema is ExtractionBatch
    assert model._structured_model.method == "function_calling"
    assert candidates[0].source_turn_index == 0
    assert json.loads(model._structured_model.messages[1][1])[0]["source_turn_index"] == 0
