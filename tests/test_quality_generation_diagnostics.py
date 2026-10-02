"""Regression tests for per-call generation protocol failure snapshots."""
import json
from datetime import datetime
import importlib
import importlib.util

from langchain_core.messages import AIMessage
from concurrent.futures import ThreadPoolExecutor
import logging

from app.services.quality.generation import (
    QUALITY_PROMPT,
    GenerationResult,
    KnowledgeAnswerService,
    StructuredGenerator,
)
from langchain_core.exceptions import OutputParserException
from app.services.quality.retrieval import Evidence


def failure_sink(path):
    module_name = 'app.services.quality.generation_diagnostics'
    assert importlib.util.find_spec(module_name) is not None, (
        'protocol failure diagnostics sink has not been implemented'
    )
    module = importlib.import_module(module_name)
    return module.JsonlGenerationFailureSink(path)


def evidence(n):
    return Evidence(
        n=n,
        chunk_id=n,
        section_path=['售后', '退款'],
        question=f'问题{n}',
        answer=f'退款以银行处理为准，证据编号{n}。',
        source_key=f'doc:refund:{n}',
        score=.9,
    )


def test_citation_protocol_failure_writes_raw_prompt_all_arranged_evidence_and_predicate(tmp_path):
    provider_raw = '{ "sufficient": true, "answer": "以银行为准 [99]", "tail": "原样保留" }\n'
    evidence_rows = [evidence(n) for n in range(1, 7)]

    class StructuredModel:
        def __init__(self):
            self.schema = None
            self.options = None
            self.messages = None

        def with_structured_output(self, schema, **options):
            self.schema, self.options = schema, options
            return self

        def invoke(self, messages):
            self.messages = messages
            return {
                'raw': AIMessage(content=provider_raw),
                'parsed': GenerationResult(
                    sufficient=True,
                    reason='',
                    answer='以银行为准 [99]',
                    cited_numbers=[99],
                ),
                'parsing_error': None,
            }

    model = StructuredModel()
    diagnostic_path = tmp_path / 'protocol-failures.jsonl'
    sink = failure_sink(diagnostic_path)
    service = KnowledgeAnswerService(
        retriever=None,
        generator=StructuredGenerator(model),
        ledger=None,
        diagnostic_sink=sink,
        model_identity='test-provider/test-model',
    )

    result = service.generate('退款具体多久到账？', evidence_rows)

    assert result.refused
    assert model.schema is GenerationResult
    assert model.options == {'method': 'json_mode', 'include_raw': True}
    assert model.messages[0]['content'] == QUALITY_PROMPT
    sent_user_payload = json.loads(model.messages[1]['content'])
    assert sent_user_payload['question'] == '退款具体多久到账？'
    assert [row['n'] for row in sent_user_payload['evidence']] == [1, 3, 5, 6, 4, 2]
    lines = diagnostic_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert datetime.fromisoformat(event['occurred_at'])
    assert event['question'] == '退款具体多久到账？'
    assert event['model_identity'] == 'test-provider/test-model'
    assert event['system_prompt'] == QUALITY_PROMPT
    assert event['raw_text'] == provider_raw
    assert event['trace_available'] is True
    assert event['conversation_key'] is None
    assert event['parsed']['answer'] == '以银行为准 [99]'
    assert event['parse_error_type'] is None
    assert [row['n'] for row in event['evidence']] == [1, 3, 5, 6, 4, 2]
    assert [row['answer'] for row in event['evidence']] == [
        '退款以银行处理为准，证据编号1。',
        '退款以银行处理为准，证据编号3。',
        '退款以银行处理为准，证据编号5。',
        '退款以银行处理为准，证据编号6。',
        '退款以银行处理为准，证据编号4。',
        '退款以银行处理为准，证据编号2。',
    ]
    assert event['inline_numbers'] == [99]
    assert event['declared_numbers'] == [99]
    assert event['allowed_numbers'] == [1, 2, 3, 4, 5, 6]
    assert event['failed_predicates'] == ['inline_citation_not_allowed']


def test_self_insufficient_and_successful_generation_do_not_write_protocol_diagnostics(tmp_path):
    evidence_rows = [evidence(1)]
    diagnostic_path = tmp_path / 'protocol-failures.jsonl'
    sink = failure_sink(diagnostic_path)
    service = KnowledgeAnswerService(
        retriever=None,
        generator=lambda question, rows: {
            'sufficient': False,
            'reason': '需要订单当前状态',
            'answer': '',
            'cited_numbers': [],
        },
        ledger=None,
        diagnostic_sink=sink,
    )

    insufficient = service.generate('订单何时送达？', evidence_rows)
    assert insufficient.refused and insufficient.source == 'self_check'
    assert not diagnostic_path.exists()

    service.generator = lambda question, rows: {
        'sufficient': True,
        'reason': '',
        'answer': '请以物流页面为准[1]',
        'cited_numbers': [1],
    }
    successful = service.generate('物流进度如何？', evidence_rows)
    assert not successful.refused
    assert not diagnostic_path.exists()


def test_retrieval_low_confidence_does_not_write_protocol_diagnostic(tmp_path):
    diagnostic_path = tmp_path / 'protocol-failures.jsonl'
    service = KnowledgeAnswerService(
        retriever=None,
        generator=None,
        ledger=None,
        min_score=.95,
        diagnostic_sink=failure_sink(diagnostic_path),
    )

    result = service.generate('退款多久到账？', [evidence(1)])

    assert result.refused and result.source == 'retrieval_low_conf'
    assert not diagnostic_path.exists()


def test_callable_without_provider_trace_records_null_raw_and_unavailable_trace(tmp_path):
    diagnostic_path = tmp_path / 'protocol-failures.jsonl'

    class Retriever:
        def retrieve(self, question, category=None):
            return [evidence(1)]

    class Ledger:
        def add_low_confidence(self, conversation_key, question, source, reason):
            assert conversation_key == 'conversation-81'

    service = KnowledgeAnswerService(
        retriever=Retriever(),
        generator=lambda question, rows: {
            'sufficient': True,
            'reason': '',
            'answer': '退款很快到账[88]',
            'cited_numbers': [88],
        },
        ledger=Ledger(),
        diagnostic_sink=failure_sink(diagnostic_path),
    )

    result = service.answer('退款何时到账？', 'conversation-81')

    assert result.refused
    event = json.loads(diagnostic_path.read_text(encoding='utf-8').splitlines()[0])
    assert event['raw_text'] is None
    assert event['trace_available'] is False
    assert event['conversation_key'] == 'conversation-81'
    assert event['parsed']['answer'] == '退款很快到账[88]'
    assert event['failed_predicates'] == ['inline_citation_not_allowed']


def test_jsonl_sink_keeps_concurrent_events_as_independent_lines(tmp_path):
    diagnostic_path = tmp_path / 'parallel.jsonl'
    sink = failure_sink(diagnostic_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda number: sink.write({'sequence': number}), range(80)))

    lines = diagnostic_path.read_text(encoding='utf-8').splitlines()
    events = [json.loads(line) for line in lines]
    assert len(events) == 80
    assert sorted(event['sequence'] for event in events) == list(range(80))


def test_diagnostic_write_failure_keeps_refusal_and_logs_only_safe_failure_signal(tmp_path, caplog):
    class FailingSink:
        def write(self, event):
            raise OSError('private filesystem detail / credential=secret')

    service = KnowledgeAnswerService(
        retriever=None,
        generator=lambda question, rows: {
            'sufficient': True,
            'reason': '',
            'answer': '答案[99]',
            'cited_numbers': [99],
        },
        ledger=None,
        diagnostic_sink=FailingSink(),
    )

    with caplog.at_level(logging.ERROR):
        result = service.generate('具体退款金额？', [evidence(1)])

    assert result.refused
    assert 'diagnostic write failed' in caplog.text.lower()
    assert 'credential=secret' not in caplog.text


def test_parse_failure_snapshot_preserves_raw_and_only_the_error_type(tmp_path):
    provider_raw = 'model emitted this exact malformed output\n'

    class StructuredModel:
        def with_structured_output(self, schema, **options):
            assert schema is GenerationResult
            assert options == {'method': 'json_mode', 'include_raw': True}
            return self

        def invoke(self, messages):
            return {
                'raw': AIMessage(content=provider_raw),
                'parsed': None,
                'parsing_error': OutputParserException('private parser detail credential=secret'),
            }

    diagnostic_path = tmp_path / 'parse-failures.jsonl'
    service = KnowledgeAnswerService(
        retriever=None,
        generator=StructuredGenerator(StructuredModel()),
        ledger=None,
        diagnostic_sink=failure_sink(diagnostic_path),
    )

    result = service.generate('未知退款状态', [evidence(1)])

    assert result.refused
    event = json.loads(diagnostic_path.read_text(encoding='utf-8').splitlines()[0])
    assert event['raw_text'] == provider_raw
    assert event['trace_available'] is True
    assert event['parsed'] is None
    assert event['parse_error_type'] == 'OutputParserException'
    assert event['failed_predicates'] == ['structured_output_parse_error']
    assert 'credential=secret' not in json.dumps(event, ensure_ascii=False)


def test_parse_failure_snapshot_preserves_raw_text_blocks(tmp_path):
    provider_raw_blocks = [
        {'type': 'text', 'text': '{"sufficient": true, '},
        {'type': 'text', 'text': '"answer": truncated'},
    ]

    class StructuredModel:
        def with_structured_output(self, schema, **options):
            return self

        def invoke(self, messages):
            return {
                'raw': AIMessage(content=provider_raw_blocks),
                'parsed': None,
                'parsing_error': OutputParserException('incomplete JSON'),
            }

    diagnostic_path = tmp_path / 'block-parse-failures.jsonl'
    service = KnowledgeAnswerService(
        retriever=None,
        generator=StructuredGenerator(StructuredModel()),
        ledger=None,
        diagnostic_sink=failure_sink(diagnostic_path),
    )

    result = service.generate('订单当前状态？', [evidence(1)])

    assert result.refused
    event = json.loads(diagnostic_path.read_text(encoding='utf-8').splitlines()[0])
    assert event['raw_content'] == provider_raw_blocks
    assert event['parse_error_type'] == 'OutputParserException'


def test_protocol_failure_snapshot_identifies_each_retrieval_strategy(tmp_path):
    diagnostic_path = tmp_path / 'strategy-failures.jsonl'
    service = KnowledgeAnswerService(
        retriever=None,
        generator=lambda question, rows: {
            'sufficient': True,
            'reason': '',
            'answer': '退款很快到账[88]',
            'cited_numbers': [88],
        },
        ledger=None,
        diagnostic_sink=failure_sink(diagnostic_path),
    )

    service.generate('退款何时到账？', [evidence(1)], strategy='hybrid')
    service.generate('退款何时到账？', [evidence(1)], strategy='hybrid_rerank')

    events = [
        json.loads(line)
        for line in diagnostic_path.read_text(encoding='utf-8').splitlines()
    ]
    assert [event['strategy'] for event in events] == ['hybrid', 'hybrid_rerank']
