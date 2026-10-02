import pytest
from app.services.quality.evaluation import retrieval_metrics, summarize, judge_faithfulness


def test_hand_checked_metrics():
    assert retrieval_metrics([9,2,4],{2,4},[1,5]) == {'recall@1':0.,'recall@5':1.,'all_evidence@1':0.,'all_evidence@5':1.,'mrr':.5}
    assert retrieval_metrics([9],set(),[1]) == {'recall@1':None,'all_evidence@1':None,'mrr':None}


def test_refusal_and_judge_failure_are_not_faithfulness_success():
    rows=[dict(bucket='A_policy',difficulty='easy',should_refuse=False,refused=True,faithfulness=None,metrics={'mrr':1,'recall@10':1},error=None),dict(bucket='D_unknown',difficulty='easy',should_refuse=True,refused=True,faithfulness=None,metrics={'mrr':None,'recall@10':None},error=None),dict(bucket='A_policy',difficulty='hard',should_refuse=False,refused=False,faithfulness=None,metrics={'mrr':0,'recall@10':0},error='judge error')]
    metrics=summarize(rows)
    assert metrics['faithfulness'] is None
    assert metrics['faithfulness_cases']==0
    assert metrics['error_count']==1
    assert metrics['known_false_refusal_rate']==.5
    assert metrics['unknown_refusal_rate']==1


def test_judge_sees_only_current_evidence():
    def judge(payload):
        assert set(payload)=={'query','answer','citations'}
        return {'claims':[{'claim':'次日到账','supported':False,'reason':'证据未承诺'}]}
    verdict=judge_faithfulness(judge,'到账','次日到账[1]',[{'n':1,'answer':'不承诺日期'}])
    assert verdict['score']==0 and '未承诺' in verdict['reason']


def test_fixture_report_is_marked_synthetic(tmp_path):
    import json
    from scripts.evaluate_ch04 import main
    assert main(['--fixture','--output-dir',str(tmp_path)])==0
    path=next(tmp_path.glob('*/report.json'))
    result=json.loads(path.read_text())
    assert result['mode']=='synthetic_fixture'
    assert set(result['summary'])=={'dense','bm25','hybrid','hybrid_rerank'}
    assert result['rows']


def test_summary_reports_denominators_and_empty_answers():
    rows=[dict(bucket='A_policy',difficulty='easy',should_refuse=False,refused=False,faithfulness=.5,claims=[{'supported':True},{'supported':False}],answer='fact',metrics={'mrr':1},error=None),dict(bucket='D_unknown',difficulty='easy',should_refuse=True,refused=True,faithfulness=None,answer='',metrics={'mrr':None},error=None)]
    m=summarize(rows)
    assert m['retrieval_cases']==1
    assert m['known_cases']==1 and m['unknown_cases']==1
    assert m['empty_answer_rate']==.5
    assert m['faithfulness_claims']==2 and m['faithfulness_micro']==.5


def test_eval_prepare_deactivates_removed_source(db_session_factory):
    from app.services.knowledge.repository import KnowledgeRepository
    from app.services.knowledge.content import ChunkDraft
    from scripts.evaluate_ch04 import prepare_evaluation_corpus
    from app.db.models import KnowledgeChunk
    from sqlalchemy import select
    repository=KnowledgeRepository(db_session_factory)
    old=ChunkDraft(source_key='doc:old.md:1',category='x',questions=['q'],answer='old',chapter_path=['x'],content_type='policy',is_critical=False)
    new=ChunkDraft(source_key='doc:new.md:1',category='x',questions=['q2'],answer='new',chapter_path=['x'],content_type='policy',is_critical=False)
    repository.upsert_drafts([old])
    prepare_evaluation_corpus(repository,[new])
    with db_session_factory() as s:
        rows={r.source_key:r.is_active for r in s.scalars(select(KnowledgeChunk))}
        assert rows=={'doc:old.md:1':False,'doc:new.md:1':True}


def test_multichunk_completeness_is_not_first_hit():
    partial = retrieval_metrics(['a', 'noise', 'a'], {'a', 'b'})
    complete = retrieval_metrics(['a', 'b'], {'a', 'b'})
    assert partial['recall@5'] == .5
    assert partial['all_evidence@5'] == 0
    assert complete['all_evidence@5'] == 1
    assert retrieval_metrics(['noise'], set())['all_evidence@5'] is None


def test_correctness_is_separate_from_faithfulness_and_requires_all_facts():
    from app.services.quality.evaluation import judge_correctness
    seen = []
    def judge(payload):
        seen.append(payload)
        return {'facts': [
            {'index': 0, 'covered': True, 'contradicted': False, 'reason': '给出了接口'},
            {'index': 1, 'covered': False, 'contradicted': False, 'reason': '没回答包装'},
        ]}
    result = judge_correctness(judge, '接口和包装？', 'USB-C[1]', ['USB-C', '不含充电器'], False, False)
    assert result['score'] == 0 and result['fact_coverage'] == .5
    assert set(seen[0]) == {'query', 'answer', 'required_facts', 'should_refuse'}
    def forbidden(payload): raise AssertionError('refusal needs no fact judge')
    assert judge_correctness(forbidden, 'q', '拒答', ['f'], False, True)['score'] == 0
    assert judge_correctness(forbidden, 'q', '拒答', [], True, True)['score'] == 1
    assert judge_correctness(forbidden, 'q', '乱答', [], True, False)['score'] == 0


def test_correctness_rejects_missing_duplicate_or_contradictory_fact_verdicts():
    from app.services.quality.evaluation import judge_correctness
    for facts in [[], [{'index': 0, 'covered': True, 'contradicted': False, 'reason': ''}]*2]:
        with pytest.raises(ValueError):
            judge_correctness(lambda _: {'facts': facts}, 'q', 'a', ['f0', 'f1'], False, False)
    result = judge_correctness(lambda _: {'facts': [dict(index=0, covered=True, contradicted=True, reason='矛盾')]}, 'q', 'a', ['f'], False, False)
    assert result['score'] == 0


def test_accuracy_summary_includes_false_refusals_and_excludes_errors():
    rows = [dict(should_refuse=False, refused=True, answer='拒答', faithfulness=None, correctness=0., metrics={}, error=None),
            dict(should_refuse=True, refused=True, answer='拒答', faithfulness=None, correctness=1., metrics={}, error=None),
            dict(should_refuse=False, refused=False, answer='a', faithfulness=1., correctness=None, metrics={}, error='JudgeError')]
    result = summarize(rows)
    assert result['answer_accuracy'] == .5
    assert result['answer_accuracy_cases'] == 2
    assert result['error_count'] == 1


def test_pair_comparison_uses_only_same_question_and_split():
    from scripts.evaluate_ch04 import report
    rows = []
    for eval_id, split, vals in [('q1', 'test', (0., 1.)), ('q2', 'calibration', (1., 0.))]:
        for strategy, value in zip(['dense', 'hybrid'], vals):
            rows.append(dict(eval_id=eval_id, split=split, bucket='A_policy', difficulty='hard',
                  challenge_tags=['near_model'], strategy=strategy, metrics={'mrr': value},
                  should_refuse=False, refused=False, answer='a', faithfulness=None, correctness=value, error=None))
    result = report(rows, 'fixture', {})
    pair = result['paired_comparisons']['test']['dense_vs_hybrid']['mrr']
    assert pair == {'pairs': 1, 'left_wins': 0, 'ties': 0, 'right_wins': 1, 'mean_delta_right_minus_left': 1.}
    assert result['groups']['hybrid']['challenge_tags']['near_model']['cases'] == 2


def test_independent_judge_guard_normalizes_identity_and_fails_before_live():
    from scripts.evaluate_ch04 import judge_identity
    from types import SimpleNamespace
    settings = SimpleNamespace(model='deepseek-chat', base_url='https://api.deepseek.com', api_key='x')
    with pytest.raises(ValueError, match='independent'):
        judge_identity(settings, {}, require_independent=True)
    with pytest.raises(ValueError, match='independent'):
        judge_identity(settings, {'JUDGE_MODEL': 'deepseek-chat', 'JUDGE_API_BASE': 'https://api.deepseek.com/v1/'}, require_independent=True)
    identity = judge_identity(settings, {'JUDGE_MODEL': 'other-judge', 'JUDGE_API_BASE': 'https://judge.example/v1'}, require_independent=True)
    assert identity['same_model_judge'] is False


def test_judge_only_preserves_answers_and_never_builds_retrieval(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import evaluate_ch04 as script
    def forbidden(*a, **k): raise AssertionError('judge-only must not build retrieval/generator')
    monkeypatch.setattr(script, 'build_live', forbidden)
    monkeypatch.setattr(script, 'build_judges', lambda *a, **k: (
        lambda _: {'claims': [dict(claim='接口USB-C', supported=True, reason='证据支持')]},
        lambda _: {'facts': [dict(index=0, covered=True, contradicted=False, reason='已回答')]},
        {'judge_model': 'external', 'judge_base_url': 'https://judge.example', 'same_model_judge': False}))
    row = dict(eval_id='V2A001', query='q', strategy='dense', split='test', bucket='A_policy', difficulty='hard',
        should_refuse=False, required_facts=['USB-C'], answer='USB-C[1]', citations=[{'n': 1, 'answer': 'USB-C'}],
        refused=False, error=None, metrics={'mrr':1.}, generator_model='original', generator_base_url='https://generator.example')
    source = tmp_path/'rows.jsonl'
    source.write_text(json.dumps(row)+'\n')
    original = source.read_bytes()
    assert script.main(['--judge-only',str(source),'--output-dir',str(tmp_path/'out')]) == 0
    result = json.loads(next((tmp_path/'out').glob('*/report.json')).read_text())
    assert source.read_bytes() == original
    assert result['mode'] == 'judge_only'
    assert result['rows'][0]['answer'] == row['answer']
    assert result['rows'][0]['generator_model'] == 'original'
    assert result['rows'][0]['correctness'] == 1
    assert result['metadata']['source_rows_sha256'] == hashlib.sha256(original).hexdigest()


def test_judge_only_rejects_bad_json_without_calling_models(tmp_path):
    from scripts.evaluate_ch04 import main
    path = tmp_path/'bad.jsonl'; path.write_text('{broken\n')
    with pytest.raises(ValueError, match='JSON'):
        main(['--judge-only',str(path),'--output-dir',str(tmp_path/'out')])


def test_record_fabrication_even_when_later_correctness_judge_failed():
    from scripts.evaluate_ch04 import persist_faith_case
    class Ledger:
        def __init__(self): self.cases = []
        def record_faith_case(self, payload): self.cases.append(payload)
    ledger = Ledger()
    row = dict(eval_id='V2A001', bucket='A_policy', query='q', strategy='hybrid_rerank',
        answer='编造[1]', judge_reason='不支持', citations=[{'n':1,'answer':'原文'}],
        judge_model='judge', faithfulness=.5, error='ParserError', error_stage='correctness_judge')
    persist_faith_case(ledger, row)
    assert ledger.cases[0]['citations'] == row['citations']
    assert ledger.cases[0]['eval_id'] == row['eval_id']
    persist_faith_case(ledger, {**row,'strategy':'dense'})
    assert len(ledger.cases) == 1


@pytest.mark.parametrize('base,alias', [
    ('https://api.deepseek.com','https://api.deepseek.com:443/v1/'),
    ('http://model.example','http://model.example:80/v1'),
])
def test_default_ports_cannot_bypass_independent_judge_guard(base, alias):
    from scripts.ch04_judging import judge_identity
    from types import SimpleNamespace
    settings=SimpleNamespace(model='same',base_url=base,api_key='x')
    with pytest.raises(ValueError,match='independent'):
        judge_identity(settings,{'JUDGE_MODEL':'same','JUDGE_API_BASE':alias},True)


def test_judge_failures_do_not_discard_or_skip_the_other_metric():
    from scripts.ch04_judging import score_answer
    from types import SimpleNamespace
    base=dict(query='q',answer='a',citations=[{'answer':'事实'}],required_facts=['事实'],
              refused=False,should_refuse=False,error=None,metrics={})
    def good_faith(payload):return {'claims':[dict(claim='a',supported=False,reason='没证据')]}
    def good_correct(payload):return {'facts':[dict(index=0,covered=True,contradicted=False,reason='答到了')]}
    def bad(payload):raise RuntimeError('judge unavailable')
    faith_success=score_answer(dict(base),good_faith,bad,'judge')
    assert faith_success['faithfulness']==0.
    summary=summarize([faith_success])
    assert summary['faithfulness']==0. and summary['faithfulness_cases']==1
    assert summary['faithfulness_claims']==1 and summary['error_count']==1
    correct_success=score_answer(dict(base),bad,good_correct,'judge')
    assert correct_success['correctness']==1.
    summary=summarize([correct_success])
    assert summary['answer_accuracy']==1. and summary['answer_accuracy_cases']==1
    assert summary['faithfulness'] is None and summary['error_count']==1
