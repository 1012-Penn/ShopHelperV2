import pytest
from app.services.quality.evaluation import retrieval_metrics, summarize, judge_faithfulness


def test_hand_checked_metrics():
    assert retrieval_metrics([9,2,4],{2,4},[1,5]) == {'recall@1':0.,'recall@5':1.,'mrr':.5}
    assert retrieval_metrics([9],set(),[1]) == {'recall@1':None,'mrr':None}


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
