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
