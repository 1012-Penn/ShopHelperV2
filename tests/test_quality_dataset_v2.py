import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.evaluate_ch04 import main
from scripts.validate_ch04_dataset import validate_dataset


def test_selected_dataset_uses_its_queries_corpus_and_output(tmp_path):
    root = tmp_path / 'selected'
    root.mkdir()
    shutil.copytree('evaluation/ch04/corpus', root / 'corpus')
    cases = json.loads(Path('evaluation/ch04/cases.json').read_text())
    cases[0]['query'] = '另一个数据版本的问题？'
    data = json.dumps(cases, ensure_ascii=False).encode()
    (root / 'cases.json').write_bytes(data)
    assert main(['--fixture', '--dataset-dir', str(root), '--limit', '1']) == 0
    result = json.loads(next((root / 'runs').glob('*/report.json')).read_text())
    assert result['rows'][0]['query'] == cases[0]['query']
    assert result['metadata']['dataset_sha256'] == hashlib.sha256(data).hexdigest()
    assert result['metadata']['dataset_dir'] == str(root)


def test_missing_selected_dataset_is_not_replaced_with_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(['--fixture', '--dataset-dir', str(tmp_path / 'missing')])


def test_dataset_targets_keep_v1_v2_separate_and_expose_overrides():
    from scripts.evaluate_ch04 import evaluation_targets
    assert evaluation_targets(Path('evaluation/ch04'), {}) == (
        'sqlite:///evaluation/ch04/eval.db', 'knowledge_ch04_eval')
    assert evaluation_targets(Path('evaluation/ch04/v2'), {}) == (
        'sqlite:///evaluation/ch04/v2/eval.db', 'knowledge_ch04_eval_v2')
    assert evaluation_targets(Path('evaluation/ch04/v2'), {
        'EVAL_DATABASE_URL': 'sqlite:///custom.db',
        'EVAL_HYBRID_COLLECTION': 'custom_eval',
    }) == ('sqlite:///custom.db', 'custom_eval')


def labeled_case(**updates):
    row = dict(eval_id='V2A001', bucket='A_policy', difficulty='easy',
               split='calibration', query='在耳机品类能退吗？', category='耳机',
               relevant_source_keys=['a'], required_facts=['七天内'],
               should_refuse=False, topic_id='a', family_id='family-a',
               challenge_tags=['category_filter'], distractor_source_keys=['b'])
    return {**row, **updates}


def test_rejects_global_family_leak_across_buckets():
    rows = [labeled_case(), labeled_case(eval_id='V2C001', query='另一个问法',
            bucket='C_colloquial', split='test', topic_id='another-topic')]
    with pytest.raises(ValueError, match='family'):
        validate_dataset(rows, {'a': '耳机', 'b': '家电'}, minimum=1)


def test_rejects_gold_outside_filter_and_gold_as_distractor():
    with pytest.raises(ValueError, match='category'):
        validate_dataset([labeled_case(category='家电')], {'a': '耳机', 'b': '家电'}, minimum=1)
    with pytest.raises(ValueError, match='distractor'):
        validate_dataset([labeled_case(distractor_source_keys=['a'])], {'a': '耳机'}, minimum=1)


def test_v2_cases_require_traceable_refusal_and_nonempty_facts():
    with pytest.raises(ValueError, match='refusal'):
        validate_dataset([labeled_case(bucket='D_unknown', relevant_source_keys=[],
              required_facts=[], should_refuse=True)], {'a': '耳机', 'b': '家电'}, minimum=1)
    with pytest.raises(ValueError, match='facts'):
        validate_dataset([labeled_case(required_facts=[' '])], {'a': '耳机', 'b': '家电'}, minimum=1)


def test_frozen_dataset_cannot_be_silently_changed(tmp_path):
    root = tmp_path / 'frozen'
    root.mkdir()
    shutil.copytree('evaluation/ch04/corpus', root / 'corpus')
    shutil.copy('evaluation/ch04/cases.json', root / 'cases.json')
    (root / 'manifest.json').write_text(json.dumps({'dataset_sha256': 'wrong', 'corpus_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='frozen'):
        main(['--fixture','--dataset-dir',str(root)])


def test_live_receives_two_disjoint_corpora_and_selected_category(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import evaluate_ch04 as script
    from scripts.validate_ch04_dataset import load_corpus
    captured=[]
    roots=[]
    for label in ['alpha','beta']:
        root=tmp_path/label; (root/'corpus').mkdir(parents=True)
        (root/'corpus'/f'{label}.md').write_text(f'# {label}\n\n## 商品\n\n### 规则\n{label} 专属原文，退货需要凭证。\n\n### 配件\n{label} 专属配件。\n')
        drafts=load_corpus(root/'corpus')
        cases=[]
        for bucket in ['A_policy','B_model','C_colloquial','D_unknown','E_multi']:
            for i in range(60):
                cases.append(dict(eval_id=f'{bucket[0]}{i:03d}',bucket=bucket,difficulty=['easy','medium','hard'][i%3],
                  split='calibration' if i<12 else 'test',query=f'{label} {bucket} {i}',category=drafts[0].category,
                  relevant_source_keys=[] if bucket=='D_unknown' else [d.source_key for d in drafts],
                  required_facts=[] if bucket=='D_unknown' else ['需要凭证'],should_refuse=bucket=='D_unknown',topic_id=f'{bucket}:{i//3}'))
        (root/'cases.json').write_text(json.dumps(cases))
        roots.append(root)
    def fake_build(root,drafts):
        captured.append((root,[d.source_key for d in drafts],[d.answer for d in drafts]))
        class Retriever:
            def retrieve_with_trace(self,query,strategy,category):
                assert root.name in query and category==drafts[0].category
                return [],[],{}
            store=SimpleNamespace(close=lambda:None,collection_name='fake')
            embeddings=SimpleNamespace(close=lambda:None)
            reranker=SimpleNamespace(close=lambda:None,model='fixed')
        service=SimpleNamespace(retriever=Retriever(),min_score=.05,
             generate=lambda *a:SimpleNamespace(answer='拒答',citations=[],refused=True,reason='test'))
        judges=(None,None,{'judge_model':'fake'})
        ledger=SimpleNamespace(record_faith_case=lambda _:None)
        return service,judges,ledger,[]
    monkeypatch.setattr(script,'build_live',fake_build)
    monkeypatch.setattr(script,'judge_identity',lambda *a,**k:{})
    for root in roots:
        assert main(['--live','--dataset-dir',str(root),'--limit','1','--workers','1'])==0
    assert set(captured[0][1]).isdisjoint(captured[1][1])
    assert all('alpha 专属' in text for text in captured[0][2])
    assert all('beta 专属' in text for text in captured[1][2])
    from scripts.evaluate_ch04 import evaluation_targets
    assert evaluation_targets(roots[0],{}) != evaluation_targets(roots[1],{})
