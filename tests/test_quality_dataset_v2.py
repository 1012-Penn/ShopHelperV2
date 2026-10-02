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
