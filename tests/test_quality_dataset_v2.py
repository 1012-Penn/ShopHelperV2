import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.evaluate_ch04 import main


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
