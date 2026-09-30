import pytest
from scripts.validate_ch04_dataset import validate_dataset


def test_rejects_missing_ground_truth_and_duplicate_ids():
    with pytest.raises(ValueError):
        validate_dataset([dict(eval_id='A1',bucket='A_policy',difficulty='easy',split='test',query='q',relevant_source_keys=[],should_refuse=False)],set(),minimum=1)
