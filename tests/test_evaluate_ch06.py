import json

from scripts.evaluate_ch06 import _write_results, summarize, summarize_expansion


def test_prompt_evaluation_writes_report_and_case_results_without_environment_secrets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-write-this-secret")
    metadata = {
        "stage": "prompts",
        "started_at": "2026-10-04T00:00:00+00:00",
        "model": "demo-model",
    }
    rows = [
        {
            "id": "CASE-1",
            "intent_json_valid": True,
            "intent_correct": True,
            "reference_json_valid": True,
            "query_correct": True,
            "reference_status_correct": True,
        }
    ]

    metrics = _write_results(tmp_path, metadata, rows)

    assert metrics["case_count"] == 1
    assert json.loads((tmp_path / "metadata.json").read_text()) == metadata
    assert json.loads((tmp_path / "results.jsonl").read_text()) == rows[0]
    report = (tmp_path / "report.md").read_text()
    assert "Intent accuracy: 100.0%" in report
    assert "do-not-write-this-secret" not in report
    assert "do-not-write-this-secret" not in (tmp_path / "metadata.json").read_text()


def test_expansion_evaluation_counts_invalid_and_incomplete_rows_explicitly(tmp_path):
    rows = [
        {
            "id": "CASE-1",
            "json_valid": True,
            "distinct_queries": True,
            "fallback_included": True,
            "queries": ["canonical", "variant"],
        },
        {
            "id": "CASE-2",
            "json_valid": False,
            "distinct_queries": True,
            "fallback_included": True,
            "queries": ["canonical"],
            "error": "ValueError",
        },
    ]
    assert summarize_expansion(rows) == {
        "case_count": 2,
        "query_json_valid": 1,
        "query_json_valid_rate": 0.5,
        "distinct_query_rate": 1.0,
        "fallback_included": 2,
        "mean_queries_including_fallback": 1.5,
    }

    metrics = _write_results(
        tmp_path,
        {"stage": "expansion", "started_at": "now", "model": "demo-model"},
        rows,
    )
    assert metrics["query_json_valid"] == 1
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2
    assert "Expansion JSON contract valid: 1/2" in (tmp_path / "report.md").read_text()


def test_prompt_summary_handles_empty_or_incomplete_model_rows():
    assert summarize([]) == {
        "case_count": 0,
        "intent_json_valid": 0,
        "intent_accuracy": 0.0,
        "reference_json_valid": 0,
        "reference_query_exact": 0.0,
        "reference_status_accuracy": 0.0,
    }
