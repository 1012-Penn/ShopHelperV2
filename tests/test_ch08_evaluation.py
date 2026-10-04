import json
from pathlib import Path


def test_cases_cover_ch08_live_workflow_boundaries():
    cases = json.loads(Path("evaluation/ch08/cases.json").read_text())
    assert len(cases) >= 16
    assert len({case["id"] for case in cases}) == len(cases)
    assert {
        "mcp_logistics",
        "mcp_after_sale",
        "dynamic_registration",
        "ticket_confirmation",
        "ticket_safety",
        "tool_validation",
        "mcp_safety",
    } <= {case["area"] for case in cases}
    assert any(
        step.get("expect", {}).get("preview") is False
        and case["area"] == "ticket_confirmation"
        for case in cases
        for step in case["steps"]
    )
    assert any(
        step["kind"] == "confirmation" and step["approve"] is True
        for case in cases
        for step in case["steps"]
    )
    assert any(
        step["kind"] == "confirmation" and step["approve"] is False
        for case in cases
        for step in case["steps"]
    )


def test_fixture_plan_is_explicitly_synthetic_and_never_scores_quality(tmp_path):
    from scripts.evaluate_ch08 import evaluate_fixture

    report = evaluate_fixture(output_dir=tmp_path)

    assert report["mode"] == "fixture"
    assert report["synthetic"] is True
    assert report["quality_metrics"] is None
    assert report["summary"]["case_count"] >= 16
    assert report["summary"]["errors"] == 0
    assert report["cases"]
    assert all(case["execution"] == "not_run" for case in report["cases"])
    assert (tmp_path / "report.json").exists()
    assert "仅验证接线" in (tmp_path / "report.md").read_text()


def test_select_cases_keeps_only_requested_cases_and_rejects_unknown_ids():
    from scripts.evaluate_ch08 import select_cases

    cases = [{"id": "one"}, {"id": "two"}]
    assert select_cases(cases, ["two"]) == [{"id": "two"}]
    try:
        select_cases(cases, ["missing"])
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("unknown case ids must be rejected")


def test_summary_keeps_errors_in_denominator_and_counts_only_completed_cases():
    from scripts.evaluate_ch08 import summarize

    result = summarize(
        [
            {"id": "ok", "status": "completed", "checks": {"case_completed": True, "tool_called": True}},
            {"id": "wrong", "status": "completed", "checks": {"case_completed": True, "tool_called": False}},
            {"id": "error", "status": "error", "checks": {"case_completed": False}, "error": "TimeoutError"},
        ]
    )

    assert result["case_count"] == 3
    assert result["completed"] == 2
    assert result["errors"] == 1
    assert result["checks"]["case_completed"] == {"passed": 2, "denominator": 3}
    assert result["checks"]["tool_called"] == {"passed": 1, "denominator": 2}


def test_live_case_runner_starts_audit_cursor_before_any_records(monkeypatch):
    from scripts import evaluate_ch08

    observed = {}
    monkeypatch.setattr(evaluate_ch08, "_ticket_count", lambda _service: 0)
    monkeypatch.setattr(evaluate_ch08, "_register_setup_tools", lambda _service, _case: lambda: None)

    def run_steps(_service, _case, _conversation_id, _tickets, audit_cursor):
        observed["audit_cursor"] = audit_cursor
        return {"status": "completed"}

    monkeypatch.setattr(evaluate_ch08, "_evaluate_case_steps", run_steps)
    result = evaluate_ch08._evaluate_case(
        object(), {"id": "cursor", "steps": []}, 1
    )

    assert result["status"] == "completed"
    assert observed["audit_cursor"] == 0


def test_sse_error_details_and_workflow_boundary_are_retained():
    from scripts.evaluate_ch08 import _run_events

    actual = _run_events(
        [
            {"event": "workflow_status", "data": {"node": "retrieve_policy"}},
            {"event": "error", "data": {"message": "暂时无法处理，请稍后再试。"}},
        ]
    )

    assert actual["error_messages"] == ["暂时无法处理，请稍后再试。"]
    assert actual["workflow_nodes"] == ["retrieve_policy"]


def test_workflow_log_trace_exposes_error_type_and_decision_path(tmp_path):
    from scripts.evaluate_ch08 import _workflow_trace

    path = tmp_path / "workflow.jsonl"
    path.write_text(
        json.dumps(
            {
                "conversation_id": "failed-case",
                "path": ["refer", "classify", "retrieve_policy"],
                "status": "error",
                "error_type": "TypeError",
                "route": "high_risk",
            }
        )
        + "\n"
    )

    trace = _workflow_trace(path, "failed-case")

    assert trace["error_type"] == "TypeError"
    assert trace["path"][-1] == "retrieve_policy"
    assert trace["route"] == "high_risk"
