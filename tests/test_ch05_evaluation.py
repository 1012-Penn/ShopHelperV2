def test_invalid_classification_is_error_not_success():
    from scripts.evaluate_ch05 import summarize

    report = summarize([{"expected": "投诉", "actual": None, "error": "invalid_json"}])
    assert report["correct"] == 0
    assert report["errors"] == 1
    assert report["accuracy"] == 0


def test_misclassification_not_counted_as_protocol_error():
    from scripts.evaluate_ch05 import summarize

    result = summarize(
        [
            {"expected": "订单", "actual": "物流", "error": None},
            {"expected": "闲聊", "actual": "闲聊", "error": None},
        ]
    )
    assert result["correct"] == 1
    assert result["errors"] == 0
    assert result["accuracy"] == 0.5


def test_labelled_cases_cover_all_seven_intents():
    import json
    from pathlib import Path

    cases = json.loads(Path("evaluation/ch05/cases.json").read_text())
    assert {c["expected"] for c in cases} == {
        "物流",
        "订单",
        "商品咨询",
        "退款退货",
        "售后",
        "投诉",
        "闲聊",
    }
    assert len({c["id"] for c in cases}) == len(cases)


def test_fixture_demo_exercises_five_acceptance_paths(tmp_path):
    from scripts.demo_ch05 import run_demo

    report = run_demo(live=False, output_dir=tmp_path)
    assert report["mode"] == "fixture"
    assert report["accepted"] is True
    assert len(report["cases"]) == 5
    assert report["cases"][4]["tool_calls"] == 2
    assert report["cases"][3]["tokens"] == 0


def test_demo_stopped_agent_is_not_accepted(tmp_path):
    from scripts.demo_ch05 import acceptance_checks, run_demo

    report = run_demo(live=False, output_dir=tmp_path)
    report["cases"][4]["stop_reason"] = "invalid_agent_signal"
    assert acceptance_checks(report["cases"])["agent_answers_completed"] is False
