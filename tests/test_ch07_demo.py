from scripts.demo_ch07 import run_demo


def test_default_and_demo_cascade(tmp_path):
    default = run_demo(tmp_path / "default", demo=False)
    assert (
        default["rounds"] >= 24
        and default["downgrades"] == 0
        and default["summaries"] == 0
    )
    small = run_demo(tmp_path / "small", demo=True)
    assert small["history_budget"] == 5650
    assert small["downgrades"] > 0 and small["summaries"] > 0
    assert small["original_messages"] == small["rounds"] * 2
    assert small["checkpoint_messages"] >= small["original_messages"]
    assert "778899" in small["last_answer"] and "未收到货" in small["last_answer"]
    assert small["max_input_tokens"] + 2000 < 18000
