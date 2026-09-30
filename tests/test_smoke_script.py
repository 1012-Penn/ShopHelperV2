from pathlib import Path


def test_smoke_script_is_executable_and_checks_both_endpoints():
    script = Path("scripts/smoke_test.sh")
    assert script.exists()
    assert script.stat().st_mode & 0o111
    content = script.read_text(encoding="utf-8")
    assert "/api/v1/chat/stream" in content
    assert "/api/v1/after-sale/extract" in content
    assert "event: token" in content
    assert "event: done" in content
    assert "order_id" in content
    assert "request_type" in content
    assert "expected_solution" in content
    assert "SERVER_URL=" in content
    assert "BASE_URL=\"http://" not in content
