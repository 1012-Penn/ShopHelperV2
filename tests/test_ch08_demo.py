"""The chapter 8 fixture demo must execute its mechanisms end to end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "entrypoint", [["-m", "scripts.demo_ch08"], ["scripts/demo_ch08.py"]]
)
def test_fixture_demo_runs_the_mechanisms_and_reports_synthetic_results(
    tmp_path, entrypoint
):
    output_dir = tmp_path / "demo-output"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            *entrypoint,
            "--fixture",
            "--output",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["mode"] == "fixture"
    assert report["synthetic"] is True
    assert report["quality_evaluation"] == "not_evaluated"
    assert report["service_reused_after_logistics_restart"] is True
    assert report["tickets"] == {"after_confirm": 1, "after_cancel": 1}
    assert report["ticket_preview_events"] == {
        "confirm": ["ticket_preview"],
        "cancel": ["ticket_preview"],
    }
    assert {
        key for key, observations in report["mcp_query_results"].items() if observations
    } == {"logistics", "warranty", "return_progress", "eta"}
    assert all(
        observation["is_error"] is False
        for observations in report["mcp_query_results"].values()
        for observation in observations
    )
    assert report["attempts"] == {
        "read_timeout": {"calls": 3, "retry_count": 2},
        "write_timeout": {"calls": 1, "retry_count": 0},
    }
    mechanism_names = {item["name"] for item in report["mechanisms"] if item["passed"]}
    assert mechanism_names == {
        "dynamic_local_registration",
        "real_mcp_logistics_and_after_sale",
        "single_server_restart_and_hot_authorization",
        "ticket_preview_confirmation_and_cancellation",
        "transient_read_retry_and_write_no_retry",
    }
    ports = {item["port"] for item in report["mcp_processes"]}
    assert ports.isdisjoint({8765, 8766})
    assert (output_dir / "report.json").is_file()
    assert report["audit_rows"]
    assert all(
        "tool_name" in row and "retry_count" in row for row in report["audit_rows"]
    )
