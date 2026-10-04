"""Run live, labelled prompt checks for chapter 7 context management."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Make direct invocation (`python scripts/evaluate_ch07.py`) work from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.services.context.budget import ContextBudget
from app.services.context.manager import ContextManager
from app.services.context.summary import SUMMARY_PROMPT, ModelSummarizer, SummaryBatch
from app.services.workflow.intents import INTENT_PROMPT, REFER_PROMPT, parse_intent
from app.services.workflow.prompts import DECISION_PROMPT


def _usage_input_tokens(response: Any) -> int | None:
    """Read actual prompt usage across ChatOpenAI's normalized usage shapes."""
    usage = getattr(response, "usage_metadata", None) or {}
    for key in ("input_tokens", "prompt_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    details = getattr(response, "response_metadata", None) or {}
    provider_usage = details.get("token_usage") or details.get("usage") or {}
    for key in ("prompt_tokens", "input_tokens"):
        value = provider_usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _chinese_chars(messages) -> int:
    text = "\n".join(str(getattr(message, "content", message)) for message in messages)
    return sum("\u3400" <= char <= "\u9fff" for char in text)


class _CapturingModel:
    def __init__(self, model, capture):
        self._model = model
        self._capture = capture

    def bind(self, **kwargs):
        return _CapturingModel(self._model.bind(**kwargs), self._capture)

    def invoke(self, messages):
        response = self._model.invoke(messages)
        self._capture(response, messages)
        return response


def _model_factory(settings, calls):
    def create():
        model = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=0,
            timeout=45,
            max_retries=0,
        )
        return _CapturingModel(
            model, lambda response, messages: calls.append((response, messages))
        )

    return create


def _messages(history):
    result = []
    for item in history:
        message_type = HumanMessage if item["role"] == "user" else AIMessage
        result.append(message_type(content=item["content"]))
    return result


def _run_case(case, settings, budget):
    calls = []
    factory = _model_factory(settings, calls)
    if case["task"] == "summary":
        batch = SummaryBatch(
            conversation_id=case["id"],
            previous_upto=None,
            from_msg_id=1,
            upto_msg_id=1,
            text=case["input"],
            background=case.get("background", ""),
        )
        actual = ModelSummarizer(factory, budget)(batch)
        captured = calls[-1] if calls else None
        # ModelSummarizer's production input shape is retained for token accounting.
        estimate_messages = [
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(
                content=f"旧摘要（仅背景）：\n{batch.background}\n待摘要原文：\n{batch.text}"
            ),
        ]
        prompt_tokens_estimate = budget.count(estimate_messages)
        raw = actual
    elif case["task"] == "refer":
        state = {
            "context_history": _messages(case.get("history", [])),
            "context_summary": case.get("background", ""),
            "question": case["input"],
        }
        prompt_messages = ContextManager.history_messages(None, state, REFER_PROMPT)
        response = (
            factory().bind(max_tokens=min(400, budget.output)).invoke(prompt_messages)
        )
        raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        captured = calls[-1] if calls else None
        prompt_tokens_estimate = budget.count(prompt_messages)
        try:
            parsed = json.loads(raw)
            actual = parsed.get("question", case["input"])
            if not isinstance(actual, str) or not actual.strip():
                actual = case["input"]
        except (ValueError, TypeError, AttributeError):
            actual = case["input"]
    elif case["task"] == "classify":
        state = {
            "context_history": _messages(case.get("history", [])),
            "context_summary": case.get("background", ""),
            "question": case["input"],
        }
        prompt_messages = ContextManager.history_messages(None, state, INTENT_PROMPT)
        response = (
            factory()
            .bind(max_tokens=budget.output, response_format={"type": "json_object"})
            .invoke(prompt_messages)
        )
        raw = response.content
        captured = calls[-1] if calls else None
        prompt_tokens_estimate = budget.count(prompt_messages)
        actual = parse_intent(raw)
    elif case["task"] == "decision":
        manager = object.__new__(ContextManager)
        manager.budget = budget
        state = {
            "context_history": _messages(case.get("history", [])),
            "context_summary": case.get("background", ""),
            "question": case["input"],
            "messages": [HumanMessage(content=case["input"], id="sql-1")],
            "current_message_id": 1,
            "evidence": [],
        }
        prompt_messages = manager.model_messages(state, DECISION_PROMPT)
        response = factory().bind(max_tokens=budget.output).invoke(prompt_messages)
        raw = response.content
        captured = calls[-1] if calls else None
        prompt_tokens_estimate = budget.count(prompt_messages)
        signal = json.loads(raw)
        if not isinstance(signal, dict) or signal.get("next") not in {
            "answer",
            "clarify",
        }:
            raise ValueError("invalid decision signal")
        actual = raw
    else:
        raise ValueError(f"unsupported task: {case['task']}")

    actual_input_tokens = _usage_input_tokens(captured[0]) if captured else None
    text = actual.strip() if isinstance(actual, str) else str(actual)
    required_missing = [
        value for value in case.get("required", []) if value not in text
    ]
    required_alternatives_not_met = [
        options
        for options in case.get("required_any", [])
        if not any(value in text for value in options)
    ]
    forbidden_found = [value for value in case.get("forbidden", []) if value in text]
    checks = {
        "length_at_most_250": len(text) <= 250,
        "required_present": not required_missing and not required_alternatives_not_met,
        "forbidden_absent": not forbidden_found,
        "actual_input_tokens_available": actual_input_tokens is not None,
    }
    return {
        "id": case["id"],
        "task": case["task"],
        "input": case["input"],
        "background": case.get("background", ""),
        "output": text,
        "raw_model_output": raw,
        "output_length": len(text),
        "required": case.get("required", []),
        "required_any": case.get("required_any", []),
        "required_missing": required_missing,
        "required_alternatives_not_met": required_alternatives_not_met,
        "forbidden": case.get("forbidden", []),
        "forbidden_found": forbidden_found,
        "assertions": checks,
        "passed": all(checks.values()),
        "token_accounting": {
            "estimated_input_tokens": prompt_tokens_estimate,
            "actual_input_tokens": actual_input_tokens,
            "estimate_minus_actual": (
                prompt_tokens_estimate - actual_input_tokens
                if actual_input_tokens is not None
                else None
            ),
            "estimate_to_actual_ratio": (
                round(prompt_tokens_estimate / actual_input_tokens, 4)
                if actual_input_tokens
                else None
            ),
            "chinese_char_count": _chinese_chars(captured[1]) if captured else None,
            "suggested_chinese_ratio": (
                round(
                    max(
                        0,
                        actual_input_tokens
                        - (
                            prompt_tokens_estimate
                            - _chinese_chars(captured[1]) * budget.chinese_ratio
                        ),
                    )
                    / _chinese_chars(captured[1]),
                    4,
                )
                if captured and actual_input_tokens and _chinese_chars(captured[1])
                else None
            ),
            "usage_source": "ChatOpenAI response usage metadata",
        },
    }


def evaluate(env_file: Path, cases_file: Path, output_root: Path) -> tuple[Path, dict]:
    if not env_file.is_file():
        raise FileNotFoundError(f"env file not found: {env_file}")
    environment = {
        key: str(value)
        for key, value in dotenv_values(env_file).items()
        if value is not None
    }
    environment["DATABASE_URL"] = "sqlite:///:memory:"
    settings = Settings.from_env(environment)
    budget = ContextBudget.from_env(environment)
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation case file must contain a non-empty JSON array")

    rows = []
    for case in cases:
        try:
            rows.append(_run_case(case, settings, budget))
        except Exception as error:  # noqa: BLE001 - avoid persisting provider headers
            rows.append(
                {
                    "id": case["id"],
                    "task": case["task"],
                    "input": case["input"],
                    "background": case.get("background", ""),
                    "output": None,
                    "raw_model_output": None,
                    "output_length": None,
                    "required": case.get("required", []),
                    "required_any": case.get("required_any", []),
                    "required_missing": case.get("required", []),
                    "required_alternatives_not_met": case.get("required_any", []),
                    "forbidden": case.get("forbidden", []),
                    "forbidden_found": [],
                    "error_type": type(error).__name__,
                    "assertions": {
                        "length_at_most_250": False,
                        "required_present": False,
                        "forbidden_absent": False,
                        "actual_input_tokens_available": False,
                    },
                    "passed": False,
                    "token_accounting": {
                        "estimated_input_tokens": None,
                        "actual_input_tokens": None,
                        "estimate_minus_actual": None,
                        "estimate_to_actual_ratio": None,
                        "chinese_char_count": None,
                        "suggested_chinese_ratio": None,
                        "usage_source": None,
                    },
                }
            )
    actual_rows = [
        row
        for row in rows
        if row["token_accounting"]["actual_input_tokens"] is not None
    ]
    comparisons = [
        row["token_accounting"]["estimate_to_actual_ratio"] for row in actual_rows
    ]
    recommended_ratios = [
        row["token_accounting"]["suggested_chinese_ratio"]
        for row in actual_rows
        if row["token_accounting"]["suggested_chinese_ratio"] is not None
    ]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live_model",
        "model": settings.model,
        "case_file": str(cases_file),
        "total": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "input_token_usage_available": len(actual_rows),
        "token_calibration": {
            "samples_with_actual_usage": len(actual_rows),
            "mean_estimate_to_actual_ratio": (
                round(sum(comparisons) / len(comparisons), 4) if comparisons else None
            ),
            "mean_estimate_minus_actual": (
                round(
                    sum(
                        row["token_accounting"]["estimate_minus_actual"]
                        for row in actual_rows
                    )
                    / len(actual_rows),
                    2,
                )
                if actual_rows
                else None
            ),
            "mean_implied_chinese_ratio": (
                round(sum(recommended_ratios) / len(recommended_ratios), 4)
                if recommended_ratios
                else None
            ),
            "recommended_chinese_token_ratio": (
                round(sum(recommended_ratios) / len(recommended_ratios), 4)
                if recommended_ratios
                else None
            ),
            "basis": "estimate residual against observed prompt_tokens, divided by Chinese character count; indicative for this prompt mix",
        },
        "samples": rows,
    }
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result_path, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="dotenv file; values are never printed",
    )
    parser.add_argument(
        "--cases", type=Path, default=Path("evaluation/ch07/cases.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("evaluation/ch07/runs")
    )
    args = parser.parse_args()
    try:
        output_path, result = evaluate(args.env_file, args.cases, args.output_root)
    except Exception as error:  # noqa: BLE001 - avoid echoing provider credentials
        print(f"live evaluation could not complete: {type(error).__name__}")
        return 2
    print(
        f"model={result['model']} cases={result['total']} passed={result['passed']} failed={result['failed']}"
    )
    print(
        f"input usage available={result['input_token_usage_available']}/{result['total']}"
    )
    print(f"result={output_path}")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
