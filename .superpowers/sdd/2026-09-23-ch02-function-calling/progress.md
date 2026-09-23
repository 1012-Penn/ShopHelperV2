# SDD ledger — plan: docs/superpowers/plans/2026-09-23-ch02-function-calling.md

## Setup

- Execution mode: Native, user explicitly chose direct work on `main`.
- Starting commit / merge base: `fdc64d2`.
- Spec: `docs/superpowers/specs/2026-09-23-ch02-function-calling-design.md` (approved by user).
- Plan: `docs/superpowers/plans/2026-09-23-ch02-function-calling.md`.
- Workspace isolation: declined by user; work directly on main.
- Interface preflight: Task 1 `Settings.database_url` -> Task 2 engine/session configuration: matches.
- Interface preflight: Task 2 `make_session_factory` and ORM models -> Task 3 business tools/runner: matches.
- Interface preflight: Task 3 `build_tools`/`ToolRunner` -> Task 4 `tool_runner_factory`: matches by injected factory.
- Interface preflight: Task 4 `ChatRequest`/event iterator -> Task 5 SSE route: matches.
- Interface preflight: Task 5 SSE event contract -> Task 6 browser parser: matches.
- Interface preflight: Task 2 `seed_faq` -> Task 7 seed CLI/evaluation: matches.
- Ruling: Installed executing-plans skill references `task-start` and `task-done`, but the mounted sibling scripts only provide `task-brief`, `sdd-workspace`, and `review-package`; use `task-brief` plus this ledger and manually record fresh full-suite results. Cost if wrong: the environment-provided ledger helper conventions may differ, but execution evidence remains explicit.
- Baseline: repository has no application/test files; initial suite check pending.

## Task 1: in_progress

Task 1: Ruling: Approved plan omitted an observable test for the promised `/health` route — add `test_health_endpoint_returns_ok` to Task 1 before implementation, because plan tasks must pin each new behavior test-first — cost if wrong: one small test and plan edit.
Task 1: Plan/environment adaptation: no `task-start` or `task-done` helper exists in the mounted skill scripts; use `task-brief`, direct commands, and record exact test results here.
Task 1: Ruling: Use `python3 -m pytest` instead of the `pytest` console script — the console script loads a globally installed `app` package before the repository package, while module invocation imports the local source — cost if wrong: developers using the console script may see confusing foreign-package failures; plan commands are updated.
Task 1: Ruling: Plan minimum Python changed from 3.11 to 3.10 — the provided runtime is Python 3.10.12, and the chosen dependency stack supports this baseline — cost if wrong: reduced access to syntax/APIs introduced only in 3.11.
Task 1: complete (commit recorded below; tests: `python3 -m pytest -q` → 3 passed). Red was verified against a temporary local app stub without Settings.from_env or /health; restored source then passed.
Task 2: Ruling: The approved test snippet inserted a message before its required conversation and would violate the declared foreign key — the RED test now creates its conversation first and adds a relationship assertion; a dedicated ticket relationship test is included — cost if wrong: slightly more fixture setup.
Task 2: Ruling: Seed CLI must not call Settings.from_env because it requires model credentials even though seed only needs MySQL — add database_url_from_env and a test proving DB setup works without model settings — cost if wrong: one small config helper and test.
Task 2: Ruling: Real MySQL rejected FAQ `TEXT UNIQUE` although SQLite accepted it — remove the unnecessary unique key (seed performs explicit existence checks), and pin the MySQL DDL contract with a regression test — cost if wrong: concurrent duplicate seed inserts are possible without a database uniqueness constraint.
Task 2: complete (tests: `python3 -m pytest -q` → 7 passed; MySQL 8 container healthy, four tables created, seed run twice leaves 4 FAQ rows; `docker compose config` passed).
Task 3: complete (tests: `python3 -m pytest -q` → 14 passed; tool-specific suite 7 passed).
Task 4: Ruling: ChatService must pass registered tools to `bind_tools`, while the registry initially exposed only private name lookup — expose a read-only `ToolRegistry.tools` property and test/consume it rather than reaching into private storage — cost if wrong: minimal public API surface expansion.
Task 4: complete (tests: `python3 -m pytest -q` → 19 passed; chat suite 5 passed).
Task 5: complete (tests: `python3 -m pytest -q` → 21 passed; API suite 2 passed).
Task 6: complete (UI Vibe Coding exception applied; `GET /` manually returned 200 text/html and page source includes SSE handling/tool status badges. Real-model browser run is blocked until MODEL/API credentials are configured; no UI test or separate UI review added).
Task 7: complete (FAQ evaluation fixture red/green; `python3 -m pytest -q` → 22 passed; `docker compose config` passed; MySQL healthy, seed repeated twice, four tables and expected hit/miss counts verified. README and `dev-notes/ch02.md` record live-model credential prerequisite. Plan's fixture/chat manual test listing was adapted to the existing `tests/test_tools.py` and actual direct UI endpoint check; no UI-specific tests/review added per user exception).
Final review follow-up: Reviewer found ticket side effects may complete after timeout; verified `Future.cancel()` does not stop a running thread. Added tests for eventual completion and for a stalled write first (red), then use a finite extra confirmation window; return confirmed success or an explicit unknown outcome instructing no duplicate submission (green). Reviewer also found missing model tool-call IDs were not normalized in persisted assistant JSON; added a regression test (red), then persist the generated ID in the tool call too (green). Restored baseline ignore patterns for `.worktrees/`, `worktrees/`, `.ruff_cache/`, and `.mypy_cache/`.
Final reviewer re-grade: no Critical/Important blocker. Final fresh verification: `python3 -m pytest -q` → 25 passed; `git diff --check` and `docker compose config` passed. Reviewer noted timed-out Python threads cannot be killed and may complete after an unknown result; caller is explicitly told not to resubmit.
