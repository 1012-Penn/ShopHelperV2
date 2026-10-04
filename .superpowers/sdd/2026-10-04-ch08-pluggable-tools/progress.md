# SDD ledger — plan: docs/superpowers/plans/2026-10-04-ch08-pluggable-tools.md
Pre-flight: Task1 definitions/registry/policy → Task2 engine → Task3 discovery/Task4 confirmation/Task5 agent, interfaces agreed.
Task 1: complete (tests: python3 -m pytest -q tests/test_tool_catalog.py → 4 passed; red: missing register/policy)
Task 2: in progress (engine tests 9 passed, integration/legacy regression remains)
Task 2: Ruling: receipt direct unique insert running instead of reserved→running CAS — one atomic claim with same exclusion — cost if wrong: crashed claim stays unknown and needs manual reconciliation.
Commits deferred until independent backend review; unrelated changes excluded.

Task 2: complete — strict engine/audit/write receipt; concurrent source/schema replacement and hot revocation regressions passed.
Task 3: complete — actual HTTP MCP integration + unchanged service on single-server restart; demo processes are isolated.
Task 4: complete — checkpointed preview/confirm/cancel, disconnect restore, five explicit revocations; ticket tests 12 passed.
Task 5: complete — dynamic schema/catalog with budget, paired observations including invalid JSON; browser ch05–ch08 19 passed.
Task 6: in progress — whole-suite rerun and independent review complete for backend issues; live original19 12passed/19completed/0errors, final core3/3passed (actual ticket created1, cancellation0). Keep failed/untriggered coverage honest.
Task 4 verification extension: concurrent approval, reopen clarification intent, both cross-resume payloads, independent complaint/chat writes — five tests pass; confirmation file17pass.
Task 5 extension: giant actual JSON Schema (short description) blocks model decision before binding; regression passes. Classification catalog stays before user question.
Task 6: complete — final Python400pass/2opt-in skips, browser21pass including actual HTTP+SQL2, scoped Ruff/diff pass, actual MCP demo5/5, final real19 cases16pass/19completed/0errors with3 uncovered probes preserved. Independent backend review no open Critical/Important. Retain codex/ch08-pluggable-tools as approved; no merge/push.
