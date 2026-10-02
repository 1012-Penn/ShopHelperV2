# Query normalization calibration capture (2026-10-02)

Total calls: 60/60, all from frozen `calibration`; test rows: 0.
Outcomes: changedAccepted=16, unchangedAccepted=26, fallback=18.

A `changedAccepted` result passed structural preservation guards; it is not a semantic-equivalence verdict. Review those rows against the original intent before treating them as good rewrites. `unchangedAccepted` means the model returned the original wording and is not counted as a successful rewrite. Fallback includes rejected, empty, unparsable, or failed model responses.

Original structurally accepted candidates (parent review now complete): V2C003, V2A002, V2A005, V2A008, V2B011, V2C006, V2C009, V2C012, V2D003, V2D006, V2D009, V2D012, V2E002, V2E005, V2E008, V2E011.
Fallback cases: V2B001, V2B003, V2B004, V2B006, V2B007, V2B009, V2B010, V2B012, V2C002, V2C004, V2C005, V2C007, V2C008, V2C010, V2E003, V2E006, V2E009, V2E012.

The guard recognizes common quantities and units and locks an atomic clause when a number is followed by an unrecognized Chinese unit. This is a conservative boundary, not a complete Chinese quantity or semantic-equivalence parser; ambiguous or unsupported forms may fall back, and changed candidates still need human review.

Context7 confirmed LangChain structured output `include_raw=True` returns raw, parsed, and parsing-error fields, and `json_mode` requires its schema to be described in the prompt: [LangChain structured output docs](https://docs.langchain.com/oss/python/langchain/models). The capture adapter uses the production `QueryNormalizer.from_model` prompt and returns the parsed object to it.

## Parent semantic review and follow-up

All 16 changed candidates were independently reviewed. Four warranty questions (V2C003/C006/C009/C012) lost the background “已用了几个月”; C009 additionally introduced a calculation interpretation. These four are not valid rewrites. Two RED→GREEN regression tests protect fuzzy durations; the final prompt explicitly preserves user state, comparison relations and scope.

Four additional real requests with the final prompt returned the original wording, preserving the information. Total requests: 64 (60 original + 4 follow-up), zero API/parse errors. Original files are retained unchanged. Reapplying the final guard to the original 60 yields 12 changed / 26 unchanged / 22 fallback. Replacing the four reviewed failures with the follow-up responses yields a fixed paired retrieval input of 12 changed / 30 unchanged / 18 fallback.

That paired input mixes 56 earlier-prompt responses and four final-prompt responses. It is suitable for comparing two retrieval limits with identical input, but is **not** a uniform final-prompt rewrite-quality estimate. Full final evaluation must run the final prompt uniformly. Per-case semantic decisions, prompt/raw/guard hashes and provenance are in `parent-review.json`, `regraded-calibration.jsonl` and `regression-after-semantic-review.metadata.json`.

The requested model remained `deepseek-chat`; captured provider response metadata reports `deepseek-flash`. The historical baseline did not retain response identity, limiting attribution of before/after differences. No model selection was changed by this work.
