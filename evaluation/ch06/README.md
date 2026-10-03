# ch06 Prompt Evaluation

`cases.json` contains 43 synthetic, manually labelled inputs across logistics, orders, product FAQ, refund/return, after-sale, complaints, chitchat, and other. Eight conversation groups (`CYCLE-01` through `CYCLE-08`) each reach at least a third user turn. `CYCLE-01` switches from logistics to a return question about the same order and back to logistics.

Each expected result includes `reference_resolved`. Complete questions without context-dependent references intentionally preserve their exact text and set this to true. If a pronoun cannot be resolved, keep the raw question and set it to false; when intent is still classifiable (for example, “它多少钱？” or “它能退吗？” without history), send that intent to the primary Agent to clarify within its normal workflow. Only genuinely unclassifiable questions use `其他` fallback. Refund and after-sale cases include `expansion_focus` tags describing retrieval aspects rather than exact generated query strings.

These labels are an engineering evaluation set, not business-approved truth. Live prompt runs must be reported separately from deterministic workflow tests.
