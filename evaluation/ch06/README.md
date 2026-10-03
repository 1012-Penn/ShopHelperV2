# ch06 Prompt Evaluation

`cases.json` contains 40 synthetic, manually labelled inputs across logistics, orders, product FAQ, refund/return, after-sale, complaints, chitchat, and other. It includes multi-turn conversations; `CYCLE-01` switches from logistics to a return question about the same order and back to logistics.

`expected.resolved_query` is the expected standalone wording. Complete questions without unresolved references intentionally preserve their exact text. Refund and after-sale cases include `expansion_focus` tags describing retrieval aspects rather than exact generated query strings.

These labels are an engineering evaluation set, not business-approved truth. Live prompt runs must be reported separately from deterministic workflow tests.
