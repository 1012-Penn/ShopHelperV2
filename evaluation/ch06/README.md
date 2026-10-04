# ch06 Prompt Evaluation

`cases.json` contains 43 synthetic, manually labelled inputs across logistics, orders, product FAQ, refund/return, after-sale, complaints, chitchat, and other. Eight conversation groups (`CYCLE-01` through `CYCLE-08`) each reach at least a third user turn. `CYCLE-01` switches from logistics to a return question about the same order and back to logistics.

Each expected result includes `reference_resolved`. Complete questions without context-dependent references intentionally preserve their exact text and set this to true. If a pronoun cannot be resolved, keep the raw question and set it to false; when intent is still classifiable (for example, “它多少钱？” or “它能退吗？” without history), send that intent to the primary Agent to clarify within its normal workflow. Only genuinely unclassifiable questions use `其他` fallback. Refund and after-sale cases include `expansion_focus` tags describing retrieval aspects rather than exact generated query strings.

These labels are an engineering evaluation set, not business-approved truth. Live prompt runs must be reported separately from deterministic workflow tests.

## Demo

Apply the additive demo refund table migration, then start the existing FastAPI app:

```bash
python3 scripts/migrate_ch06.py --dry-run
python3 scripts/migrate_ch06.py
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000`, ask `这个能退吗？`, choose a demo order card, and continue the conversation. If policy evidence and the primary Agent indicate that the order is eligible and the user wants to apply, submit the demo refund form with one of its fixed reason categories. This records a simulated application only; it does not execute a real refund. The runtime needs the configured model, database, Milvus service, and refund/after-sales policy corpus.

Run the labelled prompt and expansion evaluation with:

```bash
python3 scripts/evaluate_ch06.py --stage all --cases evaluation/ch06/cases.json --output-dir evaluation/ch06/runs/2026-10-04-final
```
