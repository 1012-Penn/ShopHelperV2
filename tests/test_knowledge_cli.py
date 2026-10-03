from pathlib import Path
from types import SimpleNamespace

from scripts import build_knowledge, evaluate_knowledge, extract_conversation_knowledge, sync_knowledge_vectors


class FakeIndexer:
    def __init__(self, settings=None):
        self.settings = settings or SimpleNamespace(
            knowledge_max_chars=1200,
            knowledge_overlap_chars=200,
            knowledge_batch_size=32,
        )
        self.imported_drafts = []
        self.faq_calls = 0
        self.sync_calls = []
        self.closed = False

    def import_drafts(self, drafts):
        self.imported_drafts.extend(drafts)
        return list(range(1, len(drafts) + 1))

    def import_markdown(self, drafts, active_sources):
        self.active_sources = active_sources
        return self.import_drafts(drafts)

    def import_faqs(self):
        self.faq_calls += 1
        return [99]

    def sync_pending(self, batch_size):
        self.sync_calls.append(batch_size)
        return SimpleNamespace(pending_before=0, vectorized=0, failed=0)

    def close(self):
        self.closed = True


def test_build_cli_imports_markdown_and_faq_then_syncs(monkeypatch, tmp_path, capsys):
    (tmp_path / "shipping.md").write_text("# 配送\n邮费以结算页为准。", encoding="utf-8")
    indexer = FakeIndexer()
    monkeypatch.setattr(build_knowledge, "build_indexer", lambda: indexer)

    result = build_knowledge.main(["--source-dir", str(tmp_path), "--batch-size", "16"])

    assert result == 0
    assert len(indexer.imported_drafts) == 1
    assert indexer.active_sources == ["shipping.md"]
    assert indexer.faq_calls == 1
    assert indexer.sync_calls == [16]
    assert indexer.closed
    output = capsys.readouterr().out
    assert "documents=1" in output and "vectorized=0" in output


def test_sync_cli_reports_safe_nonzero_failure(monkeypatch, capsys):
    class FailingIndexer(FakeIndexer):
        def sync_pending(self, batch_size):
            raise RuntimeError("sk_demo_secret upstream detail")

    monkeypatch.setattr(sync_knowledge_vectors, "build_indexer", lambda: FailingIndexer())

    assert sync_knowledge_vectors.main(["--batch-size", "32"]) == 1
    output = capsys.readouterr().out
    assert "sk_demo_secret" not in output
    assert "pending" in output


def test_missing_settings_return_nonzero_without_exception_details(monkeypatch, capsys):
    def fail_to_build():
        raise ValueError("sk_secret missing config")

    monkeypatch.setattr(sync_knowledge_vectors, "build_indexer", fail_to_build)

    assert sync_knowledge_vectors.main([]) == 1
    output = capsys.readouterr().out
    assert "sk_secret" not in output
    assert "configuration" in output.lower() or "failed" in output.lower()


def test_extraction_cli_extracts_once_and_syncs_pending(monkeypatch, capsys):
    calls = {"extract": [], "sync": []}

    class Extractor:
        def run(self, batch_size):
            calls["extract"].append(batch_size)
            return SimpleNamespace(
                messages_read=8,
                staged_pairs=2,
                deduped=1,
                inserted=1,
                skipped_tool_messages=2,
                last_message_id=42,
            )

    class Indexer(FakeIndexer):
        def sync_pending(self, batch_size):
            calls["sync"].append(batch_size)
            return SimpleNamespace(pending_before=1, vectorized=1, failed=0)

    indexer = Indexer()
    monkeypatch.setattr(extract_conversation_knowledge, "build_extractor", lambda: (Extractor(), indexer))

    assert extract_conversation_knowledge.main(["--batch-size", "7"]) == 0
    assert calls["extract"] == [7]
    assert calls["sync"] == [32]
    assert indexer.closed
    output = capsys.readouterr().out
    assert "staged=2" in output and "vectorized=1" in output


def test_labeled_evaluation_selects_threshold_with_all_positive_and_no_negative_false_hit():
    cases = [
        {"query": "邮费是多少", "expected": "hit", "source": "knowledge_docs/shipping.md", "answer_contains": "结算页"},
        {"query": "怎样给猫梳毛", "expected": "miss", "source": None, "answer_contains": ""},
    ]

    class Searcher:
        def search(self, query, limit):
            score = 0.72 if query == "邮费是多少" else 0.31
            return [SimpleNamespace(chunk_id=1 if query == "邮费是多少" else 2, score=score)]

    class Repository:
        def load_by_ids(self, ids):
            rows = {
                1: SimpleNamespace(id=1, source_key="doc:shipping.md:abc", answer="费用以结算页为准。"),
                2: SimpleNamespace(id=2, source_key="doc:returns.md:def", answer="查看店铺政策。"),
            }
            return [rows[item] for item in ids]

    result = evaluate_knowledge.evaluate_cases(cases, Searcher(), Repository(), [0.2, 0.4, 0.8])

    assert result["selected_threshold"] == 0.4
    assert result["metrics"][0.4]["answer_source_correct"] == 1
    assert result["metrics"][0.4]["negative_false_positives"] == 0
    assert result["metrics"][0.8]["answer_source_correct"] == 0
