"""Reuse chapter 3/4 retrievers without invoking their answer generation."""
from dataclasses import asdict


class EvidenceAdapter:
    def __init__(self, retriever, strategy='hybrid_rerank'):
        self.retriever, self.strategy = retriever, strategy

    def retrieve(self, question, category=None):
        if hasattr(self.retriever, 'retrieve_with_trace'):
            evidence, _, trace = self.retriever.retrieve_with_trace(question, self.strategy, category)
            return [item.snapshot() for item in evidence], {**trace, 'strategy': self.strategy}
        hits = self.retriever.search(question)
        evidence = [{**asdict(hit), 'n': i+1, 'source_key': 'legacy-faq', 'section_path': []}
                    for i, hit in enumerate(hits) if not category or hit.category == category]
        return evidence, {'strategy': 'dense', 'normalization_calls': 0}
