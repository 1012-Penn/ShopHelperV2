"""Reuse chapter 3/4 retrievers without invoking their answer generation."""

import math
from dataclasses import asdict


POLICY_CATEGORY_PREFIXES = {
    "退款退货": ("退换货与退款",),
    "售后": ("退换货与退款", "支付与售后"),
}
POLICY_CONTENT_TYPES = ("policy", "after_sales")


class EvidenceAdapter:
    def __init__(self, retriever, strategy="hybrid_rerank"):
        self.retriever, self.strategy = retriever, strategy

    def retrieve(
        self, question, category=None, *, category_prefixes=None, content_types=None
    ):
        if hasattr(self.retriever, "retrieve_with_trace"):
            if category_prefixes or content_types:
                evidence, _, trace = self.retriever.retrieve_with_trace(
                    question,
                    self.strategy,
                    category,
                    category_prefixes=category_prefixes,
                    content_types=content_types,
                )
            else:
                evidence, _, trace = self.retriever.retrieve_with_trace(
                    question, self.strategy, category
                )
            return [item.snapshot() for item in evidence], {
                **trace,
                "strategy": self.strategy,
            }
        if hasattr(self.retriever, "retrieve"):
            evidence, trace = self.retriever.retrieve(
                question,
                category,
                category_prefixes=category_prefixes,
                content_types=content_types,
            )
            if content_types:
                evidence = [
                    item for item in evidence
                    if item.get("content_type") in content_types
                    and (
                        not category_prefixes
                        or any(
                            item.get("category") == prefix
                            or item.get("category", "").startswith(prefix + " / ")
                            for prefix in category_prefixes
                        )
                    )
                ]
            return evidence, {**(trace or {}), "strategy": (trace or {}).get("strategy", self.strategy)}
        hits = self.retriever.search(question)
        evidence = [
            {
                **asdict(hit),
                "n": i + 1,
                "source_key": "legacy-faq",
                "section_path": [],
                "content_type": "product_faq",
            }
            for i, hit in enumerate(hits)
            if not category or hit.category == category
        ]
        if category_prefixes:
            evidence = [
                item
                for item in evidence
                if any(
                    item["category"] == prefix
                    or item["category"].startswith(prefix + " / ")
                    for prefix in category_prefixes
                )
            ]
        if content_types:
            evidence = [
                item for item in evidence if item["content_type"] in content_types
            ]
        return evidence, {"strategy": "dense", "normalization_calls": 0}


def retrieve_policy_queries(retriever, queries, category):
    """Retrieve all high-risk queries through policy-only category/type filters."""
    if category not in POLICY_CATEGORY_PREFIXES:
        raise ValueError("policy retrieval requires refund/return or after-sale intent")
    queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
    if not queries:
        raise ValueError("policy retrieval requires at least one query")
    prefixes = POLICY_CATEGORY_PREFIXES[category]
    found = {}
    ordered = 0
    traces = []
    for query in queries:
        evidence, trace = retriever.retrieve(
            query,
            category_prefixes=prefixes,
            content_types=POLICY_CONTENT_TYPES,
        )
        traces.append(trace)
        for item in evidence:
            ordered += 1
            chunk_id = item.get("chunk_id")
            source_key = item.get("source_key")
            key = (
                ("chunk", chunk_id)
                if chunk_id is not None
                else ("source", source_key)
                if source_key
                else ("unkeyed", ordered)
            )
            score = item.get("score")
            score = (
                float(score)
                if isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                else float("-inf")
            )
            previous = found.get(key)
            if previous is None or score > previous["_score"]:
                found[key] = {**item, "_score": score, "_order": ordered}
    merged = sorted(found.values(), key=lambda item: (-item["_score"], item["_order"]))
    result = []
    for index, item in enumerate(merged, start=1):
        item.pop("_score", None)
        item.pop("_order", None)
        item["n"] = index
        result.append(item)
    return result, {
        "strategy": "multi_query_policy",
        "policy_forced": True,
        "intent": category,
        "category_prefixes": list(prefixes),
        "content_types": list(POLICY_CONTENT_TYPES),
        "queries": queries,
        "query_traces": traces,
        "hit_count": len(result),
    }
