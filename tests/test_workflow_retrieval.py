from app.services.workflow.retrieval import retrieve_policy_queries


class RecordingRetriever:
    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query
        self.calls = []

    def retrieve(self, query, *, category_prefixes, content_types):
        self.calls.append((query, category_prefixes, content_types))
        return self.hits_by_query.get(query, []), {"query": query}


def test_policy_queries_force_policy_categories_and_merge_duplicate_chunks():
    retriever = RecordingRetriever(
        {
            "退货资格": [
                {"chunk_id": 12, "score": 0.72, "answer": "首次证据", "source_key": "p1"},
                {"chunk_id": 10, "score": 0.80, "answer": "并列第一", "source_key": "p2"},
            ],
            "退货期限": [
                {"chunk_id": 12, "score": 0.91, "answer": "高分快照", "source_key": "p1"},
                {"chunk_id": 11, "score": 0.80, "answer": "并列第二", "source_key": "p3"},
            ],
        }
    )

    evidence, trace = retrieve_policy_queries(
        retriever, ["退货资格", "退货期限"], "退款退货"
    )

    assert [call[0] for call in retriever.calls] == ["退货资格", "退货期限"]
    assert all(call[1] == ("退换货与退款",) for call in retriever.calls)
    assert all(call[2] == ("policy", "after_sales") for call in retriever.calls)
    assert [item["chunk_id"] for item in evidence] == [12, 10, 11]
    assert evidence[0]["answer"] == "高分快照"
    assert [item["n"] for item in evidence] == [1, 2, 3]
    assert trace["policy_forced"] is True
    assert trace["query_traces"] == [{"query": "退货资格"}, {"query": "退货期限"}]


def test_policy_merge_uses_source_key_if_chunk_id_is_absent_and_keeps_first_tie():
    retriever = RecordingRetriever(
        {
            "q1": [
                {"source_key": "policy:one", "score": 0.7, "answer": "first"},
                {"source_key": "policy:two", "score": 0.7, "answer": "second"},
            ],
            "q2": [
                {"source_key": "policy:one", "score": 0.7, "answer": "duplicate"}
            ],
        }
    )

    evidence, _trace = retrieve_policy_queries(retriever, ["q1", "q2"], "售后")

    assert [item["source_key"] for item in evidence] == ["policy:one", "policy:two"]
    assert evidence[0]["answer"] == "first"
    assert all(call[1] == ("退换货与退款", "支付与售后") for call in retriever.calls)


def test_policy_merge_rejects_unknown_category_and_preserves_empty_results():
    retriever = RecordingRetriever({})
    evidence, trace = retrieve_policy_queries(retriever, ["退货政策"], "退款退货")
    assert evidence == []
    assert trace["policy_forced"] is True
    assert trace["hit_count"] == 0
    try:
        retrieve_policy_queries(retriever, ["query"], "物流")
    except ValueError:
        pass
    else:
        raise AssertionError("non-policy intent must not enter policy expansion")
