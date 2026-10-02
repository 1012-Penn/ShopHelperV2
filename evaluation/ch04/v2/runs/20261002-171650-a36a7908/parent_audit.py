"""Read-only independent audit of the completed frozen 300 x 4 run."""
import hashlib
import json
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((RUN / 'report.json').read_text())
    rows = report['rows']
    frozen_path = ROOT / 'evaluation/ch04/v2/cases.json'
    cases = {c['eval_id']: c for c in json.loads(frozen_path.read_text())}
    journal = [json.loads(line) for line in (RUN / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == len(journal) == 1200
    key = lambda row: (row['eval_id'], row['strategy'])
    assert sorted(rows, key=key) == sorted(journal, key=key)
    assert len({key(row) for row in rows}) == 1200
    strategies = ('dense', 'bm25', 'hybrid', 'hybrid_rerank')
    assert Counter(row['strategy'] for row in rows) == {s: 300 for s in strategies}
    assert report['metadata']['dataset_sha256'] == digest(frozen_path)
    assert report['metadata']['hybrid_output_limit'] == 100
    for name in ('query', 'generation'):
        path = ROOT / f'app/services/quality/{name}.py'
        assert report['metadata'][name + '_implementation_sha256'] == digest(path)

    connection = sqlite3.connect(f'file:{ROOT / "evaluation/ch04/v2/eval.db"}?mode=ro', uri=True)
    db = {r[0]: dict(source_key=r[1], answer=r[2], section_path=json.loads(r[3]),
                      question=json.loads(r[4])[0], category=r[5])
          for r in connection.execute('SELECT id,source_key,answer,chapter_path,questions,category '
                                      'FROM knowledge_chunks WHERE is_active=1')}
    assert len(db) == 480
    canonical_by_case = {}
    recomputed = {}
    citation_count = 0
    for row in rows:
        case = cases[row['eval_id']]
        for field in ('query', 'split', 'bucket', 'difficulty', 'category',
                      'relevant_source_keys', 'required_facts', 'should_refuse'):
            assert row[field] == case[field]
        trace = row.get('retrieval_trace')
        if trace:
            limit = 100 if row['strategy'] in ('hybrid', 'hybrid_rerank') else 50
            assert trace['candidate_limit'] == limit
            assert trace['fusion_output_limit'] == (limit if limit == 100 else None)
            canonical = (trace['canonical_query'], trace['lexical_query'])
            assert canonical == canonical_by_case.setdefault(row['eval_id'], canonical)
            candidate_ids = row['candidate_ids']
            final_ids = row['final_ids']
            assert len(candidate_ids) <= limit and len(final_ids) <= 10
            assert set(final_ids) <= set(candidate_ids)
            assert all(i in db for i in candidate_ids)
            if row['category']:
                assert all(db[i]['category'] == row['category'] for i in candidate_ids)
            gold = set(row['relevant_source_keys'])
            final = [db[i]['source_key'] for i in final_ids]
            calculated = {}
            for k in (1, 5, 10):
                calculated[f'recall@{k}'] = len(set(final[:k]) & gold) / len(gold) if gold else None
                calculated[f'all_evidence@{k}'] = float(gold <= set(final[:k])) if gold else None
            calculated['mrr'] = next((1/i for i, s in enumerate(final, 1) if s in gold), 0.0) if gold else None
            first50 = {db[i]['source_key'] for i in candidate_ids[:50]}
            calculated['candidate_recall@50'] = len(first50 & gold) / len(gold) if gold else None
            assert calculated == row['metrics']
        for citation in row['citations']:
            source = db[citation['chunk_id']]
            for field in ('source_key', 'answer', 'section_path', 'question', 'category'):
                assert citation[field] == source[field]
            assert citation['source_url'].startswith('/api/v1/knowledge/source?source=')
            citation_count += 1
        if not row['refused'] and not row['error']:
            allowed = {c['n'] for c in row['citations']}
            cited = set(map(int, re.findall(r'\[(\d+)\]', row['answer'])))
            assert cited and cited <= allowed
            assert {c['chunk_id'] for c in row['citations']} == set(row['final_ids'])

    for strategy in strategies:
        selected = [row for row in rows if row['strategy'] == strategy]
        summary = report['summary'][strategy]
        known = [row for row in selected if not row['should_refuse']]
        unknown = [row for row in selected if row['should_refuse']]
        assert len(known) == 240 and len(unknown) == 60
        values = {}
        for name in ('recall@1', 'recall@5', 'recall@10', 'all_evidence@10', 'mrr', 'candidate_recall@50'):
            available = [row['metrics'][name] for row in selected
                         if row.get('metrics', {}).get(name) is not None]
            value = statistics.mean(available) if available else None
            assert value is None or abs(value - summary[name]) < 1e-12
            values[name] = value
        assert summary['error_count'] == sum(bool(row['error']) for row in selected)
        assert abs(summary['known_false_refusal_rate'] - sum(row['refused'] for row in known)/240) < 1e-12
        assert abs(summary['unknown_refusal_rate'] - sum(row['refused'] for row in unknown)/60) < 1e-12
        values.update(errors=summary['error_count'], known_false_refusals=sum(row['refused'] for row in known),
                      unknown_refusals=sum(row['refused'] for row in unknown))
        recomputed[strategy] = values

    unique = {row['eval_id']: row for row in rows if row.get('retrieval_trace')}
    changed = [dict(eval_id=row['eval_id'], query=row['query'],
                    canonical=row['retrieval_trace']['canonical_query'])
               for row in unique.values() if row['query'] != row['retrieval_trace']['canonical_query']]
    output = dict(verified=True, rows=len(rows), cases=len(unique), sql_chunks=len(db),
                  citation_snapshots_checked=citation_count, recomputed=recomputed,
                  changed_queries=changed, errors=[dict(eval_id=row['eval_id'], strategy=row['strategy'],
                                                       error=row['error']) for row in rows if row['error']],
                  report_sha256=digest(RUN/'report.json'), journal_sha256=digest(RUN/'rows.jsonl'),
                  production_revision='f135940',
                  limitations='Audit verifies data/metrics/provenance/citation integrity; judge verdicts remain model-based, '
                              'not independent human truth. Rewrite semantic review is recorded separately.')
    (RUN / 'parent-review.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ('changed_queries',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
