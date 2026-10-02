"""Validate the labeled corpus before any quality metric is computed."""
import json
import argparse
import hashlib
from collections import Counter
from pathlib import Path
from app.services.knowledge.chunking import split_markdown

BUCKETS={'A_policy','B_model','C_colloquial','D_unknown','E_multi'}


def validate_dataset(cases, sources, minimum=300):
    source_keys = set(sources)
    if len(cases)<minimum:
        raise ValueError('dataset below required minimum')
    if len({c['eval_id'] for c in cases})!=len(cases) or len({c['query'].strip() for c in cases})!=len(cases):
        raise ValueError('duplicate evaluation id or query')
    for c in cases:
        if c['bucket'] not in BUCKETS or c['difficulty'] not in {'easy','medium','hard'} or c['split'] not in {'calibration','test'}:
            raise ValueError('invalid bucket/difficulty/split')
        if len(c['eval_id'])>16 or len(c['query'])>512 or not c['query'].strip():
            raise ValueError('invalid case length')
        truth=set(c['relevant_source_keys'])
        if not truth<=source_keys or bool(truth)==c['should_refuse']:
            raise ValueError('missing/invalid ground truth')
        if not c['should_refuse'] and (not c.get('required_facts') or any(not f.strip() for f in c['required_facts'])):
            raise ValueError('positive case requires answer facts')
        if c.get('family_id'):
            distractors = set(c.get('distractor_source_keys', []))
            if not distractors <= source_keys:
                raise ValueError('missing distractor source')
            if truth & distractors:
                raise ValueError('gold and distractor overlap')
            if c.get('category'):
                if not isinstance(sources, dict) or any(sources[k] != c['category'] for k in truth):
                    raise ValueError('gold outside category filter')
            if not c.get('challenge_tags'):
                raise ValueError('missing challenge tags')
            if c['should_refuse'] and not c.get('refusal_rationale', '').strip():
                raise ValueError('unknown case requires refusal rationale')
        elif c['eval_id'].startswith('V2'):
            raise ValueError('v2 requires family id')
    families = {}
    for c in cases:
        if c.get('family_id'):
            families.setdefault(c['family_id'], set()).add(c['split'])
    if any(len(s) > 1 for s in families.values()):
        raise ValueError('family leaks across calibration/test globally')
    counts=Counter(c['bucket'] for c in cases)
    if minimum>=300:
        if any(counts[b]<60 for b in BUCKETS):
            raise ValueError('each bucket requires 60 cases')
        for bucket in BUCKETS:
            for difficulty in ['easy','medium','hard']:
                if not any(c['bucket']==bucket and c['difficulty']==difficulty for c in cases):
                    raise ValueError('missing difficulty')
        groups={}
        for c in cases:
            groups.setdefault(c['topic_id'],set()).add(c['split'])
        if any(len(s)>1 for s in groups.values()):
            raise ValueError('topic leaks across calibration/test within bucket')
    return {'total':len(cases),'buckets':dict(counts),'splits':dict(Counter(c['split'] for c in cases))}


def load_corpus(root=Path('evaluation/ch04/corpus')):
    drafts=[]
    for p in sorted(root.glob('*.md')):
        drafts.extend(split_markdown(p.name,p.read_text(),max_chars=1200,overlap_chars=200))
    return drafts


def verify_frozen_dataset(root, drafts):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    dataset_hash = hashlib.sha256((root / 'cases.json').read_bytes()).hexdigest()
    corpus_hash = hashlib.sha256(''.join(d.source_key+d.answer for d in drafts).encode()).hexdigest()
    if manifest.get('dataset_sha256') != dataset_hash or manifest.get('corpus_sha256') != corpus_hash:
        raise ValueError('frozen dataset/corpus hash mismatch; record a versioned annotation correction')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('evaluation/ch04'))
    args = p.parse_args(argv)
    drafts=load_corpus(args.dataset_dir / 'corpus')
    verify_frozen_dataset(args.dataset_dir, drafts)
    cases=json.loads((args.dataset_dir / 'cases.json').read_text())
    result = validate_dataset(cases,{d.source_key:d.category for d in drafts})
    result.update(corpus_chunks=len(drafts), filtered_cases=sum(bool(c['category']) for c in cases))
    print(json.dumps(result,ensure_ascii=False))
    for bucket in sorted(BUCKETS):
        for difficulty in ['easy','medium','hard']:
            c=next(c for c in cases if c['bucket']==bucket and c['difficulty']==difficulty)
            print(c['eval_id'],difficulty,c['query'],'sources=',c['relevant_source_keys'])


if __name__=='__main__':
    main()
