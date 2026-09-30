"""Validate the labeled corpus before any quality metric is computed."""
import json
from collections import Counter
from pathlib import Path
from app.services.knowledge.chunking import split_markdown

BUCKETS={'A_policy','B_model','C_colloquial','D_unknown','E_multi'}


def validate_dataset(cases, sources, minimum=300):
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
        if not truth<=sources or bool(truth)==c['should_refuse']:
            raise ValueError('missing/invalid ground truth')
        if not c['should_refuse'] and not c.get('required_facts'):
            raise ValueError('positive case requires answer facts')
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


def main():
    drafts=load_corpus()
    cases=json.loads(Path('evaluation/ch04/cases.json').read_text())
    print(json.dumps(validate_dataset(cases,{d.source_key for d in drafts}),ensure_ascii=False))
    for bucket in sorted(BUCKETS):
        for difficulty in ['easy','medium','hard']:
            c=next(c for c in cases if c['bucket']==bucket and c['difficulty']==difficulty)
            print(c['eval_id'],difficulty,c['query'],'sources=',c['relevant_source_keys'])


if __name__=='__main__':
    main()
