"""Compare four strategies on the isolated labeled corpus; preserve every run."""
import argparse
import hashlib
import json
import os
import re
import time
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from app.config import Settings
from app.db.session import create_tables,make_engine,make_session_factory
from app.services.knowledge.repository import KnowledgeRepository
from app.services.quality.index import HybridIndexer
from app.services.quality.runtime import build_answer_service,config
from app.services.quality.evaluation import retrieval_metrics,summarize,calibration_threshold,metric_available
from app.services.quality.query import NORMALIZER_PROMPT
from app.services.quality.generation import QUALITY_PROMPT
from app.services.quality.retrieval import DEFAULT_HYBRID_CANDIDATE_LIMIT
from app.services.quality.ledger import QualityLedger
from scripts.validate_ch04_dataset import load_corpus,validate_dataset,verify_frozen_dataset

from scripts.ch04_judging import (build_judges, judge_identity, public_endpoint, score_answer,
                                  load_replay_rows, rejudge_row)

STRATEGIES=['dense','bm25','hybrid','hybrid_rerank']


def evaluation_contract_metadata():
    project_root=Path(__file__).resolve().parents[1]
    query_path=project_root/'app/services/quality/query.py'
    generation_path=project_root/'app/services/quality/generation.py'
    return {
        'query_prompt_sha256':hashlib.sha256(NORMALIZER_PROMPT.encode('utf-8')).hexdigest(),
        'query_implementation_sha256':hashlib.sha256(query_path.read_bytes()).hexdigest(),
        'generation_prompt_sha256':hashlib.sha256(QUALITY_PROMPT.encode('utf-8')).hexdigest(),
        'generation_implementation_sha256':hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        'hybrid_output_limit':DEFAULT_HYBRID_CANDIDATE_LIMIT,
    }


def report(rows,mode,metadata):
    summaries={}
    groups={}
    for strategy in STRATEGIES:
        selected=[r for r in rows if r['strategy']==strategy]
        if not selected:continue
        summaries[strategy]=summarize(selected)
        groups[strategy]={'bucket':{b:summarize([r for r in selected if r['bucket']==b]) for b in sorted({r['bucket'] for r in selected})},'difficulty':{d:summarize([r for r in selected if r['difficulty']==d]) for d in ['easy','medium','hard']},'split':{split:summarize([r for r in selected if r['split']==split]) for split in ['calibration','test']}, 'challenge_tags':{tag:summarize([r for r in selected if tag in r.get('challenge_tags',[])]) for tag in sorted({tag for r in selected for tag in r.get('challenge_tags',[])})}}
    return {'mode':mode,'metadata':metadata,'summary':summaries,'groups':groups,'paired_comparisons':paired_comparisons(rows),'rows':rows}


def paired_comparisons(rows):
    output = {}
    for split in ['all','calibration','test']:
        chosen = [r for r in rows if split == 'all' or r['split'] == split]
        by_strategy = {s:{r['eval_id']:r for r in chosen if r['strategy']==s} for s in STRATEGIES}
        output[split] = {}
        for left,right in combinations(STRATEGIES,2):
            if not by_strategy[left] or not by_strategy[right]:continue
            pair = {}
            for metric in ['recall@1','recall@5','recall@10','all_evidence@10','mrr','correctness']:
                values=[]
                for id in sorted(by_strategy[left].keys() & by_strategy[right].keys()):
                    a,b=by_strategy[left][id],by_strategy[right][id]
                    if metric=='correctness' and (not metric_available(a,'correctness') or not metric_available(b,'correctness')):continue
                    av=a.get('correctness') if metric=='correctness' else a.get('metrics',{}).get(metric)
                    bv=b.get('correctness') if metric=='correctness' else b.get('metrics',{}).get(metric)
                    if av is not None and bv is not None:values.append(bv-av)
                pair[metric]={'pairs':len(values),'left_wins':sum(v < -1e-9 for v in values),
                              'ties':sum(abs(v) <= 1e-9 for v in values),'right_wins':sum(v > 1e-9 for v in values),
                              'mean_delta_right_minus_left':sum(values)/len(values) if values else None}
            output[split][left+'_vs_'+right]=pair
    return output


def markdown_report(result):
    def display(value):
        return 'N/A' if value is None else f'{value:.3f}'
    meta=result['metadata']
    identity='同模型裁判基线，存在自评偏差' if meta.get('same_model_judge') is True else ('不同模型/端点配置的裁判，仍需人工抽查' if meta.get('same_model_judge') is False else '裁判独立性未确认')
    lines=['# ch04 四策略评估', '', '模式：'+result['mode'], '', identity,
           f"语料 chunk：{meta.get('corpus_chunks')}；生成：{meta.get('generator_model')}；裁判：{meta.get('judge_model')}", '',
           '| 策略 | 题数 | Recall@1 | Recall@5 | Recall@10 | 完整证据@10 | MRR | 答案正确率 | Faithfulness | 错误 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for strategy,m in result['summary'].items():
        lines.append(f"| {strategy} | {m['cases']} | {display(m.get('recall@1'))} | {display(m.get('recall@5'))} | {display(m.get('recall@10'))} | {display(m.get('all_evidence@10'))} | {display(m.get('mrr'))} | {display(m['answer_accuracy'])} | {display(m['faithfulness'])} | {m['error_count']} |")
    for strategy,groups in result['groups'].items():
        lines+=['', '## '+strategy+' 按类型', '', '| 桶 | Recall@5 | 完整证据@10 | MRR | 正确率 | Faithfulness | 未知拒答率 | 已知误拒答率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for bucket,m in groups['bucket'].items():
            lines.append(f"| {bucket} | {display(m.get('recall@5'))} | {display(m.get('all_evidence@10'))} | {display(m.get('mrr'))} | {display(m['answer_accuracy'])} | {display(m['faithfulness'])} | {display(m['unknown_refusal_rate'])} | {display(m['known_false_refusal_rate'])} |")
    lines+=['','## Test 同题配对差异','','| 左 vs 右 | 指标 | 配对数 | 左胜 | 平 | 右胜 | 右减左均值 |','|---|---|---:|---:|---:|---:|---:|']
    for pair,metrics in result['paired_comparisons']['test'].items():
        for name,m in metrics.items():
            lines.append(f"| {pair} | {name} | {m['pairs']} | {m['left_wins']} | {m['ties']} | {m['right_wins']} | {display(m['mean_delta_right_minus_left'])} |")
    lines+=['','拒答无事实声明记 N/A，不计为忠实度通过。正例误拒答的正确性为0；未知题按显式拒答判定。正确性与证据忠实度分开，不共享 ground-truth 输入。',
            '错误按出错行及各裁判分别计数；两项裁判独立尝试，有效判定不受另一裁判失败影响。分母及事实覆盖率保存在JSON。检索指标保留已完成检索的行。',
            '类型、难度、挑战标签、test及calibration分组分别保存于JSON。受控虚构语料尚未经业务专家独立复核；不得按test成绩筛题或调参。']
    return '\n'.join(lines)+'\n'


def prepare_evaluation_corpus(repository, drafts):
    repository.upsert_drafts(drafts)
    sources=sorted({d.source_key[4:].rsplit(':',1)[0] for d in drafts})
    repository.deactivate_missing_documents(sources,[d.source_key for d in drafts])


def evaluation_targets(dataset_dir, values):
    root = Path(dataset_dir)
    if root == Path('evaluation/ch04'):
        name = 'knowledge_ch04_eval'
    elif root == Path('evaluation/ch04/v2'):
        name = 'knowledge_ch04_eval_v2'
    else:
        slug = re.sub(r'[^a-zA-Z0-9_]', '_', root.name)[:32]
        digest = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8]
        name = f'knowledge_ch04_eval_{slug}_{digest}'
    return (values.get('EVAL_DATABASE_URL', f'sqlite:///{root}/eval.db'),
            values.get('EVAL_HYBRID_COLLECTION', name))


def build_live(dataset_dir=Path('evaluation/ch04'), drafts=None):
    settings=Settings.from_env()
    values=config()
    database_url, collection = evaluation_targets(dataset_dir, values)
    eval_engine=make_engine(database_url)
    create_tables(eval_engine)
    sessions=make_session_factory(eval_engine)
    repository=KnowledgeRepository(sessions)
    drafts = drafts if drafts is not None else load_corpus(dataset_dir / 'corpus')
    prepare_evaluation_corpus(repository,drafts)
    previous = os.environ.get('HYBRID_COLLECTION')
    os.environ['HYBRID_COLLECTION'] = collection
    try:
        service=build_answer_service(sessions,settings)
    finally:
        if previous is None:
            os.environ.pop('HYBRID_COLLECTION', None)
        else:
            os.environ['HYBRID_COLLECTION'] = previous
    print('evaluation index:',HybridIndexer(sessions,service.retriever.embeddings,service.retriever.store).sync(),flush=True)
    judges=build_judges(settings,values)
    ledger_engine=make_engine(settings.database_url)
    create_tables(ledger_engine)
    ledger=QualityLedger(make_session_factory(ledger_engine))
    return service,judges,ledger,[eval_engine,ledger_engine]


def fixture_rows(cases,strategies):
    rows=[]
    for c in cases:
        for strategy in strategies:
            ranking=c['relevant_source_keys'] if not c['should_refuse'] else []
            rows.append({**c,'strategy':strategy,'refused':c['should_refuse'],'answer':'fixture only','citations':[],'faithfulness':None if c['should_refuse'] else 1.,'correctness':1.,'error':None,'metrics':retrieval_metrics(ranking,set(c['relevant_source_keys'])),'top_score':.9 if ranking else 0,'latency_seconds':0})
    return rows


def evaluate_case(case,strategies,service,judges):
    faith_judge,correctness_judge,identity=judges
    rows=[]
    for strategy in strategies:
        started=time.monotonic()
        row={**case,**identity,'strategy':strategy,'error':None,'refused':False,'faithfulness':None,'metrics':{},'answer':'','citations':[],'top_score':0}
        try:
            stage='retrieval'
            evidence,candidates,trace=service.retriever.retrieve_with_trace(case['query'],strategy,case['category'])
            row['retrieval_trace']=trace
            row['metrics']=retrieval_metrics([e.source_key for e in evidence],set(case['relevant_source_keys']))
            candidate_metrics=retrieval_metrics([e.source_key for e in candidates],set(case['relevant_source_keys']),[50])
            row['metrics']['candidate_recall@50']=candidate_metrics['recall@50']
            row['candidate_ids']=[e.chunk_id for e in candidates]
            row['final_ids']=[e.chunk_id for e in evidence]
            row['top_score']=evidence[0].score if evidence else 0
            stage='generation';generated_at=time.monotonic()
            answer=service.generate(case['query'],evidence,strategy)
            row['generation_seconds']=time.monotonic()-generated_at
            row.update(answer=answer.answer,citations=answer.citations,refused=answer.refused,refusal_reason=answer.reason)
            stage='judging'
            score_answer(row,faith_judge,correctness_judge,identity['judge_model'])
        except Exception as error:
            row['error']=type(error).__name__
            row['error_stage']=row.get('error_stage',stage)
        row['latency_seconds']=time.monotonic()-started
        rows.append(row)
    return rows


def new_run(output_dir):
    run_dir=Path(output_dir)/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    return run_dir


def is_fabrication(row):
    return (row['strategy']=='hybrid_rerank' and row.get('faithfulness') is not None
            and row['faithfulness'] < 1 and (not row.get('error')
            or row.get('error_stage')=='correctness_judge'))


def persist_faith_case(ledger, row):
    if is_fabrication(row):
        ledger.record_faith_case(dict(eval_id=row['eval_id'],bucket=row['bucket'],query=row['query'],
            strategy=row['strategy'],answer=row['answer'],reason=row['judge_reason'],
            citations=row['citations'],judge_model=row['judge_model']))


def build_quality_ledger():
    engine=make_engine(Settings.from_env().database_url)
    create_tables(engine)
    return QualityLedger(make_session_factory(engine)),engine


def replay_main(args):
    rows,original_metadata=load_replay_rows(args.judge_only)
    if any(r['strategy'] not in STRATEGIES for r in rows):raise ValueError('unknown replay strategy')
    rows=[r for r in rows if (args.split=='all' or r['split']==args.split) and (args.strategy=='all' or r['strategy']==args.strategy)]
    if not rows:raise ValueError('no replay rows selected')
    if args.limit:
        ids=list(dict.fromkeys(r['eval_id'] for r in rows))[:args.limit]
        rows=[r for r in rows if r['eval_id'] in ids]
    identities={(r.get('generator_model'),r.get('generator_base_url')) for r in rows}
    if len(identities)!=1:raise ValueError('replay requires one known generator identity')
    generator_model,generator_base=next(iter(identities))
    judges=build_judges(require_independent=args.require_independent_judge,
                       generator_identity={'generator_model':generator_model,'generator_base_url':generator_base})
    run_dir=new_run(args.output_dir or args.judge_only.parent / 'rejudged')
    output=[]
    ledger=None;ledger_engine=None
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (run_dir/'rows.jsonl').open('w') as journal:
            futures=[pool.submit(rejudge_row,r,judges) for r in rows]
            for future in as_completed(futures):
                row=future.result();output.append(row)
                journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                if is_fabrication(row):
                    if ledger is None:ledger,ledger_engine=build_quality_ledger()
                    persist_faith_case(ledger,row)
    finally:
        if ledger_engine:ledger_engine.dispose()
    output.sort(key=lambda r:(r['eval_id'],r['strategy']))
    metadata={**original_metadata,**judges[2],'generator_model':generator_model,'generator_base_url':generator_base,
              'source_rows_sha256':hashlib.sha256(args.judge_only.read_bytes()).hexdigest(),
              'source_rows':str(args.judge_only),'split':args.split,'cases':len({r['eval_id'] for r in rows})}
    result=report(output,'judge_only',metadata)
    (run_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (run_dir/'report.md').write_text(markdown_report(result))
    print('report:',run_dir/'report.md',flush=True)
    return 1 if any(r.get('error') for r in output) else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live',action='store_true');mode.add_argument('--fixture',action='store_true')
    mode.add_argument('--judge-only',type=Path)
    p.add_argument('--require-independent-judge',action='store_true')
    p.add_argument('--split',choices=['all','calibration','test'],default='all')
    p.add_argument('--strategy',choices=['all',*STRATEGIES],default='all')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--limit',type=int)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--dataset-dir', type=Path, default=Path('evaluation/ch04'))
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args(argv)
    if not 1<=args.workers<=24: p.error('workers must be 1..24')
    if args.limit is not None and args.limit < 1:p.error('limit must be positive')
    if args.require_independent_judge and args.fixture:p.error('fixture has no real judge')
    if args.judge_only:
        return replay_main(args)
    dataset_file = args.dataset_dir / 'cases.json'
    cases=json.loads(dataset_file.read_text())
    drafts=load_corpus(args.dataset_dir / 'corpus');validate_dataset(cases,{d.source_key:d.category for d in drafts})
    verify_frozen_dataset(args.dataset_dir, drafts)
    cases=[c for c in cases if args.split=='all' or c['split']==args.split]
    if args.limit:cases=cases[:args.limit]
    strategies=STRATEGIES if args.strategy=='all' else [args.strategy]
    identity=judge_identity(Settings.from_env(),config(),args.require_independent_judge) if args.live else {}
    run_dir=new_run(args.output_dir or args.dataset_dir / 'runs')
    database_url, collection = evaluation_targets(args.dataset_dir, config())
    metadata={**identity,'cases':len(cases),'strategies':strategies,'split':args.split,'dataset_dir':str(args.dataset_dir),'dataset_sha256':hashlib.sha256(dataset_file.read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256(''.join(d.source_key+d.answer for d in drafts).encode()).hexdigest(), 'corpus_chunks':len(drafts), 'collection':collection, 'evaluation_database':database_url.rsplit('@',1)[-1]}
    metadata.update(evaluation_contract_metadata())
    rows=[];resources=[];service=None;ledger=None
    try:
        if args.fixture:
            rows=fixture_rows(cases,strategies)
        else:
            service,judges,ledger,resources=build_live(args.dataset_dir, drafts)
            judge_model=judges[2]['judge_model']
            metadata.update(**judges[2],reranker=service.retriever.reranker.model,collection=service.retriever.store.collection_name,rerank_min_score=service.min_score)
            metadata['hybrid_output_limit']=service.retriever.hybrid_candidate_limit
            with ThreadPoolExecutor(max_workers=args.workers) as pool, (run_dir/'rows.jsonl').open('w') as journal:
                futures=[pool.submit(evaluate_case,c,strategies,service,judges) for c in cases]
                for completed,future in enumerate(as_completed(futures),1):
                    batch=future.result();rows.extend(batch)
                    for r in batch:
                        journal.write(json.dumps(r,ensure_ascii=False)+'\n')
                        persist_faith_case(ledger,r)
                    journal.flush()
                    if completed%10==0 or completed==len(cases):
                        print(f'completed {completed}/{len(cases)} queries; errors={sum(bool(r["error"]) for r in rows)}',flush=True)
        rows.sort(key=lambda r:(r['eval_id'],r['strategy']))
        result=report(rows,'live' if args.live else 'synthetic_fixture',metadata)
        if args.calibrate:
            result['calibration']=calibration_threshold(rows)
        (run_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (run_dir/'report.md').write_text(markdown_report(result))
        print('report:',run_dir/'report.md',flush=True)
        return 1 if any(r['error'] for r in rows) else 0
    finally:
        if service:
            service.retriever.store.close();service.retriever.embeddings.close();service.retriever.reranker.close()
        for engine in resources:engine.dispose()


if __name__=='__main__':raise SystemExit(main())
