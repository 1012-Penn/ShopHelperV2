"""Compare four strategies on the isolated labeled corpus; preserve every run."""
import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from langchain_openai import ChatOpenAI
from app.config import Settings
from app.db.session import create_tables,make_engine,make_session_factory
from app.services.knowledge.repository import KnowledgeRepository
from app.services.quality.index import HybridIndexer
from app.services.quality.runtime import build_answer_service,config
from app.services.quality.evaluation import retrieval_metrics,summarize,FaithfulnessJudge,judge_faithfulness,calibration_threshold
from app.services.quality.ledger import QualityLedger
from scripts.validate_ch04_dataset import load_corpus,validate_dataset

STRATEGIES=['dense','bm25','hybrid','hybrid_rerank']


def report(rows,mode,metadata):
    summaries={}
    groups={}
    for strategy in STRATEGIES:
        selected=[r for r in rows if r['strategy']==strategy]
        if not selected:continue
        summaries[strategy]=summarize(selected)
        groups[strategy]={'bucket':{b:summarize([r for r in selected if r['bucket']==b]) for b in sorted({r['bucket'] for r in selected})},'difficulty':{d:summarize([r for r in selected if r['difficulty']==d]) for d in ['easy','medium','hard']},'split':{split:summarize([r for r in selected if r['split']==split]) for split in ['calibration','test']}}
    return {'mode':mode,'metadata':metadata,'summary':summaries,'groups':groups,'rows':rows}


def markdown_report(result):
    lines=['# ch04 四策略评估', '', '模式：'+result['mode'], '', '| 策略 | 题数 | Recall@10 | MRR | Faithfulness | 忠实度样本 | 错误 |','|---|---:|---:|---:|---:|---:|---:|']
    for strategy,m in result['summary'].items():
        lines.append(f"| {strategy} | {m['cases']} | {m.get('recall@10')} | {m.get('mrr')} | {m['faithfulness']} | {m['faithfulness_cases']} | {m['error_count']} |")
    for strategy,groups in result['groups'].items():
        lines+=['', '## '+strategy+' 按类型', '', '| 桶 | Recall@10 | MRR | Faithfulness | 未知拒答率 | 已知误拒答率 |','|---|---:|---:|---:|---:|---:|']
        for bucket,m in groups['bucket'].items():
            lines.append(f"| {bucket} | {m.get('recall@10')} | {m.get('mrr')} | {m['faithfulness']} | {m['unknown_refusal_rate']} | {m['known_false_refusal_rate']} |")
    lines+=['','拒答无事实声明记 N/A，不计为忠实度通过。错误单独计数。合计与 test 分组分别保存于 JSON。','数据来自受控业务基线与虚构测试型号；未经业务专家独立复核。生成与默认裁判使用同一模型，存在自评偏差。']
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
    judge_model=values.get('JUDGE_MODEL',settings.model)
    judge=FaithfulnessJudge(ChatOpenAI(model=judge_model,api_key=values.get('JUDGE_API_KEY',settings.api_key),base_url=values.get('JUDGE_API_BASE',settings.base_url),temperature=0,timeout=90,max_retries=1))
    ledger_engine=make_engine(settings.database_url)
    create_tables(ledger_engine)
    ledger=QualityLedger(make_session_factory(ledger_engine))
    return service,judge,ledger,judge_model,[eval_engine,ledger_engine]


def fixture_rows(cases,strategies):
    rows=[]
    for c in cases:
        for strategy in strategies:
            ranking=c['relevant_source_keys'] if not c['should_refuse'] else []
            rows.append({**c,'strategy':strategy,'refused':c['should_refuse'],'answer':'fixture only','citations':[],'faithfulness':None if c['should_refuse'] else 1.,'error':None,'metrics':retrieval_metrics(ranking,set(c['relevant_source_keys'])),'top_score':.9 if ranking else 0,'latency_seconds':0})
    return rows


def evaluate_case(case,strategies,service,judge,judge_model):
    rows=[]
    for strategy in strategies:
        started=time.monotonic()
        row={**case,'strategy':strategy,'error':None,'refused':False,'faithfulness':None,'metrics':{},'answer':'','citations':[],'top_score':0}
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
            if not answer.refused:
                stage='judge';judged_at=time.monotonic()
                verdict=judge_faithfulness(judge,case['query'],answer.answer,answer.citations)
                row['judge_seconds']=time.monotonic()-judged_at
                row['faithfulness']=verdict['score']
                row['judge_reason']=verdict['reason'];row['claims']=verdict['claims']
                row['judge_model']=judge_model
        except Exception as error:
            row['error']=type(error).__name__
            row['error_stage']=stage
        row['latency_seconds']=time.monotonic()-started
        rows.append(row)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live',action='store_true');mode.add_argument('--fixture',action='store_true')
    p.add_argument('--split',choices=['all','calibration','test'],default='all')
    p.add_argument('--strategy',choices=['all',*STRATEGIES],default='all')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--limit',type=int)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--dataset-dir', type=Path, default=Path('evaluation/ch04'))
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args(argv)
    if not 1<=args.workers<=24: p.error('workers must be 1..24')
    dataset_file = args.dataset_dir / 'cases.json'
    cases=json.loads(dataset_file.read_text())
    drafts=load_corpus(args.dataset_dir / 'corpus');validate_dataset(cases,{d.source_key for d in drafts})
    cases=[c for c in cases if args.split=='all' or c['split']==args.split]
    if args.limit:cases=cases[:args.limit]
    strategies=STRATEGIES if args.strategy=='all' else [args.strategy]
    run_dir=(args.output_dir or args.dataset_dir / 'runs')/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8]);run_dir.mkdir(parents=True)
    database_url, collection = evaluation_targets(args.dataset_dir, config())
    metadata={'cases':len(cases),'strategies':strategies,'split':args.split,'dataset_dir':str(args.dataset_dir),'dataset_sha256':hashlib.sha256(dataset_file.read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256(''.join(d.source_key+d.answer for d in drafts).encode()).hexdigest(), 'corpus_chunks':len(drafts), 'collection':collection, 'evaluation_database':database_url.rsplit('@',1)[-1]}
    rows=[];resources=[];service=None;ledger=None
    try:
        if args.fixture:
            rows=fixture_rows(cases,strategies)
        else:
            service,judge,ledger,judge_model,resources=build_live(args.dataset_dir, drafts)
            metadata.update(generator_model=Settings.from_env().model,judge_model=judge_model,reranker=service.retriever.reranker.model,collection=service.retriever.store.collection_name,rerank_min_score=service.min_score)
            with ThreadPoolExecutor(max_workers=args.workers) as pool, (run_dir/'rows.jsonl').open('w') as journal:
                futures=[pool.submit(evaluate_case,c,strategies,service,judge,judge_model) for c in cases]
                for completed,future in enumerate(as_completed(futures),1):
                    batch=future.result();rows.extend(batch)
                    for r in batch:
                        journal.write(json.dumps(r,ensure_ascii=False)+'\n')
                        if r['strategy']=='hybrid_rerank' and r['faithfulness'] is not None and r['faithfulness']<1 and not r['error']:
                            ledger.record_faith_case(dict(eval_id=r['eval_id'],bucket=r['bucket'],query=r['query'],strategy=r['strategy'],answer=r['answer'],reason=r['judge_reason'],citations=r['citations'],judge_model=judge_model))
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
