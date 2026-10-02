"""Judge identities and replay: no retrieval or answer generation here."""
import json
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from langchain_openai import ChatOpenAI
from app.config import Settings
from app.services.quality.runtime import config
from app.services.quality.evaluation import (
    FaithfulnessJudge, CorrectnessJudge, judge_faithfulness, judge_correctness,
)


def public_endpoint(value):
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or '').lower()
    if parsed.port and (parsed.scheme.lower(), parsed.port) not in {('https',443),('http',80)}:
        host += f':{parsed.port}'
    path = parsed.path.rstrip('/')
    if path.endswith('/v1'):
        path = path[:-3]
    return urlunsplit((parsed.scheme.lower(), host, path, '', ''))


def judge_identity(settings, values, require_independent=False, generator_identity=None):
    generator = generator_identity or {
        'generator_model':settings.model, 'generator_base_url':public_endpoint(settings.base_url),
    }
    judge_model = values.get('JUDGE_MODEL') or settings.model
    judge_base = public_endpoint(values.get('JUDGE_API_BASE') or settings.base_url)
    generator_model = generator.get('generator_model')
    generator_base = generator.get('generator_base_url')
    same = None if not generator_model or not generator_base else (
        judge_model.strip().lower() == generator_model.strip().lower()
        and judge_base == public_endpoint(generator_base))
    if require_independent and (not values.get('JUDGE_MODEL') or same is not False):
        raise ValueError('independent judge requires explicit model and distinct, known generator/judge identity')
    return {**generator, 'judge_model':judge_model, 'judge_base_url':judge_base,
            'same_model_judge':same}


def build_judges(settings=None, values=None, require_independent=False, generator_identity=None):
    settings = settings or Settings.from_env()
    values = values if values is not None else config()
    identity = judge_identity(settings, values, require_independent, generator_identity)
    model = ChatOpenAI(model=identity['judge_model'],
                       api_key=values.get('JUDGE_API_KEY') or settings.api_key,
                       base_url=values.get('JUDGE_API_BASE') or settings.base_url,
                       temperature=0, timeout=90, max_retries=1)
    return FaithfulnessJudge(model), CorrectnessJudge(model), identity


def score_answer(row, faith_judge, correctness_judge, judge_model):
    row.update(faithfulness=None, correctness=None, fact_coverage=None,
               judge_model=judge_model, claims=[], correctness_facts=[],
               faithfulness_error=None, correctness_error=None, error=None)
    row.pop('error_stage', None)
    failures=[]
    if not row['refused']:
        started = time.monotonic()
        try:
            verdict = judge_faithfulness(faith_judge, row['query'], row['answer'], row['citations'])
            row.update(faithfulness=verdict['score'], judge_reason=verdict['reason'], claims=verdict['claims'])
        except Exception as e:
            row['faithfulness_error']=type(e).__name__
            failures.append(('faithfulness_judge',type(e).__name__))
        row['judge_seconds'] = time.monotonic()-started
    started = time.monotonic()
    try:
        verdict = judge_correctness(correctness_judge, row['query'], row['answer'],
                                     row['required_facts'], row['should_refuse'], row['refused'])
        row.update(correctness=verdict['score'], fact_coverage=verdict['fact_coverage'],
                   correctness_reason=verdict['reason'], correctness_facts=verdict['facts'])
    except Exception as e:
        row['correctness_error']=type(e).__name__
        failures.append(('correctness_judge',type(e).__name__))
    row['correctness_judge_seconds'] = time.monotonic()-started
    if failures:
        row['error_stage'],row['error']=failures[0]
    return row


def load_replay_rows(path):
    path = Path(path)
    metadata_path = path.parent / 'report.json'
    metadata = json.loads(metadata_path.read_text()).get('metadata', {}) if metadata_path.exists() else {}
    rows = []
    required = {'eval_id', 'query', 'strategy', 'split', 'bucket', 'difficulty',
                'should_refuse', 'required_facts', 'answer', 'citations', 'refused', 'metrics'}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f'invalid JSON at row {number}') from e
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f'invalid replay fields at row {number}')
        if (not isinstance(row['answer'],str) or not isinstance(row['citations'],list)
                or not isinstance(row['required_facts'],list)
                or not isinstance(row['metrics'],dict)
                or type(row['refused']) is not bool or type(row['should_refuse']) is not bool):
            raise ValueError(f'invalid replay types at row {number}')
        for key in ['generator_model','generator_base_url']:
            row.setdefault(key, metadata.get(key))
        rows.append(row)
    if not rows:
        raise ValueError('replay requires nonempty rows')
    pairs = [(r['eval_id'],r['strategy']) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError('duplicate replay question/strategy')
    return rows, metadata


def rejudge_row(row, judges):
    faith_judge, correctness_judge, identity = judges
    result = dict(row)
    result['previous_error'] = result.get('error')
    result['previous_error_stage'] = result.get('error_stage')
    if result.get('error') and result.get('error_stage') not in {'judge','faithfulness_judge','correctness_judge'}:
        return result
    result['error'] = None
    result['judge_base_url'] = identity.get('judge_base_url')
    result['same_model_judge'] = identity.get('same_model_judge')
    started = time.monotonic()
    try:
        score_answer(result, faith_judge, correctness_judge, identity['judge_model'])
    except Exception as e:
        result['error'] = type(e).__name__
    result['rejudge_seconds'] = time.monotonic()-started
    return result
