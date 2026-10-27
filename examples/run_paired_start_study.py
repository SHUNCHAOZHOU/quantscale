"""Preregistered three-block, four-topology paired-start continuation study."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil

from quantscale.config import Config
from quantscale.data import load_data, MarketData, prepare_partitions
from quantscale.deepseek_adapter import DeepSeekBackend
from quantscale.evaluation import evaluate, implementation_hash
from quantscale.research import run_research
from .paired_research import run_continuation, state_digest

ROOT=Path('results/sp500-deepseek-paired-start')
TOPOLOGIES=('independent','centralized','decentralized','hybrid')
SEEDS=(107,117,127)


def source_hash():
    files=[Path(__file__),Path(__file__).with_name('paired_research.py')]
    h=hashlib.sha256()
    for path in sorted(files):h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def usage(root):
    events=[]
    if (root/'trajectory.jsonl').exists():
        events=[json.loads(x) for x in (root/'trajectory.jsonl').read_text().splitlines()]
    replies=[x for x in events if x['kind']=='response']
    pending=[]
    for e in events:
        if e['kind']=='request':pending.append(e['request'])
        elif e['kind']=='response':pending.pop(0)
        elif e['kind']=='budget_exhausted' and pending and 'conservative DeepSeek input reservation' in e['reason']:pending.pop()
    return dict(confirmed_tokens=sum(x['tokens'] for x in replies),model_calls=len(replies),
        candidate_attempts=sum(x['kind']=='trial' for x in events),
        unconfirmed_requests=[dict(role=q['role'],max_total_tokens=q['max_total_tokens']) for q in pending])


def run():
    cfg=replace(Config.load('configs/sp500-open-v3.json'),max_turn_tokens=131072)
    old=Path('results/sp500-deepseek-open-pilot-v3')
    old_protocol=json.loads((old/'single-n1-s7/protocol.json').read_text())
    backend=DeepSeekBackend()
    assert backend.configuration==old_protocol['backend_configuration']
    assert implementation_hash()==json.loads((old/'pilot-protocol.json').read_text())['implementation_sha256']
    ROOT.mkdir(exist_ok=False)
    code_hash=implementation_hash();study_hash=source_hash()
    schedule=[dict(seed=seed,topology=top) for seed in SEEDS for top in TOPOLOGIES]
    random.Random(20261026).shuffle(schedule)
    protocol=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        design='Three independent common starts, each split into four paired continuations',
        seeds=list(SEEDS),topologies=list(TOPOLOGIES),schedule=schedule,concurrent_systems=4,
        initial_candidates_per_block=4,additional_candidates_per_branch=8,scheduled_continuation_rounds=2,
        initial_token_cap=270000,branch_token_cap=400000,total_known_token_cap=5610000,
        config=cfg.to_dict(),implementation_sha256=code_hash,study_source_sha256=study_hash,
        backend_configuration=backend.configuration,
        estimand='Mechanism under matched initial states and scheduled candidate opportunities; actual tokens are an outcome',
        rng_note='Seeds are repetition labels; provider randomness is uncontrolled',
        failure_policy='No replacement runs; errors consume attempts; failed manager rounds are skipped, not replaced',
        selection='Best validation Sharpe across initial four plus generated continuation; SHA tie-break',
        evaluation_policy='Finish all research and freeze roster before baseline or branch evaluation; each artifact once',
        historical_note='2024-2025 historical test was previously viewed; not a fresh blind test',
        differences_from_v3='max_total_tokens per request raised 65536->131072; token protective caps raised; exactly two scheduled continuation rounds')
    (ROOT/'study-protocol.json').write_text(json.dumps(protocol,indent=2))
    (ROOT/'study-source').mkdir()
    for filename in ['run_paired_start_study.py','paired_research.py']:
        shutil.copy2(Path(__file__).with_name(filename),ROOT/'study-source'/filename)
    research_path,evaluator_path=prepare_partitions('data/sp500-yahoo-2016-2025/prices.csv',ROOT/'operator-data',cfg)
    frame=load_data(research_path,cfg);market=MarketData(frame,cfg)
    completed=[];failures=[];baseline_states={}

    def integrity():
        assert implementation_hash()==code_hash and source_hash()==study_hash

    def baseline(seed):
        integrity();name=f'baseline-s{seed}'
        result=run_research(research_path,replace(cfg,seed=seed,backtests=4,tokens=270000),ROOT/name,'independent',4,DeepSeekBackend())
        events=[json.loads(x) for x in (ROOT/name/'trajectory.jsonl').read_text().splitlines()]
        initial=[{k:v for k,v in e.items() if k not in ('sequence','kind')} for e in events if e['kind']=='trial']
        assert len(initial)==4 and {t['agent'] for t in initial}=={0,1,2,3}
        (ROOT/name/'common-state.json').write_text(json.dumps(initial,indent=2,allow_nan=False))
        return dict(run=name,seed=seed,topology='baseline',artifact_sha256=result['artifact_sha256'],
            validation_sharpe=result['winner']['validation']['sharpe'],process=result['process']),initial

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(baseline,s):s for s in SEEDS}
        for future in as_completed(futures):
            seed=futures[future]
            try:
                row,state=future.result();completed.append(row);baseline_states[seed]=state
                print('BASELINE FROZEN '+row['run'],flush=True)
            except Exception as exc:failures.append(dict(run=f'baseline-s{seed}',seed=seed,error=str(exc)))
            (ROOT/'research-progress.json').write_text(json.dumps(dict(completed=completed,failures=failures),indent=2))

    def branch(item):
        integrity();seed=item['seed'];top=item['topology'];name=f'{top}-s{seed}'
        result=run_continuation(frame,market,replace(cfg,seed=seed,backtests=12,tokens=400000),
            ROOT/name,top,baseline_states[seed],DeepSeekBackend(),study_hash)
        assert json.loads((ROOT/name/'protocol.json').read_text())['initial_state_sha256']==state_digest(baseline_states[seed])
        return dict(run=name,seed=seed,topology=top,artifact_sha256=result['artifact_sha256'],
                    validation_sharpe=result['winner']['validation']['sharpe'],process=result['process'])

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(branch,item):item for item in schedule if item['seed'] in baseline_states}
        for future in as_completed(futures):
            item=futures[future];name=f"{item['topology']}-s{item['seed']}"
            try:row=future.result();completed.append(row);print('BRANCH FROZEN '+name,flush=True)
            except Exception as exc:failures.append(dict(run=name,**item,error=str(exc)));print('FAILED '+name+': '+str(exc),flush=True)
            (ROOT/'research-progress.json').write_text(json.dumps(dict(completed=completed,failures=failures),indent=2))
    integrity()
    roster=dict(frozen_before_any_evaluation_utc=datetime.now(timezone.utc).isoformat(),
        all_scheduled_research_finished=True,submissions={r['run']:r['artifact_sha256'] for r in completed},failures=failures)
    (ROOT/'all-strategies-frozen.json').write_text(json.dumps(roster,indent=2))
    results=[];evaluation_failures=[]
    for row in sorted(completed,key=lambda r:(r['seed'],r['topology'])):
        try:
            result=evaluate(ROOT/row['run']/'frozen.json',evaluator_path,ROOT/'hidden-evaluation'/row['run'])
            results.append(dict(**row,evaluation=result))
            print('EVALUATED '+row['run']+' hidden_sharpe=%.4f'%result['hidden']['sharpe'],flush=True)
        except Exception as exc:
            evaluation_failures.append(dict(run=row['run'],error=str(exc)))
            print('EVALUATION FAILED '+row['run']+': '+str(exc),flush=True)
        (ROOT/'evaluation-progress.json').write_text(json.dumps(dict(results=results,failures=evaluation_failures),indent=2))
    spend={p.parent.name:usage(p.parent) for p in ROOT.glob('*/trajectory.jsonl')}
    assert sum(r['confirmed_tokens'] for r in spend.values())<=5610000
    final=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),planned_baselines=3,planned_branches=12,
        research_completed=completed,research_failures=failures,evaluation_failures=evaluation_failures,
        results=results,usage=spend,confirmed_tokens=sum(r['confirmed_tokens'] for r in spend.values()),
        unconfirmed_requests=sum(len(r['unconfirmed_requests']) for r in spend.values()))
    (ROOT/'completed.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    print('STUDY COMPLETE confirmed_tokens='+str(final['confirmed_tokens']),flush=True)


if __name__=='__main__':run()
