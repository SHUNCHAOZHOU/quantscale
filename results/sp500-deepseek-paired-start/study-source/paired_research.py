"""Paired-start continuation; separate from immutable V3 implementation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from quantscale.backtest import run_backtest
from quantscale.code_strategy import CodeSpec, SandboxedStrategy
from quantscale.data import fingerprint
from quantscale.evaluation import freeze, implementation_hash
from quantscale.manager_protocol import ManagerProtocolError, parse_manager, repair_reference
from quantscale.prompts import research_contract
from quantscale.research import Budget, BudgetExhausted, Journal, process_metrics


def state_digest(initial):
    return hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run_continuation(data,market,config,output,topology,initial,backend,study_hash):
    """Two scheduled rounds; failed manager rounds are not silently replaced."""
    if topology not in ('independent','centralized','decentralized','hybrid'):
        raise ValueError('Expected one of four multi-researcher topologies')
    assert len(initial)==4 and {t['agent'] for t in initial}=={0,1,2,3}
    assert all(t['round']==0 for t in initial)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);(root/'strategies').mkdir()
    journal=Journal(root/'trajectory.jsonl')
    trials=deepcopy(initial)
    histories=[[deepcopy(t) for t in initial if t['agent']==i] for i in range(4)]
    baseline_state=state_digest(initial)
    budget=Budget(config.tokens,config.backtests,backtests=4)
    protocol=dict(schema_version=1,config=config.to_dict(),topology=topology,fleet_size=4,
        manager_count=int(topology in ('centralized','hybrid')),research_data_hash=fingerprint(data),
        backend=backend.name,backend_configuration=backend.configuration,
        token_accounting=backend.token_accounting,seed=config.seed,
        execution='close t -> next open -> following open; each split starts flat',
        selection='highest net validation Sharpe including initial four; SHA256 ascending tie-break',
        budget_mode='paired-start, 8 additional candidate attempts in exactly two scheduled rounds',
        initial_state_sha256=baseline_state,study_source_sha256=study_hash,
        initial_cost_accounting='Initial four generated once, recorded separately; not charged four times',
        manager_failure_policy='Skip failed scheduled round; at most two scheduled rounds; retain best initial strategy',
        excluded_data_sessions=[str(d.date()) for d in market.excluded_dates])
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'initial-state.json').write_text(json.dumps(initial,indent=2,allow_nan=False))
    journal.write('protocol',**protocol)
    for t in trials:
        if 'strategy' in t:(root/'strategies'/f"{t['strategy_id']}.py").write_text(t['strategy']['source'])
        journal.write('inherited_trial',**t)
    metric_cache={t['strategy_id']:(t['train'],t['validation']) for t in trials if 'validation' in t}
    train=data[(data.date>=pd.Timestamp(config.train.start)) & (data.date<=pd.Timestamp(config.train.end))]
    inspection=dict(train_rows=len(train),train_assets=int(train.ticker.nunique()),
        train_start=str(train.date.min().date()),train_end=str(train.date.max().date()),
        fields=list(data.columns),median_daily_volume=float(train.volume.median()))
    turns=communications=manager_failures=0
    start=time.monotonic()

    def invoke(role,round_number,payload,communication=False):
        nonlocal turns,communications
        remaining=budget.token_limit-budget.tokens
        if remaining<=0:raise BudgetExhausted('Token budget exhausted')
        request=dict(role=role,round=round_number,fleet_size=4,worker_ids=['0','1','2','3'],
            seed=config.seed,max_total_tokens=min(config.max_turn_tokens,remaining),
            **research_contract(config),**payload)
        journal.write('request',request=request)
        response,usage=backend.invoke(request)
        journal.write('response',role=role,response=response,tokens=usage,communication=communication)
        budget.charge_tokens(usage,communication)
        if usage>request['max_total_tokens']:raise RuntimeError('Per-request token allowance breached')
        turns+=1;communications+=int(communication)
        return response

    try:
        for round_number in (1,2):
            # Immutable whole-round snapshot, same semantics as original V3.
            snapshot=[next((t for t in reversed(h) if 'validation' in t),None) for h in histories]
            snapshot=[t for t in snapshot if t is not None]
            directions={}
            if topology in ('centralized','hybrid'):
                original=invoke('manager',round_number,dict(worker_results=deepcopy(snapshot)),True)
                response=original;reference=repair_reference(original,4);directions=None
                for attempt in range(config.manager_max_format_repairs+1):
                    try:
                        parsed,actions=parse_manager(response,4,None)
                        if attempt and parsed!=reference:raise ManagerProtocolError('Format repair changed assignments')
                        directions=parsed
                        journal.write('manager_broadcast',directions=directions,normalization=actions,
                                      format_repair_attempts=attempt,round=round_number)
                        break
                    except ManagerProtocolError as exc:
                        journal.write('manager_format_error',round=round_number,repair_attempt=attempt,error=str(exc))
                        if reference is None or attempt>=config.manager_max_format_repairs:break
                        response=invoke('manager',round_number,dict(format_repair=dict(
                            original_response=original['raw_content'],error=str(exc),attempt=attempt+1,
                            constraint='Preserve all assignments verbatim; packaging corrections only')),True)
                if directions is None:
                    manager_failures+=1
                    journal.write('manager_round_failed',round=round_number,consecutive_failures=manager_failures)
                    if manager_failures>=config.manager_max_failed_rounds:
                        journal.write('manager_protocol_exhausted',reason='Consecutive manager format failures')
                        break
                    continue
                manager_failures=0
            for agent in range(4):
                peers=deepcopy([t for t in snapshot if t['agent']!=agent]) if topology in ('decentralized','hybrid') else []
                proposal=invoke('researcher',round_number,dict(agent=agent,inspection=inspection,
                    own_results=deepcopy(histories[agent][-3:]),peer_results=peers,direction=directions.get(str(agent))),bool(peers))
                budget.charge_backtest()
                trial=dict(agent=agent,round=round_number,hypothesis=str(proposal.get('hypothesis',''))[:4000])
                try:
                    if 'adapter_error' in proposal:raise ValueError(proposal['adapter_error'])
                    spec=CodeSpec(source=proposal['strategy_code'])
                    path=root/'strategies'/f'{spec.digest}.py';path.write_text(spec.source)
                    trial.update(strategy=spec.to_dict(),strategy_id=spec.digest,source_file=str(path.relative_to(root)),
                                 cached_backtest=spec.digest in metric_cache)
                    if spec.digest not in metric_cache:
                        policy=SandboxedStrategy(spec,config)
                        train_metrics,_=run_backtest(policy,market,config.train)
                        val_metrics,_=run_backtest(policy,market,config.validation)
                        metric_cache[spec.digest]=(train_metrics,val_metrics)
                    trial['train'],trial['validation']=metric_cache[spec.digest]
                except (ValueError,KeyError,TypeError,OverflowError) as exc:trial['error']=str(exc)
                histories[agent].append(trial);trials.append(trial);journal.write('trial',**trial)
                value=trial.get('validation',{}).get('sharpe')
                print(f"pair={config.seed} {topology} added={len(trials)-4}/8 r{round_number} W{agent} validation="+(f'{value:.4f}' if value is not None else 'FAILED'),flush=True)
    except BudgetExhausted as exc:
        journal.write('budget_exhausted',reason=str(exc))
    except Exception as exc:
        journal.write('aborted',error=str(exc),tokens=budget.tokens,backtests=budget.backtests)
        raise
    assert state_digest(initial)==baseline_state
    valid=[t for t in trials if 'validation' in t]
    if not valid:raise RuntimeError('No valid initial or continuation strategy')
    winner=sorted(valid,key=lambda t:(-t['validation']['sharpe'],t['strategy_id']))[0]
    (root/'selection.json').write_text(json.dumps(winner,indent=2,allow_nan=False))
    metrics=process_metrics([t for t in trials if 'strategy_id' in t],budget,turns,communications)
    metrics.update(inherited_trials=4,additional_trials=len(trials)-4,
        complete_eight_attempts=len(trials)==12,elapsed_seconds=time.monotonic()-start,
        baseline_best_validation=max((t['validation']['sharpe'] for t in initial if 'validation' in t),default=None),
        final_validation=winner['validation']['sharpe'],winner_from_initial=winner['round']==0)
    (root/'process.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    artifact=freeze(root,protocol,winner);journal.write('frozen',artifact_sha256=artifact['artifact_sha256'])
    return dict(winner=winner,process=metrics,artifact_sha256=artifact['artifact_sha256'])
