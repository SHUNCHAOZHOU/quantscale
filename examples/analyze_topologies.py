"""Post-hoc topology mechanisms from requests and saved PnL, without reruns."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(root, records=None, aggregates=None, output='topology-analysis.json'):
    root=Path(root)
    if records is None:
        comparison=json.loads((root/'rerun-comparison.json').read_text())['new']
        records,aggregates=comparison['runs'],comparison['aggregates']
    runs=[]
    paths={}
    spy=pd.read_csv('data/sp500-yahoo-2016-2025/spy.csv',parse_dates=['date']).set_index('date').adjusted_open
    for record in records:
        name=record['run']
        research=Path(record.get('research_dir',root/name))
        evaluation=Path(record.get('evaluation_dir',root/'hidden-evaluation'/name))
        events=[json.loads(x) for x in (research/'trajectory.jsonl').read_text().splitlines()]
        trials=[x for x in events if x['kind']=='trial']
        valid=[x for x in trials if 'validation' in x]
        first_round=min((x['round'] for x in valid),default=None)
        first=[x for x in valid if x['round']==first_round]
        later=[x for x in valid if first_round is not None and x['round']>first_round]
        r=dict(run=name,label=record['label'],topology=record['topology'],status=record['status'],
               trials=len(trials),valid=len(valid),first_round=first_round,
               first_valid=len(first),later_valid=len(later),
               first_best=max((x['validation']['sharpe'] for x in first),default=None),
               later_best=max((x['validation']['sharpe'] for x in later),default=None),
               worker_attempts=dict(Counter(str(x['agent']) for x in trials)),
               worker_valid=dict(Counter(str(x['agent']) for x in valid)))
        if later:
            r['later_minus_first_best']=r['later_best']-r['first_best']
        if (research/'selection.json').exists():
            winner=json.loads((research/'selection.json').read_text())
            request=next(x['request'] for x in events if x['kind']=='request' and x['request']['role']=='researcher'
                         and x['request']['round']==winner['round'] and x['request']['agent']==winner['agent'])
            r.update(winner_worker=winner['agent'],winner_round=winner['round'],
                     winner_peers=len(request['peer_results']),
                     winner_own_valid=sum('validation' in x for x in request['own_results']),
                     winner_manager_direction=request.get('direction'),
                     winner_validation=winner['validation']['sharpe'],
                     own_previous_valid=[dict(worker=x['agent'],round=x['round'],sharpe=x['validation']['sharpe']) for x in request['own_results'] if 'validation' in x],
                     peer_previous_valid=[dict(worker=x['agent'],round=x['round'],sharpe=x['validation']['sharpe']) for x in request['peer_results']],
                     last_trial_round=max(x['round'] for x in trials),
                     last_round_workers=[x['agent'] for x in trials if x['round']==max(t['round'] for t in trials)])
        if record['status']=='evaluated':
            daily=pd.read_csv(evaluation/'hidden_daily.csv',parse_dates=['date','execution_date'])
            assert len(daily)==500
            paths[name]=daily.set_index('date').net_return
            market=spy.reindex(daily.date).to_numpy()/spy.reindex(daily.execution_date).to_numpy()-1
            assert np.isfinite(market).all()
            ret=daily.net_return.to_numpy()
            beta=float(np.cov(ret,market,ddof=1)[0,1]/np.var(market,ddof=1))
            corr=float(np.corrcoef(ret,market)[0,1])
            r.update(test=record['test'],years=record['years'],
                     market_beta=beta,market_correlation=corr,market_r_squared=corr*corr,
                     arithmetic_intercept_annualized=float((ret.mean()-beta*market.mean())*252),
                     mean_gross_exposure=float(daily.gross_exposure.mean()),
                     mean_absolute_net=float(daily.net_exposure.abs().mean()),
                     mean_turnover=float(daily.turnover.mean()),transaction_cost_sum=float(daily.cost.sum()))
        runs.append(r)
    returns=pd.DataFrame(paths)
    assert returns.notna().all().all()
    corr=returns.corr()
    pairs=[]
    for i,left in enumerate(corr.columns):
        for right in corr.columns[i+1:]:
            pairs.append(dict(left=left,right=right,correlation=float(corr.loc[left,right]),observations=len(returns)))
    means={x['topology']:x['test_sharpe_mean'] for x in aggregates if x['tested']}
    # Descriptive contrasts of selected-strategy Sharpe means. This is NOT a
    # randomized factorial estimate or a causal attribution to communication.
    factorial=dict(
        manager_without_peers=means['centralized']-means['independent'],
        manager_with_peers=means['hybrid']-means['decentralized'],
        peers_without_manager=means['decentralized']-means['independent'],
        peers_with_manager=means['hybrid']-means['centralized'],
        interaction=means['hybrid']-means['centralized']-means['decentralized']+means['independent'])
    payload=dict(runs=runs,correlation_order=list(corr.columns),correlation_matrix=corr.to_numpy().tolist(),
                 correlations=sorted(pairs,key=lambda x:x['correlation'],reverse=True),
                 factorial=factorial,scope='Post-hoc selected-strategy descriptions; no new inference, execution or causal experiment')
    (root/output).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    data=analyze(args.root)
    print(json.dumps(dict(factorial=data['factorial'],runs=[{k:v for k,v in r.items() if k not in ('winner_manager_direction','test','years')} for r in data['runs']]),ensure_ascii=False,indent=2))
