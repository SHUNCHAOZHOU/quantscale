"""Post-hoc report from frozen artifacts and saved returns; no model calls or reruns."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from quantscale.backtest import performance
from quantscale.evaluation import digest


ORDER = ['single', 'independent', 'centralized', 'decentralized', 'hybrid']
NAMES = dict(single='单 Agent', independent='独立并行', centralized='中心化',
             decentralized='去中心化', hybrid='混合')
LABELS = {
    'single-n1-s7': '单 Agent · 7', 'single-n1-s17': '单 Agent · 17',
    'independent-n4-s7': '独立并行 · 7', 'independent-n4-s17': '独立并行 · 17',
    'decentralized-n4-s7': '去中心化 · 7', 'decentralized-n4-s17': '去中心化 · 17',
    'hybrid-n4-s7': '混合 · 7',
}


def summarize_returns(values):
    values = np.asarray(values)
    return dict(sharpe=float(values.mean() / values.std(ddof=1) * np.sqrt(252)),
                annualized_return=float(np.prod(1 + values) ** (252 / len(values)) - 1),
                total_return=float(np.prod(1 + values) - 1))


def analyze(root, samples=5000, block=20):
    root = Path(root)
    raw = json.loads((root / 'open-report-data.json').read_text())
    protocol = raw['protocol']
    barrier = json.loads((root / 'all-strategies-frozen.json').read_text())
    paths, runs, histories, code_ids = {}, [], [], []
    calendar = []
    reasoning = completion = truncated_only_reasoning = 0
    spy = pd.read_csv('data/sp500-yahoo-2016-2025/spy.csv', parse_dates=['date']).set_index('date').adjusted_open
    for item in raw['runs']:
        record = dict(item)
        name = item['run']
        events = [json.loads(x) for x in (root / name / 'trajectory.jsonl').read_text().splitlines()]
        trials = [x for x in events if x['kind'] == 'trial']
        responses = [x for x in events if x['kind'] == 'response']
        record['manager_tokens'] = sum(x['tokens'] for x in responses if x['role'] == 'manager')
        record['communication_tokens_upper_bound'] = sum(x['tokens'] for x in responses if x['communication'])
        record['generated_code'] = sum('strategy_id' in x for x in trials)
        record['pure_reasoning_truncations'] = 0
        for event in responses:
            usage = event['response'].get('_provenance', {}).get('usage', {})
            ct = usage.get('completion_tokens', 0)
            rt = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
            completion += ct
            reasoning += rt
            if 'did not finish: length' in event['response'].get('adapter_error', '') and ct == rt:
                truncated_only_reasoning += 1
                record['pure_reasoning_truncations'] += 1
        record['worker_trials'] = dict(Counter(str(x['agent']) for x in trials))
        record['worker_valid'] = dict(Counter(str(x['agent']) for x in trials if 'validation' in x))
        for index, trial in enumerate(trials, 1):
            if 'strategy_id' in trial:
                code_ids.append(trial['strategy_id'])
            histories.append(dict(run=name, attempt=index, worker=trial['agent'], round=trial['round'],
                                  valid='validation' in trial, error=trial.get('error'),
                                  train_sharpe=trial.get('train', {}).get('sharpe'),
                                  validation_sharpe=trial.get('validation', {}).get('sharpe'),
                                  strategy_id=trial.get('strategy_id'), hypothesis=trial['hypothesis']))
        if item['status'] != 'evaluated':
            runs.append(record)
            continue
        evaluation = json.loads((root / 'hidden-evaluation' / name / 'evaluation.json').read_text())
        selection = json.loads((root / name / 'selection.json').read_text())
        winner_request = next(x['request'] for x in events if x['kind'] == 'request'
                              and x['request']['role'] == 'researcher'
                              and x['request']['agent'] == selection['agent']
                              and x['request']['round'] == selection['round'])
        artifact = json.loads((root / name / 'frozen.json').read_text())
        assert digest(artifact['payload']) == barrier['submissions'][name] == evaluation['artifact_sha256']
        assert artifact['payload']['implementation_sha256'] == protocol['implementation_sha256']
        assert datetime.fromisoformat(evaluation['evaluated_at']) > datetime.fromisoformat(barrier['frozen_before_any_evaluation_utc'])
        daily = pd.read_csv(root / 'hidden-evaluation' / name / 'hidden_daily.csv',
                            parse_dates=['date', 'execution_date', 'decision_date'])
        assert len(daily) == 500
        assert np.allclose(daily.net_return, daily.gross_return - daily.cost - daily.borrow_cost)
        assert np.allclose(daily.cost, daily.turnover * .0005)
        for field, val in performance(daily).items():
            assert np.isclose(val, evaluation['hidden'][field]), (name, field)
        paths[name] = daily.set_index('date').net_return
        record.update(label=LABELS[name], train_sharpe=selection['train']['sharpe'],
                      winner_attempt=next(i for i, x in enumerate(trials, 1) if x.get('strategy_id') == selection['strategy_id']),
                      winner_worker=selection['agent'], winner_round=selection['round'],
                      winner_peer_results=len(winner_request['peer_results']),
                      winner_own_valid_results=sum('validation' in x for x in winner_request['own_results']),
                      winner_received_manager_direction=bool(winner_request['direction']),
                      test=evaluation['hidden'],
                      gross=summarize_returns(daily.gross_return),
                      mean_gross_exposure=float(daily.gross_exposure.mean()),
                      mean_abs_net_exposure=float(daily.net_exposure.abs().mean()),
                      gross_positive_days=int((daily.gross_return > 0).sum()),
                      net_positive_days=int((daily.net_return > 0).sum()))
        for year, metrics in evaluation['by_calendar_year'].items():
            calendar.append(dict(run=name, label=LABELS[name], year=year, **metrics))
        nav = np.cumprod(1 + daily.net_return.to_numpy())
        peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
        dd = 1 - nav / peaks
        trough = int(np.argmax(dd))
        peak = int(np.argmax(np.r_[1., nav[:trough + 1]])) - 1
        peak_level = peaks[trough]
        recovery = np.flatnonzero(nav[trough + 1:] >= peak_level)
        record['drawdown_episode'] = dict(
            peak=daily.execution_date.iloc[0].strftime('%Y-%m-%d') if peak < 0 else daily.date.iloc[peak].strftime('%Y-%m-%d'),
            trough=daily.date.iloc[trough].strftime('%Y-%m-%d'),
            recovery=daily.date.iloc[trough + 1 + recovery[0]].strftime('%Y-%m-%d') if recovery.size else None,
            end_below_peak=float(1 - nav[-1] / max(1., nav.max())))
        monthly = daily.assign(nav=nav, drawdown=dd).groupby(daily.date.dt.to_period('M')).tail(1)
        record['monthly_nav'] = dict(dates=['2024-01-02'] + monthly.date.dt.strftime('%Y-%m-%d').tolist(),
                                     values=[1.] + monthly.nav.tolist())
        record['cost_sensitivity'] = []
        for cost_bps in [0, 2.5, 5, 10, 20]:
            # Fixed saved gross PnL and turnover. Does not rerun execution/NAV drift.
            adjusted = daily.gross_return - daily.borrow_cost - daily.turnover * cost_bps / 10000
            record['cost_sensitivity'].append(dict(cost_bps=cost_bps, **summarize_returns(adjusted)))
        record['arithmetic_breakeven_cost_bps'] = float((daily.gross_return - daily.borrow_cost).sum() / daily.turnover.sum() * 10000)
        market = spy.reindex(daily.date).to_numpy() / spy.reindex(daily.execution_date).to_numpy() - 1
        assert np.isfinite(market).all()
        ret = daily.net_return.to_numpy()
        beta = float(np.cov(ret, market, ddof=1)[0, 1] / np.var(market, ddof=1))
        record['spy'] = dict(beta=beta, correlation=float(np.corrcoef(ret, market)[0, 1]),
                             arithmetic_intercept_annualized=float((ret.mean() - beta * market.mean()) * 252),
                             **summarize_returns(market))
        runs.append(record)
    returns = pd.DataFrame(paths)
    assert returns.notna().all().all()
    valid_runs = [x for x in runs if x['status'] == 'evaluated']
    aggregates = []
    for topology in ORDER:
        selected = [x for x in runs if x['topology'] == topology]
        evaluated = [x for x in selected if x['status'] == 'evaluated']
        tokens = sum(x['tokens'] for x in selected)
        row = dict(topology=topology, label=NAMES[topology], planned=len(selected), tested=len(evaluated),
                   attempted=sum(x['trials'] for x in selected), valid=sum(x['valid'] for x in selected),
                   generated_code=sum(x['generated_code'] for x in selected), tokens=tokens,
                   truncated_calls=sum(x['truncated_calls'] for x in selected),
                   manager_tokens=sum(x['manager_tokens'] for x in selected),
                   communication_tokens_upper_bound=sum(x['communication_tokens_upper_bound'] for x in selected))
        row['valid_per_100k_tokens'] = row['valid'] / tokens * 100000
        if evaluated:
            for key in ['sharpe', 'annualized_return', 'max_drawdown']:
                row['test_' + key + '_mean'] = float(np.mean([x['test'][key] for x in evaluated]))
            row['validation_sharpe_mean'] = float(np.mean([x['validation_sharpe'] for x in evaluated]))
        aggregates.append(row)
    # Synchronous circular moving-block resampling retains cross-strategy correlation.
    # This is conditional time-series uncertainty of fixed selected strategies,
    # not an estimate of variation across independent LLM research runs.
    rng = np.random.default_rng(20261026)
    n = len(returns)
    bootstrap = {}
    values = returns.to_numpy()
    cols = list(returns.columns)
    draws = np.empty((samples, len(cols)))
    for b in range(samples):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        v = values[idx]
        draws[b] = v.mean(axis=0) / v.std(axis=0, ddof=1) * np.sqrt(252)
    for record in valid_runs:
        i = cols.index(record['run'])
        ci = np.quantile(draws[:, i], [.025, .975])
        record['sharpe_block_interval'] = ci.tolist()
    for topology in ORDER:
        ids = [cols.index(x['run']) for x in valid_runs if x['topology'] == topology]
        if ids:
            bootstrap[topology] = draws[:, ids].mean(axis=1)
    contrasts = []
    for other in ['single', 'independent', 'hybrid']:
        delta = bootstrap['decentralized'] - bootstrap[other]
        estimate = next(x for x in aggregates if x['topology']=='decentralized')['test_sharpe_mean'] - next(x for x in aggregates if x['topology']==other)['test_sharpe_mean']
        contrasts.append(dict(other=other, estimate=estimate, low=float(np.quantile(delta, .025)), high=float(np.quantile(delta, .975))))
    valid_history = [x for x in histories if x['valid']]
    connection = sqlite3.connect('file:' + str((root / 'operator-data' / '.quantscale-evaluations.sqlite3').resolve()) + '?mode=ro', uri=True)
    states = connection.execute('SELECT status,COUNT(*) FROM evaluations GROUP BY status').fetchall()
    connection.close()
    assert states == [('complete', 7)]
    audit = dict(frozen_submissions=7, missing_submissions=3, registry_complete=7,
                 hashes_verified=True, tests_after_freeze=True, all_daily_metrics_recomputed=True,
                 all_budgets_respected=all(x['tokens']<=170000 and x['trials']<=12 for x in runs),
                 excluded_sessions=['2021-05-05', '2025-11-12'], observations_per_test=500)
    payload = dict(root=str(root.resolve()), protocol=protocol, completed=raw['completed'],
                   runs=runs, aggregates=aggregates, calendar=calendar, histories=histories,
                   correlations=raw['correlations'], audit=audit,
                   bootstrap=dict(samples=samples, block=block, seed=20261026, contrasts=contrasts),
                   diagnostics=dict(reasoning_tokens=reasoning, completion_tokens=completion,
                                    reasoning_share=reasoning/completion,
                                    truncated_only_reasoning=truncated_only_reasoning,
                                    generated_code=len(code_ids), unique_code=len(set(code_ids)),
                                    valid_candidates=len(valid_history),
                                    positive_validation_candidates=sum(x['validation_sharpe']>0 for x in valid_history),
                                    positive_test_sharpe=sum(x['test']['sharpe']>0 for x in valid_runs),
                                    positive_test_compound=sum(x['test']['total_return']>0 for x in valid_runs)))
    target = root / 'detailed-analysis.json'
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    result = analyze(args.root)
    print(json.dumps(dict(audit=result['audit'], diagnostics=result['diagnostics'],
                          contrasts=result['bootstrap']['contrasts'],
                          summary=[dict(run=x['run'], test=x['test'], spy=x['spy'],
                                        interval=x['sharpe_block_interval'], costs=x['cost_sensitivity'],
                                        drawdown=x['drawdown_episode']) for x in result['runs'] if x['status']=='evaluated']), ensure_ascii=False, indent=2))
