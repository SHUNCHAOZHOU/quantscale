"""Supplement only: preserve deterministic best observed strategy after transport abort."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from quantscale.evaluation import evaluate, freeze

ROOT=Path('results/sp500-deepseek-paired-start')
TARGET=ROOT/'partial-recovery/decentralized-s107'


def preserve():
    # Declare this deterministic supplement before any main-study test revelation.
    assert not (ROOT/'all-strategies-frozen.json').exists()
    source=ROOT/'decentralized-s107'
    raw=(source/'trajectory.jsonl').read_bytes();events=[json.loads(x) for x in raw.decode().splitlines()]
    assert events[-1]['kind']=='aborted' and not (source/'frozen.json').exists()
    trials=[{k:v for k,v in x.items() if k not in ('kind','sequence')} for x in events if x['kind'] in ('inherited_trial','trial')]
    valid=[t for t in trials if 'validation' in t]
    winner=sorted(valid,key=lambda t:(-t['validation']['sharpe'],t['strategy_id']))[0]
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(supplemental_partial_recovery=True,primary_run='decentralized-s107',
        recovery_declared_at_utc=datetime.now(timezone.utc).isoformat(),
        observed_additional_attempts=sum(x['kind']=='trial' for x in events),
        source_journal_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Supplemental partial result, not included in primary paired contrasts; no extra API calls or candidate attempts',
        selection='Deterministic highest observed validation Sharpe including initial candidates; SHA ascending tie-break')
    TARGET.mkdir(parents=True,exist_ok=False)
    (TARGET/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (TARGET/'selection.json').write_text(json.dumps(winner,indent=2,allow_nan=False))
    artifact=freeze(TARGET,protocol,winner)
    print('PARTIAL SUPPLEMENT FROZEN BEFORE MAIN EVALUATION: validation=%.4f'%winner['validation']['sharpe'])
    return artifact


def test():
    assert (ROOT/'completed.json').exists()
    result=evaluate(TARGET/'frozen.json',ROOT/'operator-data/evaluator.csv',TARGET/'evaluation')
    (ROOT/'partial-recovery/result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(sharpe=result['hidden']['sharpe'],annualized_return=result['hidden']['annualized_return'],max_drawdown=result['hidden']['max_drawdown']),indent=2))


if __name__=='__main__':
    test() if '--evaluate' in sys.argv else preserve()
