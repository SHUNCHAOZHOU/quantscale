"""Reconstruct three target-weight snapshots; no returns evaluation or selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantscale.code_strategy import CodeSpec, SandboxedStrategy
from quantscale.config import Config
from quantscale.data import MarketData, load_data
from quantscale.evaluation import digest, implementation_hash
from quantscale.strategies import validate_weights


def run():
    base=Path('results/sp500-deepseek-open-pilot-v3-repeat3')
    name='decentralized-n4-s27'
    artifact=json.loads((base/name/'frozen.json').read_text())
    payload=artifact['payload']
    assert digest(payload)==artifact['artifact_sha256']
    assert implementation_hash()==payload['implementation_sha256']
    config=Config.from_dict(payload['protocol']['config'])
    market=MarketData(load_data(base/'operator-data/evaluator.csv',config),config)
    daily=pd.read_csv(base/'hidden-evaluation'/name/'hidden_daily.csv',parse_dates=['decision_date','execution_date','date'])
    selected=daily.iloc[[0,len(daily)//2,len(daily)-1]]
    policy=SandboxedStrategy(CodeSpec(source=payload['strategy']['source']),config)
    policy.prepare(market)
    snapshots=[];rows=[]
    try:
        for _,day in selected.iterrows():
            index=market.dates.get_loc(day.decision_date)
            universe=market.universe(day.decision_date)
            weights=validate_weights(policy.weights_at(index,universe),universe,config)
            assert np.isclose(weights.abs().sum(),day.gross_exposure)
            assert np.isclose(weights.sum(),day.net_exposure)
            positions=weights[weights!=0]
            longs=sorted(positions[positions>0].index.tolist())
            shorts=sorted(positions[positions<0].index.tolist())
            snapshot=dict(decision_date=str(day.decision_date.date()),execution_date=str(day.execution_date.date()),
                long_count=len(longs),short_count=len(shorts),long_weight=float(weights.clip(lower=0).sum()),
                short_weight=float(weights.clip(upper=0).sum()),longs=longs,shorts=shorts)
            snapshots.append(snapshot)
            for ticker,w in positions.items():
                rows.append(dict(decision_date=snapshot['decision_date'],execution_date=snapshot['execution_date'],
                                 ticker=ticker,side='long' if w>0 else 'short',target_weight=float(w)))
    finally:policy.close()
    target=Path('results/sp500-deepseek-open-pilot-v3/portfolio-snapshots');target.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(target/'decentralized-third-target-weights.csv',index=False)
    metadata=dict(artifact_sha256=artifact['artifact_sha256'],scope='Three target-weight snapshots reconstructed from the stateless frozen policy; not recorded trades or a new backtest',snapshots=snapshots)
    (target/'decentralized-third-target-weights.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
