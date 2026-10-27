"""Reconstruct ONLY validation paths for the 15 already-selected frozen policies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantscale.backtest import run_backtest
from quantscale.code_strategy import CodeSpec, SandboxedStrategy
from quantscale.config import Config
from quantscale.data import MarketData, load_data, fingerprint
from quantscale.evaluation import digest, implementation_hash

BASE=Path('results/sp500-deepseek-open-pilot-v3')


def run():
    payload=json.loads((BASE/'v3-three-repetitions-results.json').read_text())
    records=[r for r in payload['attempts'] if r['status']=='evaluated']
    assert len(records)==15
    first=json.loads((Path(records[0]['research_dir'])/'frozen.json').read_text())
    config=Config.from_dict(first['payload']['protocol']['config'])
    data=load_data(BASE/'operator-data/research.csv',config)
    assert data.date.max()<=pd.Timestamp(config.validation.end)
    data_hash=fingerprint(data)
    market=MarketData(data,config)
    root=BASE/'validation-curves';root.mkdir(exist_ok=True)
    plan=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Reconstruct missing daily VALIDATION paths for user-requested charts',
        no_api_calls=True,no_test_period_evaluation=True,no_new_strategy_selection=True,
        research_data_sha256=data_hash,implementation_sha256=implementation_hash(),
        validation=config.validation.__dict__,runs=[r['run'] for r in records])
    if not (root/'reconstruction-protocol.json').exists():
        (root/'reconstruction-protocol.json').write_text(json.dumps(plan,indent=2))

    def reconstruct(record):
        artifact=json.loads((Path(record['research_dir'])/'frozen.json').read_text())
        frozen=artifact['payload']
        assert digest(frozen)==artifact['artifact_sha256']
        assert frozen['implementation_sha256']==implementation_hash()
        assert frozen['protocol']['research_data_hash']==data_hash
        own=Config.from_dict(frozen['protocol']['config'])
        assert own.validation==config.validation
        expected=frozen['validation']
        folder=root/'daily'/record['run'];folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/'reconstruction.json'
        if result_path.exists():
            saved=json.loads(result_path.read_text())
            assert saved['artifact_sha256']==artifact['artifact_sha256'] and saved['matched_original']
            assert (folder/'validation_daily.csv').is_file()
            return saved
        policy=SandboxedStrategy(CodeSpec(source=frozen['strategy']['source']),own)
        metrics,daily=run_backtest(policy,market,own.validation)
        differences={k:dict(original=expected[k],reconstructed=v) for k,v in metrics.items()
                     if not np.isclose(v,expected[k],rtol=1e-9,atol=1e-10)}
        result=dict(run=record['run'],label=record['label'],topology=record['topology'],seed=record['seed'],
            artifact_sha256=artifact['artifact_sha256'],metrics=metrics,original_validation=expected,
            matched_original=not differences,differences=differences,
            reconstructed_at_utc=datetime.now(timezone.utc).isoformat())
        daily.to_csv(folder/'validation_daily.csv',index=False)
        result_path.write_text(json.dumps(result,indent=2,allow_nan=False))
        if differences:raise ValueError('Validation reconstruction mismatch: '+record['run'])
        return result

    completed=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(reconstruct,r):r for r in records}
        for future in as_completed(futures):
            record=futures[future]
            try:
                result=future.result();completed.append(result)
                print('MATCHED '+record['run']+' validation_sharpe=%.4f'%result['metrics']['sharpe'],flush=True)
            except Exception as exc:
                failures.append(dict(run=record['run'],error=str(exc)))
                print('FAILED '+record['run']+': '+str(exc),flush=True)
            (root/'progress.json').write_text(json.dumps(dict(completed=completed,failures=failures),indent=2))
    summary=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),runs=len(completed),
        planned=15,all_matched=len(completed)==15 and not failures,failures=failures,completed=completed)
    (root/'completed.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if not summary['all_matched']:raise RuntimeError('Some validation paths are not verified; inspect failures')
    print('ALL 15 VALIDATION PATHS MATCH ORIGINAL METRICS',flush=True)


if __name__=='__main__':run()
