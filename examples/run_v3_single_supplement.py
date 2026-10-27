"""One user-requested fresh V3 single-agent run; never replace original records."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from quantscale.config import Config
from quantscale.deepseek_adapter import DeepSeekBackend
from quantscale.evaluation import evaluate, implementation_hash
from quantscale.research import run_research


def run():
    base=Path('results/sp500-deepseek-open-pilot-v3')
    target=base/'supplement-single-n1-s17'
    config=replace(Config.load('configs/sp500-open-v3.json'),seed=17)
    original=json.loads((base/'single-n1-s17/protocol.json').read_text())
    pilot=json.loads((base/'pilot-protocol.json').read_text())
    assert config.to_dict()==original['config']
    assert implementation_hash()==pilot['implementation_sha256']
    backend=DeepSeekBackend()
    assert backend.configuration==original['backend_configuration']
    target.mkdir(exist_ok=False)
    (target/'supplement-protocol.json').write_text(json.dumps(dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='User-requested fresh repeat of V3 single-n1-s17 after network failure',
        original_attempt=str((base/'single-n1-s17').resolve()),
        fresh_research=True,original_results_unchanged=True,
        separate_additional_budget=dict(tokens=config.tokens,candidates=config.backtests),
        config=config.to_dict(),implementation_sha256=implementation_hash(),
        backend_configuration=backend.configuration,
        historical_test_note='Same reused historical test; supplementary, not a new blind test'),indent=2))
    start=time.monotonic()
    status={}
    try:
        result=run_research(base/'operator-data/research.csv',config,target/'research','single',1,backend)
        assert implementation_hash()==pilot['implementation_sha256']
        assert json.loads((target/'research/protocol.json').read_text())['research_data_hash']==original['research_data_hash']
        print('FROZEN supplemental single-n1-s17',flush=True)
        status['research_seconds']=time.monotonic()-start
        status['process']=result['process']
        evaluation=evaluate(target/'research/frozen.json',base/'operator-data/evaluator.csv',target/'evaluation')
        status.update(status='evaluated',evaluation=evaluation)
        print('EVALUATED supplemental single-n1-s17 hidden_sharpe=%.4f'%evaluation['hidden']['sharpe'],flush=True)
    except Exception as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        events=[]
        journal=target/'research/trajectory.jsonl'
        if journal.exists():events=[json.loads(x) for x in journal.read_text().splitlines()]
        responses=[x for x in events if x['kind']=='response']
        status.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-start,
                      confirmed_tokens=sum(x['tokens'] for x in responses),
                      model_calls=len(responses),candidate_attempts=sum(x['kind']=='trial' for x in events))
        (target/'completed.json').write_text(json.dumps(status,indent=2,allow_nan=False))


if __name__=='__main__':run()
