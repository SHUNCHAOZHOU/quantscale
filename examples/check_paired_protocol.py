"""Offline checks for paired state, snapshot timing, budgets and failure behavior."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pandas as pd

from quantscale.config import Config
from quantscale.code_strategy import CodeSpec
from .paired_research import run_continuation, state_digest

CODE='import pandas as pd\ndef generate_weights(history, universe):\n    return pd.Series(0.0, index=universe)\n'


class Backend:
    name='offline-check';configuration={};token_accounting='exact synthetic units'
    def __init__(self,fail_manager=False):self.requests=[];self.fail_manager=fail_manager
    def invoke(self,q):
        self.requests.append(deepcopy(q))
        if q['role']=='manager':return ({'raw_content':'incomplete'} if self.fail_manager else {'directions':{str(i):'Open research task' for i in range(4)}}),1
        return dict(hypothesis='Identical control; must retain initial tie winner',strategy_code=CODE),1


def run():
    spec=CodeSpec(CODE);metric=dict(sharpe=0.,active_days=0)
    initial=[dict(agent=i,round=0,hypothesis='initial',strategy=spec.to_dict(),strategy_id=spec.digest,
                  train=metric,validation=metric) for i in range(4)]
    initial_hash=state_digest(initial)
    data=pd.DataFrame(dict(date=pd.to_datetime(['2016-01-04','2022-01-03']),ticker=['X','X'],volume=[1.,1.]))
    class Market:excluded_dates=[]
    cfg=replace(Config.load('configs/sp500-open-v3.json'),tokens=400000,max_turn_tokens=131072)
    def freeze_stub(root,protocol,winner):return dict(artifact_sha256='synthetic-test-artifact')
    with tempfile.TemporaryDirectory() as scratch,patch('examples.paired_research.freeze',freeze_stub),patch('examples.paired_research.run_backtest',side_effect=AssertionError('Inherited metric cache must be reused')):
        for top in ['independent','centralized','decentralized','hybrid']:
            backend=Backend();root=Path(scratch)/top
            result=run_continuation(data,Market(),cfg,root,top,initial,backend,'offline-test')
            q=[x for x in backend.requests if x['role']=='researcher']
            assert len(q)==8 and result['process']['additional_trials']==8
            assert result['process']['complete_eight_attempts'] and result['winner']['round']==0
            assert all(len(x['own_results'])==(1 if x['round']==1 else 2) for x in q)
            for x in q:
                expected_peers=3 if top in ('decentralized','hybrid') else 0
                assert len(x['peer_results'])==expected_peers
                assert all(t['round']==x['round']-1 for t in x['peer_results'])
                assert bool(x['direction'])==(top in ('centralized','hybrid'))
            assert result['process']['total_tokens']==8+(2 if top in ('centralized','hybrid') else 0)
            assert json.loads((root/'initial-state.json').read_text())==initial
        backend=Backend(fail_manager=True)
        failed=run_continuation(data,Market(),cfg,Path(scratch)/'manager-fail','hybrid',initial,backend,'offline-test')
        assert not failed['process']['complete_eight_attempts']
        assert failed['process']['additional_trials']==0 and failed['winner']['round']==0
        assert len(backend.requests)==2
    assert state_digest(initial)==initial_hash
    print('PASS: four identical initial states; 8 candidate opportunities; immutable prior-round peers; cached initial metrics; exact cost accounting; original winner retained; manager failure disclosed')


if __name__=='__main__':run()
