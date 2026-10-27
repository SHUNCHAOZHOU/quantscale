"""One additional V3 repetition for all five topologies, without changing V3."""
import json
from pathlib import Path

from quantscale.config import Config
from quantscale.deepseek_adapter import DeepSeekBackend
from quantscale.evaluation import implementation_hash
from quantscale.pilot import run_pilot


def run():
    base = Path('results/sp500-deepseek-open-pilot-v3')
    original = json.loads((base/'single-n1-s7/protocol.json').read_text())
    pilot = json.loads((base/'pilot-protocol.json').read_text())
    assert Config.load('configs/sp500-open-v3.json').to_dict() == original['config']
    assert implementation_hash() == pilot['implementation_sha256']
    assert DeepSeekBackend().configuration == original['backend_configuration']
    return run_pilot(
        'data/sp500-yahoo-2016-2025/prices.csv', 'configs/sp500-open-v3.json',
        'results/sp500-deepseek-open-pilot-v3-repeat3', seeds=(27,), workers=4)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
