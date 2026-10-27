from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quantscale.backtest import run_backtest
from quantscale.code_strategy import CodeSpec, SandboxedStrategy
from quantscale.config import Config, Period
from quantscale.data import MarketData, synthetic_data
from quantscale.data import fingerprint, load_data
from quantscale.evaluation import freeze, evaluate
from quantscale.deepseek_adapter import DeepSeekBackend
from quantscale.prompts import research_contract, system_instructions
from quantscale.research import run_research

SOURCE = '''import pandas as pd
def generate_weights(history, universe):
    return pd.Series(0.0, index=universe)
'''


class OpenContractTests(unittest.TestCase):
    def test_default_is_open_and_has_no_example_or_family_menu(self):
        config = Config()
        self.assertEqual(config.strategy_mode, "python")
        request = research_contract(config)
        self.assertNotIn("strategy_schema", request)
        self.assertNotIn('"lookback"', json.dumps(request))
        for family in ("momentum", "reversal", "low_volatility", "volume anomaly"):
            self.assertNotIn(family, system_instructions("python"))
        self.assertIn("strategy_code", system_instructions("python"))
        self.assertIsNone(config.code_history_days)

    def test_source_hash_ignores_comments_and_layout(self):
        self.assertEqual(CodeSpec(SOURCE).digest, CodeSpec("# commentary\n" + SOURCE + "\n").digest)

    def test_arbitrary_numerical_helpers_are_allowed(self):
        code = '''import numpy as np
import pandas as pd
def combine(x):
    return np.tanh(x.ewm(span=19).mean())
def generate_weights(history, universe):
    values = combine((history['high'] - history['low']) / history['open'])
    return pd.Series(0.0, index=universe)
'''
        CodeSpec(code)

    def test_bad_interfaces_io_introspection_and_training_imports_rejected(self):
        for code in ("x=1", "def generate_weights(history): return 0", "import os\n" + SOURCE,
                     SOURCE.replace("return pd.Series", "open('secret')\n    return pd.Series"),
                     SOURCE.replace("return pd.Series", "x = history.__class__\n    return pd.Series"),
                     "import sklearn\n" + SOURCE):
            with self.assertRaises(ValueError):
                CodeSpec(code)

    def test_manager_can_assign_directions_outside_legacy_families(self):
        class Backend:
            name = "fixture"
            token_accounting = "fixture"
            def invoke(self, request):
                if request["role"] == "manager":
                    return {"directions": {"0": "Study a nonlinear interaction of your own design"}}, 10
                return {"hypothesis": "new numerical policy", "strategy_code": SOURCE}, 10
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Config(train=Period("2020-01-01", "2020-01-10"),
                            validation=Period("2020-01-13", "2020-01-20"),
                            test=Period("2020-01-21", "2020-02-01"), backtests=1)
            frame = synthetic_data("2020-01-01", "2020-01-20", 4)
            frame.to_csv(root / "research.csv", index=False)
            with patch("quantscale.research.run_backtest", return_value=({"sharpe": 0.1}, pd.DataFrame())):
                result = run_research(root / "research.csv", config, root / "run", "centralized", 1, Backend())
            self.assertEqual(result["winner"]["strategy"]["kind"], "python")
            self.assertTrue(list((root / "run/strategies").glob("*.py")))
            self.assertNotIn("family_entropy", result["process"])
            frozen = json.loads((root / "run/frozen.json").read_text())
            self.assertEqual(frozen["payload"]["strategy"]["source"], SOURCE)


@unittest.skipUnless(sys.platform == "darwin" and os.environ.get("QUANTSCALE_SANDBOX_TESTS") == "1",
                     "Set QUANTSCALE_SANDBOX_TESTS=1 outside a nested sandbox to verify OS isolation")
class SandboxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(universe_mode="fixed", universe_size=2, min_price=0, min_volume=0,
                             max_weight=0.5, code_call_timeout_seconds=10, code_backtest_timeout_seconds=30)
        self.market = MarketData(synthetic_data("2020-01-02", "2020-01-20", 2), self.config)

    def test_repaired_manager_to_open_code_and_real_sandbox_on_synthetic_data(self):
        """Provider is a fixture; parsing, budgets, execution and freeze are real."""
        calls = []
        source = '''import pandas as pd
def generate_weights(history, universe):
    weights = pd.Series(0.0, index=universe)
    if len(universe) >= 2:
        weights.iloc[0] = 0.1
        weights.iloc[-1] = -0.1
    return weights
'''
        def post(payload):
            request = json.loads(payload['messages'][-1]['content'])
            calls.append((request, payload))
            if request['role'] == 'manager':
                content = ('{"directions":{"0":"custom numerical policy A" "1":"custom numerical policy B"}}'
                           if 'format_repair' not in request else
                           '{"directions":{"0":"custom numerical policy A","1":"custom numerical policy B"}}')
            else:
                content = json.dumps({'hypothesis': 'execution fixture, not an alpha claim', 'strategy_code': source})
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fixture'}, clear=True):
            root = Path(temp)
            config = replace(self.config, train=Period('2020-01-02', '2020-01-08'),
                             validation=Period('2020-01-09', '2020-01-20'),
                             test=Period('2020-01-21', '2020-01-31'), tokens=200000, backtests=2)
            self.market.frame.to_csv(root / 'research.csv', index=False)
            result = run_research(root / 'research.csv', config, root / 'run', 'hybrid', 2,
                                  DeepSeekBackend('/nonexistent', post=post))
            self.assertEqual(result['process']['total_tokens'], 120)
            self.assertEqual(result['process']['successful_candidates'], 2)
            self.assertEqual(result['winner']['strategy']['kind'], 'python')
            self.assertGreater(result['winner']['validation']['active_days'], 0)
            self.assertTrue((root / 'run/frozen.json').exists())
            self.assertEqual([r['role'] for r, _ in calls], ['manager', 'manager', 'researcher', 'researcher'])
            self.assertEqual(calls[0][1]['thinking'], {'type': 'disabled'})
            self.assertEqual(calls[2][1]['reasoning_effort'], 'low')

    def test_child_receives_only_current_and_past_rows(self):
        code = '''import pandas as pd
def generate_weights(history, universe):
    n = len(history['close']) / 10000.0
    return pd.Series([n, -n], index=universe)
'''
        policy = SandboxedStrategy(CodeSpec(code), self.config)
        try:
            policy.prepare(self.market)
            first = policy.weights_at(5, self.market.universe(self.market.dates[5]))
            later = policy.weights_at(8, self.market.universe(self.market.dates[8]))
            self.assertAlmostEqual(first.iloc[0], 6 / 10000)
            self.assertAlmostEqual(later.iloc[0], 9 / 10000)
        finally:
            policy.close()

    def test_no_api_credentials_in_child_environment(self):
        code = '''import pandas as pd
def generate_weights(history, universe):
    leaked = 'DEEPSEEK_API_KEY' in pd.io.common.os.environ
    return pd.Series(float(leaked), index=universe)
'''
        policy = SandboxedStrategy(CodeSpec(code), self.config)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-do-not-inherit"}):
            try:
                policy.prepare(self.market)
                weights = policy.weights_at(5, self.market.universe(self.market.dates[5]))
                self.assertEqual(float(weights.sum()), 0)
            finally:
                policy.close()

    def test_host_file_is_denied_even_through_pandas(self):
        with tempfile.TemporaryDirectory() as temp:
            secret = Path(temp) / "hidden-fixture.csv"
            secret.write_text("secret\n42\n")
            code = "import pandas as pd\ndef generate_weights(history, universe):\n    pd.read_csv(%r)\n    return pd.Series(0.0,index=universe)\n" % str(secret)
            policy = SandboxedStrategy(CodeSpec(code), self.config)
            try:
                policy.prepare(self.market)
                with self.assertRaisesRegex(ValueError, "PermissionError|Operation not permitted"):
                    policy.weights_at(5, self.market.universe(self.market.dates[5]))
            finally:
                policy.close()

    def test_network_denied_even_through_pandas(self):
        code = '''import pandas as pd
def generate_weights(history, universe):
    pd.read_csv('http://127.0.0.1:9/never-access')
    return pd.Series(0.0,index=universe)
'''
        policy = SandboxedStrategy(CodeSpec(code), self.config)
        try:
            policy.prepare(self.market)
            with self.assertRaisesRegex(ValueError, "Operation not permitted|PermissionError"):
                policy.weights_at(5, self.market.universe(self.market.dates[5]))
        finally:
            policy.close()

    def test_hang_times_out_and_backtest_closes_worker(self):
        code = "def generate_weights(history, universe):\n    while True:\n        pass\n"
        config = replace(self.config, code_call_timeout_seconds=2)
        market = MarketData(self.market.frame, config)
        policy = SandboxedStrategy(CodeSpec(code), config)
        with self.assertRaisesRegex(ValueError, "timeout"):
            run_backtest(policy, market, Period("2020-01-06", "2020-01-10"))
        self.assertIsNone(policy.process)

    def test_full_engine_accepts_code_without_a_strategy_family(self):
        policy = SandboxedStrategy(CodeSpec(SOURCE), self.config)
        metrics, daily = run_backtest(policy, self.market, Period("2020-01-06", "2020-01-10"))
        self.assertEqual(metrics["total_return"], 0)
        self.assertGreater(len(daily), 0)
        self.assertIsNone(policy.process)

    def test_code_submission_can_be_frozen_and_evaluated_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = replace(self.config, train=Period("2020-01-02", "2020-01-08"),
                             validation=Period("2020-01-09", "2020-01-13"),
                             test=Period("2020-01-14", "2020-01-20"))
            self.market.frame.to_csv(root / "evaluator.csv", index=False)
            data = load_data(root / "evaluator.csv", config)
            research = data[data.date <= config.validation.end].reset_index(drop=True)
            run = root / "run"
            run.mkdir()
            freeze(run, {"config": config.to_dict(), "research_data_hash": fingerprint(research),
                         "backend": "test-only"},
                   {"strategy": CodeSpec(SOURCE).to_dict(), "validation": {"sharpe": 0.0}})
            result = evaluate(run / "frozen.json", root / "evaluator.csv", root / "evaluation")
            self.assertEqual(result["hidden"]["total_return"], 0.0)
            with self.assertRaisesRegex(ValueError, "already"):
                evaluate(run / "frozen.json", root / "evaluator.csv", root / "again")


if __name__ == "__main__":
    unittest.main()
