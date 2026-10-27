import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantscale.backtest import performance, run_backtest
from quantscale.config import Config, Period
from quantscale.data import MarketData, fingerprint, load_data, synthetic_data, validate_data
from quantscale.evaluation import evaluate, freeze
from quantscale.openai_adapter import invoke as invoke_openai
from quantscale.research import Budget, BudgetExhausted, DemoBackend, run_research
from quantscale.strategies import RuleStrategy, StrategySpec, validate_weights


def settings(**kwargs):
    defaults = dict(strategy_mode="rules", train=Period("2019-01-01", "2019-12-31"),
                    validation=Period("2020-01-01", "2020-06-30"),
                    test=Period("2020-07-01", "2020-12-31"),
                    universe_size=2, min_price=0, min_volume=0,
                    max_weight=0.5, annual_borrow_bps=0, cost_bps=10,
                    backtests=4, tokens=100000, max_turn_tokens=16000)
    defaults.update(kwargs)
    return Config(**defaults)


def small_panel():
    frame = synthetic_data("2020-01-27", "2020-02-14", 2)
    for ticker, rate in (("S0000", 1.01), ("S0001", 1.0)):
        mask = frame.ticker == ticker
        price = 10 * rate ** np.arange(mask.sum())
        for name in ("open", "close", "adjusted_open", "adjusted_close"):
            frame.loc[mask, name] = price
        frame.loc[mask, "high"] = price * 1.01
        frame.loc[mask, "low"] = price * 0.99
    return frame


class ConstantStrategy:
    def __init__(self):
        self.observed = []

    def generate_weights(self, history, universe):
        self.observed.append(history["close"].index.max())
        return pd.Series({"S0000": 0.5, "S0001": -0.5}).reindex(universe, fill_value=0.0)


class BacktestTests(unittest.TestCase):
    def test_execution_cost_and_drift(self):
        config = settings()
        market = MarketData(small_panel(), config)
        strategy = ConstantStrategy()
        metrics, daily = run_backtest(strategy, market, Period("2020-02-04", "2020-02-14"))
        self.assertAlmostEqual(daily.iloc[0].gross_return, 0.005)
        self.assertAlmostEqual(daily.iloc[0].turnover, 1)
        self.assertAlmostEqual(daily.iloc[0].net_return, 0.004)
        drift = pd.Series([0.5 * 1.01 / 1.004, -0.5 / 1.004])
        self.assertAlmostEqual(daily.iloc[1].turnover, float((pd.Series([0.5, -0.5]) - drift).abs().sum()))
        self.assertEqual(list(daily.decision_date), strategy.observed)
        self.assertTrue((daily.decision_date < daily.execution_date).all())
        self.assertTrue((daily.execution_date < daily.date).all())
        self.assertEqual(metrics["active_days"], len(daily))

    def test_future_mutation_does_not_change_past(self):
        df = small_panel()
        period = Period("2020-02-03", "2020-02-10")
        _, before = run_backtest(ConstantStrategy(), MarketData(df, settings()), period)
        changed = df.copy()
        mask = changed.date > pd.Timestamp("2020-02-10")
        for name in ("open", "high", "low", "close", "adjusted_open", "adjusted_close"):
            changed.loc[mask, name] *= 100
        _, after = run_backtest(ConstantStrategy(), MarketData(changed, settings()), period)
        pd.testing.assert_frame_equal(before, after)

    def test_missing_held_exit_fails(self):
        df = small_panel()
        df = df[~((df.date == pd.Timestamp("2020-02-05")) & (df.ticker == "S0000"))]
        with self.assertRaisesRegex(ValueError, "Missing execution/exit"):
            run_backtest(ConstantStrategy(), MarketData(df, settings()), Period("2020-02-03", "2020-02-14"))

    def test_drawdown_includes_initial_capital(self):
        daily = pd.DataFrame({"net_return": [-0.1, 0.0], "turnover": [0, 0],
                              "cost": [0, 0], "borrow_cost": [0, 0], "gross_exposure": [1, 1]})
        self.assertAlmostEqual(performance(daily)["max_drawdown"], 0.1)

    def test_borrow_cost(self):
        config = settings(annual_borrow_bps=252)
        _, daily = run_backtest(ConstantStrategy(), MarketData(small_panel(), config), Period("2020-02-04", "2020-02-14"))
        self.assertAlmostEqual(daily.iloc[0].borrow_cost, 0.5 * 0.0252 / 252)


class DataAndStrategyTests(unittest.TestCase):
    def test_universe_uses_last_month(self):
        df = small_panel()
        df.loc[df.ticker == "S0000", "market_cap"] = 100
        df.loc[df.ticker == "S0001", "market_cap"] = 10
        df.loc[(df.date >= "2020-02-01") & (df.ticker == "S0001"), "market_cap"] = 1000
        market = MarketData(df, settings(universe_size=1))
        self.assertEqual(list(market.universe(pd.Timestamp("2020-02-10"))), ["S0000"])
        self.assertEqual(list(market.universe(pd.Timestamp("2020-01-27"))), [])

    def test_input_validation(self):
        df = small_panel()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_data(pd.concat([df, df.iloc[:1]]))
        df.loc[0, "close"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_data(df)

    def test_constraints_reject_violations(self):
        config = settings(max_weight=0.02)
        for weights in (pd.Series([0.1, -0.1], index=["a", "b"]),
                        pd.Series([np.nan, 0], index=["a", "b"]),
                        pd.Series([0.01, 0.01], index=["a", "b"]),
                        pd.Series([0], index=["a"])):
            with self.assertRaises(ValueError):
                validate_weights(weights, pd.Index(["a", "b"]), config)

    def test_all_baselines_obey_limits(self):
        config = settings(universe_size=24, max_weight=0.02)
        market = MarketData(synthetic_data("2019-01-01", "2019-12-31", 24), config)
        i = len(market.dates) - 1
        universe = market.universe(market.dates[i])
        for family in ("momentum", "reversal", "low_volatility", "volume"):
            for scale in (False, True):
                spec = StrategySpec(family=family, lookback=20, volatility_scale=scale)
                weights = RuleStrategy(spec, config).generate_weights(market.history(i), universe)
                validate_weights(weights, universe, config)
                self.assertGreater(weights.abs().sum(), 0)

    def test_invalid_spec_and_period(self):
        with self.assertRaises(ValueError):
            StrategySpec(lookback=1)
        with self.assertRaises(ValueError):
            StrategySpec(family="random_forest")
        with self.assertRaises(ValueError):
            settings(validation=Period("2019-01-01", "2020-01-01"))


class FakeBackend:
    name = "test-only"
    token_accounting = "test fixture"

    def __init__(self, invalid=False, tokens=10):
        self.requests = []
        self.invalid = invalid
        self.tokens = tokens

    def invoke(self, request):
        self.requests.append(request)
        if request["role"] == "manager":
            return {"directions": {str(i): "momentum" for i in range(request["fleet_size"])}}, self.tokens
        return {"hypothesis": "test", "strategy": StrategySpec().to_dict() if not self.invalid else {"family": "bad"}}, self.tokens


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.df = synthetic_data("2018-12-03", "2020-12-31", 4)
        cls.research_path = cls.root / "research.csv"
        cls.df[cls.df.date <= "2020-06-30"].to_csv(cls.research_path, index=False)
        cls.evaluator_path = cls.root / "evaluator.csv"
        cls.df.to_csv(cls.evaluator_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_fast(self, topology, size, backend, **config):
        with tempfile.TemporaryDirectory() as temp:
            with patch("quantscale.research.run_backtest", return_value=({"sharpe": 1.0}, pd.DataFrame())):
                return run_research(self.research_path, settings(**config), Path(temp) / "run", topology, size, backend)

    def test_independent_isolation_and_allocation(self):
        backend = FakeBackend()
        result = self.run_fast("independent", 2, backend, backtests=5)
        requests = [r for r in backend.requests if r["role"] == "researcher"]
        self.assertEqual([sum(r["agent"] == i for r in requests) for i in range(2)], [3, 2])
        self.assertTrue(all(not r["peer_results"] for r in requests))
        self.assertTrue(all(all(t["agent"] == r["agent"] for t in r["own_results"]) for r in requests))
        self.assertEqual(result["process"]["total_backtests"], 5)
        self.assertEqual(result["process"]["total_tokens"], 50)

    def test_topology_visibility_and_manager_budget(self):
        for topology in ("single", "centralized", "decentralized", "hybrid"):
            backend = FakeBackend()
            size = 1 if topology == "single" else 2
            result = self.run_fast(topology, size, backend)
            requests = backend.requests
            self.assertEqual(result["process"]["total_tokens"], len(requests) * 10)
            self.assertEqual(any(r["role"] == "manager" for r in requests), topology in ("centralized", "hybrid"))
            workers = [r for r in requests if r["role"] == "researcher"]
            self.assertEqual(any(r["peer_results"] for r in workers), topology in ("decentralized", "hybrid"))
            if topology in ("decentralized", "hybrid"):
                self.assertTrue(all(t["round"] < r["round"] for r in workers for t in r["peer_results"]))

    def test_hard_token_stop(self):
        backend = FakeBackend()
        result = self.run_fast("single", 1, backend, tokens=20)
        self.assertEqual(result["process"]["total_backtests"], 2)
        self.assertEqual(result["process"]["total_tokens"], 20)

    def test_token_overrun_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "breached"):
            self.run_fast("single", 1, FakeBackend(tokens=21), tokens=20)

    def test_invalid_trials_consume_budget(self):
        backend = FakeBackend(invalid=True)
        with self.assertRaisesRegex(RuntimeError, "No valid"):
            self.run_fast("single", 1, backend, backtests=2)
        self.assertEqual(len(backend.requests), 2)

    def test_hidden_rows_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "post-validation"):
                run_research(self.evaluator_path, settings(), Path(temp) / "run")

    def test_demo_researchers_have_separate_rng(self):
        backend = DemoBackend(7)
        request = {"role": "researcher", "agent": 0, "max_total_tokens": 10000}
        first = backend.invoke(request)
        backend.invoke({**request, "agent": 1})
        second = backend.invoke(request)
        isolated = DemoBackend(7)
        self.assertEqual(first, isolated.invoke(request))
        self.assertEqual(second, isolated.invoke(request))

    def frozen_fixture(self, root):
        root.mkdir()
        config = settings()
        protocol = {"config": config.to_dict(), "research_data_hash": fingerprint(load_data(self.research_path)),
                    "backend": "test-only"}
        artifact = freeze(root, protocol, {"strategy": StrategySpec(lookback=5).to_dict(), "validation": {"sharpe": 1.0}})
        return root / "frozen.json", artifact

    def test_frozen_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, artifact = self.frozen_fixture(root / "run")
            artifact["payload"]["strategy"]["lookback"] = 20
            frozen.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "modified"):
                evaluate(frozen, self.evaluator_path, root / "eval")

    def test_hidden_evaluation_once_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, _ = self.frozen_fixture(root / "run")
            evaluator = root / "evaluator.csv"
            evaluator.write_bytes(self.evaluator_path.read_bytes())
            result = evaluate(frozen, evaluator, root / "eval")
            self.assertIn("hidden", result)
            with self.assertRaisesRegex(ValueError, "already"):
                evaluate(frozen, evaluator, root / "another-output")


class AdapterTests(unittest.TestCase):
    def test_preflight_and_real_usage(self):
        calls = []
        def post(path, payload):
            calls.append((path, payload))
            if path.endswith("input_tokens"):
                return {"input_tokens": 100}
            return {"status": "completed", "model": "pinned-model", "usage": {"total_tokens": 150},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"hypothesis":"x"}'}]}]}
        result = invoke_openai({"max_total_tokens": 300}, "pinned-model", post)
        self.assertEqual(calls[1][1]["max_output_tokens"], 200)
        self.assertEqual(result["usage"]["total_tokens"], 150)

    def test_no_inference_when_input_too_large(self):
        calls = []
        def post(path, payload):
            calls.append(path)
            return {"input_tokens": 290}
        result = invoke_openai({"max_total_tokens": 300}, "model", post)
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(calls, ["responses/input_tokens"])

    def test_incomplete_response_still_metered(self):
        def post(path, payload):
            if path.endswith("input_tokens"):
                return {"input_tokens": 10}
            return {"status": "incomplete", "usage": {"total_tokens": 300}, "output": []}
        result = invoke_openai({"max_total_tokens": 300}, "model", post)
        self.assertEqual(result["usage"]["total_tokens"], 300)
        self.assertIn("adapter_error", result["response"])


if __name__ == "__main__":
    unittest.main()
