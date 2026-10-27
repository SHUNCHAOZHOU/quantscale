from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from quantscale.backtest import run_backtest
from quantscale.config import Config, Period
from quantscale.data import MarketData, synthetic_data, validate_data
from quantscale.strategies import RuleStrategy, StrategySpec


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(universe_mode="fixed", universe_size=12, min_price=0, min_volume=0,
                            max_weight=0.2, train=Period("2020-01-01", "2020-04-30"),
                            validation=Period("2020-05-01", "2020-06-30"), test=Period("2020-07-01", "2020-12-31"))
        cls.frame = synthetic_data("2020-01-01", "2020-06-30", 12).drop(columns="market_cap")

    def test_fixed_pool_needs_no_fabricated_market_cap(self):
        market = MarketData(self.frame, self.config)
        self.assertNotIn("market_cap", market.frame)
        self.assertEqual(len(market.universe(market.dates[0])), 12)
        with self.assertRaisesRegex(ValueError, "market_cap"):
            validate_data(self.frame)

    def test_gap_gate_does_not_treat_pre_ipo_as_missing(self):
        frame = self.frame[~((self.frame.ticker == "S0000") & (self.frame.date < "2020-02-01"))]
        gap = pd.Timestamp("2020-03-03")
        frame = frame[~((frame.ticker == "S0001") & (frame.date == gap))]
        market = MarketData(frame, replace(self.config, missing_data_policy="skip_incomplete_sessions"))
        self.assertEqual(market.excluded_dates, [gap])
        self.assertNotIn(gap, market.dates)

    def test_prepared_factors_match_history_interface(self):
        market = MarketData(self.frame, self.config)
        for family in ("momentum", "reversal", "low_volatility", "volume"):
            for scale in (False, True):
                spec = StrategySpec(family=family, lookback=20, skip=3, volatility_scale=scale)
                strategy = RuleStrategy(spec, self.config)
                strategy.prepare(market)
                for i in (0, 25, 50, 80, 110):
                    universe = market.universe(market.dates[i])
                    old = strategy.generate_weights(market.history(i), universe)
                    new = strategy.weights_at(i, universe)
                    np.testing.assert_allclose(old, new, atol=1e-12)

    def test_prepared_engine_matches_reference(self):
        market = MarketData(self.frame, self.config)
        spec = StrategySpec(family="reversal", lookback=10, volatility_scale=True)
        _, old = run_backtest(RuleStrategy(spec, self.config), market, self.config.validation, use_prepared=False)
        _, new = run_backtest(RuleStrategy(spec, self.config), market, self.config.validation)
        np.testing.assert_allclose(old.net_return, new.net_return, atol=1e-12)

    def test_prepared_cache_future_changes_do_not_affect_past(self):
        changed = self.frame.copy()
        mask = changed.date > "2020-05-01"
        for name in ("open", "high", "low", "close", "adjusted_open", "adjusted_close"):
            changed.loc[mask, name] *= 5
        first = MarketData(self.frame, self.config)
        second = MarketData(changed, self.config)
        spec = StrategySpec(lookback=20, volatility_scale=True)
        a, b = RuleStrategy(spec, self.config), RuleStrategy(spec, self.config)
        a.prepare(first)
        b.prepare(second)
        for i in (40, 60, 80):
            universe = first.universe(first.dates[i])
            np.testing.assert_allclose(a.weights_at(i, universe), b.weights_at(i, universe), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
