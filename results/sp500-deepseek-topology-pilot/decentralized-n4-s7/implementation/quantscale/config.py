"""Explicit, serialized experiment settings. Dates are inclusive."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Period:
    start: str
    end: str


@dataclass(frozen=True)
class Config:
    train: Period = field(default_factory=lambda: Period("2010-01-01", "2018-12-31"))
    validation: Period = field(default_factory=lambda: Period("2019-01-01", "2021-12-31"))
    test: Period = field(default_factory=lambda: Period("2022-01-01", "2025-12-31"))
    universe_size: int = 1000
    min_price: float = 5.0
    min_volume: float = 100000.0
    gross_limit: float = 1.0
    max_weight: float = 0.02
    net_tolerance: float = 0.01
    cost_bps: float = 5.0
    annual_borrow_bps: float = 30.0
    annualization: int = 252
    backtests: int = 40
    tokens: int = 100000
    max_turn_tokens: int = 2048
    seed: int = 7
    lookback_days: int = 300
    universe_mode: str = "lagged_market_cap"
    missing_data_policy: str = "strict"

    def __post_init__(self):
        if self.universe_mode not in ("lagged_market_cap", "fixed"):
            raise ValueError("universe_mode must be lagged_market_cap or fixed")
        if self.missing_data_policy not in ("strict", "skip_incomplete_sessions"):
            raise ValueError("Unknown missing_data_policy")
        for name in ("train", "validation", "test"):
            p = getattr(self, name)
            if pd.Timestamp(p.start) > pd.Timestamp(p.end):
                raise ValueError("Period start must precede end")
        if not (pd.Timestamp(self.train.end) < pd.Timestamp(self.validation.start)
                <= pd.Timestamp(self.validation.end) < pd.Timestamp(self.test.start)):
            raise ValueError("Train, validation and test must be ordered and disjoint")
        for name in ("universe_size", "annualization", "backtests", "tokens", "max_turn_tokens", "lookback_days"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("min_price", "min_volume", "gross_limit", "max_weight", "net_tolerance", "cost_bps", "annual_borrow_bps"):
            import math
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.gross_limit == 0 or self.max_weight == 0:
            raise ValueError("Exposure limits must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        data = dict(raw)
        for name in ("train", "validation", "test"):
            if name in data:
                data[name] = Period(**data[name])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
