"""Explicit, serialized experiment settings. Dates are inclusive."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

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
    max_turn_tokens: int = 65536
    seed: int = 7
    lookback_days: int = 300
    universe_mode: str = "lagged_market_cap"
    missing_data_policy: str = "strict"
    strategy_mode: str = "python"
    code_history_days: Optional[int] = None
    code_call_timeout_seconds: int = 10
    code_backtest_timeout_seconds: int = 180
    code_memory_mb: int = 1024
    manager_max_format_repairs: int = 1
    manager_max_failed_rounds: int = 2

    def __post_init__(self):
        if type(self.manager_max_format_repairs) is not int or not 0 <= self.manager_max_format_repairs <= 2:
            raise ValueError("manager_max_format_repairs must be 0..2")
        if type(self.manager_max_failed_rounds) is not int or self.manager_max_failed_rounds < 1:
            raise ValueError("manager_max_failed_rounds must be a positive integer")
        if self.strategy_mode not in ("python", "rules"):
            raise ValueError("strategy_mode must be python or rules")
        if self.code_history_days is not None and (type(self.code_history_days) is not int or self.code_history_days < 1):
            raise ValueError("code_history_days must be null (all past history) or a positive integer")
        for name in ("code_call_timeout_seconds", "code_backtest_timeout_seconds", "code_memory_mb"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
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
