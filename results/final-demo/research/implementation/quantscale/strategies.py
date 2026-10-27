"""Auditable rule policies: no fitted parameters or trained predictive models."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    family: str = "momentum"
    lookback: int = 60
    skip: int = 1
    fraction: float = 0.25
    volatility_scale: bool = False

    def __post_init__(self):
        if self.family not in ("momentum", "reversal", "low_volatility", "volume"):
            raise ValueError("Unknown strategy family")
        if (type(self.lookback) is not int or type(self.skip) is not int
                or not 2 <= self.lookback <= 252 or not 0 <= self.skip <= 20):
            raise ValueError("lookback must be 2..252; skip must be 0..20")
        if not np.isfinite(self.fraction) or not 0 < self.fraction <= 0.5:
            raise ValueError("fraction must be in (0, .5]")
        if type(self.volatility_scale) is not bool:
            raise ValueError("volatility_scale must be boolean")

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


class RuleStrategy:
    def __init__(self, spec, config):
        self.spec = spec
        self.config = config

    def generate_weights(self, history, universe):
        spec = self.spec
        weights = pd.Series(0.0, index=universe, dtype=float)
        close = history["adjusted_close"].reindex(columns=universe)
        if len(close) < spec.lookback + spec.skip + 1 or len(universe) < 2:
            return weights
        data = close.iloc[:len(close) - spec.skip] if spec.skip else close
        window = data.iloc[-spec.lookback - 1:]
        volatility = window.pct_change(fill_method=None).std()
        if spec.family in ("momentum", "reversal"):
            signal = window.iloc[-1] / window.iloc[0] - 1
            if spec.family == "reversal":
                signal = -signal
        elif spec.family == "low_volatility":
            signal = -volatility
        else:
            volume = history["volume"].reindex(columns=universe)
            volume = volume.iloc[:len(volume) - spec.skip] if spec.skip else volume
            signal = volume.iloc[-1] / volume.iloc[-spec.lookback:].mean() - 1
        signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
        # Stable ticker order resolves ties, independent of incoming row order.
        ranked = signal.sort_index().sort_values(kind="stable")
        count = min(int(len(ranked) * spec.fraction), len(ranked) // 2)
        if count == 0:
            return weights
        shorts, longs = ranked.index[:count], ranked.index[-count:]
        for side, sign in ((longs, 1.0), (shorts, -1.0)):
            allocation = pd.Series(1.0, index=side)
            if spec.volatility_scale:
                allocation = (1 / volatility.reindex(side).clip(lower=0.001)).fillna(0)
            weights.loc[side] = sign * allocation / allocation.sum() * self.config.gross_limit / 2
        # Scale both books together; preserve neutrality and relative sizes.
        peak = weights.abs().max()
        if peak > self.config.max_weight:
            weights *= self.config.max_weight / peak
        return weights


def validate_weights(weights, universe, config):
    if not isinstance(weights, pd.Series) or weights.index.has_duplicates:
        raise ValueError("Weights must be a Series with unique ticker index")
    if set(weights.index) != set(universe):
        raise ValueError("Every universe ticker must receive exactly one weight")
    weights = weights.reindex(universe).astype(float)
    if not np.isfinite(weights).all():
        raise ValueError("Weights must be finite")
    if weights.abs().sum() > config.gross_limit + 1e-9:
        raise ValueError("Gross exposure violation")
    if (weights.abs() > config.max_weight + 1e-9).any():
        raise ValueError("Position limit violation")
    if abs(weights.sum()) > config.net_tolerance + 1e-9:
        raise ValueError("Net exposure violation")
    return weights
