"""Validated daily equity panels and lagged monthly universe membership."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

REQUIRED = ["date", "ticker", "open", "high", "low", "close", "adjusted_open", "adjusted_close", "volume", "market_cap"]


def load_data(path, config=None):
    return validate_data(pd.read_csv(path, dtype={"ticker": str}),
                         require_market_cap=config is None or config.universe_mode == "lagged_market_cap")


def validate_data(frame, require_market_cap=True):
    required = REQUIRED if require_market_cap else [name for name in REQUIRED if name != "market_cap"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    df = frame.copy()
    df["date"] = pd.to_datetime(df.date, errors="raise")
    if df.date.isna().any() or df.ticker.isna().any() or (df.ticker.astype(str).str.len() == 0).any():
        raise ValueError("Dates and tickers cannot be missing")
    if df.date.dt.tz is not None or (df.date != df.date.dt.normalize()).any():
        raise ValueError("Dates must be timezone-naive daily dates")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker rows")
    for name in required[2:]:
        df[name] = pd.to_numeric(df[name], errors="raise")
        if not np.isfinite(df[name]).all():
            raise ValueError("Nonfinite data in " + name)
        if (df[name] < 0).any() or (name not in ("volume", "market_cap") and (df[name] == 0).any()):
            raise ValueError("Invalid negative/zero data in " + name)
    if ((df.high < df[["open", "close", "low"]].max(axis=1))
            | (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC prices")
    if "eligible" not in df:
        df["eligible"] = True
    else:
        mapping = {"true": True, "false": False, "1": True, "0": False}
        df["eligible"] = df.eligible.astype(str).str.lower().map(mapping)
        if df.eligible.isna().any():
            raise ValueError("eligible must be boolean or 0/1")
    df["ticker"] = df.ticker.astype(str)
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


class MarketData:
    def __init__(self, frame, config):
        self.frame = validate_data(frame, require_market_cap=config.universe_mode == "lagged_market_cap")
        self.config = config
        self.excluded_dates = []
        if config.missing_data_policy == "skip_incomplete_sessions":
            observed = self.frame.assign(present=True).pivot(index="date", columns="ticker", values="present").notna()
            # Only absence after an asset's first observation counts as a gap;
            # pre-IPO periods never exclude a market session.
            incomplete = (observed.cummax() & ~observed).any(axis=1)
            self.excluded_dates = list(observed.index[incomplete])
            self.frame = self.frame[~self.frame.date.isin(self.excluded_dates)].reset_index(drop=True)
        self.dates = pd.DatetimeIndex(sorted(self.frame.date.unique()))
        self.fields = {name: self.frame.pivot(index="date", columns="ticker", values=name)
                       for name in REQUIRED[2:] + ["eligible"] if name in self.frame.columns}
        if config.universe_mode == "fixed":
            members = sorted(self.frame.ticker.unique())
            if len(members) > config.universe_size:
                raise ValueError("Fixed universe_size must accommodate every supplied ticker")
            self._universes = {date: members for date in self.dates}
        else:
            self._universes = self._build_universes()

    def _build_universes(self):
        """Rank a prior month's last observed trading day, never today's cap."""
        snapshots = {}
        for date, rows in self.frame.groupby("date", sort=True):
            eligible = rows[(rows.eligible) & (rows.close >= self.config.min_price)
                            & (rows.volume >= self.config.min_volume) & (rows.market_cap > 0)]
            ranked = eligible.sort_values(["market_cap", "ticker"], ascending=[False, True])
            snapshots[date.to_period("M")] = ranked.ticker.head(self.config.universe_size).tolist()
        return {date: snapshots.get(date.to_period("M") - 1, []) for date in self.dates}

    def universe(self, date):
        # Current-day suspensions/eligibility are observable at decision close.
        members = self._universes[pd.Timestamp(date)]
        close = self.fields["close"].loc[date]
        volume = self.fields["volume"].loc[date]
        eligible = self.fields["eligible"].loc[date]
        return pd.Index([t for t in members if pd.notna(close[t]) and volume[t] > 0 and eligible[t] == True
                         and (self.config.universe_mode != "fixed" or
                              (close[t] >= self.config.min_price and volume[t] >= self.config.min_volume))])

    def history(self, index):
        start = max(0, index + 1 - self.config.lookback_days)
        return {name: values.iloc[start:index + 1].copy() for name, values in self.fields.items()}


def synthetic_data(start="2017-01-02", end="2025-12-31", assets=64, seed=7):
    """Toy stationary market. Not evidence of an investable strategy or PIT vendor data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    market = rng.normal(0.0001, 0.008, (n, 1))
    returns = market + rng.normal(0.0, 0.018, (n, assets))
    for i in range(1, n):
        returns[i] += -0.06 * (returns[i - 1] - market[i - 1])
    opens = 30 * np.exp(np.cumsum(returns, axis=0))
    closes = opens * np.exp(rng.normal(0, 0.005, (n, assets)))
    shares = rng.uniform(1e7, 1e9, assets)
    return pd.DataFrame({
        "date": np.repeat(dates, assets),
        "ticker": np.tile(["S%04d" % i for i in range(assets)], n),
        "open": opens.ravel(), "close": closes.ravel(),
        "high": (np.maximum(opens, closes) * 1.01).ravel(),
        "low": (np.minimum(opens, closes) * 0.99).ravel(),
        "adjusted_open": opens.ravel(), "adjusted_close": closes.ravel(),
        "volume": rng.integers(200000, 2000000, (n, assets)).ravel(),
        "market_cap": (closes * shares).ravel(), "eligible": True,
    })


def prepare_partitions(source, directory, config):
    """Research file ends at validation end. Evaluator file has warm-up and test."""
    df = load_data(source, config)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    research = df[df.date <= pd.Timestamp(config.validation.end)]
    hidden = df[df.date <= pd.Timestamp(config.test.end)]
    if research.empty or not (df.date >= pd.Timestamp(config.test.start)).any():
        raise ValueError("Source must contain research and hidden periods")
    research.to_csv(root / "research.csv", index=False)
    hidden.to_csv(root / "evaluator.csv", index=False)
    return root / "research.csv", root / "evaluator.csv"
