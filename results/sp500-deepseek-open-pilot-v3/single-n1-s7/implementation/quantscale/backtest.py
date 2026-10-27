"""Close-t decision, open-(t+1) execution, open-(t+2) realization."""
import numpy as np
import pandas as pd

from .strategies import validate_weights


def performance(daily, annualization=252):
    if daily.empty:
        raise ValueError("Period has no complete open-to-open holding interval")
    returns = daily["net_return"].to_numpy(dtype=float)
    wealth = np.cumprod(1 + returns)
    if (wealth <= 0).any():
        raise ValueError("Portfolio insolvent")
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    # Include initial NAV so a loss on day one appears in drawdown.
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {
        "observations": int(len(returns)),
        "sharpe": float(np.mean(returns) / std * np.sqrt(annualization)) if std > 1e-12 else 0.0,
        "annualized_return": float(wealth[-1] ** (annualization / len(returns)) - 1),
        "total_return": float(wealth[-1] - 1),
        "volatility": std * float(np.sqrt(annualization)),
        "max_drawdown": float(np.max(1 - wealth / peaks)),
        "downside_risk": downside * float(np.sqrt(annualization)),
        "average_daily_turnover": float(daily.turnover.mean()),
        "total_cost": float(daily.cost.sum()),
        "total_borrow_cost": float(daily.borrow_cost.sum()),
        "active_days": int((daily.gross_exposure > 0).sum()),
    }


def run_backtest(strategy, market, period, use_prepared=True):
    try:
        return _run_backtest(strategy, market, period, use_prepared)
    finally:
        if hasattr(strategy, "close"):
            strategy.close()


def _run_backtest(strategy, market, period, use_prepared=True):
    config = market.config
    dates = market.dates
    assets = market.fields["open"].columns
    adjusted_open = market.fields["adjusted_open"]
    previous = pd.Series(0.0, index=assets)
    records = []
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
    prepared = use_prepared and hasattr(strategy, "prepare") and hasattr(strategy, "weights_at")
    if prepared:
        strategy.prepare(market)
    for i in range(len(dates) - 2):
        decision, execution, realized = dates[i:i + 3]
        # Start each evaluated partition flat; warm-up history remains available.
        if execution < start or realized > end:
            continue
        universe = market.universe(decision)
        raw_weights = strategy.weights_at(i, universe) if prepared else strategy.generate_weights(market.history(i), universe)
        weights = validate_weights(raw_weights, universe, config)
        target = weights.reindex(assets, fill_value=0.0)
        holding = target != 0
        traded = (target - previous).abs() > 1e-12
        entry = adjusted_open.iloc[i + 1]
        exit_price = adjusted_open.iloc[i + 2]
        if entry[traded | holding].isna().any() or exit_price[holding].isna().any():
            raise ValueError("Missing execution/exit price for held or traded ticker; supply delisting/terminal-price data")
        asset_return = (exit_price / entry - 1).where(holding, 0.0)
        turnover = float((target - previous).abs().sum())
        cost = turnover * config.cost_bps / 10000
        borrow = float(-target.clip(upper=0).sum()) * config.annual_borrow_bps / 10000 / config.annualization
        gross = float((target * asset_return).sum())
        net = gross - cost - borrow
        if not np.isfinite(net) or net <= -1:
            raise ValueError("Invalid PnL or portfolio insolvency")
        records.append({"decision_date": decision, "execution_date": execution,
                        "date": realized, "gross_return": gross, "net_return": net,
                        "turnover": turnover, "cost": cost, "borrow_cost": borrow,
                        "gross_exposure": float(target.abs().sum()), "net_exposure": float(target.sum())})
        # Drift holdings in NAV units before the next rebalance, including costs.
        previous = target * (1 + asset_return) / (1 + net)
    daily = pd.DataFrame(records)
    return performance(daily, config.annualization), daily
