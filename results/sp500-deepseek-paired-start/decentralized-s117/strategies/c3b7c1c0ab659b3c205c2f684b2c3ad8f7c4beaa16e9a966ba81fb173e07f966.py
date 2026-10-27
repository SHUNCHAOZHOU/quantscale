import math
import numpy as np
import pandas as pd


def _get_df(history, key, tickers):
    df = history.get(key)
    if df is None:
        return None
    try:
        return df.reindex(columns=tickers)
    except Exception:
        return None


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = _get_df(history, 'adjusted_close', tickers)
    if ac is None or ac.dropna(how='all').empty:
        ac = _get_df(history, 'close', tickers)
    if ac is None or ac.empty:
        return zero

    ac = ac.tail(420)
    if len(ac) < 80:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1

    signal_pos = last_pos
    try:
        periods = dates.to_period('M')
        cur = periods[-1]
        prior = np.where(periods < cur)[0]
        if len(prior) > 0:
            signal_pos = int(prior[-1])
    except Exception:
        signal_pos = last_pos

    if signal_pos < 60:
        signal_pos = last_pos

    price = ac.iloc[:signal_pos + 1].astype(float)
    if len(price) < 80:
        return zero

    last_sig = price.iloc[-1]
    latest_price = ac.iloc[-1].astype(float)

    elig_df = _get_df(history, 'eligible', tickers)
    if elig_df is not None and len(elig_df) > 0:
        try:
            e = elig_df.reindex(index=ac.index).iloc[-1]
            e = e.fillna(False).astype(bool)
        except Exception:
            e = pd.Series(True, index=tickers)
    else:
        e = pd.Series(True, index=tickers)

    cnt = price.notna().sum()
    valid = (
        latest_price.notna() & np.isfinite(latest_price) & (latest_price > 2.0)
        & last_sig.notna() & np.isfinite(last_sig) & (last_sig > 2.0)
        & (cnt >= 80) & e
    )

    raw = _get_df(history, 'close', tickers)
    if raw is None:
        raw = ac
    raw_sig = raw.reindex(index=ac.index).iloc[:signal_pos + 1].astype(float)

    vol_df = _get_df(history, 'volume', tickers)
    if vol_df is not None:
        v = vol_df.reindex(index=ac.index).iloc[:signal_pos + 1].tail(21).astype(float)
        c = raw_sig.tail(21)
        adv = (v * c).median()
        adv_ok = adv.notna() & np.isfinite(adv) & (adv > 0)
        if adv_ok.sum() >= 20:
            thresh = max(250000.0, float(adv[adv_ok].quantile(0.20)))
            valid = valid & adv_ok & (adv >= thresh)

    rets = price.pct_change(fill_method=None)
    vol60 = rets.tail(60).std(ddof=0).replace([np.inf, -np.inf], np.nan)
    valid = valid & vol60.notna() & (vol60 > 0.005)

    def form(k):
        if len(price) > k:
            return price.iloc[-1] / price.iloc[-1 - k] - 1.0
        return pd.Series(np.nan, index=tickers)

    r5 = form(5)
    r21 = form(21)
    r252 = form(252)

    if len(price) >= 252:
        mom = price.iloc[-22] / price.iloc[-252] - 1.0
    elif len(price) >= 126:
        mom = price.iloc[-22] / price.iloc[-126] - 1.0
    else:
        mom = r252

    max21 = rets.tail(21).max()

    daily = rets.tail(260)
    mkt_ret = daily.mean(axis=1, skipna=True).dropna()
    if len(mkt_ret) >= 60:
        mkt_level = (1.0 + mkt_ret).cumprod()
        if len(mkt_level) >= 200:
            trend_up = bool(mkt_level.iloc[-1] > mkt_level.tail(200).mean())
        else:
            trend_up = bool(mkt_level.iloc[-1] > mkt_level.mean())
        mvol = float(mkt_ret.tail(60).std(ddof=0) * math.sqrt(252.0))
        roll_vol = mkt_ret.rolling(60).std(ddof=0) * math.sqrt(252.0)
        rv = roll_vol.dropna()
        if len(rv) > 20:
            high_vol = bool(mvol > float(rv.quantile(0.75)))
        else:
            high_vol = bool(mvol > 0.20)
    else:
        trend_up = True
        high_vol = False

    rev_score = (-r21 / (vol60 * math.sqrt(21.0))).replace([np.inf, -np.inf], np.nan)
    mom_score = (mom / (vol60 * math.sqrt(252.0))).replace([np.inf, -np.inf], np.nan)
    lowvol_score = (-vol60).replace([np.inf, -np.inf], np.nan)
    max_score = (-max21).replace([np.inf, -np.inf], np.nan)

    r_rev = rev_score.where(valid).rank(method='average', pct=True).fillna(0.5)
    r_mom = mom_score.where(valid).rank(method='average', pct=True).fillna(0.5)
    r_low = lowvol_score.where(valid).rank(method='average', pct=True).fillna(0.5)
    r_max = max_score.where(valid).rank(method='average', pct=True).fillna(0.5)

    if trend_up and not high_vol:
        score = 0.45 * r_rev + 0.30 * r_mom + 0.15 * r_low + 0.10 * r_max
        gross_target = 1.0
    else:
        score = 0.20 * r_rev + 0.10 * r_mom + 0.50 * r_low + 0.20 * r_max
        gross_target = 0.60 if high_vol else 0.80

    score = score.where(valid).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(25, len(score) // 2)
    if n_side < 5:
        return zero

    ordered = score.sort_values()
    shorts = ordered.index[:n_side]
    longs = ordered.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side)) * gross_target

    weights = zero.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.astype(float)