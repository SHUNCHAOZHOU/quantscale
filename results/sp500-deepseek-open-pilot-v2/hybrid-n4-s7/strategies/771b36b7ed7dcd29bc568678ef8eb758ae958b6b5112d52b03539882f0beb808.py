import numpy as np
import pandas as pd


def _frame(history, names):
    for n in names:
        if n in history:
            obj = history[n]
            if isinstance(obj, pd.DataFrame):
                return obj
    return None


def generate_weights(history, universe):
    u = pd.Index(universe)
    out = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return out
    close = _frame(history, ['adjusted_close', 'close'])
    if close is None or close.empty:
        return out
    close = close.copy()
    cols = [c for c in u if c in close.columns]
    if len(cols) < 4:
        return out
    close = close[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    dates = close.index
    if len(dates) < 60:
        return out
    max_date = dates[-1]
    first = pd.Timestamp(year=max_date.year, month=max_date.month, day=1)
    prev_end = first - pd.Timedelta(days=1)
    cand = dates[dates <= prev_end]
    if len(cand) > 0:
        ref_date = cand[-1]
    else:
        ref_date = max_date
    px = close.loc[:ref_date]
    if len(px) < 60:
        px = close
        ref_date = max_date
    p = px.replace([np.inf, -np.inf], np.nan)
    nrows = len(p)
    mom_lb = 126 if nrows > 126 else 60
    skip = 5
    mom = p.shift(skip) / p.shift(mom_lb) - 1.0
    ret5 = p / p.shift(skip) - 1.0
    rets = p / p.shift(1) - 1.0
    vol = rets.rolling(window=60, min_periods=30).std()
    ma = p.rolling(window=100, min_periods=50).mean()
    trend = p / ma - 1.0
    last_mom = mom.iloc[-1]
    last_ret5 = ret5.iloc[-1]
    last_vol = vol.iloc[-1]
    last_trend = trend.iloc[-1]
    last_price = p.iloc[-1]
    valid = pd.Series(True, index=cols, dtype=bool)
    valid = valid & last_price.notna() & (last_price > 0)
    valid = valid & last_mom.notna() & np.isfinite(last_mom)
    valid = valid & last_ret5.notna() & np.isfinite(last_ret5)
    valid = valid & last_vol.notna() & np.isfinite(last_vol) & (last_vol > 0)
    valid = valid & last_trend.notna() & np.isfinite(last_trend)
    elig = history.get('eligible', None)
    if isinstance(elig, pd.DataFrame):
        e = elig.reindex(index=close.index, columns=cols).ffill(limit=5)
        if len(e) > 0:
            ecur = e.iloc[-1].fillna(True).astype(bool)
            valid = valid & ecur.reindex(cols).fillna(True)
            esub = e.loc[:ref_date]
            if len(esub) > 0:
                eref = esub.iloc[-1].fillna(True).astype(bool)
                valid = valid & eref.reindex(cols).fillna(True)
    volf = history.get('volume', None)
    if isinstance(volf, pd.DataFrame):
        vv = volf.reindex(index=close.index, columns=cols).loc[:ref_date]
        raw_close = _frame(history, ['close', 'adjusted_close'])
        if raw_close is not None:
            rc = raw_close.reindex(index=close.index, columns=cols).loc[:ref_date]
        else:
            rc = p
        adv = (rc * vv).rolling(20, min_periods=5).mean().iloc[-1]
        valid = valid & adv.notna() & (adv > 1e5)
    sel = list(valid[valid].index)
    if len(sel) < 4:
        return out
    m = last_mom.reindex(sel).astype(float)
    r = last_ret5.reindex(sel).astype(float)
    v = last_vol.reindex(sel).astype(float)
    t = last_trend.reindex(sel).astype(float)
    v_floor = v.quantile(0.05) if len(v) >= 20 else v.min()
    if (not np.isfinite(v_floor)) or v_floor <= 0:
        v_floor = 1e-6
    ram = m / v.clip(lower=v_floor)
    rank_ram = ram.rank(pct=True, method='average') - 0.5
    rank_rev = (-r).rank(pct=True, method='average') - 0.5
    rank_trend = t.rank(pct=True, method='average') - 0.5
    score = 0.65 * rank_ram + 0.25 * rank_rev + 0.10 * rank_trend
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return out
    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return out
    sorted_scores = score.sort_values(ascending=False)
    top = list(sorted_scores.index[:n_side])
    bottom = list(sorted_scores.index[-n_side:])
    w = min(0.02, 0.499 / float(n_side))
    if (not np.isfinite(w)) or w <= 0:
        return out
    out.loc[top] = w
    out.loc[bottom] = -w
    overlap = set(top).intersection(set(bottom))
    if len(overlap) > 0:
        for c in overlap:
            out.loc[c] = 0.0
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    gross = float(np.abs(out.values).sum())
    if gross > 1.0:
        out = out * (0.999 / gross)
    return out