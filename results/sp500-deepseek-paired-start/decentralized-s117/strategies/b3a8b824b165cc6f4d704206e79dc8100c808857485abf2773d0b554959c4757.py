import numpy as np
import pandas as pd
import math

def _rank_pct(s):
    s = pd.to_numeric(s, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index)
    return s.rank(method='average', pct=True)

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return weights

    adj = history.get('adjusted_close')
    raw = history.get('close')
    vol_df = history.get('volume')
    elig_df = history.get('eligible')

    if adj is None:
        adj = raw
    if adj is None or adj.empty:
        return weights
    try:
        adj = adj.reindex(columns=tickers)
    except Exception:
        return weights

    if raw is None:
        raw = adj
    else:
        try:
            raw = raw.reindex(columns=tickers)
        except Exception:
            raw = adj

    dates = adj.index
    if len(dates) < 70:
        return weights

    current = dates[-1]
    try:
        month_start = pd.Timestamp(year=current.year, month=current.month, day=1)
    except Exception:
        month_start = current
    prior = dates[dates < month_start]
    if len(prior) > 0:
        signal_date = prior[-1]
    else:
        signal_date = dates[-1]

    p_all = adj.loc[:signal_date]
    p = p_all.tail(320)
    if len(p) < 70:
        return weights

    r_all = raw.loc[:signal_date]
    r = r_all.reindex(index=p.index, columns=tickers)

    v = None
    if vol_df is not None:
        try:
            v = vol_df.reindex(columns=tickers).loc[:signal_date].reindex(index=p.index)
        except Exception:
            v = None

    last_adj = pd.to_numeric(p.iloc[-1], errors='coerce').astype(float)
    if r is not None and len(r) > 0:
        last_raw = pd.to_numeric(r.iloc[-1], errors='coerce').astype(float)
    else:
        last_raw = last_adj

    valid = last_adj.notna() & np.isfinite(last_adj) & (last_adj > 0.0)
    valid = valid & last_raw.notna() & np.isfinite(last_raw) & (last_raw > 2.0)

    cnt = p.notna().sum()
    valid = valid & (cnt >= min(120, max(50, len(p) // 2)))

    if elig_df is not None:
        try:
            e = elig_df.reindex(columns=tickers).loc[:signal_date]
            if len(e) > 0:
                el = e.iloc[-1].fillna(False).astype(bool)
                valid = valid & el
        except Exception:
            pass

    rets = p / p.shift(1) - 1.0
    vol = rets.tail(120).std(ddof=0) * math.sqrt(252.0)
    vol = pd.to_numeric(vol, errors='coerce').replace([np.inf, -np.inf], np.nan)
    vol = vol.clip(lower=0.05, upper=2.0)
    valid = valid & vol.notna() & np.isfinite(vol) & (vol > 0.0)

    if len(p) > 253:
        r12 = p.iloc[-22] / p.iloc[-253] - 1.0
    elif len(p) > 127:
        r12 = p.iloc[-22] / p.iloc[-127] - 1.0
    else:
        r12 = pd.Series(np.nan, index=tickers)

    if len(p) > 127:
        r6 = p.iloc[-22] / p.iloc[-127] - 1.0
    elif len(p) > 64:
        r6 = p.iloc[-22] / p.iloc[-64] - 1.0
    else:
        r6 = pd.Series(np.nan, index=tickers)

    r12a = r12 / vol
    r6a = r6 / vol

    rank12 = _rank_pct(r12a).fillna(0.5)
    rank6 = _rank_pct(r6a).fillna(0.5)
    rankvol = _rank_pct(-vol).fillna(0.5)

    score = 0.55 * rank12 + 0.30 * rank6 + 0.15 * rankvol

    if v is not None:
        try:
            adv = (r * v).tail(60).median()
            adv = pd.to_numeric(adv, errors='coerce').replace([np.inf, -np.inf], np.nan)
            if (valid & adv.notna()).sum() >= 20:
                q = adv[valid & adv.notna()].quantile(0.2)
                thr = max(5e5, float(q) if np.isfinite(q) else 0.0)
                valid = valid & adv.notna() & (adv >= thr)
        except Exception:
            pass

    valid = valid & score.notna() & np.isfinite(score)

    if valid.sum() < 6:
        valid = last_adj.notna() & np.isfinite(last_adj) & (last_adj > 0.0)
        valid = valid & last_raw.notna() & np.isfinite(last_raw) & (last_raw > 2.0)
        valid = valid & score.notna() & np.isfinite(score)

    if valid.sum() < 4:
        return weights

    s = score[valid].dropna().sort_values()
    n_side = min(30, len(s) // 2)
    if n_side < 1:
        return weights

    w = min(0.02, 0.5 / float(n_side))
    longs = s.index[-n_side:]
    shorts = s.index[:n_side]

    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)