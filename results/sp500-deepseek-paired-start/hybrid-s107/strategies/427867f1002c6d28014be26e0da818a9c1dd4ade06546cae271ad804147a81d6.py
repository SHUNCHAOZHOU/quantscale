import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None or not isinstance(ac, pd.DataFrame):
        return zero

    if not ac.index.is_monotonic_increasing:
        ac = ac.sort_index()

    cols = [c for c in tickers if c in ac.columns]
    if len(cols) == 0:
        return zero

    px = ac[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    n = len(px)
    if n < 80:
        return zero

    # Deterministic 21-session rebalance grid. Between grid dates, the
    # signal is computed through the last grid date, so weights are stale
    # by design and turnover is suppressed.
    reb = ((n - 1) // 21) * 21
    if reb < 63:
        return zero

    start = max(0, reb - 320)
    data = px.iloc[start:reb + 1]
    m = len(data)
    if m < 60:
        return zero

    last = data.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (data.tail(260).notna().sum() >= 60)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        try:
            row = elig.reindex(index=[data.index[-1]], columns=cols)
            if len(row) > 0:
                e = row.iloc[0].fillna(False).astype(bool)
                valid = valid & e
        except Exception:
            pass

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        if not vol_df.index.is_monotonic_increasing:
            vol_df = vol_df.sort_index()
        vcols = [c for c in cols if c in vol_df.columns]
        if len(vcols) > 0:
            vsub = vol_df.reindex(index=data.index, columns=vcols).astype(float)
            csub = data[vcols]
            dollar = (vsub * csub).tail(60)
            liq = dollar.median(axis=0).reindex(cols)
            valid = valid & liq.notna() & (liq > 0)
            if liq.notna().sum() >= 30:
                thresh = liq.quantile(0.20)
                valid = valid & (liq >= thresh)

    valid_names = valid[valid].index
    if len(valid_names) < 20:
        return zero

    def _mom(start_lag, end_lag):
        if m <= end_lag + 1:
            return None
        end_pos = m - 1 - end_lag
        start_pos = m - 1 - start_lag
        if start_pos < 0 or end_pos <= start_pos:
            return None
        return data.iloc[end_pos] / data.iloc[start_pos] - 1.0

    mom12 = _mom(252, 21)
    if mom12 is None:
        mom12 = _mom(126, 21)
    if mom12 is None:
        mom12 = _mom(63, 21)
    if mom12 is None:
        return zero

    def rp(x):
        return x.rank(method='average', pct=True)

    score = rp(mom12.reindex(valid_names))
    mom6 = _mom(126, 21)
    if mom6 is not None:
        score = 0.75 * score + 0.25 * rp(mom6.reindex(valid_names))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(40, len(score) // 2)
    if n_side < 5:
        return zero

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index
    selected = list(longs) + list(shorts)
    wmag = min(0.02, 0.5 / float(n_side))

    w_raw = pd.Series(0.0, index=selected, dtype=float)
    w_raw.loc[longs] = wmag
    w_raw.loc[shorts] = -wmag

    # Explicit beta neutralization against an equal-weight market proxy.
    rets = data[valid_names].pct_change().tail(126)
    if len(rets) >= 30:
        mkt = rets.mean(axis=1).fillna(0.0)
        mkt_d = mkt - mkt.mean()
        var = float((mkt_d * mkt_d).sum())
        if var > 1e-18:
            rets_f = rets.fillna(0.0)
            beta = (rets_f.mul(mkt_d, axis=0).sum()) / var
            b = beta.reindex(selected).fillna(1.0)
            b = b - b.mean()
            denom = float((b * b).sum())
            if denom > 1e-18:
                c = float((w_raw * b).sum()) / denom
                w_raw = w_raw - c * b

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[w_raw.index] = w_raw.values
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
        weights = weights.clip(-0.02, 0.02)

    net = float(weights.sum())
    if abs(net) > 1e-12:
        pos = weights[weights > 0].index
        neg = weights[weights < 0].index
        cnt = len(pos) + len(neg)
        if cnt > 0:
            d = net / float(cnt)
            weights.loc[pos] -= d
            weights.loc[neg] -= d

    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 0.01:
        pos = weights[weights > 0].index
        neg = weights[weights < 0].index
        cnt = len(pos) + len(neg)
        if cnt > 0:
            d = net / float(cnt)
            weights.loc[pos] -= d
            weights.loc[neg] -= d

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.reindex(tickers).fillna(0.0).astype(float)