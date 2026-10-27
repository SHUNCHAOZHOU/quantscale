import numpy as np
import pandas as pd


def _safe_rank(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 130:
        return zero
    if not ac.index.is_monotonic_increasing:
        ac = ac.sort_index()

    n = ac.shape[0]
    reb = ((n - 1) // 63) * 63
    if reb < 125:
        return zero
    start = max(0, reb - 300)

    px = ac.iloc[start:reb + 1].reindex(columns=tickers).astype(float).copy()
    if px.shape[0] < 125:
        return zero
    px = px.replace([np.inf, -np.inf], np.nan)
    last = px.iloc[-1]
    valid = last.notna() & (last > 0) & (px.notna().sum(axis=0) >= 120)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        esub = elig.reindex(index=px.index, columns=tickers)
        if len(esub) > 0:
            valid = valid & esub.iloc[-1].fillna(False).astype(bool)

    vol = history.get('volume')
    if isinstance(vol, pd.DataFrame):
        if not vol.index.is_monotonic_increasing:
            vol = vol.sort_index()
        vsub = vol.reindex(index=px.index, columns=tickers).astype(float)
        dollar = vsub * px
        liq = dollar.tail(63).median(axis=0)
        valid = valid & liq.notna() & (liq > 0)
        if int(valid.sum()) >= 20:
            lq = liq[valid].dropna()
            if len(lq) >= 10:
                thresh = lq.quantile(0.20)
                valid = valid & (liq >= thresh)

    div_yield = pd.Series(0.0, index=tickers)
    div = history.get('dividends')
    if isinstance(div, pd.DataFrame):
        if not div.index.is_monotonic_increasing:
            div = div.sort_index()
        dsub = div.reindex(index=px.index, columns=tickers).fillna(0.0).astype(float)
        dsum = dsub.tail(252).sum(axis=0)
        cdf = history.get('close')
        if isinstance(cdf, pd.DataFrame):
            if not cdf.index.is_monotonic_increasing:
                cdf = cdf.sort_index()
            csub = cdf.reindex(index=px.index, columns=tickers).astype(float)
            price_for_div = csub.iloc[-1]
        else:
            price_for_div = last
        dy = dsum / price_for_div.replace(0.0, np.nan)
        div_yield = dy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret = px.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    window = ret.tail(126)

    vol126 = window.std(axis=0) * np.sqrt(252.0)
    vol126 = vol126.replace([np.inf, -np.inf], np.nan)

    mkt = window.mean(axis=1)
    mkt_c = mkt - mkt.mean()
    x_c = window.sub(window.mean(axis=0), axis=1)
    denom = float((mkt_c ** 2).sum())
    if denom > 0:
        beta = x_c.mul(mkt_c, axis=0).sum(axis=0) / denom
    else:
        beta = pd.Series(0.0, index=tickers)
    beta = beta.replace([np.inf, -np.inf], np.nan)

    mean_ret = window.mean(axis=0)
    std_ret = window.std(axis=0).replace(0.0, np.nan)
    sharpe126 = mean_ret / std_ret * np.sqrt(252.0)
    sharpe126 = sharpe126.replace([np.inf, -np.inf], np.nan)

    def _ret_lag(lag):
        if len(px) > lag:
            return px.iloc[-1] / px.iloc[-1 - lag] - 1.0
        return pd.Series(np.nan, index=tickers)

    ret21 = _ret_lag(21)
    ret21 = ret21.replace([np.inf, -np.inf], np.nan)

    high = px.tail(252).max(axis=0)
    dd = last / high.replace(0.0, np.nan) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan)

    valid_idx = valid[valid].index
    if len(valid_idx) < 10:
        return zero

    vv = vol126.loc[valid_idx]
    if vv.isna().all():
        vv = pd.Series(1.0, index=valid_idx)
    else:
        vv = vv.fillna(vv.median())

    bb = beta.loc[valid_idx]
    if bb.isna().all():
        bb = pd.Series(0.0, index=valid_idx)
    else:
        bb = bb.fillna(bb.median())

    ss = sharpe126.loc[valid_idx]
    if ss.isna().all():
        ss = pd.Series(0.0, index=valid_idx)
    else:
        ss = ss.fillna(ss.median())

    ddv = dd.loc[valid_idx]
    if ddv.notna().any():
        ddv = ddv.fillna(ddv.median())
    else:
        ddv = pd.Series(0.0, index=valid_idx)

    r21 = ret21.loc[valid_idx].fillna(0.0)
    dvv = div_yield.loc[valid_idx]

    def rp(s):
        return s.replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True)

    score = (
        0.35 * rp(dvv)
        + 0.25 * (1.0 - rp(vv))
        + 0.15 * (1.0 - rp(bb))
        + 0.15 * rp(ss)
        + 0.10 * rp(ddv)
        - 0.10 * rp(r21)
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(25, len(score) // 2)
    if n_side < 1:
        return zero
    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=tickers)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    mkt_all = ret.mean(axis=1)
    mvol = mkt_all.tail(63).std() * np.sqrt(252.0)
    if np.isfinite(mvol) and mvol > 0:
        gross_target = min(1.0, max(0.60, 0.22 / float(mvol)))
    else:
        gross_target = 1.0
    weights = weights * gross_target

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    return weights.reindex(tickers).fillna(0.0).astype(float)