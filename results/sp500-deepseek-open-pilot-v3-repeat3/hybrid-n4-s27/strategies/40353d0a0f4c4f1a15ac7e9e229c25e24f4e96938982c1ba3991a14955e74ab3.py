import numpy as np
import pandas as pd

def _zeros(universe):
    return pd.Series(0.0, index=pd.Index(universe), dtype=float)

def generate_weights(history, universe):
    out = _zeros(universe)
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return out
    try:
        px_all = history.get('adjusted_close')
        if px_all is None:
            px_all = history.get('close')
    except Exception:
        return out
    if not isinstance(px_all, pd.DataFrame) or len(px_all) < 10:
        return out
    common = [t for t in tickers if t in px_all.columns]
    if len(common) < 10:
        return out
    mom_window = 120
    vol_window = 60
    beta_window = 120
    need = mom_window + 5
    px = px_all.loc[:, common]
    if len(px) > need:
        px = px.iloc[-need:]
    px = px.apply(pd.to_numeric, errors='coerce')
    if len(px) <= mom_window:
        return out
    last_px = px.iloc[-1]
    eligible = pd.Series(True, index=common)
    try:
        e_all = history.get('eligible')
        if isinstance(e_all, pd.DataFrame):
            e = e_all.reindex(index=px.index, columns=common)
            eligible = e.iloc[-1].fillna(True).astype(bool)
    except Exception:
        pass
    rets = px.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    vol60 = rets.tail(vol_window).std(ddof=0, skipna=True)
    cnt = rets.tail(mom_window).count()
    ret120 = px.iloc[-1] / px.iloc[-mom_window - 1] - 1.0
    R = rets.tail(beta_window)
    m = R.mean(axis=1)
    Rd = R.sub(R.mean(axis=0), axis=1)
    md = m - m.mean()
    denom = (md ** 2).sum()
    if denom <= 0 or not np.isfinite(denom):
        beta = pd.Series(1.0, index=common)
        mret = 0.0
    else:
        var = denom / max(len(Rd) - 1, 1)
        cov = Rd.mul(md, axis=0).sum() / max(len(Rd) - 1, 1)
        beta = cov / var
        vals = m.dropna().values
        mret = float(np.prod(1.0 + vals) - 1.0) if len(vals) else 0.0
    if not np.isfinite(mret):
        mret = 0.0
    resid_ret = ret120 - beta * mret
    risk_adj = resid_ret / (vol60 * np.sqrt(mom_window))
    mask = (eligible & np.isfinite(last_px) & (last_px > 0) & np.isfinite(vol60) & (vol60 > 0) & np.isfinite(ret120) & np.isfinite(beta) & np.isfinite(resid_ret) & np.isfinite(risk_adj) & (cnt >= 90))
    valid = pd.DataFrame({'vol': vol60, 'ra': risk_adj}, index=common)
    valid = valid[mask.reindex(valid.index).fillna(False)]
    if len(valid) < 10:
        return out
    vol_rank = valid['vol'].rank(method='average', pct=True)
    ra_rank = valid['ra'].rank(method='average', pct=True)
    score = 0.7 * (1.0 - vol_rank) + 0.3 * ra_rank
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return out
    n_side = min(20, len(score) // 2)
    if n_side < 5:
        return out
    ordered = score.sort_values(ascending=False)
    longs = ordered.iloc[:n_side].index
    shorts = ordered.iloc[-n_side:].index
    w = min(0.02, 0.4 / float(n_side))
    out.loc[longs] = w
    out.loc[shorts] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = out.abs().sum()
    if gross > 1.0:
        out = out * (1.0 / gross)
    if abs(out.sum()) > 0.01:
        return _zeros(universe)
    return out