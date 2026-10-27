import numpy as np
import pandas as pd


def _zero(index):
    return pd.Series(0.0, index=index, dtype=float)


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    out = _zero(tickers)
    if len(tickers) == 0:
        return out

    ac_df = history.get('adjusted_close')
    if ac_df is None:
        return out
    cols = [t for t in tickers if t in ac_df.columns]
    if len(cols) < 10:
        return out

    ac = ac_df.reindex(columns=cols).tail(320).astype(float)
    if len(ac) < 30:
        return out

    vol_df = history.get('volume')
    if vol_df is not None:
        vol = vol_df.reindex(columns=cols).tail(320).astype(float)
    else:
        vol = None

    last = ac.iloc[-1]
    n = len(ac)

    if n >= 253:
        mom = ac.iloc[-22] / ac.iloc[-253] - 1.0
    elif n >= 127:
        mom = ac.iloc[-22] / ac.iloc[-127] - 1.0
    else:
        mom = ac.iloc[-1] / ac.iloc[0] - 1.0

    if n >= 22:
        rev = -(ac.iloc[-1] / ac.iloc[-22] - 1.0)
    else:
        rev = pd.Series(0.0, index=ac.columns)

    ret = ac.pct_change(fill_method=None)
    vol60 = ret.tail(60).std(ddof=0)
    vol_ann = vol60 * np.sqrt(252.0)
    mom_adj = mom / (vol_ann + 0.10)

    ma = ac.rolling(200, min_periods=100).mean().iloc[-1]
    trend = (last > ma).astype(float)

    if vol is not None:
        dv = (ac * vol.reindex_like(ac)).tail(20).median()
    else:
        dv = pd.Series(0.0, index=ac.columns)

    mom = mom.replace([np.inf, -np.inf], np.nan)
    rev = rev.replace([np.inf, -np.inf], np.nan)
    vol_ann = vol_ann.replace([np.inf, -np.inf], np.nan)
    mom_adj = mom_adj.replace([np.inf, -np.inf], np.nan)
    dv = dv.replace([np.inf, -np.inf], np.nan)
    trend = trend.replace([np.inf, -np.inf], np.nan)

    valid = (
        last.notna()
        & (last > 0.0)
        & mom_adj.notna()
        & vol_ann.notna()
        & (vol_ann > 0.0)
        & dv.notna()
        & (dv > 0.0)
    )

    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.reindex(columns=cols).iloc[-1].fillna(False).astype(bool)
            valid = valid & e.reindex(valid.index).fillna(False)
        except Exception:
            pass

    if valid.sum() >= 80:
        cut = dv[valid].quantile(0.20)
        valid = valid & (dv >= cut)

    if valid.sum() < 20:
        return out

    def rp(s):
        r = s.where(valid).rank(pct=True)
        return ((r - 0.5) * 2.0).fillna(0.0)

    r_mom = rp(mom_adj)
    r_rev = rp(rev)
    r_low = rp(-vol_ann)
    r_trd = rp(trend)
    r_liq = rp(np.log1p(dv))

    score = (
        0.40 * r_mom
        + 0.25 * r_rev
        + 0.20 * r_low
        + 0.10 * r_trd
        + 0.05 * r_liq
    )

    mkt_ret = ret.mean(axis=1, skipna=True)
    if mkt_ret.notna().sum() >= 30:
        mvol = mkt_ret.tail(20).std(ddof=0) * np.sqrt(252.0)
        level = (1.0 + mkt_ret.fillna(0.0)).cumprod()
        ma_m = level.rolling(100, min_periods=60).mean().iloc[-1]
        up = level.iloc[-1] >= ma_m if np.isfinite(ma_m) else True
    else:
        mvol = 0.20
        up = True
    if not np.isfinite(mvol):
        mvol = 0.25

    if mvol > 0.35:
        gross = 0.50
    elif mvol > 0.25:
        gross = 0.70
    else:
        gross = 0.90
    if not up:
        gross *= 0.80

    ss = score.where(valid).dropna()
    if len(ss) < 20:
        return out

    k = min(35, len(ss) // 2)
    if k <= 0:
        return out

    order = ss.sort_values(ascending=False, kind='mergesort').index
    longs = order[:k]
    shorts = order[-k:]
    w = min(0.02, gross / (2.0 * k))

    out.loc[longs] = w
    out.loc[shorts] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out = out.clip(-0.02, 0.02)
    return out.astype(float)
