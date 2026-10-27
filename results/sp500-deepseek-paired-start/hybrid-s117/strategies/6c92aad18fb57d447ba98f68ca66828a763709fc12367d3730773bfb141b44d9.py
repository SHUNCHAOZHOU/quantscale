import numpy as np
import pandas as pd
import math


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return weights

    def get_df(name):
        if not isinstance(history, dict):
            return None
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            try:
                return obj.reindex(columns=tickers)
            except Exception:
                return None
        return None

    ac_full = get_df('adjusted_close')
    if ac_full is None or ac_full.dropna(how='all').shape[1] == 0:
        ac_full = get_df('close')
    if ac_full is None or len(ac_full.index) < 60:
        return weights

    ac_full = ac_full.astype(float)
    ac = ac_full.iloc[-320:].copy()
    idx = ac.index
    if len(ac) < 60:
        return weights

    raw_full = get_df('close')
    if raw_full is None:
        raw = ac.copy()
    else:
        raw = raw_full.reindex(index=idx, columns=tickers).astype(float)
        raw = raw.fillna(ac)

    vol_full = get_df('volume')
    vol = None
    if vol_full is not None:
        vol = vol_full.reindex(index=idx, columns=tickers).astype(float)

    elig_full = get_df('eligible')
    elig = None
    if elig_full is not None:
        elig = elig_full.reindex(index=idx, columns=tickers)

    last_pos = len(ac) - 1
    signal_pos = last_pos
    if isinstance(idx, pd.DatetimeIndex):
        wd = idx.weekday.values
        mons = np.where(wd[:last_pos + 1] == 0)[0]
        if len(mons) > 0:
            signal_pos = int(mons[-1])
    if signal_pos < 60:
        signal_pos = last_pos

    sig_ac = ac.iloc[:signal_pos + 1]
    n_sig = len(sig_ac)
    last = sig_ac.iloc[-1].astype(float)

    valid = last.notna() & np.isfinite(last) & (last > 1.0)
    raw_last = raw.iloc[signal_pos].astype(float)
    valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 1.0)

    if elig is not None:
        e = elig.iloc[signal_pos].reindex(tickers)
        try:
            e = e.fillna(False).astype(bool)
        except Exception:
            e = e.fillna(0).astype(float).astype(bool)
        valid = valid & e

    rets = sig_ac / sig_ac.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)

    if n_sig >= 22:
        vol126 = rets.tail(min(126, n_sig - 1)).std(ddof=0) * math.sqrt(252.0)
        vol21 = rets.tail(min(21, n_sig - 1)).std(ddof=0) * math.sqrt(252.0)
    else:
        vol126 = pd.Series(np.nan, index=tickers)
        vol21 = pd.Series(np.nan, index=tickers)
    vol126 = vol126.replace([0.0, np.inf, -np.inf], np.nan)

    def form(k):
        if n_sig > k:
            prev = sig_ac.iloc[-1 - k].astype(float)
            out = last / prev - 1.0
            return out.replace([np.inf, -np.inf], np.nan)
        return pd.Series(np.nan, index=tickers)

    r21 = form(21)
    r126 = form(126)
    r252 = form(252)

    mom = 0.5 * r126 + 0.5 * (r252 - r21)
    mom = mom.where(np.isfinite(mom), r126)
    mom = mom.where(np.isfinite(mom), r252)
    mom = mom.where(np.isfinite(mom), r21)
    mom = mom.replace([np.inf, -np.inf], np.nan)

    hi_win = min(252, n_sig)
    hi = sig_ac.tail(hi_win).max()
    trend = last / hi - 1.0
    trend = trend.replace([np.inf, -np.inf], np.nan)

    base_valid = valid.copy()
    if vol is not None:
        v_sig = vol.iloc[:signal_pos + 1].tail(60)
        c_sig = raw.iloc[:signal_pos + 1].tail(60)
        if v_sig.shape[0] >= 5:
            dv = (c_sig * v_sig).median(skipna=True)
            dv = dv.replace([np.inf, -np.inf], np.nan)
            liq_valid = dv.notna() & (dv > 0.0)
            joint = base_valid & liq_valid
            if joint.sum() >= 20:
                tmp = dv[joint]
                q = tmp.quantile(0.10)
                if not np.isfinite(q):
                    q = 0.0
                thresh = max(100000.0, float(q))
                valid = joint & (dv >= thresh)
            else:
                valid = joint
        else:
            valid = base_valid
    else:
        valid = base_valid

    if valid.sum() < 10:
        valid = base_valid

    mkt_ret = rets.mean(axis=1, skipna=True)
    if n_sig >= 10:
        mkt_vol = mkt_ret.tail(min(60, n_sig)).std(ddof=0) * math.sqrt(252.0)
        mkt_cum = (1.0 + mkt_ret.fillna(0.0)).tail(min(200, n_sig)).prod() - 1.0
    else:
        mkt_vol, mkt_cum = 0.15, 0.0
    if not np.isfinite(mkt_vol):
        mkt_vol = 0.15
    if not np.isfinite(mkt_cum):
        mkt_cum = 0.0
    risk_off = bool((mkt_vol > 0.22) or (mkt_cum < 0.0))

    rank_lowvol = (-vol126).rank(method='average', pct=True)
    rank_rev = (-r21).rank(method='average', pct=True)
    rank_mom = mom.rank(method='average', pct=True).fillna(0.5)
    rank_trend = trend.rank(method='average', pct=True).fillna(0.5)

    valid = (valid & vol126.notna() & np.isfinite(vol126) & (vol126 > 0.005)
             & r21.notna() & np.isfinite(r21)
             & trend.notna() & np.isfinite(trend))

    if risk_off:
        score = 0.75 * rank_lowvol + 0.25 * rank_trend
    else:
        score = (0.40 * rank_rev + 0.25 * rank_lowvol
                 + 0.20 * rank_mom + 0.15 * rank_trend)
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna() & np.isfinite(score)

    if valid.sum() < 8:
        score = (-vol126).replace([np.inf, -np.inf], np.nan)
        valid = (base_valid & vol126.notna() & np.isfinite(vol126)
                 & (vol126 > 0.005) & score.notna() & np.isfinite(score))

    m = int(valid.sum())
    if m < 4:
        return weights

    ordered = score[valid].dropna().sort_values(ascending=False)
    if len(ordered) < 4:
        return weights

    n_side = min(50, len(ordered) // 2)
    if n_side < 2:
        return weights

    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]
    wt = min(0.02, 0.5 / float(n_side))

    weights.loc[longs] = wt
    weights.loc[shorts] = -wt
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return weights