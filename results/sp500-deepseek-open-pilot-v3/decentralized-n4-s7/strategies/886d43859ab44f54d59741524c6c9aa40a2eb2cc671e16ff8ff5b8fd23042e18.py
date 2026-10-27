import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    zeros = pd.Series(0.0, index=tickers)
    if len(tickers) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zeros

    ac = ac.reindex(columns=tickers)
    row_has_data = ac.notna().any(axis=1)
    if not row_has_data.any():
        return zeros
    t = ac.index[row_has_data][-1]

    eligible = pd.Series(True, index=tickers)
    e = history.get('eligible')
    if e is not None:
        e = e.reindex(columns=tickers)
        if t in e.index:
            eligible = e.loc[t].fillna(False).astype(bool)

    idx = ac.index[ac.index <= t]
    if len(idx) == 0:
        return zeros

    try:
        next_bday = t + pd.offsets.BDay(1)
        is_last_bday_of_month = (next_bday.month != t.month) or (next_bday.year != t.year)
    except Exception:
        is_last_bday_of_month = False

    if is_last_bday_of_month:
        sig = t
    else:
        first_of_month = pd.Timestamp(year=t.year, month=t.month, day=1)
        prior = idx[idx < first_of_month]
        sig = prior[-1] if len(prior) > 0 else idx[0]

    sub = ac.loc[:sig]
    n = len(sub)
    if n == 0:
        return zeros

    def price_at(offset):
        pos = n - 1 - offset
        if pos < 0:
            return pd.Series(np.nan, index=tickers)
        return sub.iloc[pos]

    p0 = price_at(0)
    p5 = price_at(5)
    p21 = price_at(21)
    p63 = price_at(63)
    p126 = price_at(126)
    p252 = price_at(252)

    if n >= 2:
        recent = sub.iloc[-127:] if n >= 127 else sub
        vol = recent.pct_change().iloc[-126:].std()
    else:
        vol = pd.Series(np.nan, index=tickers)

    dvol = pd.Series(np.nan, index=tickers)
    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        v = v.reindex(columns=tickers).loc[:sig]
        c = c.reindex(columns=tickers).loc[:sig]
        if len(v) > 0 and len(c) > 0:
            vv = v.iloc[-20:]
            cc = c.iloc[-20:]
            dvol = (vv * cc).median(axis=0, skipna=True).reindex(tickers)

    valid = p0.notna() & (p0 > 1.0) & p126.notna() & vol.notna() & (vol > 0)
    valid = valid & eligible.reindex(tickers).fillna(False)

    if dvol.notna().any():
        positive_dvol = dvol[dvol > 0]
        if len(positive_dvol) > 0:
            thresh = positive_dvol.quantile(0.15)
            valid = valid & dvol.notna() & (dvol >= thresh)
        else:
            valid = valid & dvol.notna()
    valid = valid.fillna(False)

    if int(valid.sum()) < 2:
        return zeros

    with np.errstate(divide='ignore', invalid='ignore'):
        mom12 = (p21 / p252 - 1.0).replace([np.inf, -np.inf], np.nan)
        mom6 = (p0 / p126 - 1.0).replace([np.inf, -np.inf], np.nan)
        rev21 = (-(p0 / p21 - 1.0)).replace([np.inf, -np.inf], np.nan)

    mom12_filled = mom12.where(mom12.notna(), mom6)

    def centered_rank(s):
        s2 = s.copy()
        s2[~valid] = np.nan
        r = s2.rank(pct=True)
        return 2.0 * r - 1.0

    z_m12 = centered_rank(mom12_filled)
    z_m6 = centered_rank(mom6)
    z_rev = centered_rank(rev21)
    z_vol = centered_rank(vol)

    score = 0.50 * z_m12 + 0.25 * z_m6 + 0.15 * z_rev - 0.10 * z_vol
    score = score.fillna(0.0)
    score[~valid] = np.nan

    n_valid = int(score.notna().sum())
    n_side = min(60, n_valid // 2)
    if n_side <= 0:
        return zeros

    ordered = score.dropna().sort_values(ascending=False)
    longs = ordered.index[:n_side].tolist()
    shorts = ordered.index[-n_side:].tolist()

    w_side = min(0.02, 0.5 / float(n_side))
    w = pd.Series(0.0, index=tickers)
    if longs:
        w.loc[longs] = w_side
    if shorts:
        w.loc[shorts] = -w_side

    w = w.clip(-0.02, 0.02)
    return w
