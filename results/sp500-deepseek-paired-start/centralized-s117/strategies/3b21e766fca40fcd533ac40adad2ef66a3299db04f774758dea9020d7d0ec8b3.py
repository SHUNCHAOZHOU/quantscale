import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    n = len(universe)
    if n == 0:
        return pd.Series(dtype=float)
    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)
    cols = [t for t in universe if t in ac.columns]
    if len(cols) < 20:
        return pd.Series(0.0, index=universe)
    px = ac.loc[:, cols].tail(320).astype(float).copy()
    px = px.ffill(limit=5)
    if len(px) < 80:
        return pd.Series(0.0, index=universe)
    rets = px.pct_change(fill_method=None)
    last = px.iloc[-1]
    prev = px.iloc[-2]
    dret = last / prev - 1.0
    vol = rets.tail(60).std(ddof=0)
    if len(px) >= 22:
        r21 = last / px.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=cols)
    if len(px) >= 6:
        r5 = last / px.iloc[-6] - 1.0
    else:
        r5 = pd.Series(np.nan, index=cols)
    wlen = min(200, len(px))
    ma = px.tail(wlen).mean()
    trend = last / ma - 1.0
    mkt = rets.mean(axis=1, skipna=True)
    rwin = rets.tail(120)
    mwin = mkt.tail(120)
    mvar = mwin.var(ddof=0)
    if mvar is None or not np.isfinite(mvar) or mvar <= 0:
        beta = pd.Series(np.nan, index=cols)
    else:
        beta = rwin.apply(lambda s: s.cov(mwin), axis=0) / mvar
    vol_df = history.get('volume')
    close_df = history.get('close')
    if vol_df is not None and close_df is not None:
        vcols = [c for c in cols if c in vol_df.columns and c in close_df.columns]
        if len(vcols) > 0:
            v = vol_df.loc[:, vcols].tail(60).astype(float)
            c = close_df.loc[:, vcols].tail(60).astype(float)
            adv = (v * c).mean(skipna=True).reindex(cols)
        else:
            adv = pd.Series(np.nan, index=cols)
    else:
        adv = pd.Series(np.nan, index=cols)
    elig_df = history.get('eligible')
    if elig_df is not None and len(elig_df):
        elig = elig_df.iloc[-1].reindex(cols).fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=cols)
    df = pd.DataFrame({'last': last, 'dret': dret, 'r5': r5, 'r21': r21, 'trend': trend, 'vol': vol, 'beta': beta, 'adv': adv, 'eligible': elig})
    df = df.replace([np.inf, -np.inf], np.nan)
    valid = (df['eligible'] & df['last'].notna() & (df['last'] > 1.0) & df['vol'].notna() & (df['vol'] > 0) & df['beta'].notna() & df['r21'].notna() & df['r5'].notna())
    df = df[valid].copy()
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    if df['adv'].notna().sum() >= 20:
        advv = df['adv'].dropna()
        advv = advv[advv > 0]
        if len(advv) >= 20:
            q = advv.quantile(0.30)
            df = df[df['adv'].notna() & (df['adv'] >= q)]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    df['rank_trend'] = df['trend'].rank(pct=True)
    df['rank_rev21'] = (-df['r21']).rank(pct=True)
    df['rank_rev5'] = (-df['r5']).rank(pct=True)
    df['score'] = 0.50 * df['rank_trend'] + 0.35 * df['rank_rev21'] + 0.15 * df['rank_rev5']
    df['score'] = df['score'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    try:
        df['beta_q'] = pd.qcut(df['beta'].rank(method='first'), 3, labels=False)
        df['vol_q'] = pd.qcut(df['vol'].rank(method='first'), 2, labels=False)
    except Exception:
        df['beta_q'] = 0
        df['vol_q'] = 0
    df['bucket'] = df['beta_q'].astype(int).astype(str) + '_' + df['vol_q'].astype(int).astype(str)
    gross = 0.75
    mkt_valid = mkt.dropna()
    if len(mkt_valid) >= 60:
        r60 = (1.0 + mkt_valid.tail(60)).prod() - 1.0
        v20 = mkt_valid.tail(20).std(ddof=0)
        if r60 < -0.08:
            gross = 0.45
        elif r60 > 0.12:
            gross = 0.60
        if v20 > 0.025:
            gross = min(gross, 0.50)
    gross = float(np.clip(gross, 0.20, 0.90))
    side_gross = 0.5 * gross
    n_buckets = max(1, df['bucket'].nunique())
    per_bucket = max(2, int(np.ceil(30.0 / n_buckets)))
    longs = []
    shorts = []
    for b, sub in df.groupby('bucket'):
        if len(sub) < 4:
            continue
        lc = sub[(sub['dret'] > -0.12) & (sub['r5'] > -0.25)].sort_values('score', ascending=False)
        sc = sub[(sub['dret'] < 0.10) & (sub['r5'] < 0.20)].sort_values('score', ascending=True)
        k = min(per_bucket, len(lc), len(sc))
        if k > 0:
            longs.extend(lc.index[:k].tolist())
            shorts.extend(sc.index[:k].tolist())
    if len(longs) < 5 or len(shorts) < 5:
        lc = df[(df['dret'] > -0.12) & (df['r5'] > -0.25)].sort_values('score', ascending=False)
        sc = df[(df['dret'] < 0.10) & (df['r5'] < 0.20)].sort_values('score', ascending=True)
        k = min(30, len(lc), len(sc))
        longs = lc.index[:k].tolist()
        shorts = sc.index[:k].tolist()
    m = min(len(longs), len(shorts))
    if m == 0:
        return pd.Series(0.0, index=universe)
    longs = longs[:m]
    shorts = shorts[:m]
    w = min(0.02, side_gross / float(m))
    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)