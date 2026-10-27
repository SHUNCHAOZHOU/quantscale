import numpy as np
import pandas as pd


def _mom_ret(px, lookback, skip):
    """Total return from t-skip-lookback to t-skip (skips recent month)."""
    need = lookback + skip + 1
    if len(px) < need:
        return pd.Series(np.nan, index=px.columns)
    return px.iloc[-1 - skip] / px.iloc[-1 - skip - lookback] - 1.0


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
    if len(cols) < 10:
        return pd.Series(0.0, index=universe)

    px = ac.loc[:, cols].tail(300).astype(float).copy()
    if len(px) < 60:
        return pd.Series(0.0, index=universe)

    rets = px.pct_change(fill_method=None)
    last = px.iloc[-1]

    # --- raw signals over full col set ---
    sig = pd.DataFrame(index=cols)

    # multi-window (12-1, 10-1, 9-1, 6-1) momentum
    mom_lbs = [252, 210, 189, 126]
    for lb in mom_lbs:
        sig['m%d' % lb] = _mom_ret(px, lb, 21).reindex(cols)

    # proximity to high over overlapping windows
    prox_ws = [252, 210, 189]
    for w in prox_ws:
        wmax = px.tail(w).max() if len(px) >= w else px.max()
        sig['p%d' % w] = (last / wmax).reindex(cols)

    # low-volatility (120-day) stability proxy
    sig['vol'] = rets.tail(120).std(ddof=0).reindex(cols)
    sig['last'] = last.reindex(cols)

    # --- liquidity (60-day dollar volume) ---
    vol_df = history.get('volume')
    close_df = history.get('close')
    if vol_df is not None and close_df is not None:
        vcols = [c for c in cols if c in vol_df.columns]
        v = vol_df.loc[:, vcols].tail(60).astype(float)
        cser = close_df.loc[:, vcols].tail(60).astype(float)
        adv = (v * cser).mean(skipna=True)
        sig['adv'] = adv.reindex(cols)
    else:
        sig['adv'] = np.nan

    # --- eligibility ---
    elig_df = history.get('eligible')
    if elig_df is not None and len(elig_df):
        sig['eligible'] = elig_df.iloc[-1].reindex(cols).fillna(False).astype(bool)
    else:
        sig['eligible'] = True

    sig = sig.replace([np.inf, -np.inf], np.nan)

    # --- filter to tradable, priced, valid-vol, liquid names ---
    valid = (sig['eligible'].fillna(False) & sig['last'].notna()
             & sig['vol'].notna() & (sig['vol'] > 0))
    df = sig[valid].copy()
    if len(df) < 10:
        return pd.Series(0.0, index=universe)

    if df['adv'].notna().sum() >= 20:
        adv_valid = df['adv'].dropna()
        adv_valid = adv_valid[adv_valid > 0]
        if len(adv_valid) >= 20:
            q = adv_valid.quantile(0.40)  # keep the most liquid ~60%
            df = df[df['adv'].notna() & (df['adv'] >= q)]

    if len(df) < 10:
        return pd.Series(0.0, index=universe)

    # --- rank-stable composite ---
    mom_cols = ['m%d' % lb for lb in mom_lbs]
    prox_cols = ['p%d' % w for w in prox_ws]
    help_df = pd.DataFrame(index=df.index)
    help_df['mom'] = df[mom_cols].rank(pct=True).fillna(0.5).mean(axis=1)
    help_df['prox'] = df[prox_cols].rank(pct=True).fillna(0.5).mean(axis=1)
    help_df['lowvol'] = (-df['vol']).rank(pct=True).fillna(0.5)

    score = (0.40 * help_df['mom']
             + 0.30 * help_df['prox']
             + 0.30 * help_df['lowvol'])
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)

    n_side = min(50, len(df) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    ordered = score.sort_values(ascending=False)
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    w = min(0.02, 1.0 / (2.0 * n_side))

    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)
