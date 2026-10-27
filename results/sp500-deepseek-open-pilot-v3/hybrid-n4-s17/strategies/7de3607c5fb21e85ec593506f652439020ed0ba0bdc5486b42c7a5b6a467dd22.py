import numpy as np
import pandas as pd


def _get(history, key, universe):
    if key not in history:
        return None
    df = history[key]
    if not isinstance(df, pd.DataFrame):
        return None
    return df.reindex(columns=universe)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe)
    if len(universe) == 0:
        return zero

    adj = _get(history, 'adjusted_close', universe)
    if adj is None:
        adj = _get(history, 'close', universe)
    close = _get(history, 'close', universe)
    volume = _get(history, 'volume', universe)
    elig = _get(history, 'eligible', universe)
    if adj is None or close is None or volume is None:
        return zero

    adj = adj.tail(200).astype(float)
    close = close.tail(200).astype(float)
    volume = volume.tail(200).astype(float)
    if len(adj) < 30:
        return zero

    if elig is not None and len(elig) > 0:
        last_elig = elig.tail(1).iloc[0].reindex(universe).fillna(False).astype(bool)
    else:
        last_elig = pd.Series(True, index=universe)

    last_adj = adj.iloc[-1]
    last_close = close.iloc[-1]
    nobs = adj.tail(140).notna().sum()
    dvol = (close * volume).tail(20).median()
    med_dvol = dvol.median()

    valid = last_elig & last_adj.notna() & last_close.notna() & (last_close > 1.0) & (dvol > 0) & (nobs >= 100)
    if np.isfinite(med_dvol) and med_dvol > 0:
        valid = valid & (dvol >= max(1e6, 0.05 * med_dvol))
    else:
        valid = valid & (dvol >= 1e6)

    valid_names = valid[valid].index
    if len(valid_names) < 6:
        return zero

    ret = adj.pct_change()
    rev5 = adj.pct_change(5).iloc[-1]

    if len(adj) >= 141:
        mom = adj.shift(20).iloc[-1] / adj.shift(140).iloc[-1] - 1.0
    elif len(adj) >= 81:
        mom = adj.shift(20).iloc[-1] / adj.shift(80).iloc[-1] - 1.0
    else:
        mom = adj.pct_change(max(5, len(adj) - 1)).iloc[-1]

    vol_win = min(60, max(5, len(ret) - 1))
    vol60 = ret.rolling(vol_win).std().iloc[-1]

    semi_win = min(60, max(5, len(ret) - 1))
    down = ret.clip(upper=0.0).fillna(0.0)
    semi = np.sqrt((down.tail(semi_win) ** 2).mean())

    sig = pd.DataFrame({'rev': -rev5, 'mom': mom, 'lowvol': -vol60, 'lowsemi': -semi}, index=universe)
    sig = sig.loc[valid_names].dropna()
    if len(sig) < 6:
        return zero

    ranks = pd.DataFrame(index=sig.index)
    ranks['rev'] = sig['rev'].rank(pct=True, ascending=True)
    ranks['mom'] = sig['mom'].rank(pct=True, ascending=True)
    ranks['lowvol'] = sig['lowvol'].rank(pct=True, ascending=True)
    ranks['lowsemi'] = sig['lowsemi'].rank(pct=True, ascending=True)

    score = (0.35 * ranks['rev'] + 0.30 * ranks['mom'] + 0.20 * ranks['lowvol'] + 0.15 * ranks['lowsemi'])

    n = min(25, len(score) // 2)
    if n < 1:
        return zero

    sorted_score = score.sort_values(ascending=False)
    longs = sorted_score.index[:n]
    shorts = sorted_score.index[-n:]
    w = min(0.02, 0.5 / n)

    weights = zero.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.clip(-0.02, 0.02)
    return weights.fillna(0.0)