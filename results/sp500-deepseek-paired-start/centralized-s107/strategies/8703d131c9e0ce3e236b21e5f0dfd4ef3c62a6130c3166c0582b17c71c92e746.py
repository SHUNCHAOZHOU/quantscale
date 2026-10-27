import numpy as np
import pandas as pd

def _extract_panel(history, universe, field, fallback=None):
    if isinstance(history, dict):
        std_keys = ('open', 'high', 'low', 'close', 'adjusted_close', 'volume')
        field_panels = any(isinstance(history.get(k), pd.DataFrame) for k in std_keys)
        if field_panels:
            df = None
            if isinstance(history.get(field), pd.DataFrame):
                df = history[field]
            elif fallback is not None and isinstance(history.get(fallback), pd.DataFrame):
                df = history[fallback]
            if df is None:
                return pd.DataFrame()
            cols = [t for t in universe if t in df.columns]
            if not cols:
                return pd.DataFrame()
            out = df.loc[:, cols].apply(pd.to_numeric, errors='coerce')
            return out.sort_index()
        data = {}
        for t in universe:
            df = history.get(t)
            if not isinstance(df, pd.DataFrame):
                continue
            col = None
            if field in df.columns:
                col = field
            elif fallback is not None and fallback in df.columns:
                col = fallback
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) > 0:
                data[t] = s
        if data:
            return pd.DataFrame(data).sort_index()
    return pd.DataFrame()

def _zscore(s):
    s = pd.to_numeric(s, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=0)
    if (not np.isfinite(sd)) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s - mu) / sd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    closes = _extract_panel(history, universe, 'adjusted_close', 'close')
    if closes is None or closes.empty:
        return zero
    closes = closes.dropna(how='all')
    if len(closes) < 2:
        return zero
    dates = closes.index
    last_date = dates[-1]
    mask = (dates.year == last_date.year) & (dates.month == last_date.month)
    if not mask.any():
        return zero
    anchor = dates[mask][0]
    anchor_pos = closes.index.get_loc(anchor)
    if anchor_pos < 60:
        return zero
    sub = closes.iloc[:anchor_pos + 1]
    vols = _extract_panel(history, universe, 'volume')
    if vols is not None and not vols.empty:
        vols = vols.reindex(index=sub.index, columns=sub.columns)
        vols = vols.apply(pd.to_numeric, errors='coerce')
    else:
        vols = pd.DataFrame(np.nan, index=sub.index, columns=sub.columns)
    last_px = sub.iloc[-1]
    valid_price = last_px.notna() & (last_px > 0)
    n = len(sub)
    end = n - 1 - 21
    if end < 1:
        return zero
    start = max(0, end - 252)
    if end - start < 40:
        return zero
    mom = sub.iloc[end] / sub.iloc[start] - 1.0
    rets = sub / sub.shift(1) - 1.0
    vol = rets.iloc[-126:].std(ddof=0)
    if n >= 22:
        rev = sub.iloc[-1] / sub.iloc[-22] - 1.0
    else:
        rev = pd.Series(np.nan, index=sub.columns, dtype=float)
    liq = (sub * vols).iloc[-60:].median()
    if liq.notna().sum() == 0:
        liq = pd.Series(1.0, index=sub.columns, dtype=float)
    else:
        liq = liq.where(liq > 0)
    logliq = np.log1p(liq)
    sig = pd.DataFrame({'mom': mom, 'vol': vol, 'rev': rev, 'liq': logliq})
    valid = sig.notna().all(axis=1) & valid_price
    sig = sig[valid]
    if len(sig) < 2:
        return zero
    z_mom = _zscore(sig['mom'])
    z_vol = _zscore(sig['vol'])
    z_rev = _zscore(sig['rev'])
    z_liq = _zscore(sig['liq'])
    score = 1.0 * z_mom - 0.7 * z_vol - 0.2 * z_rev + 0.3 * z_liq
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 2:
        return zero
    n_side = min(24, len(score) // 2)
    if n_side <= 0:
        return zero
    ranked = score.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])
    w = pd.Series(0.0, index=universe, dtype=float)
    long_labels = [t for t in longs if t in w.index]
    short_labels = [t for t in shorts if t in w.index]
    if len(long_labels) != len(short_labels):
        m = min(len(long_labels), len(short_labels))
        long_labels = long_labels[:m]
        short_labels = short_labels[:m]
    if long_labels:
        w.loc[long_labels] = 0.02
    if short_labels:
        w.loc[short_labels] = -0.02
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w