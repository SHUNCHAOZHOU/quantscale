import numpy as np
import pandas as pd

def _get(history, key):
    try:
        return history[key]
    except Exception:
        return None

def _rank(s):
    try:
        r = s.rank(method='average', pct=True)
        return r.fillna(0.5)
    except Exception:
        return s * 0.0 + 0.5

def generate_weights(history, universe):
    zero = pd.Series(0.0, index=universe, dtype=float)
    try:
        close = _get(history, 'adjusted_close')
        if close is None:
            close = _get(history, 'close')
        raw_close = _get(history, 'close')
        volume = _get(history, 'volume')
        eligible = _get(history, 'eligible')
        if close is None:
            return zero
        try:
            nrows0 = close.shape[0]
            cols = close.columns
        except Exception:
            return zero
        if nrows0 < 40:
            return zero
        tickers = []
        for t in universe:
            if t in cols:
                tickers.append(t)
        if len(tickers) < 6:
            return zero
        p_all = close.loc[:, tickers]
        idx = p_all.index
        last_date = idx[-1]
        try:
            month_start = last_date.replace(day=1)
            prev_month_end = month_start - pd.Timedelta(days=1)
            pos = idx.searchsorted(prev_month_end, side='right') - 1
            if pos >= 0:
                signal_date = idx[pos]
            else:
                signal_date = last_date
        except Exception:
            signal_date = last_date
        p = p_all.loc[:signal_date, :]
        if p.shape[0] < 40:
            return zero
        if raw_close is not None:
            try:
                c = raw_close.reindex(index=p.index, columns=tickers)
            except Exception:
                c = p
        else:
            c = p
        if volume is not None:
            try:
                v = volume.reindex(index=p.index, columns=tickers)
            except Exception:
                v = pd.DataFrame(np.nan, index=p.index, columns=tickers)
        else:
            v = pd.DataFrame(np.nan, index=p.index, columns=tickers)
        if eligible is not None:
            try:
                el = eligible.reindex(index=p.index, columns=tickers)
                last_el = el.iloc[-1].fillna(False).astype(bool)
            except Exception:
                last_el = pd.Series(True, index=tickers)
        else:
            last_el = pd.Series(True, index=tickers)
        nrows = p.shape[0]
        last_p = p.iloc[-1]
        def _mom(lb, skip):
            if nrows > lb:
                try:
                    return p.shift(skip).iloc[-1] / p.shift(lb).iloc[-1] - 1.0
                except Exception:
                    pass
            return pd.Series(np.nan, index=tickers)
        mom12 = _mom(252, 21)
        mom6 = _mom(126, 21)
        mom3 = _mom(63, 21)
        mom = mom12.where(mom12.notna(), mom6)
        mom = mom.where(mom.notna(), mom3)
        if nrows > 21:
            try:
                rev = -(p.iloc[-1] / p.shift(21).iloc[-1] - 1.0)
            except Exception:
                rev = pd.Series(np.nan, index=tickers)
        else:
            rev = pd.Series(np.nan, index=tickers)
        rets = p.pct_change()
        if nrows >= 40:
            win = rets.iloc[-60:]
            cnt = win.notna().sum()
            vol = win.std()
            vol = vol.where(cnt >= 30)
        else:
            vol = pd.Series(np.nan, index=tickers)
        if nrows >= 20:
            try:
                dv = (c.iloc[-20:] * v.iloc[-20:]).mean()
                liq = np.log1p(dv.clip(lower=0))
            except Exception:
                liq = pd.Series(np.nan, index=tickers)
        else:
            liq = pd.Series(np.nan, index=tickers)
        mom = mom.replace([np.inf, -np.inf], np.nan)
        rev = rev.replace([np.inf, -np.inf], np.nan)
        vol = vol.replace([np.inf, -np.inf], np.nan)
        liq = liq.replace([np.inf, -np.inf], np.nan)
        last_v = v.iloc[-1].fillna(0)
        valid = last_p.notna() & (last_p > 0) & last_el & mom.notna() & rev.notna() & vol.notna() & liq.notna() & (liq > 0) & (last_v > 0)
        valid_tickers = list(valid[valid].index)
        n_valid = len(valid_tickers)
        if n_valid < 6:
            return zero
        mom_v = mom.loc[valid_tickers]
        rev_v = rev.loc[valid_tickers]
        lowvol_v = (-vol).loc[valid_tickers]
        liq_v = liq.loc[valid_tickers]
        score = 0.40 * _rank(mom_v) + 0.25 * _rank(rev_v) + 0.20 * _rank(lowvol_v) + 0.15 * _rank(liq_v)
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        n_side = int(0.12 * n_valid + 0.5)
        n_side = max(n_side, min(25, n_valid // 2))
        n_side = min(n_side, 60, n_valid // 2)
        n_side = max(n_side, 1)
        order = score.sort_values(ascending=False)
        longs = list(order.index[:n_side])
        shorts = list(order.index[-n_side:])
        w = min(0.5 / n_side, 0.02)
        weights = pd.Series(0.0, index=universe, dtype=float)
        for t in longs:
            if t in weights.index:
                weights.loc[t] = w
        for t in shorts:
            if t in weights.index:
                weights.loc[t] = -w
        return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    except Exception:
        return zero