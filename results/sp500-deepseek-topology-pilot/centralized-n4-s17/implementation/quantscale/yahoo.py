"""Resumable Yahoo daily-price acquisition for a saved S&P 500 snapshot.

No market capitalization, historical membership or missing prices are invented.
The saved panel is price-only and is NOT a PIT equity-universe dataset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICE_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                "Adj Close": "adjusted_close", "Volume": "volume", "Dividends": "dividends",
                "Stock Splits": "stock_splits", "Capital Gains": "capital_gains"}


class ConstituentsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.depth = 0
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if dict(attrs).get("id") == "constituents":
                self.inside = True
            if self.inside:
                self.depth += 1
        if not self.inside:
            return
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.inside and self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if not self.inside:
            return
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = None
        if tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self.inside = False


def parse_constituents(html):
    parser = ConstituentsParser()
    parser.feed(html)
    if not parser.rows or parser.rows[0][0] != "Symbol":
        raise ValueError("Could not find the S&P 500 constituents table")
    frame = pd.DataFrame(parser.rows[1:], columns=parser.rows[0])
    frame = frame.rename(columns={"Symbol": "ticker", "Security": "security",
                                  "GICS Sector": "sector", "GICS Sub-Industry": "industry"})
    frame["ticker"] = frame.ticker.str.replace(r"\[.*?\]", "", regex=True).str.strip()
    if frame.ticker.duplicated().any() or not frame.ticker.str.fullmatch(r"[A-Z][A-Z0-9.\-]*").all():
        raise ValueError("Invalid or duplicate constituent ticker")
    frame["yahoo_ticker"] = frame.ticker.str.replace(".", "-", regex=False)
    return frame


def normalize_prices(raw, ticker, start, end, drop_invalid_ohlc=False):
    """Preserve Yahoo observations; adjusted_open derives from the same-day factor."""
    if raw is None or raw.empty:
        raise ValueError("Yahoo returned no observations")
    frame = raw.rename(columns=PRICE_FIELDS).copy()
    required = ["open", "high", "low", "close", "adjusted_close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("Missing Yahoo fields: " + ", ".join(sorted(missing)))
    frame = frame.reindex(columns=list(PRICE_FIELDS.values()))
    # Batch downloads align tickers onto a common index. Count/drop empty rows;
    # keep raw files as evidence, and never fill historical prices.
    empty = frame[required].isna().all(axis=1)
    partial = frame[required].isna().any(axis=1) & ~empty
    frame = frame.loc[~empty & ~partial].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("No complete price observations")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame[required[:-1]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Nonfinite or invalid Yahoo prices/volume")
    invalid_ohlc = ((frame.high < frame[["open", "low", "close"]].max(axis=1) - 1e-6)
                    | (frame.low > frame[["open", "high", "close"]].min(axis=1) + 1e-6))
    if invalid_ohlc.any():
        if not drop_invalid_ohlc:
            raise ValueError("Inconsistent Yahoo OHLC")
        frame = frame.loc[~invalid_ohlc].copy()
    frame["adjusted_open"] = frame.open * (frame.adjusted_close / frame.close)
    dates = pd.DatetimeIndex(frame.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    frame.insert(0, "date", dates.normalize())
    frame.insert(1, "ticker", ticker)
    frame = frame[(frame.date >= pd.Timestamp(start)) & (frame.date < pd.Timestamp(end))]
    if frame.empty or frame.date.duplicated().any():
        raise ValueError("No in-range observations or duplicate dates")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame, {"empty_alignment_rows": int(empty.sum()), "partial_rows_dropped": int(partial.sum()),
                   "invalid_ohlc_rows_dropped": int(invalid_ohlc.sum())}


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_sp500(output, start="2016-01-01", end="2026-01-01", batch_size=20, retries=2, offline=False):
    import yfinance as yf
    if pd.Timestamp(start) >= pd.Timestamp(end) or batch_size < 1 or retries < 0:
        raise ValueError("Invalid dates, batch size or retries")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    request_settings = {"start_inclusive": start, "end_exclusive": end,
                        "auto_adjust": False, "actions": True, "repair": False, "interval": "1d"}
    settings_path = root / "request.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != request_settings:
        raise ValueError("Existing download uses different settings; choose another output directory")
    settings_path.write_text(json.dumps(request_settings, indent=2))
    yf.set_tz_cache_location(str(root / ".yfinance-cache"))
    constituents_path = root / "constituents.csv"
    if not constituents_path.exists():
        if offline:
            raise ValueError("Offline finalization requires a saved constituents snapshot")
        request = Request(CONSTITUENTS_URL, headers={"User-Agent": "QuantScale research data acquisition/0.1"})
        with urlopen(request, timeout=45) as response:
            html = response.read().decode("utf-8")
        frame = parse_constituents(html)
        if not 490 <= len(frame) <= 520:
            raise ValueError("Unexpected S&P 500 snapshot size: %d" % len(frame))
        (root / "constituents-source.html").write_text(html)
        frame.to_csv(constituents_path, index=False)
        (root / "constituents-source.json").write_text(json.dumps({
            "url": CONSTITUENTS_URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_html_sha256": file_hash(root / "constituents-source.html"),
            "membership": "current snapshot, not historical point-in-time membership"}, indent=2))
    constituents = pd.read_csv(constituents_path, dtype=str)
    raw_dir, clean_dir = root / "raw", root / "by-ticker"
    raw_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)
    quarantine_dir = root / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    errors = {}
    quality = json.loads((root / "row-quality.json").read_text()) if (root / "row-quality.json").exists() else {}

    def save(ticker, raw):
        raw.to_csv(raw_dir / (ticker + ".csv"), index_label="date")
        normalized, counts = normalize_prices(raw, ticker, start, end, drop_invalid_ohlc=True)
        raw_dates = pd.DatetimeIndex(raw.index)
        if raw_dates.tz is not None:
            raw_dates = raw_dates.tz_localize(None)
        rejected = (~raw_dates.normalize().isin(normalized.date)) & raw.reindex(columns=list(PRICE_FIELDS)).notna().any(axis=1)
        if rejected.any():
            raw.loc[rejected].to_csv(quarantine_dir / (ticker + ".csv"), index_label="date")
        temporary = clean_dir / (ticker + ".csv.tmp")
        normalized.to_csv(temporary, index=False)
        temporary.replace(clean_dir / (ticker + ".csv"))
        quality[ticker] = counts
        (root / "row-quality.json").write_text(json.dumps(quality, indent=2))
        errors.pop(ticker, None)

    pending = constituents[~constituents.ticker.map(lambda ticker: (clean_dir / (ticker + ".csv")).exists())]
    if offline:
        for ticker in pending.ticker:
            raw_path = raw_dir / (ticker + ".csv")
            try:
                cached = pd.read_csv(raw_path, index_col="date")
                # Saved Yahoo daily bars can span daylight-saving offsets.
                cached.index = pd.DatetimeIndex([pd.Timestamp(value).tz_localize(None) for value in cached.index])
                save(ticker, cached)
            except (ValueError, KeyError, OSError) as exc:
                errors[ticker] = str(exc)
        pending = pending.iloc[:0]
        retries = 0
    print("Snapshot: %d tickers; %d need download" % (len(constituents), len(pending)), flush=True)
    for offset in range(0, len(pending), batch_size):
        batch = pending.iloc[offset:offset + batch_size]
        symbols = list(batch.yahoo_ticker)
        downloaded = yf.download(symbols, start=start, end=end, interval="1d", auto_adjust=False,
                                 actions=True, repair=False, threads=4, group_by="ticker",
                                 progress=False, timeout=30, keepna=True, multi_level_index=True)
        if downloaded is None:
            downloaded = pd.DataFrame()
        for row in batch.itertuples(index=False):
            try:
                raw = downloaded[row.yahoo_ticker] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
                save(row.ticker, raw)
            except (ValueError, KeyError, TypeError) as exc:
                errors[row.ticker] = str(exc)
        print("Downloaded %d/%d tickers; pending errors %d" % (
            sum((clean_dir / (ticker + ".csv")).exists() for ticker in constituents.ticker),
            len(constituents), len(errors)), flush=True)
        (root / "download-errors.json").write_text(json.dumps(errors, indent=2))
        time.sleep(1)
    for attempt in range(retries):
        if not errors:
            break
        time.sleep(3 * (attempt + 1))
        for ticker in list(errors):
            symbol = constituents.set_index("ticker").loc[ticker, "yahoo_ticker"]
            try:
                raw = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False,
                                               actions=True, repair=False, raise_errors=True, timeout=30)
                save(ticker, raw)
            except Exception as exc:
                errors[ticker] = type(exc).__name__ + ": " + str(exc)[:500]
        (root / "download-errors.json").write_text(json.dumps(errors, indent=2))
        print("Retry %d: %d unresolved tickers" % (attempt + 1, len(errors)), flush=True)
    frames, coverage = [], []
    for ticker in constituents.ticker:
        path = clean_dir / (ticker + ".csv")
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frames.append(frame)
        coverage.append({"ticker": ticker, "rows": len(frame), "first_date": str(frame.date.min().date()),
                         "last_date": str(frame.date.max().date()), **quality.get(ticker, {})})
    if not frames:
        raise RuntimeError("No prices downloaded; see download-errors.json")
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    if combined.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate observations in merged panel")
    # Use downloaded SPY sessions as the calendar reference, not business days.
    benchmark_path = root / "spy.csv"
    if not benchmark_path.exists():
        if offline:
            raise ValueError("Offline finalization requires an existing SPY calendar")
        benchmark = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=False, actions=True,
                                             repair=False, raise_errors=True, timeout=30)
        spy, _ = normalize_prices(benchmark, "SPY", start, end)
        spy.to_csv(benchmark_path, index=False)
    sessions = pd.DatetimeIndex(pd.read_csv(benchmark_path, parse_dates=["date"]).date)
    dates_by_ticker = {ticker: pd.DatetimeIndex(frame.date) for ticker, frame in combined.groupby("ticker")}
    for row in coverage:
        observed = dates_by_ticker[row["ticker"]]
        expected = sessions[(sessions >= observed.min()) & (sessions <= observed.max())]
        missing = expected.difference(observed)
        row.update(missing_sessions_within_span=len(missing),
                   missing_dates=";".join(str(d.date()) for d in missing),
                   starts_after_first_session=bool(observed.min() > sessions.min()),
                   ends_before_last_session=bool(observed.max() < sessions.max()))
    combined.to_csv(root / "prices.csv", index=False)
    combined[combined.date < "2024-01-01"].to_csv(root / "research-prices.csv", index=False)
    combined[combined.date >= "2024-01-01"].to_csv(root / "hidden-prices.csv", index=False)
    pd.DataFrame(coverage).to_csv(root / "coverage.csv", index=False)
    (root / "download-errors.json").write_text(json.dumps(errors, indent=2))
    manifest = {"source": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), "request": request_settings,
                "requested_tickers": len(constituents), "downloaded_tickers": len(frames),
                "failed_tickers": errors, "price_rows": len(combined), "reference_sessions": len(sessions),
                "first_date": str(combined.date.min().date()), "last_date": str(combined.date.max().date()),
                "late_start_tickers": sum(row["starts_after_first_session"] for row in coverage),
                "early_end_tickers": sum(row["ends_before_last_session"] for row in coverage),
                "tickers_with_internal_gaps": sum(row["missing_sessions_within_span"] > 0 for row in coverage),
                "partial_rows_dropped": sum(row["partial_rows_dropped"] for row in coverage),
                "invalid_ohlc_rows_dropped": sum(row.get("invalid_ohlc_rows_dropped", 0) for row in coverage),
                "prices_sha256": file_hash(root / "prices.csv"),
                "limitations": ["Current constituent snapshot: survivorship and future-membership bias",
                                "Yahoo historical OHLC may already reflect split adjustments",
                                "No point-in-time market capitalization or historical shares are supplied",
                                "No missing observations are forward-filled or synthesized",
                                "Price-only panel is not directly accepted by the market-cap-based research loader"],
                "split": {"train": "2016-01-01..2021-12-31", "validation": "2022-01-01..2023-12-31",
                          "hidden": "2024-01-01..2025-12-31"}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/sp500-yahoo-2016-2025")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild outputs from cached raw files, without network requests")
    args = parser.parse_args()
    try:
        manifest = download_sp500(args.output, args.start, args.end, args.batch_size, args.retries, args.finalize_only)
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        parser.exit(2, "Yahoo download: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
