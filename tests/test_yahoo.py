import unittest

import numpy as np
import pandas as pd

from quantscale.yahoo import normalize_prices, parse_constituents


class YahooTests(unittest.TestCase):
    def test_constituent_parser_and_share_class_mapping(self):
        html = '<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>' \
               '<tr><td><a>BRK.B</a></td><td>Berkshire Hathaway</td><td>Financials</td></tr></table>'
        frame = parse_constituents(html)
        self.assertEqual(frame.iloc[0].ticker, "BRK.B")
        self.assertEqual(frame.iloc[0].yahoo_ticker, "BRK-B")
        self.assertEqual(frame.iloc[0].sector, "Financials")

    def test_duplicate_constituents_rejected(self):
        html = '<table id="constituents"><tr><th>Symbol</th></tr>' \
               '<tr><td>A</td></tr><tr><td>A</td></tr></table>'
        with self.assertRaises(ValueError):
            parse_constituents(html)

    def panel(self):
        return pd.DataFrame({"Open": [100., 110.], "High": [104., 115.],
                             "Low": [98., 105.], "Close": [102., 112.],
                             "Adj Close": [51., 56.], "Volume": [1000, 2000],
                             "Dividends": [0., 0.5], "Stock Splits": [0., 0.]},
                            index=pd.date_range("2020-01-02", periods=2, tz="America/New_York"))

    def test_adjusted_open_and_exclusive_end(self):
        frame, quality = normalize_prices(self.panel(), "A", "2020-01-01", "2020-01-03")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].adjusted_open, 50)
        self.assertIsNone(frame.date.dt.tz)
        self.assertNotIn("market_cap", frame.columns)
        self.assertEqual(quality["partial_rows_dropped"], 0)

    def test_partial_and_empty_rows_not_filled(self):
        raw = self.panel().reindex(pd.date_range("2020-01-02", periods=3, tz="America/New_York"))
        raw.iloc[1, raw.columns.get_loc("Open")] = np.nan
        frame, quality = normalize_prices(raw, "A", "2020-01-01", "2020-02-01")
        self.assertEqual(len(frame), 1)
        self.assertEqual(quality["partial_rows_dropped"], 1)
        self.assertEqual(quality["empty_alignment_rows"], 1)

    def test_unadjusted_fallback_not_silent(self):
        with self.assertRaisesRegex(ValueError, "Missing Yahoo fields"):
            normalize_prices(self.panel().drop(columns="Adj Close"), "A", "2020-01-01", "2020-02-01")

    def test_invalid_ohlc_can_be_quarantined_without_losing_ticker(self):
        raw = self.panel()
        raw.iloc[0, raw.columns.get_loc("Low")] = 101
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            normalize_prices(raw, "A", "2020-01-01", "2020-02-01")
        frame, counts = normalize_prices(raw, "A", "2020-01-01", "2020-02-01", drop_invalid_ohlc=True)
        self.assertEqual(len(frame), 1)
        self.assertEqual(counts["invalid_ohlc_rows_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
