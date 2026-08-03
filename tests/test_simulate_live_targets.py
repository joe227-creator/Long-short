"""Regression tests for live-execution target construction."""

import unittest

import numpy as np
import pandas as pd

from prepare import ETF_TICKERS
from simulate import build_live_execution_targets


class LiveExecutionTargetTests(unittest.TestCase):
    def test_weekly_targets_enter_next_open_and_exit_weekly_close(self):
        dates = pd.bdate_range("2026-01-05", periods=7)
        data = {}
        for ticker in ETF_TICKERS:
            data[f"{ticker}_Open"] = [10, 11, 12, 13, 14, 15, 16]
            data[f"{ticker}_Close"] = [10, 12, 13, 14, 15, 16, 17]
        etf_df = pd.DataFrame(data, index=dates)

        targets = build_live_execution_targets(etf_df, trade_frequency="weekly")

        expected = np.log(16) - np.log(11)
        self.assertAlmostEqual(targets.iloc[0]["SSO_fwd_ret"], expected)


if __name__ == "__main__":
    unittest.main()
