"""Regression tests for chronological model-training splits."""

import unittest

import numpy as np
import pandas as pd

from prepare import make_dataloaders


class DataSplittingTests(unittest.TestCase):
    def test_weekly_dataloaders_respect_custom_fold_dates(self):
        dates = pd.bdate_range("2019-01-01", "2022-02-01")
        features = pd.DataFrame({"feature_a": np.arange(len(dates), dtype=float)}, index=dates)
        targets = pd.DataFrame({"target_a": np.arange(len(dates), dtype=float)}, index=dates)

        train_end = "2020-01-01"
        val_end = "2020-07-01"

        train_loader, val_loader, test_loader, _, _ = make_dataloaders(
            features,
            targets,
            lookback=2,
            batch_size=128,
            trade_frequency="weekly",
            train_end=train_end,
            val_end=val_end,
        )

        weekly_idx = features.index[::5]
        expected_train_count = int((weekly_idx < train_end).sum())
        expected_val_count = int(((weekly_idx >= train_end) & (weekly_idx < val_end)).sum())
        expected_test_count = int((weekly_idx >= val_end).sum())

        self.assertEqual(len(train_loader.dataset.features), expected_train_count)
        self.assertEqual(len(val_loader.dataset.features), expected_val_count)
        self.assertEqual(len(test_loader.dataset.features), expected_test_count)


if __name__ == "__main__":
    unittest.main()
