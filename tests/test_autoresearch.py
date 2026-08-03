"""Regression tests for the autoresearch harness contract."""

import pathlib
import unittest


class AutoresearchHarnessTests(unittest.TestCase):
    def test_autoresearch_optimizes_validation_not_test(self):
        script = pathlib.Path("autoresearch.sh").read_text(encoding="utf-8")
        self.assertIn("backtest.py --val", script)


if __name__ == "__main__":
    unittest.main()
