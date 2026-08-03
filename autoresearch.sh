#!/usr/bin/env bash
# ===========================================================================
# Autoresearch harness — Hedge Portfolio Test
# ===========================================================================
# Workload: train the LSTM+VSN ensemble → backtest on validation period
# Primary metric:  CAGR          (higher is better)
# Secondary:        CVaR 95%, Skewness (higher is better)
#                    + Sharpe, Max Drawdown, Turnover (informational)
#
# Determinism: fixed seeds (torch.manual_seed per seed), cached price/macro data
# (no live network).  PYTHONHASHSEED=0 for reproducible hashing.
#
# Iteration-speed: research defaults keep each run to a few minutes (5 seeds ×
# 60 s).  Edit these values for production-quality runs, or override before
# calling:  ARC_TIME_BUDGET=300 ARC_MIN_TRAIN_TIME=120 bash autoresearch.sh
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONHASHSEED=0
export PYTHON="${PYTHON:-.venv/Scripts/python.exe}"

# --- Research iteration-speed defaults (override by setting these env vars) ---
# 5-seed ensemble, 60 s/seed → ~5 min per run.  ~320 s well within timeouts.
export ARC_TIME_BUDGET="${ARC_TIME_BUDGET:-60}"
export ARC_SEEDS="${ARC_SEEDS:-6,42,123,7,99}"
export ARC_PATIENCE="${ARC_PATIENCE:-5}"
export ARC_MIN_TRAIN_TIME="${ARC_MIN_TRAIN_TIME:-30}"

# 1. Train the ensemble (fixed seeds → deterministic given the config in train.py)
"$PYTHON" train.py

# 2. Backtest on validation period → emits METRIC lines for hyperparameter selection.
# Keep the test split for final, manual confirmation only.
"$PYTHON" backtest.py --val
