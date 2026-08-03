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
# CPU contract: use shared WSL venv and model defaults without per-run overrides.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONHASHSEED=0
PYTHON="/home/user/venv/bin/python"

# 1. Train the ensemble (fixed seeds → deterministic given the config in train.py)
"$PYTHON" train.py

# 2. Backtest validation first; this is the selection metric.
"$PYTHON" backtest.py --val

# 3. Backtest held-out test period for final confirmation only.
"$PYTHON" backtest.py
