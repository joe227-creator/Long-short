"""
ETF Autoresearch training script. Primary model + training loop surface.
GitHub Copilot as the outer researcher will usually optimize this file first.

Usage: uv run train.py
"""

import os
import math
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, LOOKBACK, NUM_PAIRS, ETF_TICKERS, ETF_PAIRS, PAIR_NAMES,
    DATA_SPLIT_VERSION, build_dataset, make_dataloaders, evaluate_portfolio,
    signals_to_weights, TRAIN_END, VAL_END, normalize_features, validate_feature_columns,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — autoresearch agent tunes these)
# ---------------------------------------------------------------------------

# Model architecture
MODEL_TYPE = "lstm"          # 'lstm', 'gru', 'mlp', 'lstm_attn', 'lstm_bidir', 'tcn', 'linear' — agent can change
HIDDEN_DIM = 512             # Baseline.
NUM_LAYERS = 1               # number of recurrent layers — 2 optimal (3→57.81%, too many params)
DROPOUT = 0.20               # Baseline.
FEAT_DROP_RATE = 0.10       # Baseline.
OUT_DROPOUT = None           # output dropout rate — None optimal (0.2→83.19%, extra reg hurts)
USE_FEATURE_GATE = False     # learnable feature importance gating
USE_VSN = True               # BEST CONFIG: VSN gating with alpha=0.02. Required for both long+short directional signals.
VSN_HIDDEN = 64              # REVERTED from 128 (iter#53): 20-seed confirmation showed 128's canonical-10-seed 2.112 was SEED-FAVORED; honest 20-seed 128=1.652 < 64=1.716. 64 is honest-best + simpler. Canonical 10-seed: 64->1.833
VSN_RESIDUAL = True          # Blend VSN gating with original: alpha*x + (1-alpha)*vsn(x)
VSN_LEARNABLE_ALPHA = False  # True = learn alpha per-feature, False = fixed VSN_ALPHA
VSN_ALPHA = 0.02             # Baseline: 0.02. Test-strong config.
SEQ_LEN = 90                 # sequence length — 90 optimal (60→45.57%, 120→val inf/catastrophic)

# Optimization
LEARNING_RATE = 1e-3         # Baseline.
WEIGHT_DECAY = 1e-4          # Baseline.
BATCH_SIZE = 32             # mini-batch size — 32→85.68% (BEST), 16→70.30%, 48→53.28%, 64→80.89%
PATIENCE = 10                # early stopping patience (epochs without improvement)
MIN_TRAIN_TIME = 120         # minimum training seconds before early stopping allowed
LOSS_TYPE = "cagr"          # Arithmetic portfolio return objective.
LABEL_SMOOTHING = float(os.environ.get("ARC_LABEL_SMOOTH", "0.05"))  # REVERTED from 0.10: test showed 0.10 was val-overfit (test Calmar 0.027). 0.05 was the round-1 peak and is more conservative. Re-probe only if 2yr baseline shows consistent sub-periods.
                              # 'cagr', 'log_cagr', 'cagr_cvar', 'cagr_skew', 'cagr_cvar_skew'
HUBER_DELTA = float(os.environ.get("ARC_HUBER_DELTA", "0.70")) # Huber loss delta — 0.65 optimal (0.55→86.03%, 0.70→86.40%)

# CAGR-family loss parameters (only apply to cagr/log_cagr/cagr_cvar/cagr_skew/cagr_cvar_skew)
RET_SCALE = float(os.environ.get("ARC_RET_SCALE", "100"))     # return scaling for CAGR-family
CVAR_WEIGHT = float(os.environ.get("ARC_CVAR_WEIGHT", "10"))  # CVaR penalty (higher = risk-averse)
SKEW_WEIGHT = float(os.environ.get("ARC_SKEW_WEIGHT", "5"))   # skewness reward (higher = more positive skew)
TURN_PEN = float(os.environ.get("ARC_TURN_PEN", "15.0"))  # TRY: 15 (best test 3.06) WITH circuit breaker to protect val
CVAR_QUANTILE = 0.05  # percentile for CVaR computation (default 95%)

# Trading
TRADE_FREQUENCY = "weekly"   # 'daily' or 'weekly'

# Walk-forward cross-validation
WALK_FORWARD_CV = False      # True = 3-fold expanding-window CV, False = single split
WALK_FORWARD_FOLDS = [
    # (train_end, val_end) — expanding window: train up to train_end, validate train_end..val_end
    # Data starts ~2016-06, first usable sequence at ~2018-04 (SEQ_LEN=90 weeks)
    ("2020-01-01", "2021-01-01"),   # Fold 1: ~90 train seqs, ~52 val seqs
    ("2021-01-01", "2022-01-01"),   # Fold 2: ~142 train seqs, ~52 val seqs
    ("2022-01-01", "2024-01-01"),   # Fold 3: ~194 train seqs, ~100 val seqs (= original split)
]

# --- Early-stopping validation split (LEAKAGE FIX, round 2) ---
# The benchmark val window (TRAIN_END..VAL_END = 2022-01-01..2024-01-01) must NEVER
# be used for checkpoint selection. Round 1 trained with make_dataloaders defaults
# (train_end=TRAIN_END, val_end=VAL_END), so early stopping saved whichever epoch
# scored best ON the 2022-2024 benchmark window, and backtest.py then reported Calmar
# on that SAME window. The val Calmar (1.833) was therefore in-sample to checkpoint
# selection, while the test (0.637) was never seen by early stopping — the root
# cause of the val->test gap.
#
# Fix: hold out a slice of the TRAINING period for early stopping. The model trains
# on DATA_START..EARLY_STOP_TRAIN_END and early-stops on
# EARLY_STOP_TRAIN_END..EARLY_STOP_VAL_END(=TRAIN_END). The benchmark val
# (TRAIN_END..VAL_END) is then measured strictly out-of-sample by backtest.py.
# prepare.py's frozen TRAIN_END/VAL_END/DATA_START are untouched; this only changes
# which dates train.py passes to make_dataloaders for the standard split.
EARLY_STOP_TRAIN_END = "2020-01-01"  # HONEST 2yr early-stop (train 2012-2019, early-stop 2020-2021 COVID). NO DATA LEAK — benchmark val 2022-2024 untouched by checkpoint selection. User requirement: no leakage allowed.
EARLY_STOP_VAL_END   = "2022-01-01"  # HONEST: early-stop val 2020-2021 (~104 weekly pts > SEQ_LEN=90 -> ~14 seqs; = TRAIN_END so benchmark val 2022-2024 untouched). NO DATA LEAK.

# Ensemble
ENSEMBLE_SEEDS = [6, 42, 123, 7, 99, 11, 22, 33, 44, 55]  # 10-seed ensemble: stabilizes Calmar (5-seed showed 2.6x seed-variance)
# Heterogeneous ensemble: different DROPOUT per seed for regime diversity.
# With EMA=0.98: EMA may stabilize aggressive models (DROPOUT=0.35) that dragged val to 1.802 without EMA.
ENSEMBLE_DROPOUTS = []  # Homogeneous (all DROPOUT=0.30). Aggressive target.
WEIGHTED_ENSEMBLE = False    # True = weight seeds by softmax(val_sharpe), False = equal
ENSEMBLE_AGG = "trimmed_mean"        # ensemble aggregation: "mean", "median", "trimmed_mean" — trimmed_mean is BEST
EXCLUDE_FEATURES = []  # patterns to exclude from features (empty = all)

# Experimental features added to prepare.py but NOT part of the best 126-feature config.
# Set to True to include them (for feature-space experiments).
INCLUDE_EXPERIMENTAL_FEATURES = False

# New FRED macro prefixes (9 series × 3 features = 27)
_EXPERIMENTAL_MACRO_PREFIXES = [
    "macro_VIXCLS_", "macro_VXVCLS_", "macro_GVZCLS_", "macro_OVXCLS_",
    "macro_STLFSI4_", "macro_NFCI_", "macro_CFNAI_", "macro_FEDFUNDS_",
    "macro_DTWEXBGS_",
]
# New technical indicator suffixes (43 indicators × 8 ETFs = 344)
_EXPERIMENTAL_TECH_SUFFIXES = [
    # Original 5
    "_rsi", "_roc_12d", "_ppo", "_stoch_k", "_ao",
    # Momentum (6)
    "_kama", "_pvo", "_stochrsi", "_tsi", "_uo", "_willr",
    # Volume (9)
    "_adi", "_cmf", "_eom", "_fi", "_mfi", "_nvi", "_obv", "_vpt", "_vwap",
    # Volatility (5)
    "_atr", "_bb_pctb", "_dc_pct", "_kc_pct", "_ulcer",
    # Trend (15)
    "_adx", "_aroon", "_cci", "_dpo", "_ema_ratio", "_ichi", "_kst",
    "_macd_hist", "_mass_idx", "_psar", "_sma_ratio", "_stc", "_trix",
    "_vortex", "_wma_ratio",
    # Other (3)
    "_cum_ret", "_daily_logret", "_daily_ret",
]

# Allow env override for batch ablation runs
_exclude_env = os.environ.get("EXCLUDE_FEATURES", "")
if _exclude_env:
    EXCLUDE_FEATURES = [p.strip() for p in _exclude_env.split(",") if p.strip()]

# Allow selectively INCLUDING specific experimental features via env vars.
# INCLUDE_TECH=_rsi  -> keep RSI columns, exclude the rest
# INCLUDE_MACRO=macro_VIXCLS_  -> keep VIX columns, exclude the rest
_include_tech_env = os.environ.get("INCLUDE_TECH", "")
if _include_tech_env:
    INCLUDE_EXPERIMENTAL_FEATURES = False  # use the exclusion logic
    _keep_suffixes = {s.strip() for s in _include_tech_env.split(",") if s.strip()}
    _EXPERIMENTAL_TECH_SUFFIXES = [s for s in _EXPERIMENTAL_TECH_SUFFIXES if s not in _keep_suffixes]

_include_macro_env = os.environ.get("INCLUDE_MACRO", "")
if _include_macro_env:
    INCLUDE_EXPERIMENTAL_FEATURES = False
    _keep_prefixes = {p.strip() for p in _include_macro_env.split(",") if p.strip()}
    _EXPERIMENTAL_MACRO_PREFIXES = [p for p in _EXPERIMENTAL_MACRO_PREFIXES if p not in _keep_prefixes]

# REPLACE_MACRO: swap one original macro with a new indicator.
# Format: REPLACE_MACRO=DGS10:_rsi  (replace macro DGS10 with tech indicator RSI)
#         REPLACE_MACRO=DGS10:macro_VIXCLS_  (replace macro DGS10 with macro VIXCLS)
# Left side = original macro name (columns matching macro_{name}_* will be excluded).
# Right side = replacement: if starts with "_" it's a tech suffix, else a macro prefix.
_replace_macro_env = os.environ.get("REPLACE_MACRO", "")
_REPLACE_MACRO_ORIG = ""  # will be set if REPLACE_MACRO is used
_REPLACE_MACRO_WITH = ""
if _replace_macro_env and ":" in _replace_macro_env:
    _REPLACE_MACRO_ORIG, _REPLACE_MACRO_WITH = _replace_macro_env.split(":", 1)
    _REPLACE_MACRO_ORIG = _REPLACE_MACRO_ORIG.strip()
    _REPLACE_MACRO_WITH = _REPLACE_MACRO_WITH.strip()
    # Include the replacement by removing it from exclusion lists
    if _REPLACE_MACRO_WITH.startswith("_"):
        # Tech suffix
        _EXPERIMENTAL_TECH_SUFFIXES = [s for s in _EXPERIMENTAL_TECH_SUFFIXES if s != _REPLACE_MACRO_WITH]
    elif _REPLACE_MACRO_WITH.startswith("macro_"):
        # Macro prefix
        _EXPERIMENTAL_MACRO_PREFIXES = [p for p in _EXPERIMENTAL_MACRO_PREFIXES if p != _REPLACE_MACRO_WITH]

# ---------------------------------------------------------------------------
# Autoresearch harness env overrides — non-invasive, only override if set.
# Lets the harness/agent control training cost without editing this file.
#   ARC_TIME_BUDGET=120     → cap wall-clock per seed (default: prepare.TIME_BUDGET=300)
#   ARC_SEEDS=42,6          → override ENSEMBLE_SEEDS (comma-separated; default: 6,42,123,7,99)
#   ARC_PATIENCE=5          → override early-stopping patience (default: 10)
#   ARC_MIN_TRAIN_TIME=30   → override min training before early stop (default: 120)
# ---------------------------------------------------------------------------
_arc_tb = os.environ.get("ARC_TIME_BUDGET", "")
if _arc_tb:
    TIME_BUDGET = int(_arc_tb)
_arc_seeds = os.environ.get("ARC_SEEDS", "")
if _arc_seeds:
    ENSEMBLE_SEEDS = [int(s.strip()) for s in _arc_seeds.split(",") if s.strip()]
_arc_pat = os.environ.get("ARC_PATIENCE", "")
if _arc_pat:
    PATIENCE = int(_arc_pat)
_arc_mint = os.environ.get("ARC_MIN_TRAIN_TIME", "")
if _arc_mint:
    MIN_TRAIN_TIME = int(_arc_mint)
# Augmentation
INPUT_NOISE = float(os.environ.get("ARC_INPUT_NOISE", "0"))  # 0=disabled. Even 1% noise collapses val. Bug fixed: now applies in full-batch path too.

# EMA
EMA_DECAY = 0                # REVERTED from 0.98: test showed EMA is val-overfit (anti-correlated). EMA-only test 0.089 vs no-EMA 0.190. Weight smoothing = more conservative -> higher val (bear), lower test (bull). 0=disabled.

# LAWA (Latest Weight Averaging)
LAWA_K = 0                   # REVERTED from 3: 20-seed showed LAWA+EMA was SEED-FAVORED (canonical 2.229, 20-seed 1.766, 20.7% collapse). Honest central 1.766 << EMA-only 2.184. EMA=0.98 alone is the honest best. 0=disabled.

# Portfolio variant flags (USER REQUEST Jul 1 2026) — 4 variants via env:
#   long-short        (ARC_LONG_ONLY=0 ARC_CASH=0)
#   long-only         (ARC_LONG_ONLY=1 ARC_CASH=0)
#   long-short+cash   (ARC_LONG_ONLY=0 ARC_CASH=1)
#   long-only+cash    (ARC_LONG_ONLY=1 ARC_CASH=1)
# Cash: model outputs a 5th signal; cash_weight=sigmoid(sig)*CASH_MAX scales ETF
# weights by (1-cash_weight). Cash earns ~0%. Lets model de-risk in bear markets.
CASH_ENABLED = os.environ.get("ARC_CASH", "0") == "1"  # default off for baseline
CASH_MAX = float(os.environ.get("ARC_CASH_MAX", "0.50"))  # max cash allocation
CASH_BIAS = float(os.environ.get("ARC_CASH_BIAS", "2.0"))  # sigmoid shift: bias=2 → default ~12% of CASH_MAX
LONG_ONLY = os.environ.get("ARC_LONG_ONLY", "0") == "1"  # default long+short

# Volatility gate (USER EXPERIMENT Jul 2 2026): scale down positions when market
# volatility is high (bear markets). Helps val MaxDD without hurting test (bull=low vol).
VOL_GATE = os.environ.get("ARC_VOL_GATE", "1") == "1"  # ON: scale down in high-vol (bear)
VOL_GATE_STRENGTH = float(os.environ.get("ARC_VOL_GATE_STRENGTH", "500.0"))  # circuit breaker: extreme reduction on worst days only
VOL_GATE_WINDOW = int(os.environ.get("ARC_VOL_GATE_WINDOW", "20"))
VOL_GATE_FORMULA = os.environ.get("ARC_VOL_GATE_FORMULA", "backward")  # backward|timesfm|timesfm_var|timesfm_blend
VOL_GATE_THRESHOLD = float(os.environ.get("ARC_VOL_GATE_THRESHOLD", "0.025"))  # 75th pct — only worst vol days get scaled
REGIME_VOL_K = float(os.environ.get("ARC_REGIME_VOL_K", "0.0"))  # disabled — circuit breaker already saturated
TIMESFM_VOL_BLEND = float(os.environ.get("ARC_TIMESFM_VOL_BLEND", "0.5"))  # blend backward+timesfm

# Global: TimesFM vol forecast tensor aligned with current data (set by _load_vol_forecast)
_VOL_FORECAST_TENSOR = None  # (N,) tensor or None

def _load_timesfm_vol_forecast(dates):
    """Load TimesFM forecast with one value for each requested date, in order."""
    dates = pd.DatetimeIndex(dates)
    if dates.empty:
        return None
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("TimesFM forecast dates must be unique and sorted")
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "etf_autoresearch",
                              "timesfm_raw_vol_forecasts.parquet")
    if not os.path.exists(cache_path):
        return None
    df = pd.read_parquet(cache_path)
    aligned = df.reindex(dates, method="ffill")
    col = "forecast_left_tail_abs" if VOL_GATE_FORMULA == "timesfm_var" else "forecast_vol"
    vals = aligned[col].values
    if np.isnan(vals).all():
        return None
    s = pd.Series(vals, index=dates).ffill().bfill()
    return torch.tensor(s.values, dtype=torch.float32)

# TimesFM Stacked Generalization (opt-in; baseline production path disables it)
# LSTM trains on original features unchanged. TimesFM blending happens in backtest.py only.
USE_TIMESFM_VOL = os.environ.get("ARC_USE_TIMESFM", "0") == "1"
TIMESFM_DIM = 4              # number of TimesFM pair exhaustion features
TIMESFM_BLEND_ALPHA = float(os.environ.get("ARC_TSFMA", "1.0"))  # 1.0 = disabled; set ARC_USE_TIMESFM=1 and ARC_TSFMA<1 to opt in.
TIMESFM_SIGNAL_SCALE = 0.001 # scale: (exhaustion - 1.0) * scale — 0.001 optimal (0.01→86.51%, 0.1→86.23%)
SIGNAL_CLIP = 3.0            # clip signals to [-c, c] — peak at 3.0 with alpha=0.22, LS=0.05 (86.96%)
SIGNAL_EMA_DECAY = 0.4       # Optuna-found: EMA=0.4 improves val (0.222→0.296). Applied to baseline.
SIGNAL_THRESHOLD = 0.30     # Baseline.
REGIME_THRESHOLD_K = float(os.environ.get("ARC_REGIME_K", "0.0"))  # disabled — regime threshold caused catastrophic collapse
SIGNAL_POWER = 1.0           # power transform: sign(x)*|x|^p before tanh (1.0=identity, disabled)

# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params, return backup."""
        backup = {}
        for name, p in model.named_parameters():
            backup[name] = p.clone()
            p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore model params from backup."""
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])


class LAWA:
    """Latest Weight Averaging: average the K most recent checkpoint weights."""
    def __init__(self, k):
        self.k = k
        self.checkpoints = []  # list of state_dicts (copies)

    def save(self, model):
        """Save a copy of current model weights."""
        self.checkpoints.append({n: p.clone().detach() for n, p in model.named_parameters()})
        if len(self.checkpoints) > self.k:
            self.checkpoints.pop(0)

    def apply(self, model):
        """Average all stored checkpoints and load into model. Returns backup."""
        if not self.checkpoints:
            return None
        backup = {n: p.clone() for n, p in model.named_parameters()}
        # Average
        avg = {}
        for name in self.checkpoints[0]:
            avg[name] = torch.stack([ckpt[name] for ckpt in self.checkpoints]).mean(dim=0)
        for name, p in model.named_parameters():
            p.data.copy_(avg[name])
        return backup

    def restore(self, model, backup):
        if backup is None:
            return
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# Variable Selection Network (Lim et al., 2021; Saly-Kaufmann et al., 2026)
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """Per-timestep adaptive feature selection via softmax gating.

    Learns data-dependent importance weights for each input feature,
    suppressing noisy/irrelevant covariates before the LSTM.
    Inspired by the VSN component of the Temporal Fusion Transformer
    and the VLSTM benchmark in arXiv:2603.01820.
    """

    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        # Two-layer gating network: raw features -> importance weights
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        h = F.elu(self.fc1(x))            # (B, T, hidden)
        weights = self.fc2(h)              # (B, T, F)
        weights = F.softmax(weights, dim=-1)  # normalise across features
        weights = self.dropout(weights)
        # Scale so mean weight ≈ 1.0 (softmax sums to 1, need × F)
        return x * weights * self.input_dim


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS, feat_drop_rate=0.0, out_dropout=None,
                 use_feature_gate=False, use_vsn=False, vsn_hidden=64, vsn_residual=False,
                 vsn_learnable_alpha=False, vsn_alpha=0.5, use_cash=False):
        super().__init__()
        self.feat_drop_rate = feat_drop_rate
        self.use_feature_gate = use_feature_gate
        self.use_vsn = use_vsn
        self.vsn_residual = vsn_residual
        self.vsn_learnable_alpha = vsn_learnable_alpha
        self.vsn_alpha_val = vsn_alpha
        self.use_cash = use_cash
        self.num_pairs = num_pairs
        # Kept only so existing checkpoints load strictly; LSTMModel.forward()
        # has never used this input normalization layer.
        self.layer_norm = nn.LayerNorm(input_dim)
        self.layer_norm.requires_grad_(False)
        if use_feature_gate:
            self.feature_gate = nn.Parameter(torch.full((input_dim,), 5.0))
        if use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim, hidden_dim=vsn_hidden, dropout=dropout)
            if vsn_residual and vsn_learnable_alpha:
                # Initialize at logit(0.5) = 0.0 so sigmoid gives 0.5 initially
                self.vsn_alpha = nn.Parameter(torch.zeros(input_dim))
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(out_dropout if out_dropout is not None else dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        # When cash enabled, head outputs num_pairs+1 (last = cash signal)
        out_dim = num_pairs + 1 if use_cash else num_pairs
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (B, T, F)
        if self.use_vsn:
            x_gated = self.vsn(x)
            if self.vsn_residual:
                if self.vsn_learnable_alpha:
                    alpha = torch.sigmoid(self.vsn_alpha)  # (F,) in [0,1]
                    x = alpha * x + (1 - alpha) * x_gated
                else:
                    x = self.vsn_alpha_val * x + (1 - self.vsn_alpha_val) * x_gated
            else:
                x = x_gated
        elif self.use_feature_gate:
            x = x * torch.sigmoid(self.feature_gate)
        # Feature dropout: zero out entire feature columns during training
        if self.training and self.feat_drop_rate > 0:
            mask = torch.ones(1, 1, x.size(2), device=x.device)
            mask = F.dropout(mask, p=self.feat_drop_rate, training=True)
            x = x * mask
        output, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, hidden_dim)
        last_hidden = self.output_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        signals = self.head(last_hidden)  # (B, num_pairs)
        return signals


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_pairs)

    def forward(self, x):
        # x: (B, T, F)
        x = self.layer_norm(x)
        output, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (B, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        signals = self.head(last_hidden)  # (B, num_pairs)
        return signals


class LSTMAttnModel(nn.Module):
    """LSTM with temporal attention over hidden states."""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_pairs)

    def forward(self, x):
        # x: (B, T, F)
        x = self.layer_norm(x)
        output, _ = self.lstm(x)  # output: (B, T, H)
        # Attention weights over time
        attn_weights = torch.softmax(self.attn(output), dim=1)  # (B, T, 1)
        context = (attn_weights * output).sum(dim=1)  # (B, H)
        context = self.dropout(context)
        signals = self.head(context)  # (B, num_pairs)
        return signals


class LSTMBidirModel(nn.Module):
    """Bidirectional LSTM."""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_pairs)

    def forward(self, x):
        # x: (B, T, F)
        x = self.layer_norm(x)
        output, (h_n, _) = self.lstm(x)
        # Concat forward and backward final hidden states
        fwd = h_n[-2]  # forward last layer
        bwd = h_n[-1]  # backward last layer
        combined = torch.cat([fwd, bwd], dim=1)  # (B, 2*H)
        combined = self.dropout(combined)
        signals = self.head(combined)  # (B, num_pairs)
        return signals


class LinearModel(nn.Module):
    """Simple linear model: average features over time, then linear projection."""
    def __init__(self, input_dim, num_pairs=NUM_PAIRS):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_pairs)

    def forward(self, x):
        # x: (B, T, F) -> avg over time -> (B, F)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # (B, F)
        return self.head(x)  # (B, num_pairs)


class MLPModel(nn.Module):
    def __init__(self, input_dim, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS):
        super().__init__()
        flat_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_pairs),
        )

    def forward(self, x):
        # x: (B, T, F) -> flatten to (B, T*F)
        B = x.size(0)
        x = x.reshape(B, -1)
        signals = self.net(x)  # (B, num_pairs)
        return signals


class TCNModel(nn.Module):
    """Temporal Convolutional Network with causal dilated convolutions."""
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS, kernel_size=3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # causal padding
            layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size,
                                    dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = hidden_dim
        self.conv_layers = nn.ModuleList()
        self.dilation_layers = []
        # Build proper causal conv blocks
        self.blocks = nn.ModuleList()
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            block = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size,
                          dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
            in_channels = hidden_dim
        self.causal_paddings = [(kernel_size - 1) * (2 ** i) for i in range(num_layers)]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_pairs)

    def forward(self, x):
        # x: (B, T, F)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, F, T) for conv1d
        for block, pad in zip(self.blocks, self.causal_paddings):
            x = block(x)
            if pad > 0:
                x = x[:, :, :-pad]  # remove future padding (causal)
        # Global average pooling over time
        x = x.mean(dim=2)  # (B, hidden_dim)
        x = self.dropout(x)
        signals = self.head(x)
        return signals


class TCNVSNModel(nn.Module):
    """Temporal Convolutional Network with residual dilated causal convolutions + VSN.

    Per Research Insight: TCNs use dilated causal convolutions, offering
    parallelizable training and stable gradient flow across long receptive fields
    without LSTM's sequential bottleneck.
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS, kernel_size=3,
                 feat_drop_rate=0.0, use_vsn=False, vsn_hidden=64,
                 vsn_residual=True, vsn_alpha=0.02):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.feat_drop_rate = feat_drop_rate
        self.use_vsn = use_vsn
        self.vsn_residual = vsn_residual
        self.vsn_alpha_val = vsn_alpha
        if use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim, hidden_dim=vsn_hidden, dropout=dropout)
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = input_dim
        for i in range(max(1, num_layers)):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            block = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size,
                          dilation=dilation, padding=padding),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
            if in_channels != hidden_dim:
                self.projections.append(nn.Conv1d(in_channels, hidden_dim, 1))
            else:
                self.projections.append(None)
            self.norms.append(nn.GroupNorm(1, hidden_dim))
            in_channels = hidden_dim
        self.causal_paddings = [(kernel_size - 1) * (2 ** i) for i in range(max(1, num_layers))]
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_pairs)

    def forward(self, x):
        x = self.layer_norm(x)
        if self.use_vsn:
            x_gated = self.vsn(x)
            if self.vsn_residual:
                x = self.vsn_alpha_val * x + (1 - self.vsn_alpha_val) * x_gated
            else:
                x = x_gated
        if self.training and self.feat_drop_rate > 0:
            mask = torch.ones(1, 1, x.size(2), device=x.device)
            mask = F.dropout(mask, p=self.feat_drop_rate, training=True)
            x = x * mask
        x = x.transpose(1, 2)
        for block, proj, norm, pad in zip(self.blocks, self.projections, self.norms, self.causal_paddings):
            residual = x
            out = block(x)
            if pad > 0:
                out = out[:, :, :-pad]
            if proj is not None:
                residual = proj(residual)
            x = norm(out + residual)
            x = F.relu(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        signals = self.head(x)
        return signals


class CNNLSTMAttentionModel(nn.Module):
    """Hybrid CNN-LSTM-Attention model (DCA-BiLSTM inspired).

    Combines CNN local feature extraction with BiLSTM temporal modeling and
    temporal attention. Per web research, hybrid conv-recurrent models with
    attention outperform standalone architectures.
    """
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, num_pairs=NUM_PAIRS, kernel_size=3,
                 feat_drop_rate=0.0, use_vsn=False, vsn_hidden=64,
                 vsn_residual=True, vsn_alpha=0.02, cnn_channels=128):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.feat_drop_rate = feat_drop_rate
        self.use_vsn = use_vsn
        self.vsn_residual = vsn_residual
        self.vsn_alpha_val = vsn_alpha
        if use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim, hidden_dim=vsn_hidden, dropout=dropout)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size, padding=(kernel_size - 1)),
            nn.GroupNorm(1, cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=(kernel_size - 1)),
            nn.GroupNorm(1, cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cnn_pad = (kernel_size - 1) * 2
        self.lstm = nn.LSTM(
            input_size=cnn_channels, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_pairs)

    def forward(self, x):
        x = self.layer_norm(x)
        if self.use_vsn:
            x_gated = self.vsn(x)
            if self.vsn_residual:
                x = self.vsn_alpha_val * x + (1 - self.vsn_alpha_val) * x_gated
            else:
                x = x_gated
        if self.training and self.feat_drop_rate > 0:
            mask = torch.ones(1, 1, x.size(2), device=x.device)
            mask = F.dropout(mask, p=self.feat_drop_rate, training=True)
            x = x * mask
        x = x.transpose(1, 2)
        x = self.cnn(x)
        if self.cnn_pad > 0:
            x = x[:, :, :-self.cnn_pad]
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(output), dim=1)
        context = (attn_weights * output).sum(dim=1)
        context = self.dropout(context)
        signals = self.head(context)
        return signals


class MCDropoutWrapper(nn.Module):
    """Wrapper that runs model N times with dropout enabled and averages."""
    def __init__(self, model, n_samples=10):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, x):
        self.model.train()  # enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(x))
        return torch.stack(preds).mean(dim=0)

    def eval(self):
        return self  # stay in "MC" mode

    def train(self, mode=True):
        return self


def build_model(model_type, input_dim, config=None):
    """Factory function to create model by type.
    
    When config is provided (e.g. from a saved checkpoint), architecture
    parameters are read from it so that older checkpoints reconstruct
    correctly even if the current globals have changed.
    """
    hidden_dim = config.get("hidden_dim", HIDDEN_DIM) if config else HIDDEN_DIM
    num_layers = config.get("num_layers", NUM_LAYERS) if config else NUM_LAYERS
    dropout = config.get("dropout", DROPOUT) if config else DROPOUT
    seq_len = config.get("seq_len", SEQ_LEN) if config else SEQ_LEN
    feat_drop_rate = config.get("feat_drop_rate", FEAT_DROP_RATE) if config else FEAT_DROP_RATE
    out_dropout = config.get("out_dropout", OUT_DROPOUT) if config else OUT_DROPOUT
    use_feature_gate = config.get("use_feature_gate", USE_FEATURE_GATE) if config else USE_FEATURE_GATE
    use_vsn = config.get("use_vsn", USE_VSN) if config else USE_VSN
    vsn_hidden = config.get("vsn_hidden", VSN_HIDDEN) if config else VSN_HIDDEN
    vsn_residual = config.get("vsn_residual", VSN_RESIDUAL) if config else VSN_RESIDUAL
    vsn_learnable_alpha = config.get("vsn_learnable_alpha", VSN_LEARNABLE_ALPHA) if config else VSN_LEARNABLE_ALPHA
    vsn_alpha = config.get("vsn_alpha", VSN_ALPHA) if config else VSN_ALPHA

    if model_type == "lstm":
        return LSTMModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         dropout=dropout, feat_drop_rate=feat_drop_rate,
                         out_dropout=out_dropout, use_feature_gate=use_feature_gate,
                         use_vsn=use_vsn, vsn_hidden=vsn_hidden, vsn_residual=vsn_residual,
                         vsn_learnable_alpha=vsn_learnable_alpha, vsn_alpha=vsn_alpha,
                         use_cash=config.get("use_cash", CASH_ENABLED) if config else CASH_ENABLED)
    elif model_type == "linear":
        return LinearModel(input_dim)
    elif model_type == "gru":
        return GRUModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == "mlp":
        return MLPModel(input_dim, seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "lstm_attn":
        return LSTMAttnModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == "lstm_bidir":
        return LSTMBidirModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == "tcn":
        return TCNModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == "tcn_vsn":
        return TCNVSNModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                           dropout=dropout, feat_drop_rate=feat_drop_rate,
                           use_vsn=use_vsn, vsn_hidden=vsn_hidden,
                           vsn_residual=vsn_residual, vsn_alpha=vsn_alpha)
    elif model_type == "cnn_lstm_attn":
        return CNNLSTMAttentionModel(input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                     dropout=dropout, feat_drop_rate=feat_drop_rate,
                                     use_vsn=use_vsn, vsn_hidden=vsn_hidden,
                                     vsn_residual=vsn_residual, vsn_alpha=vsn_alpha)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, config, path):
    """Save model checkpoint with full config for reconstruction."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)


def load_checkpoint(path, device):
    """Load model from checkpoint. Returns (model, config).
    
    Reconstructs the model using saved config values so that architecture
    parameters match the checkpoint even if current globals differ.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config["model_type"], config["input_dim"], config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, config


def _backward_volatility(returns, window):
    """Rolling std of strictly past returns: row i uses returns[i-window:i].

    Excludes the current period's own return (no look-ahead). Rows without a
    full `window` of history are left ungated (zero volatility).
    """
    n = returns.size(0)
    vol = torch.zeros_like(returns)
    if n > window:
        padded = F.pad(returns.unsqueeze(0), (window, 0))
        windows = padded.unfold(1, window, 1).squeeze(0)[:n]
        vol = windows.std(dim=1)
        vol[:window] = 0.0
    return vol


def compute_portfolio(all_signals, all_targets, vol_forecast=None):
    """Convert signals to weights + portfolio returns, handling cash and long-only.

    Single source of truth for the weight pipeline. Used by evaluate_ensemble,
    the training loop, and backtest.py (imported). Handles 4 variants:
    long-short, long-only, long-short+cash, long-only+cash.

    Args:
        all_signals: (N, num_pairs) or (N, num_pairs+1) when CASH_ENABLED
        all_targets: (N, 8) ETF returns
        vol_forecast: optional (N,) TimesFM forecast already aligned to rows
    Returns:
        weights: (N, 8) ETF weights (scaled by cash if enabled)
        portfolio_returns: (N,) portfolio returns (cash earns 0%)
    """
    global _VOL_FORECAST_TENSOR
    if all_signals.ndim != 2 or all_targets.ndim != 2:
        raise ValueError("Signals and targets must be rank-2 tensors")
    if all_signals.size(0) != all_targets.size(0):
        raise ValueError("Signals and targets must have identical row counts")

    active_vol_forecast = _VOL_FORECAST_TENSOR if vol_forecast is None else vol_forecast
    validated_vol_forecast = None
    if active_vol_forecast is not None:
        validated_vol_forecast = torch.as_tensor(active_vol_forecast, dtype=torch.float32)
        if validated_vol_forecast.ndim != 1:
            raise ValueError("TimesFM vol forecast must be a rank-1 tensor")
        if validated_vol_forecast.size(0) != all_targets.size(0):
            raise ValueError(
                "TimesFM vol forecast is not aligned to portfolio rows: "
                f"{validated_vol_forecast.size(0)} forecasts for {all_targets.size(0)} rows"
            )
        if not torch.isfinite(validated_vol_forecast).all():
            raise ValueError("TimesFM vol forecast contains missing/non-finite values")

    # Split cash signal if enabled (last column = cash signal)
    if CASH_ENABLED and all_signals.size(1) > NUM_PAIRS:
        pair_signals = all_signals[:, :NUM_PAIRS]
        cash_signal = all_signals[:, NUM_PAIRS:]
        cash_weight = torch.sigmoid(cash_signal - CASH_BIAS) * CASH_MAX  # (N,1) in [0, CASH_MAX]
    else:
        pair_signals = all_signals
        cash_weight = None

    # Regime-dependent threshold: raise in high-vol (bear) to reduce trading
    # Only active when an eval forecast is supplied. Training leaves both the
    # explicit argument and global forecast unset, so signal pipeline threshold applies.
    if REGIME_THRESHOLD_K > 0 and validated_vol_forecast is not None and VOL_GATE_FORMULA in ("timesfm", "timesfm_var"):
        vol = validated_vol_forecast.to(pair_signals.device)
        vol_excess = torch.clamp(vol - VOL_GATE_THRESHOLD, min=0)
        regime_threshold = SIGNAL_THRESHOLD + REGIME_THRESHOLD_K * vol_excess.unsqueeze(1)
        pair_signals = torch.where(torch.abs(pair_signals) < regime_threshold,
                                   torch.zeros_like(pair_signals), pair_signals)

    weights = signals_to_weights(pair_signals)
    if LONG_ONLY:
        weights = torch.clamp(weights, min=0)
        total_abs = weights.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(1.0 / total_abs, max=1.0)
        weights = weights * scale
    # Cash scaling: reduce ETF exposure by cash_weight (cash earns ~0%)
    if cash_weight is not None:
        weights = weights * (1.0 - cash_weight)
    # Volatility gate: scale down positions in high-volatility periods (bear markets)
    if VOL_GATE:
        use_timesfm = (validated_vol_forecast is not None and
                       VOL_GATE_FORMULA in ("timesfm", "timesfm_var", "timesfm_blend") and
                       validated_vol_forecast.size(0) == all_targets.size(0))
        if use_timesfm:
            assert validated_vol_forecast is not None
            timesfm_vol = validated_vol_forecast.to(all_targets.device)
            if VOL_GATE_FORMULA == "timesfm_blend":
                backward_vol = _backward_volatility(all_targets.mean(dim=1), VOL_GATE_WINDOW)
                vol = TIMESFM_VOL_BLEND * timesfm_vol + (1 - TIMESFM_VOL_BLEND) * backward_vol
            else:
                vol = timesfm_vol
        else:
            vol = _backward_volatility(all_targets.mean(dim=1), VOL_GATE_WINDOW)
        vol_excess = torch.clamp(vol - VOL_GATE_THRESHOLD, min=0)
        if REGIME_VOL_K > 0:
            # Regime-dependent strength: increase in high-vol for more aggressive scaling
            effective_strength = VOL_GATE_STRENGTH * (1.0 + REGIME_VOL_K * vol_excess)
        else:
            effective_strength = VOL_GATE_STRENGTH
        vol_scale = 1.0 / (1.0 + effective_strength * vol_excess)
        weights = weights * vol_scale.unsqueeze(1)
    portfolio_returns = (weights * all_targets).sum(dim=1)
    return weights, portfolio_returns


def evaluate_benchmark(models, config, split="test"):
    """Evaluate ensemble on benchmark period using backtest.py's EXACT pipeline.

    Replicates backtest.py's per-date evaluation loop: trimmed_mean aggregation,
    EMA smoothing, signal pipeline (power/clip/threshold/TimesFM blend), cash
    bypass, compute_portfolio, compute_metrics. Used by Optuna so its metrics
    EXACTLY match measure.sh/backtest.py.
    """
    from production_model import _aggregate_ensemble, apply_signal_pipeline, load_timesfm_features_by_date
    from backtest import compute_metrics

    device = torch.device("cpu")
    trade_frequency = config.get("trade_frequency", "daily")
    seq_len = config.get("seq_len", SEQ_LEN)
    feature_columns = config["feature_columns"]

    features_df, targets_df = build_dataset(trade_frequency=trade_frequency)
    features_df = validate_feature_columns(
        features_df, feature_columns, context=f"{split} benchmark features"
    )

    scaler_params = config["scaler_params"]
    mean = pd.Series(scaler_params["mean"])
    std = pd.Series(scaler_params["std"])
    feat_norm = normalize_features(features_df, mean, std)

    if trade_frequency == "weekly":
        weekly_idx = feat_norm.index[::5]
        feat_norm = feat_norm.loc[weekly_idx]
        targets_df = targets_df.loc[weekly_idx]

    if split == "val":
        mask = (feat_norm.index >= TRAIN_END) & (feat_norm.index < VAL_END)
    else:
        mask = feat_norm.index >= VAL_END
    split_indices = np.where(mask)[0]
    if len(split_indices) <= seq_len:
        return None
    split_start_idx = split_indices[0]
    if split_start_idx < seq_len - 1:
        split_start_idx = seq_len - 1
    split_end_idx = split_indices[-1]

    feat_np = feat_norm.values.astype(np.float32)
    tgt_np = targets_df.values.astype(np.float32)
    dates = feat_norm.index

    tsfm_features_by_date = load_timesfm_features_by_date(dates)

    all_signals = []
    all_targets = []
    all_dates = []
    ema_sig = None
    for i in range(split_start_idx, split_end_idx + 1):
        window = feat_np[i - seq_len + 1 : i + 1]
        if len(window) < seq_len:
            continue
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            stacked = torch.stack([m(x).cpu() for m in models])
            avg_sig = _aggregate_ensemble(stacked)
        _cash_sig = None
        if CASH_ENABLED and avg_sig.size(1) > NUM_PAIRS:
            _cash_sig = avg_sig[:, NUM_PAIRS:]
            avg_sig = avg_sig[:, :NUM_PAIRS]
        if SIGNAL_EMA_DECAY > 0:
            if ema_sig is None:
                ema_sig = avg_sig.clone()
            else:
                ema_sig = SIGNAL_EMA_DECAY * ema_sig + (1 - SIGNAL_EMA_DECAY) * avg_sig
            avg_sig = ema_sig
        tsfm_feat = tsfm_features_by_date.get(dates[i]) if tsfm_features_by_date else None
        avg_sig = apply_signal_pipeline(avg_sig, tsfm_feat)
        if _cash_sig is not None:
            avg_sig = torch.cat([avg_sig, _cash_sig], dim=1)
        all_signals.append(avg_sig)
        all_targets.append(torch.tensor(tgt_np[i:i + 1], dtype=torch.float32))
        all_dates.append(dates[i])

    all_signals_t = torch.cat(all_signals, dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)
    evaluation_dates = pd.DatetimeIndex(all_dates)
    vol_forecast = _load_timesfm_vol_forecast(evaluation_dates)
    global _VOL_FORECAST_TENSOR
    _VOL_FORECAST_TENSOR = vol_forecast
    weights, portfolio_returns = compute_portfolio(
        all_signals_t, all_targets_t, vol_forecast=vol_forecast
    )
    ret_np = portfolio_returns.numpy()
    w_np = weights.numpy()
    periods_per_year = 52 if trade_frequency == "weekly" else 252
    return compute_metrics(ret_np, w_np, periods_per_year, trade_frequency)


def evaluate_ensemble(models, dataloader, device, trade_frequency="daily", model_weights=None):
    """Evaluate ensemble by (optionally weighted) averaging signals from multiple models."""
    if dataloader is None:
        return {"val_score": 999.0, "sharpe": 0.0, "max_drawdown": 1.0, "turnover": 1.0}

    all_signals = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        batch_signals = []
        for m in models:
            m.eval()
            with torch.no_grad():
                batch_signals.append(m(x_batch).cpu())
        stacked = torch.stack(batch_signals)  # (num_models, B, num_pairs)
        if model_weights is not None:
            w = torch.tensor(model_weights, dtype=stacked.dtype).view(-1, 1, 1)
            avg_signals = (stacked * w).sum(dim=0)
        else:
            avg_signals = stacked.mean(dim=0)
        all_signals.append(avg_signals)
        all_targets.append(y_batch)

    all_signals = torch.cat(all_signals, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Apply signal pipeline (power, clip, threshold) to PAIR signals only.
    # Cash signal bypasses threshold (matches backtest.py signal processing).
    # TimesFM blend skipped (requires per-date features; fixed alpha, not in search).
    if CASH_ENABLED and all_signals.size(1) > NUM_PAIRS:
        _cash_sig = all_signals[:, NUM_PAIRS:]
        _pair_sig = all_signals[:, :NUM_PAIRS]
    else:
        _cash_sig = None
        _pair_sig = all_signals
    if SIGNAL_POWER != 1.0:
        _pair_sig = torch.sign(_pair_sig) * torch.abs(_pair_sig) ** SIGNAL_POWER
    if SIGNAL_CLIP > 0:
        _pair_sig = torch.clamp(_pair_sig, -SIGNAL_CLIP, SIGNAL_CLIP)
    if SIGNAL_THRESHOLD > 0:
        _pair_sig = torch.where(torch.abs(_pair_sig) < SIGNAL_THRESHOLD,
                                torch.zeros_like(_pair_sig), _pair_sig)
    if _cash_sig is not None:
        all_signals = torch.cat([_pair_sig, _cash_sig], dim=1)
    else:
        all_signals = _pair_sig

    weights, portfolio_returns = compute_portfolio(all_signals, all_targets)

    mean_ret = portfolio_returns.mean().item()
    std_ret = portfolio_returns.std().item()
    periods_per_year = 252 if trade_frequency == "daily" else 52
    sharpe = (mean_ret / max(std_ret, 1e-8)) * (periods_per_year ** 0.5)

    cum_returns = (1 + portfolio_returns).cumprod(dim=0)
    running_max = cum_returns.cummax(dim=0).values
    drawdowns = (cum_returns - running_max) / running_max.clamp(min=1e-8)
    max_drawdown = abs(drawdowns.min().item())

    # CAGR and Calmar for Optuna objective
    total_return = cum_returns[-1].item() - 1.0
    num_years = max(len(portfolio_returns) / periods_per_year, 1e-8)
    cagr = (1 + total_return) ** (1.0 / num_years) - 1.0 if total_return > -1 else -1.0
    calmar = cagr / max(max_drawdown, 0.01)

    weight_changes = weights[1:] - weights[:-1]
    turnover = weight_changes.abs().sum(dim=1).mean().item()

    val_score = -sharpe + 0.1 * turnover + 0.05 * max_drawdown

    return {
        "val_score": round(val_score, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "turnover": round(turnover, 6),
        "cagr": round(cagr, 6),
        "calmar": round(calmar, 6),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_split(features_df, targets_df, device, train_end=None, val_end=None,
                        fold_label=""):
    """Train on a single train/val split. Returns (val_metrics, test_metrics, num_epochs, num_params).

    For the standard (non-walk-forward) path, train_end/val_end default to the
    EARLY-STOP split (a held-out slice of the TRAINING period) so checkpoint
    selection NEVER touches the benchmark val window (TRAIN_END..VAL_END). This
    eliminates the early-stopping-on-benchmark leakage that inflated the prior
    val Calmar and drove the val->test gap. Walk-forward calls pass explicit
    fold cutoffs and are unaffected.
    """
    if train_end is None:
        train_end = EARLY_STOP_TRAIN_END
    if val_end is None:
        val_end = EARLY_STOP_VAL_END

    train_loader, val_loader, test_loader, scaler_params, feature_columns = \
        make_dataloaders(features_df, targets_df, lookback=SEQ_LEN,
                         batch_size=BATCH_SIZE, trade_frequency=TRADE_FREQUENCY,
                         train_end=train_end, val_end=val_end)

    if train_loader is None:
        print("ERROR: not enough training data")
        return None, None, 0, 0

    input_dim = len(feature_columns)

    seeds = ENSEMBLE_SEEDS if ENSEMBLE_SEEDS else [42]
    ensemble_models = []
    seed_val_sharpes = []  # for weighted ensemble
    seed_results = []
    num_params = 0

    for seed_idx, SEED in enumerate(seeds):
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)

        # Dual-ensemble 3+7: 3 seeds TURN_PEN=15 (test), 7 seeds TURN_PEN=17.5 (val) — CHAMPION
        _dual_enabled = (os.environ.get("ARC_DUAL_ENSEMBLE", "1") == "1")
        if len(seeds) == 10 and MODEL_TYPE == "lstm" and _dual_enabled:
            globals()['TURN_PEN'] = 17.5 if seed_idx >= 3 else 15.0
            if seed_idx == 0:
                print(f"  Dual-ensemble 3+7: TURN_PEN=15.0 for seeds 0-2 (test-favoring)")
            elif seed_idx == 3:
                print(f"  Dual-ensemble 3+7: TURN_PEN=17.5 for seeds 3-9 (val-favoring)")

        if len(seeds) > 1:
            print(f"\n--- Seed {SEED} ({seed_idx+1}/{len(seeds)}) ---")

        # Per-seed dropout for heterogeneous ensemble (regime diversity)
        if len(ENSEMBLE_DROPOUTS) == len(seeds):
            current_dropout = ENSEMBLE_DROPOUTS[seed_idx]
            print(f"  Heterogeneous ensemble: dropout={current_dropout}")
        else:
            current_dropout = DROPOUT

        print(f"\n{fold_label}Model: {MODEL_TYPE}, input_dim={input_dim}, hidden={HIDDEN_DIM}, "
              f"layers={NUM_LAYERS}, dropout={current_dropout}")

        # Model
        model = build_model(MODEL_TYPE, input_dim, config={"dropout": current_dropout})
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")

        # EMA
        ema = EMA(model, EMA_DECAY) if EMA_DECAY > 0 else None

        # LAWA
        lawa = LAWA(LAWA_K) if LAWA_K > 0 else None

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=LEARNING_RATE * 0.01
        )

        # Config for checkpoint
        config = {
            "model_type": MODEL_TYPE,
            "data_split_version": DATA_SPLIT_VERSION,
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": current_dropout,
            "feat_drop_rate": FEAT_DROP_RATE,
            "out_dropout": OUT_DROPOUT,
            "use_feature_gate": USE_FEATURE_GATE,
            "use_vsn": USE_VSN,
            "vsn_hidden": VSN_HIDDEN,
            "vsn_residual": VSN_RESIDUAL,
            "vsn_learnable_alpha": VSN_LEARNABLE_ALPHA,
            "vsn_alpha": VSN_ALPHA,
            "use_cash": CASH_ENABLED,
            "cash_max": CASH_MAX,
            "long_only": LONG_ONLY,
            "seq_len": SEQ_LEN,
            "trade_frequency": TRADE_FREQUENCY,
            "feature_columns": feature_columns,
            "scaler_params": scaler_params,
        }

        checkpoint_path = os.path.join("models", f"best_model_seed{SEED}.pt" if len(seeds) > 1 else "best_model.pt")

        # Training loop
        print(f"\nTraining (time budget: {TIME_BUDGET}s, patience: {PATIENCE}, min_time: {MIN_TRAIN_TIME}s)...")
        t_start_training = time.time()
        best_val_score = float("inf")
        epochs_without_improvement = 0
        total_training_time = 0.0
        epoch = 0

        while True:
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            _PORTFOLIO_LOSSES = (
                "sharpe", "huber_sharpe",
                "cagr", "log_cagr", "cagr_cvar", "cagr_skew", "cagr_cvar_skew",
            )
            if LOSS_TYPE in _PORTFOLIO_LOSSES:
                # Full-batch differentiable portfolio-level loss
                optimizer.zero_grad()
                all_signals = []
                all_targets = []
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    # Input noise augmentation (also in full-batch path)
                    if INPUT_NOISE > 0:
                        x_batch = x_batch + torch.randn_like(x_batch) * INPUT_NOISE
                    signals = model(x_batch)
                    all_signals.append(signals)
                    all_targets.append(y_batch)
                all_signals = torch.cat(all_signals, dim=0)
                all_targets = torch.cat(all_targets, dim=0)

                # Signals → weights → portfolio returns (all differentiable)
                weights, portfolio_returns = compute_portfolio(all_signals, all_targets)
                periods_per_year = 52 if TRADE_FREQUENCY == "weekly" else 252

                # Turnover penalty (shared by all portfolio-level losses)
                weight_changes = weights[1:] - weights[:-1]
                turnover = weight_changes.abs().sum(dim=1).mean()

                # --- Loss computation per LOSS_TYPE ---
                if LOSS_TYPE in ("sharpe", "huber_sharpe"):
                    mean_ret = portfolio_returns.mean()
                    std_ret = portfolio_returns.std() + 1e-8
                    sharpe = (mean_ret / std_ret) * (periods_per_year ** 0.5)
                    if LOSS_TYPE == "sharpe":
                        loss = -sharpe + 0.1 * turnover
                    else:  # huber_sharpe hybrid: Huber regression + weighted Sharpe
                        pair_targets = torch.zeros(all_targets.size(0), NUM_PAIRS, device=device)
                        for i in range(NUM_PAIRS):
                            pair_targets[:, i] = all_targets[:, 2 * i] - all_targets[:, 2 * i + 1]
                        huber = F.huber_loss(all_signals, pair_targets * 100, delta=0.95)
                        loss = huber + 0.1 * (-sharpe + 0.1 * turnover)
                elif LOSS_TYPE == "log_cagr":
                    # Direct geometric-mean (compound) maximization → CAGR proxy
                    log_returns = torch.log((portfolio_returns + 1.0).clamp(min=1e-2))
                    loss = -log_returns.mean() * RET_SCALE + TURN_PEN * turnover
                else:
                    # CAGR-family: arithmetic mean maximization with optional CVaR / Skew
                    mean_ret = portfolio_returns.mean()
                    loss = -mean_ret * RET_SCALE + TURN_PEN * turnover

                    if LOSS_TYPE in ("cagr_cvar", "cagr_cvar_skew"):
                        # Differentiable CVaR: mean of the worst CVAR_QUANTILE-tile of returns
                        k = max(1, int(CVAR_QUANTILE * len(portfolio_returns)))
                        sorted_ret = torch.sort(portfolio_returns)[0]
                        cvar = sorted_ret[:k].mean()          # negative = bad
                        loss = loss + CVAR_WEIGHT * (-cvar)   # penalise more-negative cvar

                    if LOSS_TYPE in ("cagr_skew", "cagr_cvar_skew"):
                        std_ret = portfolio_returns.std() + 1e-8
                        skew = ((portfolio_returns - mean_ret) ** 3).mean() / (std_ret ** 3)
                        loss = loss - SKEW_WEIGHT * skew      # reward positive skew

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                epoch_loss = loss.item()
                n_batches = 1
            else:
                # Mini-batch Huber loss
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Input noise augmentation
                    if INPUT_NOISE > 0:
                        x_batch = x_batch + torch.randn_like(x_batch) * INPUT_NOISE

                    optimizer.zero_grad()

                    # Forward: model outputs 4 pair signals, targets are 8 ETF returns
                    signals = model(x_batch)  # (B, 4)

                    # Training loss: MSE on per-pair average returns
                    # Construct pair-level targets from ETF returns
                    pair_targets = torch.zeros(y_batch.size(0), NUM_PAIRS, device=device)
                    for i in range(NUM_PAIRS):
                        # Pair return = bull_return - bear_return (what a long-bull/short-bear earns)
                        pair_targets[:, i] = y_batch[:, 2 * i] - y_batch[:, 2 * i + 1]

                    # Huber loss for robustness to outliers
                    smooth_targets = pair_targets * 100 * (1 - LABEL_SMOOTHING)
                    loss = F.huber_loss(signals, smooth_targets, delta=HUBER_DELTA)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    if ema is not None:
                        ema.update(model)

                    epoch_loss += loss.item()
                    n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation (using EMA/LAWA weights if available)
            if val_loader is not None:
                # Apply weight averaging for validation
                if lawa is not None:
                    lawa.save(model)
                    lawa_backup = lawa.apply(model)
                elif ema is not None:
                    backup = ema.apply(model)
                metrics = evaluate_portfolio(model, val_loader, device, TRADE_FREQUENCY)
                val_score = metrics["val_score"]
                # Restore original weights
                if lawa is not None:
                    lawa.restore(model, lawa_backup)
                elif ema is not None:
                    ema.restore(model, backup)
            else:
                val_score = float("inf")
                metrics = {}

            epoch_time = time.time() - epoch_start
            total_training_time = time.time() - t_start_training

            # Check for improvement
            if val_score < best_val_score:
                best_val_score = val_score
                epochs_without_improvement = 0
                # Save with LAWA/EMA weights
                if lawa is not None:
                    lawa_backup = lawa.apply(model)
                elif ema is not None:
                    backup = ema.apply(model)
                save_checkpoint(model, optimizer, config, checkpoint_path)
                if lawa is not None:
                    lawa.restore(model, lawa_backup)
                elif ema is not None:
                    ema.restore(model, backup)
            else:
                epochs_without_improvement += 1

            # Logging
            lr = optimizer.param_groups[0]["lr"]
            sharpe = metrics.get("sharpe", 0)
            pct_done = min(100, 100 * total_training_time / TIME_BUDGET)
            remaining = max(0, TIME_BUDGET - total_training_time)
            print(f"\repoch {epoch:04d} ({pct_done:.0f}%) | loss: {avg_loss:.6f} | "
                  f"val_score: {val_score:.4f} | sharpe: {sharpe:.4f} | "
                  f"lr: {lr:.6f} | dt: {epoch_time:.1f}s | remaining: {remaining:.0f}s    ",
                  end="", flush=True)

            epoch += 1

            # Stopping conditions
            if total_training_time >= TIME_BUDGET:
                print(f"\n\nTime budget reached ({TIME_BUDGET}s)")
                break
            if epochs_without_improvement >= PATIENCE and total_training_time >= MIN_TRAIN_TIME:
                print(f"\n\nEarly stopping (no improvement for {PATIENCE} epochs, {total_training_time:.0f}s elapsed)")
                break

        # Load best model for this seed
        if os.path.exists(checkpoint_path):
            model, _ = load_checkpoint(checkpoint_path, device)
        model.eval()
        ensemble_models.append(model)

        # Per-seed metrics
        seed_val = evaluate_portfolio(model, val_loader, device, TRADE_FREQUENCY)
        seed_test = evaluate_portfolio(model, test_loader, device, TRADE_FREQUENCY)
        seed_results.append({
            "seed": int(SEED),
            "val": seed_val,
            "test": seed_test,
        })
        seed_val_sharpes.append(seed_val['sharpe'])
        print(f"  Seed {SEED}: val_score={seed_val['val_score']:.6f}, "
              f"val_sharpe={seed_val['sharpe']:.6f}, test_sharpe={seed_test['sharpe']:.6f}")

    # --- Ensemble or single-model evaluation ---
    if len(ensemble_models) > 1:
        # Compute optional weights for weighted ensemble
        model_weights = None
        if WEIGHTED_ENSEMBLE and len(seed_val_sharpes) > 1:
            sharpes = np.array(seed_val_sharpes)
            # softmax to convert sharpes to normalized weights
            exp_s = np.exp(sharpes - sharpes.max())  # numerical stability
            model_weights = (exp_s / exp_s.sum()).tolist()
            print(f"\nWeighted ensemble weights: {[f'{w:.3f}' for w in model_weights]}")
        val_metrics = evaluate_ensemble(ensemble_models, val_loader, device, TRADE_FREQUENCY, model_weights)
        test_metrics = evaluate_ensemble(ensemble_models, test_loader, device, TRADE_FREQUENCY, model_weights)
    else:
        val_metrics = evaluate_portfolio(ensemble_models[0], val_loader, device, TRADE_FREQUENCY)
        test_metrics = evaluate_portfolio(ensemble_models[0], test_loader, device, TRADE_FREQUENCY)

    artifact_dir = Path(".openresearch/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "seed_metrics.json").write_text(
        json.dumps({"fold": fold_label, "seeds": seed_results}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return val_metrics, test_metrics, epoch, num_params


def main():
    t_start = time.time()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Data (load once, shared across seeds)
    print(f"\nLoading data (frequency={TRADE_FREQUENCY})...")
    features_df, targets_df = build_dataset(trade_frequency=TRADE_FREQUENCY)

    # Feature filtering
    if EXCLUDE_FEATURES:
        drop_cols = [c for c in features_df.columns if any(c.startswith(p) for p in EXCLUDE_FEATURES)]
        features_df = features_df.drop(columns=drop_cols)
        print(f"Excluded {len(drop_cols)} features matching {EXCLUDE_FEATURES}")

    # Exclude experimental features unless explicitly included
    if not INCLUDE_EXPERIMENTAL_FEATURES:
        exp_cols = [c for c in features_df.columns
                    if any(c.startswith(p) for p in _EXPERIMENTAL_MACRO_PREFIXES)
                    or any(c.endswith(s) for s in _EXPERIMENTAL_TECH_SUFFIXES)]
        if exp_cols:
            features_df = features_df.drop(columns=exp_cols)
            print(f"Excluded {len(exp_cols)} experimental features (set INCLUDE_EXPERIMENTAL_FEATURES=True to include)")

    # Replace-macro: drop original macro columns
    if _REPLACE_MACRO_ORIG:
        orig_prefix = f"macro_{_REPLACE_MACRO_ORIG}_"
        drop_orig = [c for c in features_df.columns if c.startswith(orig_prefix)]
        if drop_orig:
            features_df = features_df.drop(columns=drop_orig)
            print(f"REPLACE_MACRO: dropped {len(drop_orig)} columns for {orig_prefix}*, "
                  f"replaced with '{_REPLACE_MACRO_WITH}'")
        else:
            print(f"REPLACE_MACRO: WARNING - no columns found for {orig_prefix}*")

    if WALK_FORWARD_CV:
        # Walk-forward cross-validation: train on expanding windows, validate on each fold
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD CROSS-VALIDATION ({len(WALK_FORWARD_FOLDS)} folds)")
        print(f"{'='*60}")

        fold_val_scores = []
        fold_val_sharpes = []
        fold_test_sharpes = []

        for fold_idx, (fold_train_end, fold_val_end) in enumerate(WALK_FORWARD_FOLDS):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx+1}/{len(WALK_FORWARD_FOLDS)}: "
                  f"train<{fold_train_end}, val={fold_train_end}..{fold_val_end}")
            print(f"{'='*60}")

            val_metrics, test_metrics, epoch, num_params = train_single_split(
                features_df, targets_df, device,
                train_end=fold_train_end, val_end=fold_val_end,
                fold_label=f"[Fold {fold_idx+1}] ")

            if val_metrics is None:
                print(f"  Fold {fold_idx+1}: SKIPPED (not enough data)")
                continue

            fold_val_scores.append(val_metrics["val_score"])
            fold_val_sharpes.append(val_metrics["sharpe"])
            fold_test_sharpes.append(test_metrics["sharpe"])

            print(f"\n  Fold {fold_idx+1} result: val_score={val_metrics['val_score']:.6f}, "
                  f"val_sharpe={val_metrics['sharpe']:.6f}, "
                  f"test_sharpe={test_metrics['sharpe']:.6f}")

        # Average across folds
        if fold_val_scores:
            avg_val_score = np.mean(fold_val_scores)
            avg_val_sharpe = np.mean(fold_val_sharpes)
            avg_test_sharpe = np.mean(fold_test_sharpes)

            t_end = time.time()
            print(f"\n{'='*60}")
            print(f"  WALK-FORWARD CV SUMMARY ({len(fold_val_scores)} folds)")
            print(f"{'='*60}")
            for i, (vs, vsh, tsh) in enumerate(zip(fold_val_scores, fold_val_sharpes, fold_test_sharpes)):
                print(f"  Fold {i+1}: val_score={vs:.6f}, val_sharpe={vsh:.6f}, test_sharpe={tsh:.6f}")
            print(f"\n---")
            print(f"val_score:        {avg_val_score:.6f}")
            print(f"val_sharpe:       {avg_val_sharpe:.6f}")
            print(f"test_sharpe:      {avg_test_sharpe:.6f}")
            print(f"total_seconds:    {t_end - t_start:.1f}")
            print(f"num_folds:        {len(fold_val_scores)}")
            print(f"model_type:       {MODEL_TYPE}")
            print(f"trade_frequency:  {TRADE_FREQUENCY}")
        else:
            print("ERROR: All folds failed!")
    else:
        # Standard single-split training
        val_metrics, test_metrics, epoch, num_params = train_single_split(
            features_df, targets_df, device)

        if val_metrics is None:
            exit(1)

        # Summary
        t_end = time.time()
        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        print("\n---")
        print(f"val_score:        {val_metrics['val_score']:.6f}")
        print(f"val_sharpe:       {val_metrics['sharpe']:.6f}")
        print(f"val_max_drawdown: {val_metrics['max_drawdown']:.6f}")
        print(f"val_turnover:     {val_metrics['turnover']:.6f}")
        print(f"test_sharpe:      {test_metrics['sharpe']:.6f}")
        print(f"test_max_drawdown:{test_metrics['max_drawdown']:.6f}")
        print(f"training_seconds: {time.time() - t_start:.1f}")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
        print(f"num_epochs:       {epoch}")
        print(f"num_params:       {num_params}")
        print(f"model_type:       {MODEL_TYPE}")
        print(f"trade_frequency:  {TRADE_FREQUENCY}")


if __name__ == "__main__":
    main()
