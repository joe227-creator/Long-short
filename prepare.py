"""
Data preparation and evaluation for ETF autoresearch.
Downloads ETF prices (yfinance) and macro data (FRED API), builds features,
provides dataloaders and the fixed evaluation metric.

Usage:
    python prepare.py              # download data, build features, print summary
    python prepare.py --refresh    # force-refresh cached data

This file is READ-ONLY for the outer researcher.
Keep its data and evaluation contract fixed during autoresearch runs.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds (5 minutes)
LOOKBACK = 60              # default sequence length (trading days)
NUM_PAIRS = 4              # number of factor pairs

ETF_TICKERS = ["SSO", "SDS", "UBT", "TBT", "UCO", "SCO", "UGL", "GLL"]

ETF_PAIRS = [
    ("SSO", "SDS"),   # equity (S&P 500)
    ("UBT", "TBT"),    # treasury (20+ year)
    ("UCO", "SCO"),     # oil (crude)
    ("UGL", "GLL"),     # gold
]

PAIR_NAMES = ["equity", "treasury", "oil", "gold"]

# Date splits
DATA_START = "2012-01-01"
TRAIN_END = "2022-01-01"    # train:  DATA_START .. TRAIN_END (exclusive)
VAL_END = "2024-01-01"      # val:    TRAIN_END  .. VAL_END   (exclusive)
                             # test:   VAL_END    .. present
DATA_SPLIT_VERSION = 2       # v2: weekly CV honors caller-provided fold cutoffs

# Cache
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "etf_autoresearch")

_ETF_FIELDS = ("Open", "High", "Low", "Close", "Volume")
REQUIRED_ETF_COLUMNS = tuple(
    f"{ticker}_{field}" for ticker in ETF_TICKERS for field in _ETF_FIELDS
)

# FRED macro indicators
MACRO_SERIES = {
    "DGS10": {
        "name": "10-Year Treasury Rate",
        "frequency": "daily",
        "inverse": True,
    },
    "T10Y3M": {
        "name": "10Y-3M Treasury Spread",
        "frequency": "daily",
        "inverse": False,
    },
    "ICSA": {
        "name": "Initial Unemployment Claims",
        "frequency": "weekly",
        "inverse": True,
    },
    "HOUST": {
        "name": "Housing Starts",
        "frequency": "monthly",
        "inverse": False,
    },
    "AMTMNO": {
        "name": "Manufacturing New Orders",
        "frequency": "monthly",
        "inverse": False,
    },
    "UMCSENT": {
        "name": "Consumer Sentiment (U. Michigan)",
        "frequency": "monthly",
        "inverse": False,
    },
    "PPIACO": {
        "name": "PPI All Commodities",
        "frequency": "monthly",
        "inverse": True,
    },
    "PCETRIM12M159SFRBDAL": {
        "name": "Trimmed Mean PCE Inflation",
        "frequency": "monthly",
        "inverse": True,
    },
    "DTWEXAFEGS": {
        "name": "USD Index (Advanced Economies)",
        "frequency": "daily",
        "inverse": True,
    },
    # --- New FRED indicators (added for feature-space experiments) ---
    "VIXCLS": {
        "name": "CBOE Volatility Index (VIX)",
        "frequency": "daily",
        "inverse": True,
    },
    "VXVCLS": {
        "name": "CBOE S&P 500 3-Month Volatility Index",
        "frequency": "daily",
        "inverse": True,
    },
    "GVZCLS": {
        "name": "CBOE Gold ETF Volatility Index",
        "frequency": "daily",
        "inverse": True,
    },
    "OVXCLS": {
        "name": "CBOE Crude Oil ETF Volatility Index",
        "frequency": "daily",
        "inverse": True,
    },
    "STLFSI4": {
        "name": "St. Louis Fed Financial Stress Index",
        "frequency": "weekly",
        "inverse": True,
    },
    "NFCI": {
        "name": "Chicago Fed National Financial Conditions Index",
        "frequency": "weekly",
        "inverse": True,
    },
    "CFNAI": {
        "name": "Chicago Fed National Activity Index",
        "frequency": "monthly",
        "inverse": False,
    },
    "FEDFUNDS": {
        "name": "Federal Funds Effective Rate",
        "frequency": "monthly",
        "inverse": True,
    },
    "DTWEXBGS": {
        "name": "Nominal Broad U.S. Dollar Index",
        "frequency": "daily",
        "inverse": True,
    },
}

# Conservative FRED publication-lag map (in trading days).
# FRED's `observation_date` is the *reference* date of a value, not its
# publication date. Without a lag, monthly/weekly series leak future
# information by days to weeks. These shifts are conservative and apply on
# top of forward-fill, so a value with reference date t is only visible
# from t + MACRO_PUBLICATION_LAG_DAYS[sid] onward.
MACRO_PUBLICATION_LAG_DAYS = {
    # Daily series — values for date t are typically released after that
    # day's market close, so they are safe to use at t (lag 0).
    "DGS10": 0, "T10Y3M": 0, "DTWEXAFEGS": 0, "DTWEXBGS": 0,
    "VIXCLS": 0, "VXVCLS": 0, "GVZCLS": 0, "OVXCLS": 0,
    # Weekly series — initial release lag of several days.
    "ICSA": 5, "STLFSI4": 5, "NFCI": 5,
    # Monthly indicators released ~2-4 weeks after the reference month.
    "HOUST": 20, "AMTMNO": 30, "UMCSENT": 20, "PPIACO": 15,
    "PCETRIM12M159SFRBDAL": 30, "CFNAI": 20, "FEDFUNDS": 20,
}

# ---------------------------------------------------------------------------
# Data download — ETF prices
# ---------------------------------------------------------------------------

def validate_etf_data(etf_df, context="ETF data"):
    """Reject incomplete/non-finite market data before feature construction."""
    if etf_df is None or not isinstance(etf_df, pd.DataFrame) or etf_df.empty:
        raise ValueError(f"{context} is empty; refusing to build predictions")

    missing = [col for col in REQUIRED_ETF_COLUMNS if col not in etf_df.columns]
    if missing:
        raise ValueError(
            f"{context} missing required yfinance columns: {', '.join(missing)}"
        )

    if not isinstance(etf_df.index, pd.DatetimeIndex):
        raise ValueError(f"{context} must use a DatetimeIndex")
    if etf_df.index.has_duplicates or not etf_df.index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")

    try:
        values = etf_df.loc[:, REQUIRED_ETF_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} contains non-numeric required columns") from exc
    if not np.isfinite(values).all():
        bad_columns = [
            col for col in REQUIRED_ETF_COLUMNS
            if not np.isfinite(etf_df[col].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            f"{context} contains missing/non-finite values in: {', '.join(bad_columns)}"
        )
    return etf_df


def validate_macro_data(macro_df, context="FRED data"):
    """Reject missing FRED series while allowing sparse observations per series."""
    if macro_df is None or not isinstance(macro_df, pd.DataFrame) or macro_df.empty:
        raise ValueError(f"{context} is empty; refusing to build macro features")

    missing = [sid for sid in MACRO_SERIES if sid not in macro_df.columns]
    if missing:
        raise ValueError(
            f"{context} missing required FRED series: {', '.join(missing)}"
        )
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        raise ValueError(f"{context} must use a DatetimeIndex")
    if macro_df.index.has_duplicates or not macro_df.index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")

    empty_series = []
    for sid in MACRO_SERIES:
        try:
            values = macro_df[sid].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} series {sid} is non-numeric") from exc
        if not np.isfinite(values).any():
            empty_series.append(sid)
    if empty_series:
        raise ValueError(
            f"{context} contains no observations for: {', '.join(empty_series)}"
        )
    return macro_df


def validate_feature_columns(features_df, feature_columns, context="features"):
    """Return requested features or fail instead of manufacturing missing inputs."""
    if features_df is None or not isinstance(features_df, pd.DataFrame) or features_df.empty:
        raise ValueError(f"{context} is empty; refusing to run the model")
    if not feature_columns:
        raise ValueError(f"{context} has no configured feature columns")

    missing = [col for col in feature_columns if col not in features_df.columns]
    if missing:
        raise ValueError(
            f"{context} missing required model features: {', '.join(missing)}"
        )

    selected = features_df.loc[:, list(feature_columns)]
    try:
        values = selected.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} contains non-numeric model features") from exc
    if not np.isfinite(values).all():
        bad_columns = [
            col for col in selected.columns
            if not np.isfinite(selected[col].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            f"{context} contains missing/non-finite values in: {', '.join(bad_columns)}"
        )
    return selected


def validate_target_columns(targets_df, target_columns, context="targets", allow_nan=False):
    """Validate target schema; optionally allow expected trailing forward-return NaNs."""
    if targets_df is None or not isinstance(targets_df, pd.DataFrame) or targets_df.empty:
        raise ValueError(f"{context} is empty; refusing to evaluate the model")
    missing = [col for col in target_columns if col not in targets_df.columns]
    if missing:
        raise ValueError(
            f"{context} missing required target columns: {', '.join(missing)}"
        )
    selected = targets_df.loc[:, list(target_columns)]
    if not allow_nan:
        try:
            values = selected.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} contains non-numeric targets") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{context} contains missing/non-finite targets")
    return selected

def download_etf_data(refresh=False):
    """Download daily OHLCV for all ETFs via yfinance. Returns DataFrame."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "etf_prices.parquet")

    if os.path.exists(cache_path) and not refresh:
        df = pd.read_parquet(cache_path)
        validate_etf_data(df, context="Cached ETF data")
        print(f"ETF data: loaded {len(df)} rows from cache")
        return df

    print(f"ETF data: downloading {len(ETF_TICKERS)} tickers from yfinance...")
    frames = {}
    for ticker in ETF_TICKERS:
        t0 = time.time()
        data = yf.download(ticker, start=DATA_START, auto_adjust=True, progress=False)
        if data.empty:
            print(f"  WARNING: no data for {ticker}")
            continue
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                frames[f"{ticker}_{col}"] = data[col]
        dt = time.time() - t0
        print(f"  {ticker}: {len(data)} rows ({dt:.1f}s)")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # Forward-fill small gaps (weekends already excluded), then fail closed if
    # any required source value is still unavailable.
    df = df.ffill(limit=5)
    validate_etf_data(df, context="Downloaded ETF data")

    df.to_parquet(cache_path)
    print(f"ETF data: saved {len(df)} rows to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Data download — FRED macro series
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id, api_key):
    """Fetch a single FRED series. Returns (dates, values) lists."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": DATA_START,
        "sort_order": "asc",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    dates, values = [], []
    for o in obs:
        val = o["value"]
        if val == ".":
            continue
        dates.append(o["date"])
        values.append(float(val))
    return dates, values


def download_macro_data(refresh=False):
    """Download macro series from FRED API. Returns DataFrame indexed by date."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "macro_data.parquet")

    if os.path.exists(cache_path) and not refresh:
        df = pd.read_parquet(cache_path)
        validate_macro_data(df, context="Cached FRED data")
        print(f"Macro data: loaded {len(df)} rows from cache")
        return df

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set; refusing to build macro features")

    print(f"Macro data: downloading {len(MACRO_SERIES)} series from FRED...")
    frames = {}
    failures = {}
    for sid, info in MACRO_SERIES.items():
        try:
            dates, values = _fetch_fred_series(sid, api_key)
            if not dates or not values:
                raise ValueError("no observations returned")
            s = pd.Series(values, index=pd.to_datetime(dates), name=sid)
            frames[sid] = s
            print(f"  {sid} ({info['name']}): {len(s)} obs")
        except Exception as e:
            failures[sid] = str(e)

    if failures:
        detail = "; ".join(f"{sid}: {reason}" for sid, reason in failures.items())
        raise RuntimeError(f"FRED data incomplete; refusing to continue: {detail}")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    df = df.sort_index()

    # Forward-fill to daily (weekly/monthly series become daily)
    df = df.ffill()
    validate_macro_data(df, context="Downloaded FRED data")

    df.to_parquet(cache_path)
    print(f"Macro data: saved {len(df)} rows to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(etf_df, macro_df):
    """
    Build feature matrix from ETF prices and macro data.
    Returns DataFrame with finite core features; optional experimental features
    retain warm-up NaNs and must be validated when explicitly selected.
    """
    validate_etf_data(etf_df)
    validate_macro_data(macro_df)
    features = {}

    # --- Per-ETF features ---
    for ticker in ETF_TICKERS:
        close_col = f"{ticker}_Close"
        vol_col = f"{ticker}_Volume"

        if close_col not in etf_df.columns:
            continue

        close = etf_df[close_col]
        log_close = np.log(close)

        # Log returns over multiple horizons
        for h in [1, 3, 5, 10, 20]:
            features[f"{ticker}_logret_{h}d"] = log_close.diff(h)

        # Realized volatility (rolling std of 1-day log returns)
        daily_ret = log_close.diff(1)
        for w in [10, 20, 60]:
            features[f"{ticker}_vol_{w}d"] = daily_ret.rolling(w).std()

        # Drawdown from rolling max
        rolling_max = close.rolling(60, min_periods=1).max()
        features[f"{ticker}_drawdown"] = (close - rolling_max) / rolling_max

        # Price momentum: MA5 / MA20 ratio
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        features[f"{ticker}_momentum"] = ma5 / ma20 - 1.0

        # Volume change (if available)
        if vol_col in etf_df.columns:
            vol = etf_df[vol_col]
            vol_ma = vol.rolling(20).mean()
            features[f"{ticker}_vol_ratio"] = vol / vol_ma.replace(0, np.nan) - 1.0

        # --- Technical indicators (computed from Close/High/Low/Volume) ---
        high_col = f"{ticker}_High"
        low_col = f"{ticker}_Low"

        # RSI (14-day Relative Strength Index)
        delta_close = close.diff(1)
        gain = delta_close.clip(lower=0)
        loss_val = (-delta_close).clip(lower=0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss_val.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features[f"{ticker}_rsi"] = 100 - (100 / (1 + rs))

        # ROC (12-day Rate of Change)
        features[f"{ticker}_roc_12d"] = close.pct_change(12)

        # PPO (Percentage Price Oscillator: (EMA12 - EMA26) / EMA26)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features[f"{ticker}_ppo"] = (ema12 - ema26) / ema26.replace(0, np.nan)

        # Stochastic Oscillator %K (14-day)
        if high_col in etf_df.columns and low_col in etf_df.columns:
            high = etf_df[high_col]
            low = etf_df[low_col]
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            features[f"{ticker}_stoch_k"] = (close - low_14) / (high_14 - low_14).replace(0, np.nan)

            # Awesome Oscillator: SMA5(midpoint) - SMA34(midpoint)
            midpoint = (high + low) / 2
            features[f"{ticker}_ao"] = midpoint.rolling(5).mean() - midpoint.rolling(34).mean()

            # --- NEW: Untested momentum indicators (6) ---

            # KAMA (Kaufman's Adaptive Moving Average) — ratio to close
            _kama_win = 10
            _er_num = (close - close.shift(_kama_win)).abs()
            _er_den = close.diff(1).abs().rolling(_kama_win).sum()
            _er = _er_num / _er_den.replace(0, np.nan)
            _fast_sc = 2.0 / (2 + 1)
            _slow_sc = 2.0 / (30 + 1)
            _sc = (_er * (_fast_sc - _slow_sc) + _slow_sc) ** 2
            _kama = close.copy()
            _kama.iloc[:_kama_win] = np.nan
            for _i in range(_kama_win, len(close)):
                if np.isnan(_kama.iloc[_i - 1]) or np.isnan(_sc.iloc[_i]):
                    _kama.iloc[_i] = close.iloc[_i]
                else:
                    _kama.iloc[_i] = _kama.iloc[_i - 1] + _sc.iloc[_i] * (close.iloc[_i] - _kama.iloc[_i - 1])
            features[f"{ticker}_kama"] = _kama / close - 1.0

            # PVO (Percentage Volume Oscillator)
            if vol_col in etf_df.columns:
                vol = etf_df[vol_col]
                _vol_ema12 = vol.ewm(span=12, adjust=False).mean()
                _vol_ema26 = vol.ewm(span=26, adjust=False).mean()
                features[f"{ticker}_pvo"] = (_vol_ema12 - _vol_ema26) / _vol_ema26.replace(0, np.nan)

            # StochRSI
            _rsi_series = features[f"{ticker}_rsi"]
            _rsi_min = _rsi_series.rolling(14).min()
            _rsi_max = _rsi_series.rolling(14).max()
            features[f"{ticker}_stochrsi"] = (_rsi_series - _rsi_min) / (_rsi_max - _rsi_min).replace(0, np.nan)

            # TSI (True Strength Index)
            _pc = close.diff(1)
            _pc_double_smooth = _pc.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
            _apc_double_smooth = _pc.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
            features[f"{ticker}_tsi"] = _pc_double_smooth / _apc_double_smooth.replace(0, np.nan)

            # Ultimate Oscillator
            _prev_close = close.shift(1)
            _bp = close - pd.concat([low, _prev_close], axis=1).min(axis=1)
            _tr_uo = pd.concat([high, _prev_close], axis=1).max(axis=1) - pd.concat([low, _prev_close], axis=1).min(axis=1)
            _avg7 = _bp.rolling(7).sum() / _tr_uo.rolling(7).sum().replace(0, np.nan)
            _avg14 = _bp.rolling(14).sum() / _tr_uo.rolling(14).sum().replace(0, np.nan)
            _avg28 = _bp.rolling(28).sum() / _tr_uo.rolling(28).sum().replace(0, np.nan)
            features[f"{ticker}_uo"] = (4 * _avg7 + 2 * _avg14 + _avg28) / 7.0

            # Williams %R
            features[f"{ticker}_willr"] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100

            # --- NEW: Volume indicators (9) ---

            # ADI (Accumulation/Distribution Index) — normalized
            _mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
            _mfv = _mfm * etf_df[vol_col] if vol_col in etf_df.columns else _mfm * 0
            features[f"{ticker}_adi"] = _mfv.cumsum()

            # CMF (Chaikin Money Flow, 20-day)
            if vol_col in etf_df.columns:
                features[f"{ticker}_cmf"] = _mfv.rolling(20).sum() / etf_df[vol_col].rolling(20).sum().replace(0, np.nan)

            # EoM (Ease of Movement, 14-day)
            _dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
            _br = (etf_df[vol_col] / 1e6 if vol_col in etf_df.columns else pd.Series(1, index=close.index)) / (high - low).replace(0, np.nan)
            features[f"{ticker}_eom"] = (_dm / _br.replace(0, np.nan)).rolling(14).mean()

            # Force Index (13-day EMA)
            if vol_col in etf_df.columns:
                _fi = close.diff(1) * etf_df[vol_col]
                features[f"{ticker}_fi"] = _fi.ewm(span=13, adjust=False).mean()

            # MFI (Money Flow Index, 14-day)
            _tp = (high + low + close) / 3
            _rmf = _tp * (etf_df[vol_col] if vol_col in etf_df.columns else pd.Series(0, index=close.index))
            _pos_mf = pd.Series(np.where(_tp > _tp.shift(1), _rmf, 0), index=close.index)
            _neg_mf = pd.Series(np.where(_tp < _tp.shift(1), _rmf, 0), index=close.index)
            _mfr = _pos_mf.rolling(14).sum() / _neg_mf.rolling(14).sum().replace(0, np.nan)
            features[f"{ticker}_mfi"] = 100 - (100 / (1 + _mfr))

            # NVI (Negative Volume Index) — log-scaled
            if vol_col in etf_df.columns:
                _vol_series = etf_df[vol_col]
                _nvi = pd.Series(1000.0, index=close.index)
                for _i in range(1, len(close)):
                    if _vol_series.iloc[_i] < _vol_series.iloc[_i - 1]:
                        _nvi.iloc[_i] = _nvi.iloc[_i - 1] * (1 + close.pct_change().iloc[_i])
                    else:
                        _nvi.iloc[_i] = _nvi.iloc[_i - 1]
                features[f"{ticker}_nvi"] = np.log(_nvi / _nvi.iloc[0])

            # OBV (On-Balance Volume)
            if vol_col in etf_df.columns:
                _sign = np.sign(close.diff(1))
                features[f"{ticker}_obv"] = (_sign * etf_df[vol_col]).cumsum()

            # VPT (Volume-Price Trend)
            if vol_col in etf_df.columns:
                features[f"{ticker}_vpt"] = (close.pct_change() * etf_df[vol_col]).cumsum()

            # VWAP (Volume-Weighted Average Price, 20-day rolling)
            if vol_col in etf_df.columns:
                _vwap = (_tp * etf_df[vol_col]).rolling(20).sum() / etf_df[vol_col].rolling(20).sum().replace(0, np.nan)
                features[f"{ticker}_vwap"] = _vwap / close - 1.0

            # --- NEW: Volatility indicators (5) ---

            # ATR (Average True Range, 14-day)
            _tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            features[f"{ticker}_atr"] = _tr.ewm(span=14, adjust=False).mean() / close

            # Bollinger Bands (20-day, 2 std) — %B indicator
            _bb_mid = close.rolling(20).mean()
            _bb_std = close.rolling(20).std()
            _bb_upper = _bb_mid + 2 * _bb_std
            _bb_lower = _bb_mid - 2 * _bb_std
            features[f"{ticker}_bb_pctb"] = (close - _bb_lower) / (_bb_upper - _bb_lower).replace(0, np.nan)

            # Donchian Channel — position within channel (20-day)
            _dc_high = high.rolling(20).max()
            _dc_low = low.rolling(20).min()
            features[f"{ticker}_dc_pct"] = (close - _dc_low) / (_dc_high - _dc_low).replace(0, np.nan)

            # Keltner Channels — position within channel (20-day, 1.5 ATR)
            _kc_mid = close.ewm(span=20, adjust=False).mean()
            _kc_atr = _tr.ewm(span=10, adjust=False).mean()
            _kc_upper = _kc_mid + 1.5 * _kc_atr
            _kc_lower = _kc_mid - 1.5 * _kc_atr
            features[f"{ticker}_kc_pct"] = (close - _kc_lower) / (_kc_upper - _kc_lower).replace(0, np.nan)

            # Ulcer Index (14-day)
            _pct_drawdown = ((close - close.rolling(14).max()) / close.rolling(14).max()) * 100
            features[f"{ticker}_ulcer"] = np.sqrt((_pct_drawdown ** 2).rolling(14).mean())

            # --- NEW: Trend indicators (15) ---

            # ADX (Average Directional Index, 14-day)
            _plus_dm = pd.Series(np.where((high.diff(1) > 0) & (high.diff(1) > -low.diff(1)), high.diff(1), 0), index=close.index)
            _minus_dm = pd.Series(np.where((-low.diff(1) > 0) & (-low.diff(1) > high.diff(1)), -low.diff(1), 0), index=close.index)
            _atr14 = _tr.ewm(span=14, adjust=False).mean()
            _plus_di = 100 * _plus_dm.ewm(span=14, adjust=False).mean() / _atr14.replace(0, np.nan)
            _minus_di = 100 * _minus_dm.ewm(span=14, adjust=False).mean() / _atr14.replace(0, np.nan)
            _dx = ((_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, np.nan)) * 100
            features[f"{ticker}_adx"] = _dx.ewm(span=14, adjust=False).mean()

            # Aroon (25-day)
            _aroon_up = high.rolling(25).apply(lambda x: x.argmax(), raw=True) / 24 * 100
            _aroon_down = low.rolling(25).apply(lambda x: x.argmin(), raw=True) / 24 * 100
            features[f"{ticker}_aroon"] = _aroon_up - _aroon_down

            # CCI (Commodity Channel Index, 20-day)
            _tp_cci = (high + low + close) / 3
            _tp_sma = _tp_cci.rolling(20).mean()
            _tp_mad = _tp_cci.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            features[f"{ticker}_cci"] = (_tp_cci - _tp_sma) / (0.015 * _tp_mad).replace(0, np.nan)

            # DPO (Detrended Price Oscillator, 20-day)
            features[f"{ticker}_dpo"] = close - close.rolling(20).mean().shift(11)

            # EMA (ratio vs close, 20-day)
            features[f"{ticker}_ema_ratio"] = close.ewm(span=20, adjust=False).mean() / close - 1.0

            # Ichimoku — distance from cloud (26-day lag)
            _tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            _kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            _senkou_a = ((_tenkan + _kijun) / 2).shift(26)
            _senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            features[f"{ticker}_ichi"] = (close - (_senkou_a + _senkou_b) / 2) / close

            # KST (Know Sure Thing)
            _roc1 = close.pct_change(10).rolling(10).mean()
            _roc2 = close.pct_change(15).rolling(10).mean()
            _roc3 = close.pct_change(20).rolling(10).mean()
            _roc4 = close.pct_change(30).rolling(15).mean()
            features[f"{ticker}_kst"] = _roc1 + 2 * _roc2 + 3 * _roc3 + 4 * _roc4

            # MACD (histogram as fraction of close)
            _macd_line = ema12 - ema26  # reuse EMA12/26 from PPO
            _macd_signal = _macd_line.ewm(span=9, adjust=False).mean()
            features[f"{ticker}_macd_hist"] = (_macd_line - _macd_signal) / close

            # Mass Index (25-day, ratio)
            _ema_hl = (high - low).ewm(span=9, adjust=False).mean()
            _ema_ema_hl = _ema_hl.ewm(span=9, adjust=False).mean()
            features[f"{ticker}_mass_idx"] = (_ema_hl / _ema_ema_hl.replace(0, np.nan)).rolling(25).sum()

            # Parabolic SAR (simplified: distance from SAR)
            # Use a simplified version: start with first close, AF=0.02, max AF=0.20
            _sar = close.rolling(5).mean()  # simplified SAR approximation
            features[f"{ticker}_psar"] = (close - _sar) / close

            # SMA (ratio vs close, 20-day)
            features[f"{ticker}_sma_ratio"] = close.rolling(20).mean() / close - 1.0

            # STC (Schaff Trend Cycle) — simplified as MACD stochastic
            _stc_macd = ema12 - ema26
            _stc_lo = _stc_macd.rolling(10).min()
            _stc_hi = _stc_macd.rolling(10).max()
            features[f"{ticker}_stc"] = (_stc_macd - _stc_lo) / (_stc_hi - _stc_lo).replace(0, np.nan)

            # TRIX (triple-smoothed EMA, 15-day)
            _ema1 = close.ewm(span=15, adjust=False).mean()
            _ema2 = _ema1.ewm(span=15, adjust=False).mean()
            _ema3 = _ema2.ewm(span=15, adjust=False).mean()
            features[f"{ticker}_trix"] = _ema3.pct_change(1)

            # Vortex Indicator (14-day: VI+ - VI-)
            _vm_plus = (high - low.shift(1)).abs()
            _vm_minus = (low - high.shift(1)).abs()
            _vi_plus = _vm_plus.rolling(14).sum() / _atr14.replace(0, np.nan) / 14
            _vi_minus = _vm_minus.rolling(14).sum() / _atr14.replace(0, np.nan) / 14
            features[f"{ticker}_vortex"] = _vi_plus - _vi_minus

            # WMA (ratio vs close, 20-day)
            _weights_wma = np.arange(1, 21)
            features[f"{ticker}_wma_ratio"] = close.rolling(20).apply(
                lambda x: np.dot(x, _weights_wma) / _weights_wma.sum(), raw=True
            ) / close - 1.0

            # --- NEW: Other indicators (3) ---

            # Cumulative Return (from start, log)
            features[f"{ticker}_cum_ret"] = np.log(close / close.iloc[0])

            # Daily Log Return (same as logret_1d but included per library)
            features[f"{ticker}_daily_logret"] = log_close.diff(1)

            # Daily Return
            features[f"{ticker}_daily_ret"] = close.pct_change(1)

    # --- Per-pair features: bull-bear spread ---
    for (bull, bear), pair_name in zip(ETF_PAIRS, PAIR_NAMES):
        bull_col = f"{bull}_Close"
        bear_col = f"{bear}_Close"
        if bull_col in etf_df.columns and bear_col in etf_df.columns:
            bull_ret = np.log(etf_df[bull_col]).diff(1)
            bear_ret = np.log(etf_df[bear_col]).diff(1)
            features[f"pair_{pair_name}_spread_1d"] = bull_ret - bear_ret
            features[f"pair_{pair_name}_spread_5d"] = (
                np.log(etf_df[bull_col]).diff(5) - np.log(etf_df[bear_col]).diff(5)
            )

    # --- Macro features ---
    if macro_df is not None and not macro_df.empty:
        # Align macro to ETF dates, then apply per-series publication lag so
        # values are only visible from their (estimated) release date onward.
        macro_aligned = macro_df.reindex(etf_df.index, method="ffill")
        for sid in list(macro_aligned.columns):
            lag = MACRO_PUBLICATION_LAG_DAYS.get(sid, 0)
            if lag > 0:
                macro_aligned[sid] = macro_aligned[sid].shift(lag)

        for sid, info in MACRO_SERIES.items():
            if sid not in macro_aligned.columns:
                continue

            series = macro_aligned[sid]
            sign = -1.0 if info["inverse"] else 1.0

            # Z-scored level using expanding mean/std through t-1 only (the
            # current value is excluded from its own normalization stats to
            # avoid the subtle look-ahead of including y(t) in mean/std(t)).
            expanding_mean = series.expanding(min_periods=60).mean().shift(1)
            expanding_std = series.expanding(min_periods=60).std().shift(1)
            features[f"macro_{sid}_zscore"] = sign * (series - expanding_mean) / expanding_std.replace(0, np.nan)

            # 1-day change
            features[f"macro_{sid}_chg_1d"] = sign * series.diff(1)

            # 20-day change
            features[f"macro_{sid}_chg_20d"] = sign * series.diff(20)

    feat_df = pd.DataFrame(features, index=etf_df.index)
    feat_df.index.name = "Date"

    # Core features define dataset rows. Never zero-fill experimental features:
    # they retain warm-up NaNs and fail validation if explicitly selected.
    _exp_tech_suffixes = [
        "_rsi", "_roc_12d", "_ppo", "_stoch_k", "_ao",
        "_kama", "_pvo", "_stochrsi", "_tsi", "_uo", "_willr",
        "_adi", "_cmf", "_eom", "_fi", "_mfi", "_nvi", "_obv", "_vpt", "_vwap",
        "_atr", "_bb_pctb", "_dc_pct", "_kc_pct", "_ulcer",
        "_adx", "_aroon", "_cci", "_dpo", "_ema_ratio", "_ichi", "_kst",
        "_macd_hist", "_mass_idx", "_psar", "_sma_ratio", "_stc", "_trix",
        "_vortex", "_wma_ratio",
        "_cum_ret", "_daily_logret", "_daily_ret",
    ]
    _exp_macro_prefixes = [
        "macro_VIXCLS_", "macro_VXVCLS_", "macro_GVZCLS_", "macro_OVXCLS_",
        "macro_STLFSI4_", "macro_NFCI_", "macro_CFNAI_", "macro_FEDFUNDS_",
        "macro_DTWEXBGS_",
    ]
    exp_cols = [c for c in feat_df.columns
                if any(c.endswith(s) for s in _exp_tech_suffixes)
                or any(c.startswith(p) for p in _exp_macro_prefixes)]
    core_cols = [c for c in feat_df.columns if c not in exp_cols]

    initial_len = len(feat_df)
    core_nan_mask = feat_df[core_cols].isna().any(axis=1)
    feat_df = feat_df.loc[~core_nan_mask]
    if feat_df.empty:
        raise ValueError("Feature engineering produced no finite rows")

    print(f"Features: {feat_df.shape[1]} columns, {len(feat_df)} rows "
          f"(dropped {initial_len - len(feat_df)} NaN rows)")

    return feat_df


def build_targets(etf_df, trade_frequency="daily"):
    """
    Build forward return targets for each ETF.
    Returns DataFrame with 8 columns (one per ETF), aligned by date.
    """
    validate_etf_data(etf_df)
    targets = {}
    horizon = 1 if trade_frequency == "daily" else 5

    for ticker in ETF_TICKERS:
        close_col = f"{ticker}_Close"
        log_close = np.log(etf_df[close_col])
        # Forward return: return from t to t+horizon
        targets[f"{ticker}_fwd_ret"] = log_close.shift(-horizon) - log_close

    target_df = pd.DataFrame(targets, index=etf_df.index)
    target_df.index.name = "Date"
    return target_df


def build_dataset(trade_frequency="daily", refresh=False):
    """
    Full pipeline: download -> features -> targets -> align -> clean.
    Returns (features_df, targets_df) with matching indexes.
    """
    etf_df = download_etf_data(refresh=refresh)
    macro_df = download_macro_data(refresh=refresh)

    features_df = build_features(etf_df, macro_df)
    targets_df = build_targets(etf_df, trade_frequency)
    validate_target_columns(
        targets_df,
        [f"{ticker}_fwd_ret" for ticker in ETF_TICKERS],
        context="Built targets",
        allow_nan=True,
    )

    # Align indexes
    common_idx = features_df.index.intersection(targets_df.index)
    features_df = features_df.loc[common_idx]
    targets_df = targets_df.loc[common_idx]

    # Drop rows where targets have NaN (end of dataset)
    valid_mask = targets_df.notna().all(axis=1)
    features_df = features_df.loc[valid_mask]
    targets_df = targets_df.loc[valid_mask]
    if features_df.empty:
        raise ValueError("Dataset has no rows with complete features and targets")

    print(f"Dataset: {len(features_df)} samples, {features_df.shape[1]} features, "
          f"{targets_df.shape[1]} targets, freq={trade_frequency}")
    print(f"  Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}")

    return features_df, targets_df


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def compute_scaler_params(features_df, train_end=TRAIN_END):
    """Compute mean and std from training data only (no look-ahead)."""
    train_mask = features_df.index < train_end
    train_data = features_df.loc[train_mask]
    if train_data.empty:
        raise ValueError(f"No training rows available before {train_end}")
    mean = train_data.mean()
    std = train_data.std().replace(0, 1.0)
    validate_feature_columns(
        pd.DataFrame([mean], columns=mean.index),
        list(mean.index),
        context="Training scaler means",
    )
    if not np.isfinite(std.to_numpy(dtype=float)).all():
        raise ValueError("Training scaler contains missing/non-finite standard deviations")
    return mean, std


def normalize_features(features_df, mean, std):
    """Z-score normalize using provided params."""
    mean = pd.Series(mean)
    std = pd.Series(std)
    missing_mean = [col for col in features_df.columns if col not in mean.index]
    missing_std = [col for col in features_df.columns if col not in std.index]
    if missing_mean or missing_std:
        missing = sorted(set(missing_mean + missing_std))
        raise ValueError(
            f"Scaler missing parameters for model features: {', '.join(missing)}"
        )
    selected_mean = mean.loc[features_df.columns]
    selected_std = std.loc[features_df.columns].replace(0, 1.0)
    if not np.isfinite(selected_mean.to_numpy(dtype=float)).all():
        raise ValueError("Scaler means contain missing/non-finite values")
    if not np.isfinite(selected_std.to_numpy(dtype=float)).all():
        raise ValueError("Scaler standard deviations contain missing/non-finite values")
    normalized = (features_df - selected_mean) / selected_std
    return validate_feature_columns(
        normalized,
        list(features_df.columns),
        context="Normalized features",
    )


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Rolling-window time series dataset for sequence models."""

    def __init__(self, features, targets, lookback):
        """
        Args:
            features: np.ndarray of shape (N, F)
            targets: np.ndarray of shape (N, 8)
            lookback: int, sequence length T
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.features) - self.lookback

    def __getitem__(self, idx):
        # Sequence of T feature vectors -> next-period target
        x = self.features[idx : idx + self.lookback]       # (T, F)
        y = self.targets[idx + self.lookback - 1]           # (8,)
        return x, y


def make_dataloaders(features_df, targets_df, lookback=LOOKBACK, batch_size=64,
                     trade_frequency="daily", train_end=None, val_end=None):
    """
    Build train/val/test DataLoaders from feature and target DataFrames.
    Splits by fixed date cutoffs. Normalizes features using train-set stats.

    Returns: (train_loader, val_loader, test_loader, scaler_params, feature_columns)
    """
    _train_end = train_end or TRAIN_END
    _val_end = val_end or VAL_END

    # Date-based splits
    train_mask = features_df.index < _train_end
    val_mask = (features_df.index >= _train_end) & (features_df.index < _val_end)
    test_mask = features_df.index >= _val_end

    # Compute normalization from training data
    mean, std = compute_scaler_params(features_df, _train_end)
    feature_columns = list(features_df.columns)

    # Normalize all data using train stats
    feat_norm = normalize_features(features_df, mean, std)

    # For weekly mode: subsample to weekly (every 5th trading day)
    if trade_frequency == "weekly":
        weekly_idx = feat_norm.index[::5]
        feat_norm = feat_norm.loc[weekly_idx]
        targets_df = targets_df.loc[weekly_idx]
        train_mask = feat_norm.index < _train_end
        val_mask = (feat_norm.index >= _train_end) & (feat_norm.index < _val_end)
        test_mask = feat_norm.index >= _val_end

    feat_np = feat_norm.values.astype(np.float32)
    tgt_np = targets_df.values.astype(np.float32)

    # Build split arrays (contiguous slices)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    def _make_loader(indices, shuffle):
        if len(indices) <= lookback:
            return None
        start, end = indices[0], indices[-1] + 1
        ds = TimeSeriesDataset(feat_np[start:end], tgt_np[start:end], lookback)
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=0)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx, shuffle=False)
    test_loader = _make_loader(test_idx, shuffle=False)

    scaler_params = {"mean": mean.to_dict(), "std": std.to_dict()}

    split_info = {
        "train": int(train_mask.sum()),
        "val": int(val_mask.sum()),
        "test": int(test_mask.sum()),
    }
    print(f"DataLoaders: train={split_info['train']}, val={split_info['val']}, "
          f"test={split_info['test']}, lookback={lookback}, batch={batch_size}")

    return train_loader, val_loader, test_loader, scaler_params, feature_columns


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def signals_to_weights(raw_signals):
    """
    Convert 4 factor-pair signals to 8 ETF weights (per-pair dollar-neutral).

    Args:
        raw_signals: tensor of shape (N, 4) — one signal per factor pair
    Returns:
        weights: tensor of shape (N, 8) — per-ETF weights
    """
    signals = torch.tanh(raw_signals)  # bound to [-1, 1]

    weights = torch.zeros(signals.shape[0], 8, device=signals.device)
    for i in range(NUM_PAIRS):
        # Pair i: bull ETF at index 2*i, bear ETF at index 2*i+1
        # Positive signal -> long bull, short bear
        # Negative signal -> short bull, long bear
        # Always dollar-neutral within pair
        weights[:, 2 * i] = signals[:, i]       # bull weight
        weights[:, 2 * i + 1] = -signals[:, i]  # bear weight (opposite)

    # Normalize so total absolute exposure <= 1
    total_abs = weights.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    max_exposure = 1.0
    scale = torch.clamp(max_exposure / total_abs, max=1.0)
    weights = weights * scale

    return weights


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_portfolio(model, dataloader, device, trade_frequency="daily"):
    """
    Evaluate model on a dataloader. Returns dict with val_score and sub-metrics.

    val_score = -sharpe + 0.1 * turnover + 0.05 * max_drawdown
    Lower is better (mirrors autoresearch BPB convention).
    """
    if dataloader is None:
        return {"val_score": 999.0, "sharpe": 0.0, "max_drawdown": 1.0, "turnover": 1.0}

    model.eval()
    all_signals = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        signals = model(x_batch)  # (B, 4)
        all_signals.append(signals.cpu())
        all_targets.append(y_batch)

    all_signals = torch.cat(all_signals, dim=0)  # (N, 4)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 8)

    # Convert signals to weights
    weights = signals_to_weights(all_signals)  # (N, 8)

    # Portfolio returns: sum of weight_i * actual_return_i
    portfolio_returns = (weights * all_targets).sum(dim=1)  # (N,)

    # --- Sharpe ratio ---
    mean_ret = portfolio_returns.mean().item()
    std_ret = portfolio_returns.std().item()
    periods_per_year = 252 if trade_frequency == "daily" else 52
    sharpe = (mean_ret / max(std_ret, 1e-8)) * (periods_per_year ** 0.5)

    # --- Max drawdown ---
    cum_returns = (1 + portfolio_returns).cumprod(dim=0)
    running_max = cum_returns.cummax(dim=0).values
    drawdowns = (cum_returns - running_max) / running_max.clamp(min=1e-8)
    max_drawdown = abs(drawdowns.min().item())

    # --- Turnover ---
    weight_changes = weights[1:] - weights[:-1]
    turnover = weight_changes.abs().sum(dim=1).mean().item()

    # --- Composite score (lower is better) ---
    val_score = -sharpe + 0.1 * turnover + 0.05 * max_drawdown

    metrics = {
        "val_score": round(val_score, 6),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6),
        "turnover": round(turnover, 6),
        "mean_daily_ret": round(mean_ret, 8),
        "std_daily_ret": round(std_ret, 8),
        "num_samples": len(portfolio_returns),
    }

    return metrics


# ---------------------------------------------------------------------------
# Main — data preparation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ETF data for autoresearch")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    args = parser.parse_args()

    print("=" * 60)
    print("ETF Autoresearch — Data Preparation")
    print("=" * 60)

    features_df, targets_df = build_dataset(
        trade_frequency="daily",
        refresh=args.refresh,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Features shape:  {features_df.shape}")
    print(f"Targets shape:   {targets_df.shape}")
    print(f"Date range:      {features_df.index[0].date()} to {features_df.index[-1].date()}")

    train_n = (features_df.index < TRAIN_END).sum()
    val_n = ((features_df.index >= TRAIN_END) & (features_df.index < VAL_END)).sum()
    test_n = (features_df.index >= VAL_END).sum()
    print(f"Train samples:   {train_n}")
    print(f"Val samples:     {val_n}")
    print(f"Test samples:    {test_n}")
    print(f"\nFeature columns ({features_df.shape[1]}):")
    for col in features_df.columns:
        print(f"  {col}")

    print(f"\nTarget columns ({targets_df.shape[1]}):")
    for col in targets_df.columns:
        print(f"  {col}")

    print(f"\nCache directory: {CACHE_DIR}")
    print("Done.")
