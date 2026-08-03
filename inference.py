"""
Generate trading signals from the saved production ensemble.

This CLI shares the live loading and inference path used by trade.py via
production_model.py, so checkpoint discovery, config validation, feature
alignment, and weekly subsampling stay identical across both live scripts.

Usage: uv run inference.py
"""

import sys

import torch

from production_model import describe_model, load_production_ensemble, run_live_ensemble
from prepare import (
    ETF_PAIRS, PAIR_NAMES,
    download_etf_data, download_macro_data,
    build_features,
)
from train import load_checkpoint

MODELS_DIR = "models"


def load_ensemble(device):
    """Load all seed models for the production ensemble."""
    return load_production_ensemble(
        device,
        models_dir=MODELS_DIR,
        checkpoint_loader=load_checkpoint,
    )


def generate_signals():
    """Load ensemble, fetch latest data, output trading signals."""

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        models, config = load_ensemble(device)
    except FileNotFoundError:
        print("ERROR: No trained models found in models/")
        print("Run 'uv run train.py' first to train a model.")
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    loaded_seeds = config.get("loaded_seeds", [])

    trade_frequency = config.get("trade_frequency", "daily")

    # Download fresh data
    print("Downloading latest market data...")
    etf_df = download_etf_data(refresh=True)
    macro_df = download_macro_data(refresh=True)

    # Build features
    feat_df = build_features(etf_df, macro_df)
    try:
        decision = run_live_ensemble(models, config, feat_df, device)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    missing_cols = decision["missing_columns"]
    if missing_cols:
        raise ValueError(f"Live model features missing: {missing_cols}")

    raw_signals_np = decision["raw_signals"]
    signals_np = decision["signals"]
    weights_np = decision["weights"]

    # Output
    latest_date = decision["latest_date"]
    print(f"\n{'='*60}")
    print(f"  Trading Signals for {latest_date}")
    print(f"  Frequency: {trade_frequency}")
    print(f"  Model: {describe_model(config)}")
    if loaded_seeds:
        print(f"  Ensemble: {len(models)} loaded seed checkpoints {loaded_seeds}")
    else:
        print(f"  Ensemble: {len(models)} loaded checkpoint")
    print(f"{'='*60}\n")

    print("  Decision path: raw model score -> tanh-bounded signal -> normalized ETF weights")
    print(f"{'Pair':<12} {'Raw':>8} {'Signal':>8}   {'Bull ETF':<10} {'Weight':>8}   {'Bear ETF':<10} {'Weight':>8}")
    print("-" * 70)

    total_long = 0.0
    total_short = 0.0

    for i, (pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        raw_sig = raw_signals_np[i]
        sig = signals_np[i]
        w_bull = weights_np[2 * i]
        w_bear = weights_np[2 * i + 1]

        total_long += max(w_bull, 0) + max(w_bear, 0)
        total_short += abs(min(w_bull, 0)) + abs(min(w_bear, 0))

        print(f"{pair_name:<12} {raw_sig:+.4f} {sig:+.4f}   {bull:<10} {w_bull:+.4f}   {bear:<10} {w_bear:+.4f}")

    print("-" * 70)
    print(f"{'Total long exposure:':<30} {total_long:.4f}")
    print(f"{'Total short exposure:':<30} {total_short:.4f}")
    print(f"{'Net exposure:':<30} {total_long - total_short:.4f}")
    print()


if __name__ == "__main__":
    generate_signals()
