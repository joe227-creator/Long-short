"""
Weekly trading script — run on Sunday at 21:00 via Windows Task Scheduler (see run_trade_scheduled.ps1).

Discovers the saved production checkpoints from models/ via production_model.py,
validates that the checkpoint configs match, downloads fresh data, generates target
portfolio weights, and maintains a position change log showing the latest 4 weeks.

Usage: uv run trade.py

Output:
    - Target end-state portfolio positions (per-ETF weights)
    - From-flat trading actions for users not already holding the ETFs
    - 4-week position change log (historical positions from trade_log.json)
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

from production_model import describe_model, load_production_ensemble, run_live_ensemble
from prepare import (
    ETF_PAIRS,
    PAIR_NAMES,
    download_etf_data,
    download_macro_data,
    build_features,
)
from train import load_checkpoint
from drift_monitor import evaluate_drift, print_drift_report
from research.execution_controls import (
    apply_live_weight_band,
    apply_partial_adjustment,
    load_live_execution_controls,
)

MODELS_DIR = "models"
LOG_FILE = "trade_log.json"
HISTORY_WEEKS = 4


def load_ensemble(device):
    """Load all seed models for the production ensemble."""
    return load_production_ensemble(
        device,
        models_dir=MODELS_DIR,
        checkpoint_loader=load_checkpoint,
    )


def load_trade_log():
    """Load existing trade log or return empty list."""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"WARNING: {LOG_FILE} is corrupt ({exc}); starting fresh log.")
            return []
    return []


def save_trade_log(log):
    """Save trade log to disk."""
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def plot_positions(weights_np, latest_date, chart_file="trade_chart.png"):
    """Generate and save a portfolio allocation pie chart."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, save to file
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    etfs = [etf for pair in ETF_PAIRS for etf in pair]
    sizes = [abs(weights_np[i]) * 100 for i in range(len(etfs))]
    is_long = [weights_np[i] > 0 for i in range(len(etfs))]

    # Single green for all long positions, single red for all short positions
    COLOR_LONG = "#2ecc71"
    COLOR_SHORT = "#e74c3c"
    colors = [COLOR_LONG if flag else COLOR_SHORT for flag in is_long]

    total_exposure = sum(sizes)
    total_long = sum(sizes[i] for i in range(len(etfs)) if weights_np[i] > 0)
    total_short = sum(sizes[i] for i in range(len(etfs)) if weights_np[i] < 0)
    labels = []
    for etf, size, long in zip(etfs, sizes, is_long):
        side = "LONG" if long else "SHORT"
        # Hide labels for positions below 0.5% of capital to avoid overlap.
        if size < 0.5:
            labels.append("")
        else:
            labels.append(f"{etf}\n{side} {size:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"color": "white", "fontsize": 9, "fontweight": "bold"},
        wedgeprops={"linewidth": 1.5, "edgecolor": "#1a1a2e"},
    )

    ax.set_title(
        f"Target Portfolio Allocation — {latest_date}\n"
        "Green = target LONG   |   Red = target SHORT\n"
        f"Long {total_long:.1f}%  +  Short {total_short:.1f}%  =  {total_exposure:.1f}% gross exposure",
        fontsize=12, color="white", fontweight="bold", pad=20,
    )

    legend_elements = [
        Patch(facecolor="#2ecc71", label="LONG  — target position is long"),
        Patch(facecolor="#e74c3c", label="SHORT — target position is short"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#2e2e4e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(chart_file, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    return chart_file


def target_side(weight):
    """Return a plain-English side label for a target weight."""
    if weight > 0:
        return "LONG"
    if weight < 0:
        return "SHORT"
    return "FLAT"


def flat_action(weight):
    """Return the action needed if starting from zero holdings."""
    if weight > 0:
        return "BUY"
    if weight < 0:
        return "SHORT"
    return "HOLD"


def apply_live_execution_controls(weights_np, previous_log, controls=None):
    """Apply selected partial adjustment and target retention to live weights."""
    controls = load_live_execution_controls() if controls is None else controls
    target = torch.as_tensor(weights_np, dtype=torch.float32)
    if not previous_log:
        return target.numpy()

    previous_record = previous_log[-1].get("weights", {})
    etfs = [etf for pair in ETF_PAIRS for etf in pair]
    if any(etf not in previous_record for etf in etfs):
        return target.numpy()
    previous = torch.tensor(
        [float(previous_record[etf]) for etf in etfs],
        dtype=target.dtype,
    )

    partial_rate = controls.get("partial_adjustment")
    if partial_rate is not None:
        target = apply_partial_adjustment(
            torch.stack([previous, target], dim=0), partial_rate
        )[-1]
    target = apply_live_weight_band(
        previous,
        target,
        controls.get("weight_band", 0.0),
    )
    return target.numpy()


def generate_positions():
    """Generate current positions from the ensemble and update the trade log."""
    device = torch.device("cpu")

    # Load ensemble
    print("Loading ensemble models...")
    try:
        models, config = load_ensemble(device)
    except FileNotFoundError:
        print("ERROR: No ensemble models found. Run 'uv run train.py' first.")
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    loaded_seeds = config.get("loaded_seeds", [])
    if loaded_seeds:
        print(f"  Ensemble size: {len(models)} models (loaded seeds: {loaded_seeds})")
    else:
        print(f"  Ensemble size: {len(models)} model")

    trade_frequency = config.get("trade_frequency", "daily")

    # Download fresh data
    print("Downloading latest market data...")
    etf_df = download_etf_data(refresh=True)
    macro_df = download_macro_data(refresh=True)

    # Keep the TimesFM caches fresh (vol gate + signal blend). Non-fatal: if this
    # fails, inference still runs but with forward-filled (stale) TimesFM values.
    try:
        from generate_timesfm_vol import refresh_timesfm_caches
        refresh_timesfm_caches(etf_df)
    except Exception as exc:
        print(f"WARNING: TimesFM cache refresh failed ({exc}); "
              "vol gate/blend will use the last cached (possibly stale) values.")

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
    base_weights_np = decision["weights"]
    latest_date = decision["latest_date"]

    # Load prior targets before applying stateful live controls.
    log = load_trade_log()
    execution_controls = load_live_execution_controls()
    same_date = bool(log and log[-1].get("date") == latest_date)
    if same_date:
        weights_np = np.array([
            float(log[-1]["weights"][etf])
            for pair in ETF_PAIRS
            for etf in pair
        ])
    else:
        weights_np = apply_live_execution_controls(
            base_weights_np,
            log,
            execution_controls,
        )

    # Build current position record
    current = {
        "date": latest_date,
        "timestamp": datetime.now().isoformat(),
        "raw_signals": {},
        "signals": {},
        "base_weights": {},
        "weights": {},
        "execution_controls": execution_controls,
    }

    for i, (pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        current["raw_signals"][pair_name] = round(float(raw_signals_np[i]), 6)
        current["signals"][pair_name] = round(float(signals_np[i]), 6)
        current["base_weights"][bull] = round(float(base_weights_np[2 * i]), 6)
        current["base_weights"][bear] = round(float(base_weights_np[2 * i + 1]), 6)
        current["weights"][bull] = round(float(weights_np[2 * i]), 6)
        current["weights"][bear] = round(float(weights_np[2 * i + 1]), 6)

    # Avoid duplicate entries for the same date
    if log and log[-1]["date"] == latest_date:
        log[-1] = current
    else:
        log.append(current)

    save_trade_log(log)

    # ---- Display ----

    print(f"\n{'=' * 70}")
    print(f"  WEEKLY TRADING POSITIONS — {latest_date}")
    print(f"  Model: {describe_model(config)}")
    if loaded_seeds:
        print(f"  Ensemble: {len(models)} loaded seed checkpoints {loaded_seeds}")
    else:
        print(f"  Ensemble: {len(models)} loaded checkpoint")
    print(f"  Frequency: {trade_frequency}")
    print(f"{'=' * 70}\n")

    # Target pair book — weights shown as % of total capital
    print("  Decision path: raw model score -> tanh signal -> normalized target -> live controls.")
    print("  TARGET PAIR BOOK  (these are end-state portfolio targets, not trade deltas):")
    print(f"  {'Pair':<12} {'Raw':>8} {'Signal':>8}   {'Bull ETF':<8} {'Target%':>8}   {'Bear ETF':<8} {'Target%':>8}   Pair target")
    print(f"  {'-' * 72}")

    total_long = 0.0
    total_short = 0.0

    for i, (pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        raw_sig = raw_signals_np[i]
        sig = signals_np[i]
        w_bull = weights_np[2 * i]
        w_bear = weights_np[2 * i + 1]

        total_long += max(w_bull, 0) + max(w_bear, 0)
        total_short += abs(min(w_bull, 0)) + abs(min(w_bear, 0))

        if np.isclose(w_bull, 0.0) and np.isclose(w_bear, 0.0):
            pair_target = "FLAT"
        elif raw_sig >= 0:
            pair_target = f"LONG {bull} / SHORT {bear}"
        else:
            pair_target = f"SHORT {bull} / LONG {bear}"
        print(
            f"  {pair_name:<12} {raw_sig:+.4f} {sig:+.4f}   "
            f"{bull:<8} {w_bull*100:>+7.2f}%   {bear:<8} {w_bear*100:>+7.2f}%   {pair_target}"
        )

    print(f"  {'-' * 72}")
    print(f"  {'LONG  (buy positions):':<38} {total_long*100:>6.2f}% of capital")
    print(f"  {'SHORT (short positions):':<38} {total_short*100:>6.2f}% of capital")
    print(f"  {'Gross exposure:':<38} {(total_long + total_short)*100:>6.2f}% of capital")
    print(f"  {'Net exposure:':<38} {(total_long - total_short)*100:>+6.2f}% of capital")
    print("  Gross 100% means the portfolio uses 100% total capital in absolute terms.")
    print("  Here that is 50% long + 50% short, so the portfolio is market-neutral overall.")

    # Per-ETF target book in plain English
    print("\n  TARGET ETF BOOK  (what each number means):")
    print(f"  {'ETF':<6} {'Target%':>9} {'Side':>8}   Meaning")
    print(f"  {'-' * 72}")
    for i, (_pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        for etf, widx in [(bull, 2 * i), (bear, 2 * i + 1)]:
            weight = weights_np[widx]
            side = target_side(weight)
            pct = abs(weight) * 100
            if side == "LONG":
                meaning = f"end state: hold {pct:.2f}% of capital as a long {etf} position"
            elif side == "SHORT":
                meaning = f"end state: hold {pct:.2f}% of capital as a short {etf} position"
            else:
                meaning = f"end state: no {etf} position"
            print(f"  {etf:<6} {weight*100:>+8.2f}% {side:>8}   {meaning}")

    print("\n  HOW TO READ NEGATIVE TARGETS:")
    print("  - A negative target does NOT mean 'move money from this ETF into the other ETF'.")
    print("  - It means the desired final portfolio should be SHORT that ETF by that % of capital.")
    print("  - If you already own that ETF long, sell it down to zero and then continue until you are short.")
    print("  - If you do not own it, open a new short position of that size.")
    print("  - If you are already short, rebalance that short until it matches the target size.")
    print("  - The script does NOT inspect your brokerage account; it only outputs the target portfolio.")

    print("\n  IF YOU START FROM ZERO HOLDINGS TODAY:")
    print(f"  {'ETF':<6} {'Action':<7} {'Size':>9}   Plain instruction")
    print(f"  {'-' * 72}")
    for i, (_pair_name, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        for etf, widx in [(bull, 2 * i), (bear, 2 * i + 1)]:
            weight = weights_np[widx]
            action = flat_action(weight)
            pct = abs(weight) * 100
            if action == "BUY":
                instruction = f"buy {etf} using {pct:.2f}% of your capital"
            elif action == "SHORT":
                instruction = f"short {etf} using {pct:.2f}% of your capital"
            else:
                instruction = f"do not open a position in {etf}"
            print(f"  {etf:<6} {action:<7} {pct:>8.2f}%   {instruction}")

    # ASCII allocation bar chart
    max_w = max(abs(w) for w in weights_np) if any(abs(w) > 0 for w in weights_np) else 1.0
    BAR_MAX = 22
    print("\n  ALLOCATION BARS  (# target long  . target short):")
    print(f"  {'ETF':<6}  {'':4}  {'':22}  {'% of capital':>12}")
    print(f"  {'-' * 48}")
    for i, (_pair, (bull, bear)) in enumerate(zip(PAIR_NAMES, ETF_PAIRS)):
        for etf, widx in [(bull, 2 * i), (bear, 2 * i + 1)]:
            w = weights_np[widx]
            pct = abs(w) * 100
            bar_len = max(1, int(abs(w) / max_w * BAR_MAX))
            bar = "#" * bar_len if w > 0 else "." * bar_len
            direction = "LONG" if w > 0 else "SHORT"
            print(f"  {etf:<6}  {direction}  {bar:<22}  {pct:>9.2f}%")
    print(f"  {'-' * 48}")
    print(f"  LONG {total_long*100:.2f}%  +  SHORT {total_short*100:.2f}%  =  {(total_long+total_short)*100:.2f}% gross")

    # Generate and open pie chart
    chart_file = f"trade_chart_{latest_date}.png"
    saved = plot_positions(weights_np, latest_date, chart_file)
    print(f"\n  Pie chart saved → {saved}")
    if os.environ.get("HEDGE_PORTFOLIO_SCHEDULED") == "1":
        print("  Chart auto-open skipped for scheduled run.")
    else:
        try:
            import subprocess
            subprocess.Popen(["cmd", "/c", "start", "", saved])
        except Exception:
            pass

    # 4-week change log
    recent = log[-HISTORY_WEEKS - 1 :]  # up to 5 entries to show 4 weeks of changes
    if len(recent) > 1:
        print(f"\n  {'=' * 70}")
        print(f"  4-WEEK POSITION CHANGE LOG")
        print(f"  {'=' * 70}\n")

        # Header
        print(f"  {'Date':<14}", end="")
        for pair_name in PAIR_NAMES:
            print(f" {pair_name:>10}", end="")
        print()
        print(f"  {'-' * (14 + 10 * len(PAIR_NAMES))}")

        for entry in recent:
            print(f"  {entry['date']:<14}", end="")
            for pair_name in PAIR_NAMES:
                sig = entry["signals"].get(pair_name, 0.0)
                print(f" {sig:+10.4f}", end="")
            print()

        # Show weight changes between consecutive weeks
        if len(recent) >= 2:
            print(f"\n  WEIGHT CHANGES (latest vs prior week, in % of capital):")
            prev = recent[-2]
            curr = recent[-1]

            print(f"  {'ETF':<8} {'Side':<8} {'Prior%':>8} {'Now%':>8} {'Change':>10}")
            print(f"  {'-' * 44}")

            for pair in ETF_PAIRS:
                for etf in pair:
                    w_prev = prev["weights"].get(etf, 0.0)
                    w_curr = curr["weights"].get(etf, 0.0)
                    delta = w_curr - w_prev
                    side = target_side(w_curr)
                    marker = " ***" if abs(delta) > 0.05 else ""
                    print(f"  {etf:<8} {side:<8} {w_prev*100:>+7.2f}%  {w_curr*100:>+7.2f}%  {delta*100:>+7.2f}pp{marker}")
    else:
        print("\n  (First run — no prior history for change log)")
        print("  Run again next week to see position changes.")

    print(f"\n{'=' * 70}")
    print(f"  Log saved to: {LOG_FILE}")
    print(f"{'=' * 70}")

    # ---- Concept drift monitor (uses log + already-downloaded etf_df) ----
    try:
        drift_summary = evaluate_drift(log, etf_df, latest_date)
        print_drift_report(drift_summary)
    except Exception as exc:
        print(f"\n  [drift monitor skipped: {exc}]\n")


if __name__ == "__main__":
    generate_positions()
