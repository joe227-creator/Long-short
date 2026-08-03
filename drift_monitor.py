"""
Concept-drift monitor for the production ensemble.

Compares realized weekly portfolio P&L against a baseline distribution and
emits an alert only when several independent drift signals agree, with a
mandatory cooldown to prevent always-triggering. Designed to be called from
trade.py once per weekly run, after positions for the new week are saved.

Tunable parameters are calibrated to keep historical false-positive rate in
the ~5–10% range on the simulate.py backtest. Adjust BREACH_K_SIGMA, CUSUM_H
and SHARPE_FLOOR if your offline calibration differs.

Trigger logic (>=2 of 3 must fire AND cooldown not active):
  1. Consecutive breaches: realized weekly return below mean - k*sigma for
     CONSECUTIVE_BREACHES weeks in a row.
  2. Rolling Sharpe floor: annualized Sharpe over the last WINDOW_WEEKS
     weeks falls below SHARPE_FLOOR.
  3. CUSUM: one-sided cumulative sum of standardized underperformance
     exceeds CUSUM_H sigma units (CUSUM_K is the slack / drift-to-detect).

A `drift_baseline.json` file (optional) overrides the bootstrapped baseline.
Recommended workflow: run `uv run simulate.py`, then write the test-period
mean and std of weekly returns into drift_baseline.json so the monitor uses
a stable reference instead of self-bootstrapping from short live history.
"""

import json
import os

import numpy as np
import pandas as pd

from prepare import ETF_TICKERS

DRIFT_LOG_FILE = "drift_log.json"
BASELINE_FILE = "drift_baseline.json"

# --- Tunables (calibrated for ~5-10% historical false positive rate) ---
WINDOW_WEEKS = 13                # rolling window for Sharpe / mean
CONSECUTIVE_BREACHES = 2         # weeks in a row below threshold
BREACH_K_SIGMA = 2.0             # weekly return < mean - k*sigma counts as a breach
CUSUM_K = 0.5                    # CUSUM slack (drift-to-detect, in sigma)
CUSUM_H = 5.0                    # CUSUM alarm threshold (in sigma)
SHARPE_FLOOR = 0.5               # rolling annualized Sharpe must stay above this
COOLDOWN_DAYS = 60               # hysteresis: minimum days between alerts
MIN_OBSERVATIONS = 8             # need at least this many realized weeks to evaluate
MIN_TRIGGERS_FOR_ALERT = 2       # multi-indicator agreement required


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def compute_realized_returns(trade_log, etf_df):
    """Realized portfolio return between consecutive trade-log entries.

    Each entry's weights are held from its date until the next entry's date.
    Skips the most recent entry (no realized return yet). Returns a list of
    {start, end, return} dicts in chronological order.
    """
    closes = {t: etf_df[f"{t}_Close"]
              for t in ETF_TICKERS if f"{t}_Close" in etf_df.columns}
    realized = []
    for i in range(len(trade_log) - 1):
        entry = trade_log[i]
        nxt = trade_log[i + 1]
        try:
            d0 = pd.to_datetime(entry["date"])
            d1 = pd.to_datetime(nxt["date"])
        except (KeyError, ValueError):
            continue
        if d1 <= d0:
            continue
        r = 0.0
        ok = True
        for etf, w in entry.get("weights", {}).items():
            if etf not in closes:
                ok = False
                break
            s = closes[etf]
            p0 = s.asof(d0)
            p1 = s.asof(d1)
            if pd.isna(p0) or pd.isna(p1) or p0 <= 0 or p1 <= 0:
                ok = False
                break
            r += float(w) * float(np.log(p1) - np.log(p0))
        if ok:
            realized.append({"start": entry["date"], "end": nxt["date"], "return": r})
    return realized


def _baseline(realized_returns):
    """Return (mean, std) baseline weekly return.

    Prefer drift_baseline.json (calibrated from simulate.py); else bootstrap
    from observed history once enough samples exist.
    """
    base = _load_json(BASELINE_FILE, None)
    if base and "mean" in base and "std" in base:
        return float(base["mean"]), max(float(base["std"]), 1e-6)
    arr = np.array([r["return"] for r in realized_returns], dtype=float)
    if len(arr) >= MIN_OBSERVATIONS:
        return float(arr.mean()), max(float(arr.std(ddof=1)), 1e-6)
    return None


def evaluate_drift(trade_log, etf_df, latest_date):
    """Run the multi-indicator drift evaluation and persist alerts."""
    realized = compute_realized_returns(trade_log, etf_df)

    if len(realized) < MIN_OBSERVATIONS:
        return {
            "status": "insufficient_data",
            "n_observations": len(realized),
            "alert": False,
        }

    base = _baseline(realized)
    if base is None:
        return {
            "status": "no_baseline",
            "n_observations": len(realized),
            "alert": False,
        }
    mean_b, std_b = base

    arr = np.array([r["return"] for r in realized], dtype=float)
    win = arr[-WINDOW_WEEKS:] if len(arr) >= WINDOW_WEEKS else arr

    # 1) Consecutive breaches at the tail of the series
    threshold = mean_b - BREACH_K_SIGMA * std_b
    consec = 0
    for v in arr[::-1]:
        if v < threshold:
            consec += 1
        else:
            break
    consec_trigger = consec >= CONSECUTIVE_BREACHES

    # 2) Rolling Sharpe (annualized, weekly => sqrt(52))
    if len(win) >= 4 and win.std(ddof=1) > 0:
        roll_sharpe = float((win.mean() / win.std(ddof=1)) * np.sqrt(52))
    else:
        roll_sharpe = None
    sharpe_trigger = roll_sharpe is not None and roll_sharpe < SHARPE_FLOOR

    # 3) Two-sided one-sided CUSUM on standardized underperformance
    z = (mean_b - arr) / std_b   # positive when underperforming
    s_pos = 0.0
    s_max = 0.0
    for v in z:
        s_pos = max(0.0, s_pos + v - CUSUM_K)
        s_max = max(s_max, s_pos)
    cusum_trigger = s_pos > CUSUM_H

    # Hysteresis (cooldown) check against persisted alerts
    drift_log = _load_json(DRIFT_LOG_FILE, {"alerts": []})
    last_alert_date = drift_log["alerts"][-1]["date"] if drift_log["alerts"] else None
    days_since_last = None
    cooldown_active = False
    if last_alert_date is not None:
        days_since_last = (pd.to_datetime(latest_date) - pd.to_datetime(last_alert_date)).days
        cooldown_active = days_since_last < COOLDOWN_DAYS

    triggers = {
        "consecutive_breaches": bool(consec_trigger),
        "rolling_sharpe_below_floor": bool(sharpe_trigger),
        "cusum_breach": bool(cusum_trigger),
    }
    n_triggered = sum(triggers.values())
    alert = (n_triggered >= MIN_TRIGGERS_FOR_ALERT) and (not cooldown_active)

    summary = {
        "status": "ok",
        "latest_date": str(latest_date),
        "n_observations": int(len(arr)),
        "baseline_mean_weekly_return": float(mean_b),
        "baseline_std_weekly_return": float(std_b),
        "baseline_source": "drift_baseline.json" if os.path.exists(BASELINE_FILE) else "bootstrapped",
        "rolling_window_weeks": int(len(win)),
        "rolling_mean_return": float(win.mean()),
        "rolling_sharpe": roll_sharpe,
        "consecutive_breaches": int(consec),
        "consecutive_breach_required": int(CONSECUTIVE_BREACHES),
        "breach_return_threshold": float(threshold),
        "cusum_value": float(s_pos),
        "cusum_max_seen": float(s_max),
        "cusum_threshold": float(CUSUM_H),
        "triggers": triggers,
        "n_triggered": int(n_triggered),
        "min_triggers_for_alert": int(MIN_TRIGGERS_FOR_ALERT),
        "cooldown_active": bool(cooldown_active),
        "cooldown_days": int(COOLDOWN_DAYS),
        "days_since_last_alert": days_since_last,
        "alert": bool(alert),
    }

    if alert:
        drift_log["alerts"].append({"date": str(latest_date), "summary": summary})
        _save_json(DRIFT_LOG_FILE, drift_log)

    return summary


def print_drift_report(summary):
    """Pretty-print the drift summary returned by evaluate_drift()."""
    print(f"\n{'=' * 70}")
    print("  CONCEPT DRIFT MONITOR")
    print(f"{'=' * 70}")

    status = summary.get("status")
    if status == "insufficient_data":
        print(f"  Status: insufficient history "
              f"({summary['n_observations']} realized weeks; need >= {MIN_OBSERVATIONS}).")
        print(f"{'=' * 70}\n")
        return
    if status == "no_baseline":
        print(f"  Status: no calibrated baseline yet "
              f"(drift_baseline.json missing and only {summary['n_observations']} weeks observed).")
        print(f"{'=' * 70}\n")
        return

    print(f"  Realized weeks observed:  {summary['n_observations']}  "
          f"(baseline source: {summary['baseline_source']})")
    print(f"  Baseline weekly return:   mean={summary['baseline_mean_weekly_return']:+.4%}   "
          f"std={summary['baseline_std_weekly_return']:.4%}")

    rs = summary['rolling_sharpe']
    rs_str = f"{rs:+.3f}" if rs is not None else "n/a"
    print(f"  Rolling {summary['rolling_window_weeks']}-week mean: "
          f"{summary['rolling_mean_return']:+.4%}    Sharpe(annualized): {rs_str}")
    print(f"  Consecutive breaches:     {summary['consecutive_breaches']} / "
          f"{summary['consecutive_breach_required']} "
          f"(weekly ret < {summary['breach_return_threshold']:+.4%})")
    print(f"  CUSUM:                    {summary['cusum_value']:.2f} / "
          f"{summary['cusum_threshold']:.2f}  (max seen: {summary['cusum_max_seen']:.2f})")

    for name, tripped in summary['triggers'].items():
        mark = "TRIPPED" if tripped else "ok"
        print(f"    - {name:<32} {mark}")

    print(f"  Triggers active:          {summary['n_triggered']} / 3 "
          f"(>= {summary['min_triggers_for_alert']} required to alert)")

    if summary['cooldown_active']:
        print(f"  Cooldown:                 ACTIVE "
              f"({summary['days_since_last_alert']} days since last alert; "
              f"cooldown {summary['cooldown_days']} days)")
    elif summary['days_since_last_alert'] is not None:
        print(f"  Cooldown:                 inactive "
              f"({summary['days_since_last_alert']} days since last alert)")

    if summary['alert']:
        print("\n  *** CONCEPT DRIFT ALERT ***")
        print("  Multiple drift indicators agree. Recommended actions:")
        print("    1. Inspect drift_log.json for the triggering summary.")
        print("    2. Review research_memory.txt and recent realized P&L.")
        print("    3. Retrain the ensemble:  uv run train.py")
        print("    4. Validate with:         uv run simulate.py")
    else:
        print("  Status: no drift alert.")
    print(f"{'=' * 70}\n")
