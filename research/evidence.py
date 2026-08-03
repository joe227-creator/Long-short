"""Write reproducible robustness-adjusted backtest evidence."""

import csv
import json
from pathlib import Path

import numpy as np


WINDOW_TRADING_SESSIONS = 126


def _window_periods(trade_frequency):
    if trade_frequency == "daily":
        return WINDOW_TRADING_SESSIONS
    if trade_frequency == "weekly":
        return int(np.ceil(WINDOW_TRADING_SESSIONS / 5))
    raise ValueError(f"Unsupported trade frequency: {trade_frequency}")


def _iso_date(value):
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def _window_records(returns, dates, window_periods):
    records = []
    for start in range(0, len(returns), window_periods):
        stop = start + window_periods
        if stop > len(returns):
            break
        window_returns = returns[start:stop]
        equity = np.cumprod(1.0 + window_returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.maximum(running_max, 1e-8)
        records.append({
            "window_index": len(records),
            "start": _iso_date(dates[start]),
            "end": _iso_date(dates[stop - 1]),
            "periods": window_periods,
            "return": float(equity[-1] - 1.0),
            "max_drawdown": float(drawdowns.min()),
            "positive": bool(equity[-1] > 1.0),
        })
    if not records:
        raise ValueError("Evaluation has no complete 126-session windows")
    return records


def _baseline_reference(metrics):
    path = Path("research/baseline_reference.json")
    if path.exists():
        reference = json.loads(path.read_text(encoding="utf-8"))
        return float(reference["baseline_turnover"]), str(reference.get("source", path))
    return float(metrics["turnover"]), "current evaluation fallback"


def _score(metrics, windows, baseline_turnover):
    mean_return = float(np.mean([row["return"] for row in windows]))
    maximum_drawdown = -abs(float(metrics["max_drawdown"]))
    return_on_risk = mean_return / max(abs(maximum_drawdown), 1e-6)
    win_rate = float(np.mean([row["positive"] for row in windows]))
    drawdown_penalty = max(0.0, -0.50 - maximum_drawdown)
    sharpe_penalty = max(0.0, 0.80 - float(metrics["sharpe"]))
    turnover_penalty = max(0.0, float(metrics["turnover"]) - baseline_turnover)
    score = (
        mean_return
        + 0.20 * return_on_risk
        + 0.10 * win_rate
        - 0.35 * drawdown_penalty
        - 0.15 * sharpe_penalty
        - 0.10 * turnover_penalty
    )
    return {
        "mean_rolling_6m_return": mean_return,
        "return_on_risk": return_on_risk,
        "win_rate_126_session": win_rate,
        "maximum_drawdown": maximum_drawdown,
        "sharpe": float(metrics["sharpe"]),
        "turnover": float(metrics["turnover"]),
        "baseline_turnover": baseline_turnover,
        "drawdown_penalty": drawdown_penalty,
        "sharpe_penalty": sharpe_penalty,
        "turnover_penalty": turnover_penalty,
        "research_score": float(score),
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_curves(directory, split, returns, dates, weights):
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.maximum(running_max, 1e-8)
    curve_path = directory / f"equity_curve_{split}.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "return", "equity", "drawdown", "gross_exposure"])
        for date, ret, eq, dd, weight in zip(dates, returns, equity, drawdowns, weights):
            writer.writerow([_iso_date(date), f"{ret:.12g}", f"{eq:.12g}", f"{dd:.12g}", f"{np.abs(weight).sum():.12g}"])


def _write_windows(directory, split, windows):
    path = directory / f"rolling_126_session_windows_{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)


def _write_markdown(directory, split, summary):
    name = "EVAL.md" if split == "val" else "TEST_EVAL.md"
    score = summary["score"]
    text = "\n".join([
        f"# {split.upper()} Evaluation",
        "",
        f"- Period: `{summary['start']}` to `{summary['end']}`",
        f"- Complete 126-session windows: `{summary['complete_windows']}`",
        f"- Mean rolling 6m return: `{score['mean_rolling_6m_return']:.8f}`",
        f"- Return on risk: `{score['return_on_risk']:.8f}`",
        f"- Window win rate: `{score['win_rate_126_session']:.8f}`",
        f"- Maximum drawdown: `{score['maximum_drawdown']:.8f}`",
        f"- Sharpe: `{score['sharpe']:.8f}`",
        f"- Turnover: `{score['turnover']:.8f}`",
        f"- Baseline turnover: `{score['baseline_turnover']:.8f}`",
        f"- Research score: `{score['research_score']:.8f}`",
        "",
        "Artifacts: resolved config, equity/drawdown curve, rolling-window CSV, and seed metrics.",
        "",
    ])
    (directory / name).write_text(text, encoding="utf-8")


def write_evidence(split, metrics, returns, weights, dates, config):
    """Persist score inputs and text evidence for one evaluation split."""
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if returns.ndim != 1 or weights.ndim != 2 or len(returns) != len(weights) or len(returns) != len(dates):
        raise ValueError("Returns, weights, and dates must have aligned dimensions")
    if not np.isfinite(returns).all() or not np.isfinite(weights).all():
        raise ValueError("Evaluation returns and weights must be finite")

    directory = Path(".openresearch/artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    frequency = config.get("trade_frequency", "daily")
    window_periods = _window_periods(frequency)
    windows = _window_records(returns, dates, window_periods)
    baseline_turnover, baseline_source = _baseline_reference(metrics)
    score = _score(metrics, windows, baseline_turnover)
    summary = {
        "split": split,
        "start": _iso_date(dates[0]),
        "end": _iso_date(dates[-1]),
        "trade_frequency": frequency,
        "periods_per_year": 52 if frequency == "weekly" else 252,
        "window_trading_sessions": WINDOW_TRADING_SESSIONS,
        "window_periods": window_periods,
        "complete_windows": len(windows),
        "metrics": metrics,
        "score": score,
        "baseline_turnover_source": baseline_source,
    }
    _write_json(directory / f"research_score_{split}.json", summary)
    _write_json(directory / f"resolved_config_{split}.json", config)
    _write_curves(directory, split, returns, dates, weights)
    _write_windows(directory, split, windows)
    _write_markdown(directory, split, summary)
    return summary
