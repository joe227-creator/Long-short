"""Fast Optuna screen for cached TimesFM signal blending."""

import json
import os
import subprocess
from pathlib import Path

import optuna


SEED = 20260803
SEEDS = "6,42,123,7,99,11,22,33"
ARTIFACT_DIR = Path(".openresearch/artifacts")


def _score(output):
    for line in output.splitlines():
        if line.startswith("METRIC research_score="):
            return float(line.split("=", 1)[1])
    return None


def _run(alpha):
    env = os.environ.copy()
    env.update({
        "ARC_USE_TIMESFM": "1",
        "ARC_TSFMA": str(alpha),
        "ARC_SEEDS": SEEDS,
        "PYTHONHASHSEED": "0",
    })
    result = subprocess.run(
        "/home/user/venv/bin/python backtest.py --val",
        shell=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    return _score(output), output


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="timesfm_signal_blend",
        storage=f"sqlite:///{(ARTIFACT_DIR / 'timesfm_signal_blend.db').resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        alpha = trial.suggest_float("ARC_TSFMA", 0.0, 0.9)
        score, output = _run(alpha)
        if score is None:
            print(output[-1000:])
            raise optuna.TrialPruned()
        print(f"Trial {trial.number}: ARC_TSFMA={alpha:.8g} score={score:.8f}")
        return score

    study.optimize(objective, n_trials=6, catch=(Exception,))
    best = study.best_trial
    result = {
        "parameter": "ARC_TSFMA",
        "value": best.params["ARC_TSFMA"],
        "research_score": best.value,
        "trial": best.number,
        "n_trials": len(study.trials),
        "seeds": SEEDS,
    }
    (ARTIFACT_DIR / "timesfm_signal_blend_best.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BEST ARC_TSFMA={result['value']:.8g} score={result['research_score']:.8f}")


if __name__ == "__main__":
    main()
