"""Optuna toggle for dormant ten-seed dual ensemble configuration."""

import glob
import json
import os
import subprocess
from pathlib import Path

import optuna


SEED = 20260803
SEEDS = "6,42,123,7,99,11,22,33,44,55"
TIME_BUDGET = 120
ARTIFACT_DIR = Path(".openresearch/artifacts")


def _clear_models():
    for path in glob.glob("models/best_model_seed*.pt"):
        os.remove(path)


def _score(output):
    for line in output.splitlines():
        if line.startswith("METRIC research_score="):
            return float(line.split("=", 1)[1])
    return None


def _run(enabled):
    _clear_models()
    env = os.environ.copy()
    env.update({
        "ARC_DUAL_ENSEMBLE": str(enabled),
        "ARC_SEEDS": SEEDS,
        "ARC_TIME_BUDGET": str(TIME_BUDGET),
        "PYTHONHASHSEED": "0",
    })
    result = subprocess.run(
        "/home/user/venv/bin/python train.py && /home/user/venv/bin/python backtest.py --val",
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
        study_name="dual_ensemble",
        storage=f"sqlite:///{(ARTIFACT_DIR / 'dual_ensemble.db').resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    def objective(trial):
        enabled = trial.suggest_int("ARC_DUAL_ENSEMBLE", 0, 1)
        score, output = _run(enabled)
        if score is None:
            print(output[-1000:])
            raise optuna.TrialPruned()
        print(f"Trial {trial.number}: ARC_DUAL_ENSEMBLE={enabled} score={score:.8f}")
        return score

    study.optimize(objective, n_trials=2, catch=(Exception,))
    best = study.best_trial
    result = {
        "parameter": "ARC_DUAL_ENSEMBLE",
        "value": best.params["ARC_DUAL_ENSEMBLE"],
        "research_score": best.value,
        "trial": best.number,
        "n_trials": len(study.trials),
        "seeds": SEEDS,
        "time_budget": TIME_BUDGET,
    }
    (ARTIFACT_DIR / "dual_ensemble_best.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"BEST ARC_DUAL_ENSEMBLE={result['value']} score={result['research_score']:.8f}")


if __name__ == "__main__":
    main()
