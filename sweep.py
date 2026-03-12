"""
Optuna hyperparameter sweep for train_model.py.

Usage:
  python sweep_params.py [--n_trials 100] [--db sqlite:///sweep.db] [--save_dir sweep_runs]

Parallel workers: run multiple processes pointing at the same --db.
Monitor with: optuna-dashboard <db>
"""

import argparse
import hashlib
import json
import os

import optuna
from train_model import Config, Pruned, _log, _preprocess_data, train


class SweepConfig:
    n_trials: int = 100
    db: str = "sqlite:///sweep.db"
    save_dir: str = "sweep_runs"
    study_name: str = "ppo_stock_sweep"
    sampler_startup_trials: int = 10
    pruner_startup_trials: int = 5
    pruner_warmup_steps: int = 20


SEARCH_SPACE = {
    "dropout":       (0.0, 0.2, False),
    "train_peers":   (1, 10, False),
    "value_coeff":   (0.01, 1.0, True),
    "entropy_coeff": (0.001, 0.1, True),
    "lr":            (1e-5, 1e-3, True),
}


def _param_hash(params):
    keys = sorted(SEARCH_SPACE.keys())
    serialized = json.dumps({k: params[k] for k in keys}).encode()
    return hashlib.md5(serialized).hexdigest()[:12]


def objective(trial, preprocess_path, base_save_dir):
    cfg = Config()

    for name, (low, high, log) in SEARCH_SPACE.items():
        setattr(cfg, name, trial.suggest_float(name, low, high, log=log))

    cfg.save_dir = os.path.join(base_save_dir, _param_hash(trial.params))
    cfg.ablation = False

    _log()
    _log(f"[Trial {trial.number}]")

    def epoch_callback(epoch, score):
        trial.report(score, epoch)
        if trial.should_prune():
            _log(f"    Pruned at epoch {epoch}")
            raise Pruned(f"Pruned at epoch {epoch}")

    try:
        best_score = train(
            cfg,
            preprocess_path=preprocess_path,
            epoch_callback=epoch_callback,
        )
    except Pruned:
        raise optuna.TrialPruned()

    return best_score


def main():
    scfg = SweepConfig()
    parser = argparse.ArgumentParser(description="Optuna sweep for train_model")

    for name, ann_type in SweepConfig.__annotations__.items():
        default = getattr(scfg, name)
        flag = f"--{name}"
        parser.add_argument(flag, type=ann_type, default=default,
                            help=f"(default: {default})")

    args = parser.parse_args()
    for name in SweepConfig.__annotations__:
        setattr(scfg, name, getattr(args, name))

    os.makedirs(scfg.save_dir, exist_ok=True)

    cfg = Config()
    cfg.save_dir = scfg.save_dir
    preprocess_path = _preprocess_data(cfg)

    study = optuna.create_study(
        study_name=scfg.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=scfg.sampler_startup_trials),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=scfg.pruner_startup_trials,
            n_warmup_steps=scfg.pruner_warmup_steps,
        ),
        storage=scfg.db,
        load_if_exists=True,
    )

    # Re-enqueue interrupted trials that have a checkpoint on disk
    retryable = (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)
    for t in study.trials:
        if t.state in retryable and t.params:
            h = _param_hash(t.params)
            if os.path.exists(os.path.join(scfg.save_dir, h, "model_latest.pt")):
                study.enqueue_trial(t.params)
                _log()
                _log(f"Re-enqueued trial {t.number} ({h})")

    study.optimize(
        lambda trial: objective(trial, preprocess_path, scfg.save_dir),
        n_trials=scfg.n_trials,
    )

    _log()
    _log("[Sweep Summary]")
    _log(f"    {'Best trial':<20s}: {study.best_trial.number}")
    _log(f"    {'Best score':<20s}: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        _log(f"    {key:<20s}: {value}")


if __name__ == "__main__":
    main()
