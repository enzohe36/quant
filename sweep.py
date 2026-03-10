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

_SWEEP_KEYS = [
    "dropout", "peer_dropout", "value_coeff", "entropy_coeff", "lr",
]


def _param_hash(params):
    serialized = json.dumps({k: params[k] for k in sorted(_SWEEP_KEYS)}).encode()
    return hashlib.md5(serialized).hexdigest()[:12]


def objective(trial, preprocess_path, base_save_dir):
    cfg = Config()

    # Search space
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.2)
    cfg.peer_dropout = trial.suggest_float("peer_dropout", 0.0, 0.8)
    cfg.value_coeff = trial.suggest_float("value_coeff", 0.01, 1.0, log=True)
    cfg.entropy_coeff = trial.suggest_float("entropy_coeff", 0.001, 0.1, log=True)
    cfg.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

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
    parser = argparse.ArgumentParser(description="Optuna sweep for train_model")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--db", type=str, default="sqlite:///sweep.db")
    parser.add_argument("--save_dir", type=str, default="sweep_runs")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    cfg = Config()
    cfg.save_dir = args.save_dir
    preprocess_path = _preprocess_data(cfg)

    study = optuna.create_study(
        study_name="ppo_stock_sweep",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=20,
        ),
        storage=args.db,
        load_if_exists=True,
    )

    # Re-enqueue interrupted trials that have a checkpoint on disk
    retryable = (optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING)
    for t in study.trials:
        if t.state in retryable and t.params:
            h = _param_hash(t.params)
            if os.path.exists(os.path.join(args.save_dir, h, "model_latest.pt")):
                study.enqueue_trial(t.params)
                _log()
                _log(f"Re-enqueued trial {t.number} ({h})")

    study.optimize(
        lambda trial: objective(trial, preprocess_path, args.save_dir),
        n_trials=args.n_trials,
    )

    _log()
    _log("[Sweep Summary]")
    _log(f"    {'Best trial':<20s}: {study.best_trial.number}")
    _log(f"    {'Best score':<20s}: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        _log(f"    {key:<20s}: {value}")


if __name__ == "__main__":
    main()
