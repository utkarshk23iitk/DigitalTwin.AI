"""
optuna_tune.py — tune the virtual-sensor / feature pipeline and the defect
model, with resumable Optuna studies and saved best-parameter checkpoints.

Usage:
    python src/optuna_tune.py --study all --virtual-trials 25 --defect-trials 40
    python src/optuna_tune.py --study virtual --virtual-trials 30
    python src/optuna_tune.py --study defect --defect-trials 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from defect_model import DefectModel
from feature_engineering import build_features
from train_defect_model import (NON_FEATURE_COLS, _channel_of_feature,
                                _load_decoy_channels, build_topk,
                                evaluate_holdout, load_split,
                                save_trained_artifacts)
from tuning_config import (DEFECT_PARAMS_PATH, VIRTUAL_SENSOR_PARAMS_PATH,
                           best_payload, checkpoint_path, optuna_storage_uri,
                           save_json)
from virtual_sensor import fit_virtual_sensors, validate


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"


def _load_manifest() -> dict:
    with open(DATA_DIR / "manifest.json") as fh:
        return json.load(fh)


def ensure_feature_table() -> None:
    path = DATA_DIR / "model_features.csv"
    if path.exists():
        return
    features = build_features(verbose=True)
    features.to_csv(path, index=False)
    print(f"[write] {path}")


def _feature_split(features: pd.DataFrame):
    manifest = _load_manifest()
    X = features.drop(columns=NON_FEATURE_COLS)
    y = features["response"].astype(int)
    train_mask = features["session_id"].isin(manifest["train_sessions"])
    test_mask = features["session_id"].isin(manifest["test_sessions"])
    return X, y, train_mask, test_mask


def _real_only(X: pd.DataFrame) -> pd.DataFrame:
    decoys = _load_decoy_channels()
    if not decoys:
        return X
    cols = [c for c in X.columns if _channel_of_feature(c) not in decoys]
    return X[cols]


def _sensor_gain(report: pd.DataFrame) -> tuple[float, float]:
    gains = []
    for _, row in report.iterrows():
        mae = row.get("mae")
        base = row.get("baseline_mae")
        if pd.notna(mae) and pd.notna(base) and float(base) > 0:
            gains.append(max(-1.0, min(1.0, (float(base) - float(mae)) / float(base))))
    coverage = 1.0 - float((report["method"] == "unrecoverable").mean()) if len(report) else 0.0
    return (float(np.mean(gains)) if gains else 0.0, coverage)


def _save_best_callback(target_path: Path, metric_name: str, param_transform=None):
    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_trial.number != trial.number:
            return
        params = dict(trial.params)
        if param_transform is not None:
            params = param_transform(params)
        payload = best_payload(
            study_name=study.study_name,
            metric_name=metric_name,
            metric_value=float(study.best_value),
            params=params,
            extra={
                "best_trial_number": trial.number,
                "n_trials_finished": len(study.trials),
            },
        )
        save_json(target_path, payload)
        save_json(checkpoint_path(study.study_name), payload)
    return _callback


def tune_virtual_sensor(trial: optuna.Trial) -> float:
    params = {
        "corr_threshold": trial.suggest_float("corr_threshold", 0.15, 0.55),
        "min_pair_rows": trial.suggest_int("min_pair_rows", 20, 80),
        "q_scale": trial.suggest_float("q_scale", 0.2, 2.5, log=True),
        "r_scale": trial.suggest_float("r_scale", 0.5, 2.5, log=True),
        "r_floor": trial.suggest_float("r_floor", 1e-6, 1e-2, log=True),
        "tier_a_window_ticks": trial.suggest_int("tier_a_window_ticks", 4, 16),
        "tier_a_max_staleness": trial.suggest_float("tier_a_max_staleness", 30.0, 180.0),
        "tier_c_window_checks": trial.suggest_int("tier_c_window_checks", 2, 8),
        "tier_c_max_staleness": trial.suggest_float("tier_c_max_staleness", 600.0, 7200.0),
    }
    features = build_features(verbose=False, params=params)
    X, y, train_mask, test_mask = _feature_split(features)
    X_real = _real_only(X)
    model = DefectModel().fit(X_real[train_mask], y[train_mask])
    out = model.predict_with_confidence(X_real[test_mask])
    y_test = y[test_mask].to_numpy()
    try:
        auc = float(roc_auc_score(y_test, out["risk_score"]))
    except ValueError:
        auc = 0.5
    top20 = build_topk(out["risk_score"].to_numpy(), y_test)[0.20]
    top20_recall = top20[0] / max(1, top20[1])
    sensor_report = validate(build_virtual_sensors(params))
    sensor_gain, coverage = _sensor_gain(sensor_report)
    score = 0.65 * auc + 0.20 * top20_recall + 0.10 * sensor_gain + 0.05 * coverage
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("top20_recall", top20_recall)
    trial.set_user_attr("sensor_gain", sensor_gain)
    trial.set_user_attr("coverage", coverage)
    return float(score)


def build_virtual_sensors(params: dict):
    return fit_virtual_sensors(verbose=False, params=params)


def tune_defect_model(trial: optuna.Trial) -> float:
    ensure_feature_table()
    q_low = trial.suggest_float("threshold_quantile_low", 0.55, 0.85)
    q_high = trial.suggest_float("threshold_quantile_high", max(0.90, q_low + 0.05), 0.9995)
    params = {
        "xgb_params": {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "tree_method": "hist",
            "objective": "binary:logistic",
            "n_jobs": 4,
        },
        "threshold_params": {
            "quantile_low": q_low,
            "quantile_high": q_high,
            "num_thresholds": trial.suggest_int("num_thresholds", 60, 220),
        },
    }

    _, X, y, train_mask, test_mask = load_split()
    X_real = _real_only(X)
    model = DefectModel(xgb_params=params["xgb_params"],
                        threshold_params=params["threshold_params"]).fit(
                            X_real[train_mask], y[train_mask])
    out = model.predict_with_confidence(X_real[test_mask])
    y_test = y[test_mask].to_numpy()
    try:
        auc = float(roc_auc_score(y_test, out["risk_score"]))
    except ValueError:
        auc = 0.5
    top20 = build_topk(out["risk_score"].to_numpy(), y_test)[0.20]
    top20_recall = top20[0] / max(1, top20[1])
    mcc = max(0.0, float(model.metrics.mcc if model.metrics is not None else 0.0))
    score = 0.55 * auc + 0.25 * top20_recall + 0.20 * mcc
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("top20_recall", top20_recall)
    trial.set_user_attr("mcc", mcc)
    return float(score)


def run_study(name: str, objective, n_trials: int, timeout: int | None,
              checkpoint_target: Path, metric_name: str, param_transform=None) -> optuna.Study:
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        storage=optuna_storage_uri(),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_save_best_callback(checkpoint_target, metric_name, param_transform)],
        show_progress_bar=False,
    )
    return study


def _flatten_virtual_payload(params: dict) -> dict:
    return params


def _flatten_defect_payload(params: dict) -> dict:
    q_low = params.pop("threshold_quantile_low")
    q_high = params.pop("threshold_quantile_high")
    num_thresholds = params.pop("num_thresholds")
    return {
        "xgb_params": params,
        "threshold_params": {
            "quantile_low": q_low,
            "quantile_high": q_high,
            "num_thresholds": num_thresholds,
        },
    }


def rebuild_best_features() -> None:
    payload = json.loads(Path(VIRTUAL_SENSOR_PARAMS_PATH).read_text()) \
        if Path(VIRTUAL_SENSOR_PARAMS_PATH).exists() else {"params": {}}
    features = build_features(verbose=True, params=payload.get("params", {}))
    out_path = DATA_DIR / "model_features.csv"
    features.to_csv(out_path, index=False)
    print(f"[write] {out_path}")


def train_final_model() -> None:
    ensure_feature_table()
    _, X, y, train_mask, test_mask = load_split()
    X_real = _real_only(X)
    model = DefectModel().fit(X_real[train_mask], y[train_mask])
    evaluate_holdout(model, X_real[test_mask], y[test_mask], "FINAL tuned production model")
    save_trained_artifacts(model, list(X_real.columns), int(train_mask.sum()), int(test_mask.sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", choices=["virtual", "defect", "all"], default="all")
    ap.add_argument("--virtual-trials", type=int, default=25)
    ap.add_argument("--defect-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None,
                    help="global per-study timeout in seconds")
    ap.add_argument("--skip-final-train", action="store_true",
                    help="tune only; do not rebuild features + save final model")
    args = ap.parse_args()

    if args.study in {"virtual", "all"}:
        study = run_study(
            name="digitaltwin_virtual_sensor",
            objective=tune_virtual_sensor,
            n_trials=args.virtual_trials,
            timeout=args.timeout,
            checkpoint_target=VIRTUAL_SENSOR_PARAMS_PATH,
            metric_name="composite_virtual_score",
            param_transform=_flatten_virtual_payload,
        )
        save_json(
            VIRTUAL_SENSOR_PARAMS_PATH,
            best_payload(
                study_name=study.study_name,
                metric_name="composite_virtual_score",
                metric_value=float(study.best_value),
                params=_flatten_virtual_payload(dict(study.best_trial.params)),
                extra={"best_trial_number": study.best_trial.number},
            ),
        )
        rebuild_best_features()
        print(f"\nBest virtual-sensor score: {study.best_value:.4f}")

    if args.study in {"defect", "all"}:
        study = run_study(
            name="digitaltwin_defect_model",
            objective=tune_defect_model,
            n_trials=args.defect_trials,
            timeout=args.timeout,
            checkpoint_target=DEFECT_PARAMS_PATH,
            metric_name="composite_defect_score",
            param_transform=_flatten_defect_payload,
        )
        save_json(
            DEFECT_PARAMS_PATH,
            best_payload(
                study_name=study.study_name,
                metric_name="composite_defect_score",
                metric_value=float(study.best_value),
                params=_flatten_defect_payload(dict(study.best_trial.params)),
                extra={"best_trial_number": study.best_trial.number},
            ),
        )
        print(f"\nBest defect-model score: {study.best_value:.4f}")

    if not args.skip_final_train:
        rebuild_best_features()
        train_final_model()


if __name__ == "__main__":
    main()
