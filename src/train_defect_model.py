"""
train_defect_model.py — retrain DefectModel on the trend-aware, tier-complete
feature table from feature_engineering.py, with an honest chronological
holdout.

This is deliberately separate from defect_model.py's own __main__ (which
demos the class against the Bosch-faithful get_data.py table for the
offline-notebook story). Here the model trains on data/simulated/'s TRAIN
sessions only; the held-out TEST session is never touched until final
scoring -- DefectModel.fit()'s internal calibration split happens entirely
within the train sessions, so nothing about the reported metrics below has
seen session 14 in any form.

Usage:
    python src/train_defect_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defect_model import DefectModel, load_tuned_defect_config  # noqa: E402
from tuning_config import (TRAINED_MODEL_META_PATH, TRAINED_MODEL_PATH,  # noqa: E402
                           ensure_artifact_dirs, save_json)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"
NON_FEATURE_COLS = ["session_id", "unit_id", "response", "defect_occurred_at", "defect_caught_at"]
PERM_REPEATS = 30       # shuffles per feature; more = stabler, slower


def _channel_of_feature(feat: str) -> str:
    """S{station}_{channel}_{suffix} -> channel. Station and channel are each a
    single underscore-free token (torque, humidity, ...), so index 1 is it."""
    parts = feat.split("_")
    return parts[1] if len(parts) >= 2 else feat


def _load_decoy_channels() -> set[str]:
    """Ground-truth set of irrelevant channels, or empty if not tagged (older
    datasets generated before decoys existed)."""
    path = DATA_DIR / "channel_registry.csv"
    if not path.exists():
        return set()
    reg = pd.read_csv(path)
    return set(reg.loc[reg["is_decoy"], "channel"])


def report_decoy_separation(importances: pd.Series, decoy_channels: set[str]) -> None:
    """Did the model correctly learn to ignore the irrelevant (decoy) channels?

    We know the ground truth (channel_registry.csv), so we can score feature
    selection directly: real channels should hold almost all the importance,
    and a simple 'drop everything below median importance' rule should drop
    the decoys while keeping the real signal. Channel-level headline: of the
    Q decoy channel-instances, how many land in the low-importance half we'd
    ignore."""
    if not decoy_channels:
        print("\n(no channel_registry.csv -- regenerate data to score decoy separation)")
        return

    is_decoy = importances.index.map(lambda f: _channel_of_feature(f) in decoy_channels)
    is_decoy = pd.Series(is_decoy, index=importances.index)
    real_imp, decoy_imp = importances[~is_decoy], importances[is_decoy]

    print("\n=== decoy separation (feature-selection sanity check) ===")
    print(f"  feature columns: {len(importances)}  "
          f"(real-derived {int((~is_decoy).sum())}, decoy-derived {int(is_decoy.sum())})")
    print(f"  importance held by real-derived  : {real_imp.sum() * 100:5.1f}%")
    print(f"  importance held by decoy-derived : {decoy_imp.sum() * 100:5.1f}%")

    # Feature-level: a plain 'drop below median importance' selection rule.
    thr = float(importances.median())
    decoy_dropped = int((is_decoy & (importances < thr)).sum())
    real_kept = int((~is_decoy & (importances >= thr)).sum())
    print(f"  drop-below-median rule (thr={thr:.4f}):")
    print(f"    decoy-derived features dropped : {decoy_dropped}/{int(is_decoy.sum())}")
    print(f"    real-derived features kept     : {real_kept}/{int((~is_decoy).sum())}")

    # Channel-instance headline: group importance by the S{st}_{ch} prefix,
    # flag the bottom half as 'would ignore', and see how many are truly decoy.
    inst = importances.groupby(
        importances.index.map(lambda f: "_".join(f.split("_")[:2]))).sum()
    inst_is_decoy = inst.index.map(lambda ci: ci.split("_")[1] in decoy_channels)
    inst_is_decoy = pd.Series(inst_is_decoy, index=inst.index)
    flagged_low = inst < inst.median()
    caught = int((flagged_low & inst_is_decoy).sum())
    total = int(inst_is_decoy.sum())
    print(f"  channel instances flagged low-value that are truly decoys: "
          f"{caught}/{total}")


def _decoy_feature_mask(columns, decoy_channels: set[str]) -> np.ndarray:
    """Boolean array: True where a feature column derives from a decoy channel."""
    return np.array([_channel_of_feature(c) in decoy_channels for c in columns])


def report_permutation_importance(model: DefectModel, X_test: pd.DataFrame,
                                  y_test: pd.Series, decoy_channels: set[str]) -> None:
    """Permutation importance on the HELD-OUT sessions: shuffle each feature and
    measure the drop in test AUC. Unlike gain importance (measured on train, and
    fooled by noisy splits), this asks 'does this feature actually help on data
    the model has never seen?' -- so genuine decoys should score ~0."""
    if not decoy_channels:
        print("\n(no channel_registry.csv -- skipping permutation importance)")
        return
    print(f"\n=== permutation importance on held-out set ({PERM_REPEATS} shuffles) ===")
    result = permutation_importance(
        model.model, X_test, y_test, scoring="roc_auc",
        n_repeats=PERM_REPEATS, random_state=0, n_jobs=4)
    imp = pd.Series(result.importances_mean, index=X_test.columns)
    is_decoy = _decoy_feature_mask(X_test.columns, decoy_channels)
    real_imp, decoy_imp = imp[~is_decoy], imp[is_decoy]

    print(f"  mean AUC-drop when shuffled -- real features : {real_imp.mean():+.4f}")
    print(f"                                 decoy features : {decoy_imp.mean():+.4f}")
    print(f"  real features with a positive (helpful) score : "
          f"{int((real_imp > 0).sum())}/{len(real_imp)}")
    print(f"  decoy features scoring ~0 or negative (ignored): "
          f"{int((decoy_imp <= 0).sum())}/{len(decoy_imp)}")

    # Top real vs any decoy that leaked in -- the honest side-by-side.
    top = imp.sort_values(ascending=False).head(8)
    print("  most-useful features by permutation importance:")
    for name, val in top.items():
        tag = "  <-- DECOY" if _channel_of_feature(name) in decoy_channels else ""
        print(f"    {name:<26} {val:+.4f}{tag}")


def build_topk(scores: np.ndarray, y_test: np.ndarray) -> dict:
    order = np.argsort(-scores)
    y_sorted = y_test[order]
    n, total_pos = len(y_sorted), int(y_test.sum())
    out = {}
    for k in (0.01, 0.05, 0.10, 0.20):
        k_n = max(1, int(round(n * k)))
        out[k] = (int(y_sorted[:k_n].sum()), total_pos)
    return out


def evaluate_holdout(model: DefectModel, X_test: pd.DataFrame, y_test_s: pd.Series,
                     title: str) -> dict:
    """Score a fitted model on the held-out sessions; print + return key metrics."""
    out = model.predict_with_confidence(X_test)
    y_test = y_test_s.to_numpy()
    pred = out["prediction"].to_numpy()
    scores = out["risk_score"].to_numpy()
    try:
        auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = float("nan")
    topk = build_topk(scores, y_test)
    print(f"\n=== {title} ===")
    print(f"  AUC={auc:.3f}  MCC={matthews_corrcoef(y_test, pred):.3f}  "
          f"recall={recall_score(y_test, pred, zero_division=0):.3f}  "
          f"n_pos={int(y_test.sum())}")
    print("  recall at top-K% by risk score:")
    for k, (caught, total) in topk.items():
        print(f"    top {int(k*100):>2}% : {caught}/{total} defects "
              f"({caught / max(1, total) * 100:.1f}%)")
    return {
        "auc": float(auc),
        "mcc": float(matthews_corrcoef(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "threshold": float(model.threshold),
        "n_test": int(len(y_test)),
        "n_pos_test": int(y_test.sum()),
        "topk": topk,
    }


def load_split():
    features = pd.read_csv(DATA_DIR / "model_features.csv")
    with open(DATA_DIR / "manifest.json") as fh:
        manifest = json.load(fh)

    X = features.drop(columns=NON_FEATURE_COLS)
    y = features["response"].astype(int)
    development_sessions = manifest["train_sessions"] + manifest.get("validation_sessions", [])
    train_mask = features["session_id"].isin(development_sessions)
    test_mask = features["session_id"].isin(manifest["test_sessions"])
    return features, X, y, train_mask, test_mask


def save_trained_artifacts(model: DefectModel, feature_columns: list[str],
                           train_rows: int, test_rows: int,
                           holdout_metrics: dict | None = None) -> None:
    ensure_artifact_dirs()
    model.model.save_model(TRAINED_MODEL_PATH)
    meta = {
        "threshold": model.threshold,
        "feature_names": feature_columns,
        "xgb_params": model.xgb_params,
        "threshold_params": model.threshold_params,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "metrics": holdout_metrics or (vars(model.metrics) if model.metrics is not None else None),
        "internal_calibration_metrics": (
            vars(model.metrics) if model.metrics is not None else None
        ),
        "evaluation_protocol": (
            "Chronological session holdout evaluated once after model and threshold selection."
        ),
    }
    save_json(TRAINED_MODEL_META_PATH, meta)
    print(f"\n[write] {TRAINED_MODEL_PATH}")
    print(f"[write] {TRAINED_MODEL_META_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-tuned", action="store_true",
                    help="load best saved Optuna params from artifacts/tuning/")
    args = ap.parse_args()

    features, X, y, train_mask, test_mask = load_split()

    print(f"train rows: {train_mask.sum()} ({y[train_mask].sum()} defects)")
    print(f"test  rows: {test_mask.sum()} ({y[test_mask].sum()} defects)  "
          f"<- held out, never seen until here\n")

    tuned = load_tuned_defect_config() if args.use_tuned else {
        "xgb_params": None,
        "threshold_params": None,
    }
    model = DefectModel(xgb_params=tuned["xgb_params"],
                        threshold_params=tuned["threshold_params"],
                        auto_load_tuned=args.use_tuned).fit(
                            X[train_mask], y[train_mask])
    print("--- internal calibration-split metrics (within train sessions only) ---")
    m = model.metrics
    print(f"  MCC={m.mcc:.3f}  precision={m.precision:.3f}  recall={m.recall:.3f}  "
          f"AUC={m.auc:.3f}  n_test={m.n_test}  n_pos={m.n_pos_test}\n")

    out = model.predict_with_confidence(X[test_mask])
    y_test = y[test_mask].to_numpy()
    pred = out["prediction"].to_numpy()

    mcc = matthews_corrcoef(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, out["risk_score"])
    except Exception:
        auc = float("nan")

    print("=== HELD-OUT test session(s), never touched before now ===")
    print(f"  MCC (fixed threshold) : {mcc:.3f}")
    print(f"  precision             : {prec:.3f}")
    print(f"  recall                : {rec:.3f}")
    print(f"  AUC                   : {auc:.3f}")
    print(f"  n_test                : {len(y_test)}   n_pos: {int(y_test.sum())}")

    # A single fixed threshold is fragile with this few positives -- one flip
    # either way swings precision/recall by several points. Recall-at-top-K%
    # is the standard, more stable way to report rare-event ranking quality:
    # "if we act on/review the riskiest K% of units, what fraction of real
    # defects do we actually catch?" This is also the natural language for
    # effective_trust.py's action-gating -- it consumes a continuous risk
    # score + confidence, not a hard yes/no.
    scores = out["risk_score"].to_numpy()
    order = np.argsort(-scores)
    y_sorted = y_test[order]
    n = len(y_sorted)
    print("\n  recall at top-K% by risk score (rank-based, threshold-free):")
    for k in (0.01, 0.05, 0.10, 0.20):
        k_n = max(1, int(round(n * k)))
        caught = int(y_sorted[:k_n].sum())
        total_pos = int(y_test.sum())
        print(f"    top {int(k*100):>2}% ({k_n:>4} units): catches {caught}/{total_pos} "
              f"defects ({caught / max(1, total_pos) * 100:.1f}%)")

    # Delayed-discovery breakdown: of the true positives in the held-out
    # session, how many were caught at a LATER station than where the
    # defect actually occurred -- and did the model still catch them from
    # the full-trip trend features?
    test_feats = features[test_mask].reset_index(drop=True)
    tp_mask = (y_test == 1) & (pred == 1)
    fn_mask = (y_test == 1) & (pred == 0)
    occurred = test_feats["defect_occurred_at"]
    caught = test_feats["defect_caught_at"]
    delayed = (caught != occurred) & occurred.notna()
    print(f"\n  delayed-discovery defects in test session : {int(delayed.sum())} "
          f"of {int(y_test.sum())}")
    print(f"  of those, caught by the model (TP)         : "
          f"{int((delayed.to_numpy() & tp_mask).sum())}")
    print(f"  of those, missed by the model (FN)         : "
          f"{int((delayed.to_numpy() & fn_mask).sum())}")

    # Feature importance: are trend/imputed features actually earning their
    # keep, or is the model just leaning on raw tier-A means?
    importances = pd.Series(model.model.feature_importances_, index=X.columns)
    decoy_channels = _load_decoy_channels()
    top = importances.sort_values(ascending=False).head(10)
    print("\n=== top 10 feature importances ===")
    for name, val in top.items():
        kind = ("trend (rolling)" if name.endswith(("_mean", "_std", "_slope"))
               else "virtual-sensor estimate" if name.endswith(("_est", "_conf"))
               else "other")
        tag = "  <-- DECOY" if _channel_of_feature(name) in decoy_channels else ""
        print(f"  {name:<28} {val:.4f}   [{kind}]{tag}")

    report_decoy_separation(importances, decoy_channels)

    # The honest selector: permutation importance on held-out data, which
    # gain importance (measured on train) could not deliver.
    report_permutation_importance(model, X[test_mask], y[test_mask], decoy_channels)

    # Payoff: drop the decoy-derived features, retrain on real channels only,
    # and compare held-out performance. A cleaner model with no noise to overfit.
    if decoy_channels:
        decoy_mask = _decoy_feature_mask(X.columns, decoy_channels)
        real_cols = X.columns[~decoy_mask]
        print(f"\n=== drop {int(decoy_mask.sum())} decoy features -> retrain on "
              f"{len(real_cols)} real features ===")
        before = evaluate_holdout(model, X[test_mask], y[test_mask],
                                  "WITH decoys (current model)")
        model_real = DefectModel(
            xgb_params=tuned["xgb_params"],
            threshold_params=tuned["threshold_params"],
            auto_load_tuned=args.use_tuned,
        ).fit(X.loc[train_mask, real_cols], y[train_mask])
        after = evaluate_holdout(model_real, X.loc[test_mask, real_cols], y[test_mask],
                                 "WITHOUT decoys (real channels only)")
        b10, a10 = before["topk"][0.10], after["topk"][0.10]
        b20, a20 = before["topk"][0.20], after["topk"][0.20]
        print(f"\n  summary  AUC {before['auc']:.3f} -> {after['auc']:.3f}   "
              f"top10% {b10[0]}/{b10[1]} -> {a10[0]}/{a10[1]}   "
              f"top20% {b20[0]}/{b20[1]} -> {a20[0]}/{a20[1]}")
        save_trained_artifacts(
            model_real, list(real_cols), int(train_mask.sum()), int(test_mask.sum()),
            holdout_metrics={key: value for key, value in after.items() if key != "topk"},
        )
    else:
        final_metrics = evaluate_holdout(model, X[test_mask], y[test_mask], "FINAL production model")
        save_trained_artifacts(
            model, list(X.columns), int(train_mask.sum()), int(test_mask.sum()),
            holdout_metrics={key: value for key, value in final_metrics.items() if key != "topk"},
        )


if __name__ == "__main__":
    main()
