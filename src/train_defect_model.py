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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defect_model import DefectModel  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"
NON_FEATURE_COLS = ["session_id", "unit_id", "response", "defect_occurred_at", "defect_caught_at"]


def load_split():
    features = pd.read_csv(DATA_DIR / "model_features.csv")
    with open(DATA_DIR / "manifest.json") as fh:
        manifest = json.load(fh)

    X = features.drop(columns=NON_FEATURE_COLS)
    y = features["response"].astype(int)
    train_mask = features["session_id"].isin(manifest["train_sessions"])
    test_mask = features["session_id"].isin(manifest["test_sessions"])
    return features, X, y, train_mask, test_mask


def main():
    features, X, y, train_mask, test_mask = load_split()

    print(f"train rows: {train_mask.sum()} ({y[train_mask].sum()} defects)")
    print(f"test  rows: {test_mask.sum()} ({y[test_mask].sum()} defects)  "
          f"<- held out, never seen until here\n")

    model = DefectModel().fit(X[train_mask], y[train_mask])
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
    top = importances.sort_values(ascending=False).head(10)
    print("\n=== top 10 feature importances ===")
    for name, val in top.items():
        kind = ("trend (rolling)" if name.endswith(("_mean", "_std", "_slope"))
               else "virtual-sensor estimate" if name.endswith(("_est", "_conf"))
               else "other")
        print(f"  {name:<28} {val:.4f}   [{kind}]")


if __name__ == "__main__":
    main()
