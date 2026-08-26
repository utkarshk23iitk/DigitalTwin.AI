"""
defect_model.py — Station-level defect prediction with calibrated confidence.

Pipeline:
  - Train an XGBoost classifier on Bosch(-faithful) station features.
  - Handle extreme class imbalance with scale_pos_weight + MCC-oriented
    threshold selection (accuracy is meaningless at 0.58% positives).
  - Attach a *calibrated confidence* to every prediction using split-conformal
    style scoring: we hold out a calibration set, then express model confidence
    as how far a prediction's score sits from the decision boundary relative to
    the calibration distribution. This is the "model confidence" input that
    effective_trust.py later fuses with input (data) trust.

Why not raw predict_proba? Gradient-boosted probabilities are poorly calibrated
under heavy imbalance. The conformal-style score gives an honest, monotone
confidence that behaves sensibly near the boundary.

Public API:
    model = DefectModel().fit(X, y)
    out   = model.predict_with_confidence(X_new)
        -> DataFrame[risk_score, prediction, confidence]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def _best_threshold_mcc(y_true, scores):
    """
    Pick an operating threshold maximising MCC over a wide quantile range.
    At extreme imbalance MCC can be degenerate, so we skip thresholds that flag
    nothing. Under-detecting defects is worse than over-flagging (asymmetric
    cost, per the brief), so the wide search leans toward catching positives.
    """
    best_t, best_mcc = 0.5, -1.0
    for t in np.quantile(scores, np.linspace(0.70, 0.999, 120)):
        pred = (scores >= t).astype(int)
        if pred.sum() == 0:
            continue
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t), float(best_mcc)


@dataclass
class DefectMetrics:
    mcc: float
    precision: float
    recall: float
    threshold: float
    auc: float
    n_test: int
    n_pos_test: int


class DefectModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: XGBClassifier | None = None
        self.threshold: float = 0.5
        self.metrics: DefectMetrics | None = None
        self._cal_scores: np.ndarray | None = None   # calibration score dist.
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DefectModel":
        self.feature_names = list(X.columns)
        # train / calibration / test split (stratified on the rare positive)
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.40, random_state=self.seed, stratify=y)
        X_cal, X_te, y_cal, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=self.seed, stratify=y_tmp)

        pos = int(y_tr.sum())
        neg = int((y_tr == 0).sum())
        spw = max(1.0, neg / max(1, pos))   # scale_pos_weight for imbalance

        # Regularised for a rare-event, few-positive problem: shallow trees +
        # a large min_child_weight stop the model memorising individual
        # defects (an unregularised depth-6 fit hit train-AUC 1.0 / held-out
        # 0.55 -- pure overfit). gamma/reg_lambda penalise marginal splits so
        # noise features can't buy their way in.
        self.model = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=10, gamma=1.0, reg_lambda=3.0,
            scale_pos_weight=spw, eval_metric="aucpr",
            missing=np.nan, n_jobs=4, random_state=self.seed,
        )
        self.model.fit(X_tr, y_tr)

        # threshold on calibration set (MCC-optimal)
        cal_scores = self.model.predict_proba(X_cal)[:, 1]
        self.threshold, _ = _best_threshold_mcc(y_cal, cal_scores)
        self._cal_scores = np.sort(cal_scores)

        # evaluate on held-out test set
        te_scores = self.model.predict_proba(X_te)[:, 1]
        te_pred = (te_scores >= self.threshold).astype(int)
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_te, te_scores))
        except Exception:
            auc = float("nan")
        self.metrics = DefectMetrics(
            mcc=float(matthews_corrcoef(y_te, te_pred)),
            precision=float(precision_score(y_te, te_pred, zero_division=0)),
            recall=float(recall_score(y_te, te_pred, zero_division=0)),
            threshold=self.threshold, auc=auc,
            n_test=int(len(y_te)), n_pos_test=int(y_te.sum()),
        )
        return self

    def _confidence(self, scores: np.ndarray) -> np.ndarray:
        """
        Conformal-style confidence in [0,1]: how decisively a score sits away
        from the decision threshold, ranked against the calibration score
        distribution. Scores near the boundary -> low confidence; scores far
        into either region -> high confidence.
        """
        cal = self._cal_scores
        # empirical CDF position of each score within calibration distribution
        pos = np.searchsorted(cal, scores) / max(1, len(cal))
        # distance of that percentile from the threshold's percentile
        thr_pos = np.searchsorted(cal, self.threshold) / max(1, len(cal))
        conf = np.abs(pos - thr_pos) / max(thr_pos, 1 - thr_pos, 1e-9)
        return np.clip(conf, 0.0, 1.0)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model not fitted")
        scores = self.model.predict_proba(X)[:, 1]
        pred = (scores >= self.threshold).astype(int)
        conf = self._confidence(scores)
        return pd.DataFrame({
            "risk_score": np.round(scores, 4),
            "prediction": pred,
            "confidence": np.round(conf, 3),
        }, index=X.index)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
    from get_data import load_defect_data

    X, y, meta = load_defect_data()
    m = DefectModel().fit(X, y)
    met = m.metrics
    print(f"data source : {meta['source']}")
    print(f"test set    : {met.n_test} parts, {met.n_pos_test} defects")
    print(f"MCC         : {met.mcc:.3f}")
    print(f"precision   : {met.precision:.3f}")
    print(f"recall      : {met.recall:.3f}")
    print(f"AUC         : {met.auc:.3f}")
    print(f"threshold   : {met.threshold:.4f}")
    out = m.predict_with_confidence(X.head(8))
    print("\nsample predictions:")
    print(out.to_string())
