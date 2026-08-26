"""
effective_trust.py — fuse input-trust x model-confidence, then gate the action.

The differentiator of this twin is that it does not treat every prediction as
equally trustworthy. Two independent things can be weak:

  1. INPUT TRUST  — is the data feeding this prediction reliable? A unit whose
     evidence is all real, directly-measured sensors is fully trusted; one that
     leans on a *virtual-sensor estimate* is only as trustworthy as that
     estimate's confidence (from virtual_sensor.py / the *_conf columns).
  2. MODEL CONFIDENCE — is the model itself sure? The conformal-style score
     from defect_model.py: how decisively the risk sits away from the boundary.

    Effective Trust = input_trust  x  model_confidence      (MULTIPLY, not average)

We multiply on purpose. If either factor is weak, effective trust must collapse
-- a confident prediction built on an unreliable inferred reading is NOT
trustworthy, and vice versa. Averaging (0.9 + 0.1)/2 = 0.5 would hide the weak
factor; multiplying 0.9 x 0.1 = 0.09 surfaces it, which is the honest behaviour.

Effective Trust then GATES the action via the Risk x Trust matrix:

                    high TRUST                 low TRUST
   high RISK   -> AUTO-ACT (auto reject     HUMAN-VERIFY (flag, but a person
                  / hold the part)           checks before acting)
   low  RISK   -> PASS (let it through)     MONITOR (probably fine, but the
                                             data is shaky -- keep watching)

This is deliberately NOT "act on risk alone": a high-risk flag built on
low-trust data goes to a human, not an automatic reject. That is what makes the
twin safe to wire to an actuator.

Blind spots: a station that is *unrecoverable* (no sensor of its own and no
correlated neighbour -- e.g. tier-B S9 on the current data) contributes no
features at all. The model cannot lean on it, so it cannot inflate trust with a
confident-looking guess about it -- the honest failure mode.

Run standalone to see the gating validated on held-out data:
    python src/effective_trust.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defect_model import DefectModel  # noqa: E402
from train_defect_model import (_channel_of_feature, _load_decoy_channels,  # noqa: E402
                                load_split)


def load_production_split():
    """load_split, then drop the decoy-derived features. The production model
    uses REAL channels only -- feature selection (permutation importance +
    ablation, in train_defect_model.py) showed the decoys add no value and
    dropping them improves held-out AUC. Every downstream module (this one,
    personas.py, the dashboard) trains on this same real-only set so the
    numbers are consistent with the headline metrics."""
    features, X, y, train_mask, test_mask = load_split()
    decoys = _load_decoy_channels()
    real_cols = [c for c in X.columns if _channel_of_feature(c) not in decoys]
    return features, X[real_cols], y, train_mask, test_mask

# Action labels (the Risk x Trust matrix cells).
AUTO_ACT = "AUTO-ACT"          # high risk, high trust -> automatic hold/reject
HUMAN_VERIFY = "HUMAN-VERIFY"  # high risk, low trust  -> flag for a person
MONITOR = "MONITOR"            # low risk, low trust   -> pass but keep watching
PASS = "PASS"                  # low risk, high trust  -> let it through


def compute_input_trust(X: pd.DataFrame, importances: pd.Series) -> np.ndarray:
    """Per-unit input trust in [0, 1]: an importance-weighted average of each
    feature's own trust, over the features actually present for that unit.

    A feature's trust is 1.0 if it is a real, directly-measured reading, or the
    virtual-sensor's own confidence (the paired *_conf value) if it is an
    inferred *_est feature. Weighting by importance means a low-confidence
    estimate only dents input trust to the extent the model actually relies on
    that feature -- an unreliable reading the model ignores shouldn't matter.
    """
    cols = list(X.columns)
    imp = importances.reindex(cols).fillna(0.0).to_numpy()
    present = X.notna().to_numpy()                    # (N, F)
    trust = np.ones(X.shape, dtype=float)             # (N, F), real -> 1.0

    for j, col in enumerate(cols):
        if col.endswith("_est"):
            conf_col = col[:-4] + "_conf"
            if conf_col in X.columns:
                c = np.clip(X[conf_col].to_numpy(dtype=float), 0.0, 1.0)
                trust[:, j] = np.nan_to_num(c, nan=0.0)

    num = (present * trust) @ imp                     # weighted, present-only
    den = present @ imp
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, 1.0)       # no weighted evidence -> neutral
    return np.clip(out, 0.0, 1.0)


def fuse_trust(input_trust: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
    """Effective Trust = input_trust x model_confidence (multiply, not average)."""
    return np.clip(input_trust, 0, 1) * np.clip(model_confidence, 0, 1)


def gate_actions(risk_score: np.ndarray, effective_trust: np.ndarray,
                 risk_thr: float, trust_thr: float) -> np.ndarray:
    """Map (risk, effective-trust) to an action label via the Risk x Trust matrix."""
    risk_high = risk_score >= risk_thr
    trust_high = effective_trust >= trust_thr
    out = np.empty(len(risk_score), dtype=object)
    out[risk_high & trust_high] = AUTO_ACT
    out[risk_high & ~trust_high] = HUMAN_VERIFY
    out[~risk_high & ~trust_high] = MONITOR
    out[~risk_high & trust_high] = PASS
    return out


def assess(model: DefectModel, X: pd.DataFrame, importances: pd.Series,
           trust_thr: float | None = None) -> pd.DataFrame:
    """Full per-unit assessment: risk, confidence, input trust, effective trust,
    and the gated action. If trust_thr is None it defaults to the median
    effective trust among high-risk units, so the high-risk band splits into a
    balanced auto-act / human-verify comparison (a tunable operating point)."""
    out = model.predict_with_confidence(X)
    risk = out["risk_score"].to_numpy()
    conf = out["confidence"].to_numpy()
    inp = compute_input_trust(X, importances)
    eff = fuse_trust(inp, conf)

    risk_thr = model.threshold
    if trust_thr is None:
        hi = eff[risk >= risk_thr]
        trust_thr = float(np.median(hi)) if len(hi) else 0.5
    actions = gate_actions(risk, eff, risk_thr, trust_thr)

    return pd.DataFrame({
        "risk_score": np.round(risk, 4),
        "model_confidence": np.round(conf, 3),
        "input_trust": np.round(inp, 3),
        "effective_trust": np.round(eff, 3),
        "action": actions,
    }, index=X.index), risk_thr, trust_thr


def _report(assessment: pd.DataFrame, y_true: np.ndarray,
            risk_thr: float, trust_thr: float) -> None:
    print(f"\noperating point: risk >= {risk_thr:.4f}, trust >= {trust_thr:.3f}")
    print(f"held-out units: {len(assessment)}  defects: {int(y_true.sum())}")

    print("\n=== action distribution ===")
    for act in (AUTO_ACT, HUMAN_VERIFY, MONITOR, PASS):
        m = (assessment["action"] == act).to_numpy()
        n = int(m.sum())
        if n == 0:
            print(f"  {act:<13} {n:>6}")
            continue
        dr = y_true[m].mean() * 100
        print(f"  {act:<13} {n:>6} units   defect rate {dr:5.2f}%  "
              f"({int(y_true[m].sum())} defects)")

    # The payoff: among HIGH-RISK units, does trust separate reliable flags from
    # shaky ones? Auto-act (high trust) should be a cleaner catch than the
    # human-verify (low trust) pile -- otherwise the trust score is not earning
    # its place in the loop.
    aa = (assessment["action"] == AUTO_ACT).to_numpy()
    hv = (assessment["action"] == HUMAN_VERIFY).to_numpy()
    base = y_true.mean() * 100
    print("\n=== does trust gate correctly? (precision within high-risk) ===")
    print(f"  baseline defect rate (all units)     : {base:5.2f}%")
    if aa.sum():
        print(f"  AUTO-ACT   (high risk + HIGH trust)  : {y_true[aa].mean()*100:5.2f}%  "
              f"<- should be the cleanest catch")
    if hv.sum():
        print(f"  HUMAN-VERIFY (high risk + LOW trust) : {y_true[hv].mean()*100:5.2f}%  "
              f"<- shakier; routed to a person")
    if aa.sum() and hv.sum():
        lift = y_true[aa].mean() / max(1e-9, y_true[hv].mean())
        print(f"  -> auto-act is {lift:.2f}x as precise as human-verify")

    # Illustrate multiply-not-average on the actual data: units where one factor
    # is weak but the other strong -- averaging would call these trustworthy.
    a = assessment
    masked = a[(a["input_trust"] < 0.5) | (a["model_confidence"] < 0.5)]
    if len(masked):
        avg = (masked["input_trust"] + masked["model_confidence"]) / 2
        hidden = int((avg >= 0.5).sum())
        print(f"\n  multiply-not-average: {len(masked)} units have one weak factor; "
              f"averaging would still call {hidden} of them >=0.5 trust, "
              f"multiplying calls {int((masked['effective_trust']>=0.5).sum())}.")


def main() -> None:
    features, X, y, train_mask, test_mask = load_production_split()
    print(f"training defect model on {int(train_mask.sum())} units "
          f"({int(y[train_mask].sum())} defects)...")
    model = DefectModel().fit(X[train_mask], y[train_mask])
    importances = pd.Series(model.model.feature_importances_, index=X.columns)

    assessment, risk_thr, trust_thr = assess(model, X[test_mask], importances)
    y_test = y[test_mask].to_numpy()
    _report(assessment, y_test, risk_thr, trust_thr)

    print("\n=== sample assessments (highest risk first) ===")
    top = assessment.sort_values("risk_score", ascending=False).head(8)
    print(top.to_string())


if __name__ == "__main__":
    main()
