"""
personas.py — three stakeholder views over ONE shared model state.

The same underlying signals (bottleneck detection + defect risk + Effective
Trust gating) serve three audiences who each need a different altitude:

  SUPERVISOR  (shop floor, right now)  — what needs a human in the next few
              minutes: the current/forming bottleneck, the specific units to
              hold or check, and which stations are blind spots.
  MANAGER     (this week, trends)      — is the line getting better or worse:
              per-shift defect and action volumes, how much of the decisioning
              leans on inferred (virtual-sensor) data.
  LEADERSHIP  (business, ROI)          — the numbers that justify the system:
              how many defects are caught early, how much is auto-handled vs.
              needs people, and how that maps onto the cost of a stopped line.

Nothing here re-computes the model; each view is a projection of the same
per-unit assessment (from effective_trust.py) plus the bottleneck report. That
is the point — one twin, three lenses, consistent numbers across all of them.

Cost figures in the leadership view are CONTEXT anchors (cited, with stated
assumptions), never fabricated performance claims — see the note there.

    python src/personas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottleneck_detect import bottleneck_report  # noqa: E402
from defect_model import DefectModel  # noqa: E402
from effective_trust import (AUTO_ACT, HUMAN_VERIFY, MONITOR, PASS,  # noqa: E402
                             assess, load_production_split)
from line_sim import default_line  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"

# Cited context anchor (NOT a claimed saving): Siemens "True Cost of Downtime
# 2024" puts a stopped automotive line near $2.3M/hour (~$600/second). Used
# only to frame scale; any euro/dollar figure below is explicitly illustrative.
DOWNTIME_COST_PER_HOUR = 2_300_000


def _recall_at_k(assessment: pd.DataFrame, k: float) -> tuple[int, int]:
    n = max(1, int(round(len(assessment) * k)))
    top = assessment.nlargest(n, "risk_score")
    return int(top["defect"].sum()), int(assessment["defect"].sum())


# ------------------------------- views ---------------------------------- #

def supervisor_view(assessment: pd.DataFrame, registry: pd.DataFrame,
                    brep: dict) -> None:
    print("\n" + "=" * 66)
    print("SUPERVISOR VIEW  —  shop floor, right now")
    print("=" * 66)

    p = brep["primary"]
    tag = "" if p["has_sensor"] else "  [SENSOR-POOR — verdict is inferred]"
    print(f"\n  Primary bottleneck : #{p['index']} {p['name']} "
          f"(util {p['utilisation']*100:.0f}%){tag}")
    if brep["forming"]:
        f = brep["forming"][0]
        print(f"  Forming (watch)    : #{f['index']} {f['name']} "
              f"(queue growing {f['queue_slope']:+.2f}/100s)")

    auto = assessment[assessment["action"] == AUTO_ACT]
    human = assessment[assessment["action"] == HUMAN_VERIFY]
    print(f"\n  Action queue this window:")
    print(f"    AUTO-ACT (held automatically) : {len(auto):>4} units")
    print(f"    HUMAN-VERIFY (please check)   : {len(human):>4} units")
    if len(human):
        print("    top items to check now (high risk, but trust too low to auto-act):")
        for _, r in human.nlargest(3, "risk_score").iterrows():
            print(f"      unit {r.name:<6} risk {r['risk_score']:.2f}  "
                  f"trust {r['effective_trust']:.2f}")

    blind = registry[registry["tier"] != "A"]
    if len(blind):
        names = ", ".join(f"#{r.station} {r['name']} (tier {r.tier})"
                          for _, r in blind.iterrows())
        print(f"\n  Blind spots (no/poor sensor — decisions here carry lower trust):")
        print(f"    {names}")


def manager_view(assessment: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("MANAGER VIEW  —  trend across shifts (each session = one shift)")
    print("=" * 66)

    print(f"\n  {'shift':>5} {'units':>6} {'defects':>8} {'auto':>6} "
          f"{'human':>6} {'meanTrust':>10}")
    for sess, g in assessment.groupby("session_id"):
        print(f"  {int(sess):>5} {len(g):>6} {int(g['defect'].sum()):>8} "
              f"{int((g['action'] == AUTO_ACT).sum()):>6} "
              f"{int((g['action'] == HUMAN_VERIFY).sum()):>6} "
              f"{g['effective_trust'].mean():>10.2f}")

    # How much of the decisioning materially leans on inferred data -- i.e.
    # input trust pulled down far enough to matter (below 0.9), not just a
    # hair under 1.0 because every unit's path touches one imputed station.
    infer_load = (assessment["input_trust"] < 0.9).mean() * 100
    print(f"\n  Decisions materially leaning on inferred data (trust<0.9) : "
          f"{infer_load:.0f}%")
    print(f"  Mean input trust / mean effective trust                   : "
          f"{assessment['input_trust'].mean():.2f} / "
          f"{assessment['effective_trust'].mean():.2f}")


def leadership_view(assessment: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("LEADERSHIP VIEW  —  business case")
    print("=" * 66)

    caught10, total = _recall_at_k(assessment, 0.10)
    caught20, _ = _recall_at_k(assessment, 0.20)
    auto = int((assessment["action"] == AUTO_ACT).sum())
    human = int((assessment["action"] == HUMAN_VERIFY).sum())
    autom = auto / max(1, auto + human) * 100

    print(f"\n  Defects caught EARLY (before final QC):")
    print(f"    reviewing the riskiest 10% of units catches {caught10}/{total} "
          f"({caught10/max(1,total)*100:.0f}%) of defects")
    print(f"    riskiest 20%                             catches {caught20}/{total} "
          f"({caught20/max(1,total)*100:.0f}%)")

    print(f"\n  Decision automation (of all high-risk flags, trust policy >= 0.5):")
    print(f"    auto-handled (no human needed) : {autom:.0f}%  ({auto} units)")
    print(f"    escalated to a person          : {100-autom:.0f}%  "
          f"({human} units) — the trust gate keeps a human on the shaky calls")

    print(f"\n  Business context (cited anchor, illustrative — not a claimed saving):")
    print(f"    A stopped automotive line runs ~${DOWNTIME_COST_PER_HOUR:,}/hour "
          f"(Siemens True Cost of Downtime 2024).")
    print(f"    Value comes from (a) catching the ~{caught20/max(1,total)*100:.0f}% of "
          f"defects flagged early before they propagate downstream, and")
    print(f"    (b) flagging the constraint before it stalls the line. Concrete "
          f"euro figures require plant-specific throughput/scrap data (stated as")
    print(f"    an assumption, never fabricated).")


def main() -> None:
    features, X, y, train_mask, test_mask = load_production_split()
    print(f"training on {int(train_mask.sum())} units, assessing "
          f"{int(test_mask.sum())} held-out units...")
    model = DefectModel().fit(X[train_mask], y[train_mask])
    importances = pd.Series(model.model.feature_importances_, index=X.columns)

    # Fixed, interpretable trust policy: auto-act only when combined trust
    # clears 0.5. (effective_trust.py's own validation uses a median split to
    # balance the precision comparison; an operational KPI needs a real, fixed
    # bar, otherwise the auto/human ratio is a tautology.)
    assessment, _, _ = assess(model, X[test_mask], importances, trust_thr=0.5)
    meta = features[test_mask][["session_id", "response"]].reset_index()
    assessment = assessment.reset_index().merge(meta, on="index").set_index("index")
    assessment = assessment.rename(columns={"response": "defect"})

    # A representative "live" run for the current bottleneck picture.
    line = default_line(seed=999).run(until=8000)
    brep = bottleneck_report(line)
    registry = pd.read_csv(DATA_DIR / "station_registry.csv")

    supervisor_view(assessment, registry, brep)
    manager_view(assessment)
    leadership_view(assessment)


if __name__ == "__main__":
    main()
