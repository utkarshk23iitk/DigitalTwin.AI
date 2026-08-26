"""
app.py — DigitalTwin.ai dashboard (the demo).

A Streamlit view of the whole twin on one screen: the simulated line and its
bottleneck, per-unit defect risk, and — the differentiator — the Effective
Trust that gates every action. Three persona tabs (Supervisor / Manager /
Leadership) are three lenses on ONE shared model state, and a trust-policy
slider re-gates the action matrix live so a viewer can see risk-vs-trust
trade-offs move in real time.

Run:
    streamlit run app.py

All numbers come from an actual run of the pipeline (60-session held-out data,
real channels only — the same production model personas.py uses); nothing here
is hard-coded or fabricated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bottleneck_detect import bottleneck_report  # noqa: E402
from defect_model import DefectModel  # noqa: E402
from effective_trust import (AUTO_ACT, HUMAN_VERIFY, MONITOR, PASS,  # noqa: E402
                             assess, gate_actions, load_production_split)
from line_sim import default_line  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "simulated"
NAVY, AMBER, RED, GREEN, GREY = "#10233F", "#F5A623", "#D64545", "#3FA34D", "#8A94A6"
ACTION_COLORS = {AUTO_ACT: RED, HUMAN_VERIFY: AMBER, MONITOR: GREY, PASS: GREEN}


# ------------------------- state (pipeline, cached) ---------------------- #

def _build_state() -> dict:
    """Train the production defect model and assemble everything the dashboard
    renders. Pure logic (no Streamlit calls) so it can be tested headless."""
    features, X, y, train_mask, test_mask = load_production_split()
    model = DefectModel().fit(X[train_mask], y[train_mask])
    importances = pd.Series(model.model.feature_importances_, index=X.columns)

    assessment, risk_thr, _ = assess(model, X[test_mask], importances, trust_thr=0.5)
    meta = features[test_mask][["session_id", "response"]].reset_index()
    assessment = (assessment.reset_index().merge(meta, on="index")
                  .rename(columns={"response": "defect"}))

    line = default_line(seed=999).run(until=8000)
    brep = bottleneck_report(line)
    line_stats = pd.DataFrame(line.station_stats())
    registry = pd.read_csv(DATA_DIR / "station_registry.csv")

    # True held-out AUC on the 10 test sessions (matches the headline metric),
    # not DefectModel's internal train/cal/test split AUC.
    from sklearn.metrics import roc_auc_score
    try:
        held_out_auc = float(roc_auc_score(assessment["defect"], assessment["risk_score"]))
    except ValueError:
        held_out_auc = float("nan")

    return {
        "assessment": assessment, "risk_thr": float(risk_thr),
        "brep": brep, "line_stats": line_stats, "registry": registry,
        "held_out_auc": held_out_auc,
        "n_defects": int(assessment["defect"].sum()),
    }


@st.cache_resource(show_spinner="Training the twin on 60 sessions of line data…")
def get_state() -> dict:
    return _build_state()


def regate(assessment: pd.DataFrame, risk_thr: float, trust_thr: float) -> pd.DataFrame:
    """Re-apply the Risk × Trust matrix at a chosen trust policy (for the slider)."""
    a = assessment.copy()
    a["action"] = gate_actions(a["risk_score"].to_numpy(),
                               a["effective_trust"].to_numpy(), risk_thr, trust_thr)
    return a


def _recall_curve(a: pd.DataFrame) -> pd.DataFrame:
    s = a.sort_values("risk_score", ascending=False)
    n, total = len(s), max(1, int(s["defect"].sum()))
    frac_reviewed = np.arange(1, n + 1) / n
    frac_caught = np.cumsum(s["defect"].to_numpy()) / total
    return pd.DataFrame({"reviewed": frac_reviewed, "caught": frac_caught})


# ------------------------------- personas -------------------------------- #

def render_supervisor(state: dict, a: pd.DataFrame) -> None:
    brep, registry = state["brep"], state["registry"]
    p = brep["primary"]

    c1, c2, c3, c4 = st.columns(4)
    sensor_note = "sensor-poor" if not p["has_sensor"] else "instrumented"
    c1.metric("Primary bottleneck", f"#{p['index']} {p['name']}",
              f"{p['utilisation']*100:.0f}% util · {sensor_note}")
    c2.metric("AUTO-ACT (auto-held)", int((a["action"] == AUTO_ACT).sum()))
    c3.metric("HUMAN-VERIFY (check)", int((a["action"] == HUMAN_VERIFY).sum()))
    c4.metric("Blind spots", int((registry["tier"] != "A").sum()),
              "stations with no/poor sensor")

    st.subheader("The line — utilisation by station")
    ls = state["line_stats"].copy()
    ls["role"] = np.where(ls["index"] == p["index"], "bottleneck",
                          np.where(ls["tier"] != "A", "sensor-poor", "normal"))
    fig = px.bar(ls, x="name", y="utilisation", color="role",
                 color_discrete_map={"bottleneck": RED, "sensor-poor": AMBER,
                                     "normal": NAVY},
                 labels={"utilisation": "utilisation", "name": ""})
    fig.update_layout(height=320, showlegend=True, legend_title_text="",
                      margin=dict(t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Action queue — where risk meets trust")
    left, right = st.columns([3, 2])
    with left:
        fig2 = px.scatter(
            a[a["risk_score"] >= state["risk_thr"] * 0.6], x="effective_trust",
            y="risk_score", color="action", color_discrete_map=ACTION_COLORS,
            hover_data=["input_trust", "model_confidence"],
            labels={"effective_trust": "Effective Trust", "risk_score": "Defect risk"})
        fig2.add_hline(y=state["risk_thr"], line_dash="dot", line_color=GREY)
        fig2.update_layout(height=340, margin=dict(t=10, b=0))
        st.plotly_chart(fig2, width="stretch")
    with right:
        st.caption("High-risk items to verify now (trust too low to auto-act):")
        hv = a[a["action"] == HUMAN_VERIFY].nlargest(8, "risk_score")
        st.dataframe(hv[["risk_score", "input_trust", "model_confidence",
                         "effective_trust"]].round(2), width="stretch")

    blind = registry[registry["tier"] != "A"]
    st.info("**Blind spots:** " + " · ".join(
        f"#{r.station} {r['name']} (tier {r.tier})" for _, r in blind.iterrows())
        + " — decisions resting on these carry lower input-trust.")


def render_manager(state: dict, a: pd.DataFrame) -> None:
    rows = []
    for sess, g in a.groupby("session_id"):
        rows.append({
            "shift": int(sess), "units": len(g), "defects": int(g["defect"].sum()),
            "auto_act": int((g["action"] == AUTO_ACT).sum()),
            "human_verify": int((g["action"] == HUMAN_VERIFY).sum()),
            "mean_trust": round(g["effective_trust"].mean(), 3),
        })
    tbl = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Shifts tracked", len(tbl))
    c2.metric("Defects / shift (avg)", f"{tbl['defects'].mean():.1f}")
    c3.metric("Mean effective trust", f"{a['effective_trust'].mean():.2f}")

    st.subheader("Defects and action volume per shift")
    fig = go.Figure()
    fig.add_bar(x=tbl["shift"], y=tbl["auto_act"], name="AUTO-ACT", marker_color=RED)
    fig.add_bar(x=tbl["shift"], y=tbl["human_verify"], name="HUMAN-VERIFY",
                marker_color=AMBER)
    fig.add_trace(go.Scatter(x=tbl["shift"], y=tbl["defects"], name="actual defects",
                             mode="lines+markers", line=dict(color=NAVY, width=3),
                             yaxis="y2"))
    fig.update_layout(barmode="stack", height=360, margin=dict(t=10, b=0),
                      xaxis_title="shift", yaxis_title="flagged units",
                      yaxis2=dict(title="defects", overlaying="y", side="right"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Mean Effective Trust per shift")
    figt = px.line(tbl, x="shift", y="mean_trust", markers=True)
    figt.update_traces(line_color=AMBER)
    figt.update_layout(height=240, margin=dict(t=10, b=0), yaxis_range=[0, 1])
    st.plotly_chart(figt, width="stretch")
    st.dataframe(tbl, width="stretch", hide_index=True)


def render_leadership(state: dict, a: pd.DataFrame) -> None:
    curve = _recall_curve(a)
    total = max(1, int(a["defect"].sum()))
    caught20 = int(a.nlargest(max(1, int(len(a) * 0.2)), "risk_score")["defect"].sum())
    auto = int((a["action"] == AUTO_ACT).sum())
    human = int((a["action"] == HUMAN_VERIFY).sum())
    autom = auto / max(1, auto + human) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Held-out AUC", f"{state['held_out_auc']:.2f}", "real, stable signal")
    c2.metric("Defects caught early", f"{caught20/total*100:.0f}%",
              "reviewing riskiest 20%")
    c3.metric("Decisions auto-handled", f"{autom:.0f}%", "no human needed")
    c4.metric("Escalated to a person", f"{100-autom:.0f}%", f"{human} units")

    st.subheader("Early-catch lift — defects caught vs. units reviewed")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["reviewed"] * 100, y=curve["caught"] * 100,
                             mode="lines", line=dict(color=NAVY, width=3),
                             name="the twin"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color=GREY, dash="dash"), name="random"))
    fig.add_vline(x=20, line_dash="dot", line_color=AMBER)
    fig.update_layout(height=340, margin=dict(t=10, b=0),
                      xaxis_title="% of units reviewed (riskiest first)",
                      yaxis_title="% of defects caught")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
        **Business context** *(cited anchor, illustrative — not a claimed saving)*
        A stopped automotive line runs **~\\$2.3M/hour** (Siemens *True Cost of
        Downtime 2024*). The twin's value is (a) catching the **{caught20/total*100:.0f}%**
        of defects flagged early before they propagate downstream, and (b)
        surfacing the constraint before it stalls the line. Concrete euro figures
        need plant-specific throughput/scrap data — stated as an assumption,
        never fabricated.
        """)


# --------------------------------- main ---------------------------------- #

def main() -> None:
    st.set_page_config(page_title="DigitalTwin.ai", page_icon="🏭", layout="wide")
    st.title("🏭 DigitalTwin.ai — an assembly-line twin that knows its limits")
    st.caption("Bottleneck detection · defect prediction · **Effective Trust** "
               "gating every action, honest where sensors are missing.")

    state = get_state()

    with st.sidebar:
        st.header("View")
        persona = st.radio("Persona", ["Supervisor", "Manager", "Leadership"])
        st.divider()
        st.header("Trust policy")
        trust_thr = st.slider(
            "Auto-act only above this Effective Trust", 0.0, 1.0, 0.5, 0.05,
            help="Lower = more automation, more risk of acting on shaky data. "
                 "Higher = more sent to humans. Re-gates the action matrix live.")
        st.caption(f"Held-out AUC **{state['held_out_auc']:.2f}** · "
                   f"{state['n_defects']} defects in the held-out set")

    a = regate(state["assessment"], state["risk_thr"], trust_thr)

    if persona == "Supervisor":
        render_supervisor(state, a)
    elif persona == "Manager":
        render_manager(state, a)
    else:
        render_leadership(state, a)


if __name__ == "__main__":
    main()
