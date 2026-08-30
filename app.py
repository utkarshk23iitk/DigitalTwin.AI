"""
app.py — DigitalTwin.ai dashboard (the demo).

A Streamlit control-room view of the twin: live simulated line playback,
station-by-station bottleneck pressure, per-unit defect risk, and the
Effective Trust gate that decides whether the system acts automatically or
hands the case to a person.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
import json
import ast

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
from line_sim import BLOCKED, DOWN, STARVED, WORKING, default_line  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "simulated"
DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo_live"
DATA_SCRIPT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DEMO_DURATION = 8000.0
DEFAULT_DEMO_SEED = 999
DEMO_BASE_FILES = [
    DEMO_DIR / "line_events.csv",
    DEMO_DIR / "buffer_history.csv",
    DEMO_DIR / "health_log.csv",
    DEMO_DIR / "sensor_log.csv",
    DEMO_DIR / "station_registry.csv",
    DEMO_DIR / "station_stats.csv",
    DEMO_DIR / "unit_features.csv",
    DEMO_DIR / "unit_visit_times.csv",
    DEMO_DIR / "bottleneck_report.json",
    DEMO_DIR / "manifest.json",
]
DEMO_INFERENCE_INPUTS = [
    DEMO_DIR / "unit_features.csv",
    DEMO_DIR / "station_registry.csv",
    DEMO_DIR / "sensor_log.csv",
    DEMO_DIR / "unit_visit_times.csv",
]
DEMO_INFERENCE_OUTPUTS = [
    DEMO_DIR / "model_features.csv",
    DEMO_DIR / "demo_assessment.csv",
    DEMO_DIR / "virtual_sensor_events.csv",
]
NAVY = "#10233F"
AMBER = "#F5A623"
RED = "#D64545"
GREEN = "#3FA34D"
GREY = "#8A94A6"
SLATE = "#22324B"
CREAM = "#F7F3EA"
STATE_COLORS = {
    WORKING: GREEN,
    BLOCKED: RED,
    STARVED: GREY,
    DOWN: AMBER,
    "STARTUP": "#5E6C84",
}
ACTION_COLORS = {AUTO_ACT: RED, HUMAN_VERIFY: AMBER, MONITOR: GREY, PASS: GREEN}


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-size: 18px;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245,166,35,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(16,35,63,0.16), transparent 32%),
                linear-gradient(180deg, #f7f3ea 0%, #f2efe6 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        p, li, label, .stMarkdown, .stCaption, .stRadio, .stSelectbox, .stSlider {
            font-size: 1rem !important;
        }
        h1 {
            font-size: 2.35rem !important;
        }
        h2, h3 {
            font-size: 1.45rem !important;
        }
        .ops-banner {
            border: 1px solid rgba(16,35,63,0.10);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            background: linear-gradient(135deg, rgba(16,35,63,0.98), rgba(34,50,75,0.94));
            color: #f7f3ea;
            box-shadow: 0 18px 40px rgba(16,35,63,0.12);
            margin-bottom: 0.9rem;
        }
        .ops-banner h3 {
            margin: 0;
            color: #f7f3ea;
            letter-spacing: 0.02em;
        }
        .ops-banner p {
            margin: 0.35rem 0 0 0;
            color: rgba(247,243,234,0.82);
            font-size: 1rem;
        }
        .ops-card {
            border: 1px solid rgba(16,35,63,0.10);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(6px);
            box-shadow: 0 10px 28px rgba(16,35,63,0.06);
            min-height: 122px;
        }
        .ops-label {
            color: #5b6678;
            font-size: 0.88rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .ops-value {
            color: #10233F;
            font-size: 1.95rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .ops-sub {
            color: #465468;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .explain-box {
            border-left: 6px solid #10233F;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: rgba(255,255,255,0.78);
            box-shadow: 0 10px 24px rgba(16,35,63,0.05);
            margin-bottom: 1rem;
        }
        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(16,35,63,0.10);
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            color: #22324B;
            font-size: 0.95rem;
        }
        .legend-dot {
            width: 0.85rem;
            height: 0.85rem;
            border-radius: 999px;
            display: inline-block;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(16,35,63,0.08);
            padding: 0.8rem 0.9rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(16,35,63,0.05);
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.98rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(247,243,234,0.98));
            border-right: 1px solid rgba(16,35,63,0.08);
        }
        section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(16,35,63,0.06);
            padding: 0.35rem 0.75rem;
            transition: all 0.2s ease;
        }
        section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
            background: #10233F;
            color: #F7F3EA;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(16,35,63,0.12);
            border-radius: 14px;
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.45rem;
            box-shadow: 0 8px 18px rgba(16,35,63,0.04);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: #10233F;
            border-color: #10233F;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {
            color: #F7F3EA !important;
        }
        .station-card {
            border: 1px solid rgba(16,35,63,0.12);
            border-radius: 20px;
            padding: 0.85rem 0.95rem;
            background: rgba(255,255,255,0.74);
            box-shadow: 0 10px 24px rgba(16,35,63,0.05);
            min-height: 118px;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .station-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(16,35,63,0.10);
            border-color: rgba(16,35,63,0.24);
        }
        .station-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #10233F;
            margin-bottom: 0.2rem;
        }
        .station-card-meta {
            font-size: 0.92rem;
            color: #5B6678;
            margin-bottom: 0.45rem;
        }
        .station-card-state {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.85rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.45rem;
        }
        .station-card-copy {
            font-size: 0.92rem;
            color: #22324B;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stage_of(index: int, total: int) -> str:
    third = max(1, total // 3)
    if index < third:
        return "Body"
    if index < 2 * third:
        return "Paint"
    return "Final"


def _buffer_history_frame(line) -> pd.DataFrame:
    rows = []
    n = len(line.cfg.stations)
    for rec in line.buffer_log:
        row = {"t": float(rec["t"])}
        for i in range(n):
            row[f"buffer_{i}"] = int(rec["levels"][i])
        rows.append(row)
    return pd.DataFrame(rows)


def _latest_mtime(paths: list[Path]) -> float:
    return max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)


def _ensure_demo_assets() -> None:
    if str(DATA_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_SCRIPT_DIR))

    if not all(path.exists() for path in DEMO_BASE_FILES):
        from generate_demo_data import generate_demo  # noqa: WPS433
        generate_demo(DEFAULT_DEMO_DURATION, DEFAULT_DEMO_SEED, DEMO_DIR)

    outputs_missing = not all(path.exists() for path in DEMO_INFERENCE_OUTPUTS)
    outputs_stale = _latest_mtime(DEMO_INFERENCE_OUTPUTS) < _latest_mtime(DEMO_INFERENCE_INPUTS)
    if outputs_missing or outputs_stale:
        from build_demo_inference import build_demo_inference  # noqa: WPS433
        build_demo_inference(verbose=False)


def _load_demo_stream() -> dict | None:
    if not all(path.exists() for path in DEMO_BASE_FILES):
        return None

    line_events = pd.read_csv(DEMO_DIR / "line_events.csv")
    buffer_history = pd.read_csv(DEMO_DIR / "buffer_history.csv")
    if "levels" in buffer_history.columns:
        rows = []
        for rec in buffer_history.to_dict(orient="records"):
            row = {"t": float(rec["t"])}
            levels = rec.get("levels", [])
            if isinstance(levels, str):
                levels = ast.literal_eval(levels)
            for i, level in enumerate(levels):
                row[f"buffer_{i}"] = int(level)
            rows.append(row)
        buffer_history = pd.DataFrame(rows)
    health_history = pd.read_csv(DEMO_DIR / "health_log.csv")
    sensor_history = pd.read_csv(DEMO_DIR / "sensor_log.csv")
    registry = pd.read_csv(DEMO_DIR / "station_registry.csv")
    line_stats = pd.read_csv(DEMO_DIR / "station_stats.csv")
    with open(DEMO_DIR / "bottleneck_report.json") as fh:
        brep = json.load(fh)
    with open(DEMO_DIR / "manifest.json") as fh:
        manifest = json.load(fh)
    demo_assessment = pd.read_csv(DEMO_DIR / "demo_assessment.csv") \
        if (DEMO_DIR / "demo_assessment.csv").exists() else pd.DataFrame()
    virtual_sensor_events = pd.read_csv(DEMO_DIR / "virtual_sensor_events.csv") \
        if (DEMO_DIR / "virtual_sensor_events.csv").exists() else pd.DataFrame()

    timeline_min = float(buffer_history["t"].min()) if not buffer_history.empty else 0.0
    timeline_max = float(buffer_history["t"].max()) if not buffer_history.empty else float(
        manifest.get("duration_s", 8000.0)
    )
    return {
        "brep": brep,
        "line_stats": line_stats,
        "registry": registry,
        "line_events": line_events,
        "health_history": health_history,
        "sensor_history": sensor_history,
        "buffer_history": buffer_history,
        "timeline_min": timeline_min,
        "timeline_max": timeline_max,
        "buffer_capacity": int(manifest.get("buffer_capacity", 4)),
        "demo_source": str(DEMO_DIR),
        "demo_assessment": demo_assessment,
        "virtual_sensor_events": virtual_sensor_events,
    }


def _line_snapshot(state: dict, t_now: float) -> pd.DataFrame:
    base = state["line_stats"].copy()
    base["stage"] = base["index"].map(lambda i: _stage_of(int(i), len(base)))
    base["state"] = "STARTUP"
    base["health_true"] = 1.0
    base["buffer_in"] = 0
    base["buffer_out"] = 0

    events = state["line_events"]
    if not events.empty:
        seen = events[events["t"] <= t_now].sort_values(["station", "t"]).groupby("station").tail(1)
        if not seen.empty:
            base = base.merge(
                seen[["station", "state"]].rename(columns={"station": "index"}),
                on="index", how="left", suffixes=("", "_event"),
            )
            base["state"] = base["state_event"].fillna(base["state"])
            base = base.drop(columns=["state_event"])

    health = state["health_history"]
    if not health.empty:
        seen = health[health["t"] <= t_now].sort_values(["station", "t"]).groupby("station").tail(1)
        if not seen.empty:
            base = base.merge(
                seen[["station", "health_true"]].rename(columns={"station": "index"}),
                on="index", how="left", suffixes=("", "_obs"),
            )
            base["health_true"] = base["health_true_obs"].fillna(base["health_true"])
            base = base.drop(columns=["health_true_obs"])

    buffers = state["buffer_history"]
    if not buffers.empty:
        buf_row = buffers[buffers["t"] <= t_now].tail(1)
        if not buf_row.empty:
            buf_row = buf_row.iloc[0]
            n = len(base)
            base["buffer_in"] = [int(buf_row.get(f"buffer_{i}", 0)) for i in range(n)]
            base["buffer_out"] = [
                int(buf_row.get(f"buffer_{i + 1}", 0)) if i + 1 < n else 0
                for i in range(n)
            ]

    base["pressure"] = (
        0.55 * base["buffer_in"] / max(1, state["buffer_capacity"])
        + 0.30 * base["utilisation"].astype(float)
        + 0.15 * (1.0 - base["health_true"].astype(float))
    )
    base["pressure"] = base["pressure"].clip(0.0, 1.0)
    return base.sort_values("index").reset_index(drop=True)


def _recent_events(state: dict, t_now: float, limit: int = 10) -> pd.DataFrame:
    events = state["line_events"]
    if events.empty:
        return pd.DataFrame(columns=["t", "station", "name", "state", "buffer_in", "buffer_out"])
    out = events[events["t"] <= t_now].tail(limit).copy()
    return out.merge(state["registry"][["station", "name"]], on="station", how="left")


def _live_warning_board(snapshot: pd.DataFrame, buffer_capacity: int) -> pd.DataFrame:
    state_penalty = {
        WORKING: 0.08,
        BLOCKED: 1.00,
        STARVED: 0.35,
        DOWN: 0.90,
        "STARTUP": 0.15,
    }
    board = snapshot.copy()
    board["state_penalty"] = board["state"].map(state_penalty).fillna(0.1)
    board["live_failure_signal"] = (
        0.42 * (1.0 - board["health_true"].astype(float))
        + 0.28 * (board["buffer_in"].astype(float) / max(1, buffer_capacity))
        + 0.20 * board["state_penalty"].astype(float)
        + 0.10 * board["pressure"].astype(float)
    ).clip(0.0, 1.0)

    def classify(row: pd.Series) -> tuple[str, str, str]:
        if row["state"] == DOWN:
            return ("Breakdown risk", "Immediate technician check", "Station is down in the replay")
        if row["state"] == BLOCKED and row["buffer_in"] >= max(1, buffer_capacity - 1):
            return ("Spillback bottleneck", "Clear downstream congestion", "Work cannot leave the station")
        if row["health_true"] < 0.88:
            return ("Quality drift", "Inspect units from this station", "Health is degraded enough to raise defect risk")
        if row["pressure"] >= 0.62 or row["buffer_in"] >= max(1, buffer_capacity - 1):
            return ("Forming bottleneck", "Watch queues and cycle time", "Pressure is building before a stoppage")
        return ("Stable", "Monitor only", "No strong warning at this moment")

    triples = board.apply(classify, axis=1)
    board["warning_type"] = [t[0] for t in triples]
    board["recommended_action"] = [t[1] for t in triples]
    board["reason"] = [t[2] for t in triples]
    board["urgency"] = pd.cut(
        board["live_failure_signal"],
        bins=[-0.01, 0.33, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return board.sort_values(["live_failure_signal", "buffer_in"], ascending=[False, False]).reset_index(drop=True)


def _warning_bar_fig(warnings: pd.DataFrame) -> go.Figure:
    top = warnings.head(6).copy()
    top["label"] = [f"S{i} {n}" for i, n in zip(top["index"], top["name"])]
    color_map = {"High": RED, "Medium": AMBER, "Low": GREEN}
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["live_failure_signal"],
            y=top["label"],
            orientation="h",
            marker_color=[color_map.get(u, GREY) for u in top["urgency"]],
            customdata=np.stack([top["warning_type"], top["reason"], top["recommended_action"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Signal: %{x:.2f}<br>"
                "Type: %{customdata[0]}<br>"
                "Reason: %{customdata[1]}<br>"
                "Action: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(t=20, b=0, l=0, r=10),
        xaxis=dict(title="Live failure / bottleneck signal", range=[0, 1], tickfont=dict(size=14)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _live_prediction_queue(state: dict, t_now: float) -> pd.DataFrame:
    a = state.get("demo_assessment", pd.DataFrame())
    if a is None or a.empty or "latest_t" not in a.columns:
        return pd.DataFrame()
    out = a[a["latest_t"] <= t_now].copy()
    if out.empty:
        return out
    return out.sort_values(["risk_score", "effective_trust"], ascending=[False, False])


def _virtual_sensor_snapshot(state: dict, station: int, t_now: float) -> pd.DataFrame:
    events = state.get("virtual_sensor_events", pd.DataFrame())
    if events is None or events.empty:
        return pd.DataFrame()
    sub = events[(events["station"] == station) & (events["t_global"] <= t_now)].copy()
    if sub.empty:
        return sub
    latest = sub.sort_values(["channel", "t_global"]).groupby("channel").tail(1)
    return latest.sort_values("channel").reset_index(drop=True)


def _virtual_sensor_history(state: dict, station: int, t_now: float, window_s: float) -> pd.DataFrame:
    events = state.get("virtual_sensor_events", pd.DataFrame())
    if events is None or events.empty:
        return pd.DataFrame()
    start_t = max(float(state["timeline_min"]), float(t_now) - float(window_s))
    return events[
        (events["station"] == station)
        & (events["t_global"] >= start_t)
        & (events["t_global"] <= t_now)
    ].copy().sort_values("t_global")


def _virtual_sensor_observations(state: dict, station: int, t_now: float, window_s: float) -> pd.DataFrame:
    sensor_history = state.get("sensor_history", pd.DataFrame())
    if sensor_history is None or sensor_history.empty:
        return pd.DataFrame()
    start_t = max(float(state["timeline_min"]), float(t_now) - float(window_s))
    return sensor_history[
        (sensor_history["station"] == station)
        & (sensor_history["t_global"] >= start_t)
        & (sensor_history["t_global"] <= t_now)
    ].copy().sort_values("t_global")


def _virtual_sensor_station_choices(state: dict) -> list[int]:
    events = state.get("virtual_sensor_events", pd.DataFrame())
    if events is None or events.empty or "station" not in events.columns:
        return []
    return sorted(events["station"].dropna().astype(int).unique().tolist())


def _virtual_sensor_confidence_fig(vhist: pd.DataFrame) -> go.Figure:
    if vhist.empty:
        return go.Figure()
    latest = vhist.sort_values(["channel", "t_global"]).groupby("channel").tail(1).copy()
    latest["label"] = latest["channel"].str.title()
    latest["band"] = pd.cut(
        latest["confidence"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    color_map = {"Low": RED, "Medium": AMBER, "High": GREEN}
    fig = go.Figure(
        go.Bar(
            x=latest["confidence"],
            y=latest["label"],
            orientation="h",
            marker_color=[color_map.get(b, GREY) for b in latest["band"]],
            customdata=np.stack([latest["method"], latest["estimate"], latest["t_global"], latest["band"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Confidence: %{x:.2f}<br>"
                "Method: %{customdata[0]}<br>"
                "Estimate: %{customdata[1]:.3f}<br>"
                "Updated at t=%{customdata[2]:.1f}<br>"
                "Band: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.35, line_color=RED, line_dash="dot")
    fig.add_vline(x=0.65, line_color=GREEN, line_dash="dot")
    fig.update_layout(
        height=230,
        margin=dict(t=20, b=0, l=0, r=0),
        xaxis=dict(title="Inference confidence", range=[0, 1], tickfont=dict(size=14)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15, color=SLATE),
    )
    return fig


def _virtual_sensor_trend_fig(vhist: pd.DataFrame, t_now: float) -> go.Figure:
    if vhist.empty:
        return go.Figure()
    plot_df = vhist.copy().sort_values(["channel", "t_global"])
    color_map = {"torque": NAVY, "vibration": RED, "temperature": AMBER}
    fig = go.Figure()
    for channel, sub in plot_df.groupby("channel"):
        fig.add_trace(
            go.Scatter(
                x=sub["t_global"],
                y=sub["estimate"],
                mode="lines+markers",
                name=f"{channel} fill",
                line=dict(color=color_map.get(channel, SLATE), width=3, shape="hv"),
                marker=dict(size=7, color=color_map.get(channel, SLATE)),
                customdata=np.stack([sub["method"], sub["confidence"]], axis=1),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Filled value: %{y:.3f}<br>"
                    "Passed forward at t=%{x:.1f}<br>"
                    "Method: %{customdata[0]}<br>"
                    "Confidence: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=t_now, line_color=GREEN, line_dash="dot", line_width=2)
    fig.update_layout(
        height=340,
        margin=dict(t=20, b=0, l=0, r=0),
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15, color=SLATE),
        xaxis_title="Simulation time (s)",
        yaxis_title="Filled value carried into downstream logic",
    )
    return fig


def _virtual_sensor_fill_flow_fig(vhist: pd.DataFrame, observed: pd.DataFrame, t_now: float) -> go.Figure:
    fig = go.Figure()
    if vhist.empty:
        return fig

    channel_order = ["torque", "vibration", "temperature"]
    y_pos = {ch: idx for idx, ch in enumerate(channel_order)}
    color_map = {"torque": NAVY, "vibration": RED, "temperature": AMBER}

    for channel, sub in vhist.groupby("channel"):
        y = y_pos.get(channel, len(y_pos))
        fig.add_trace(
            go.Scatter(
                x=sub["t_global"],
                y=[y] * len(sub),
                mode="markers",
                name=f"{channel} filled",
                marker=dict(size=11, color=color_map.get(channel, SLATE), symbol="diamond"),
                customdata=np.stack([sub["estimate"], sub["confidence"], sub["method"]], axis=1),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Fill used at t=%{x:.1f}<br>"
                    "Estimate: %{customdata[0]:.3f}<br>"
                    "Confidence: %{customdata[1]:.2f}<br>"
                    "Method: %{customdata[2]}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    if not observed.empty:
        for channel, sub in observed.groupby("channel"):
            y = y_pos.get(channel, len(y_pos))
            fig.add_trace(
                go.Scatter(
                    x=sub["t_global"],
                    y=[y] * len(sub),
                    mode="markers",
                    name=f"{channel} real check",
                    marker=dict(
                        size=14,
                        color="white",
                        symbol="circle-open",
                        line=dict(color=color_map.get(channel, SLATE), width=3),
                    ),
                    customdata=np.stack([sub["value"]], axis=1),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Observed at t=%{x:.1f}<br>"
                        "Real value: %{customdata[0]:.3f}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    fig.add_vline(x=t_now, line_color=GREEN, line_dash="dot", line_width=2)
    fig.update_layout(
        height=240,
        margin=dict(t=20, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15, color=SLATE),
        xaxis_title="Simulation time (s)",
        yaxis=dict(
            title="Channel",
            tickmode="array",
            tickvals=list(y_pos.values()),
            ticktext=[ch.title() for ch in y_pos.keys()],
            range=[-0.6, max(y_pos.values()) + 0.6],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _station_strip_fig(snapshot: pd.DataFrame, highlight_station: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=snapshot["index"],
            y=[0] * len(snapshot),
            mode="lines",
            line=dict(color="rgba(16,35,63,0.18)", width=8),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=snapshot["index"],
            y=[0] * len(snapshot),
            mode="markers+text",
            text=[f"S{i}" for i in snapshot["index"]],
            textposition="top center",
            marker=dict(
                size=24 + snapshot["buffer_in"].astype(float) * 6,
                color=[STATE_COLORS.get(s, GREY) for s in snapshot["state"]],
                symbol=snapshot["tier"].map({"A": "circle", "B": "diamond", "C": "square"}).tolist(),
                line=dict(
                    width=[4 if int(i) == highlight_station else 1.5 for i in snapshot["index"]],
                    color=[NAVY if int(i) == highlight_station else "rgba(16,35,63,0.28)"
                           for i in snapshot["index"]],
                ),
            ),
            customdata=np.stack([
                snapshot["name"],
                snapshot["state"],
                snapshot["tier"],
                snapshot["health_true"].round(3),
                snapshot["buffer_in"],
                snapshot["buffer_out"],
                snapshot["pressure"].round(3),
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "State: %{customdata[1]}<br>"
                "Tier: %{customdata[2]}<br>"
                "Health: %{customdata[3]}<br>"
                "Buffer in/out: %{customdata[4]}/%{customdata[5]}<br>"
                "Pressure: %{customdata[6]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for _, row in snapshot.iterrows():
        fig.add_annotation(
            x=row["index"],
            y=-0.26,
            text=f"{row['name']}<br>B{int(row['buffer_in'])}",
            showarrow=False,
            font=dict(size=12, color=SLATE),
            align="center",
        )

    fig.update_layout(
        height=260,
        margin=dict(t=20, b=10, l=20, r=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-0.45, 0.45]),
        font=dict(size=15, color=SLATE),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _buffer_heatmap_fig(state: dict, t_now: float, window_s: float) -> go.Figure:
    hist = state["buffer_history"]
    if hist.empty:
        return go.Figure()
    view = hist[(hist["t"] >= max(hist["t"].min(), t_now - window_s)) & (hist["t"] <= t_now)].copy()
    if view.empty:
        view = hist.tail(20).copy()
    buffer_cols = [c for c in view.columns if c.startswith("buffer_")]
    long = view[["t"] + buffer_cols].melt(id_vars="t", var_name="buffer", value_name="level")
    long["station"] = long["buffer"].str.replace("buffer_", "", regex=False).astype(int)
    z = (long.pivot(index="station", columns="t", values="level")
         .sort_index(ascending=False))
    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=z.columns,
            y=[f"S{i}" for i in z.index],
            colorscale=[
                [0.0, "#F5F1E8"],
                [0.35, "#F5A623"],
                [0.7, "#E26F3F"],
                [1.0, "#D64545"],
            ],
            colorbar=dict(title="Buffer"),
            zmin=0,
            zmax=max(1, state["buffer_capacity"]),
        )
    )
    fig.add_vline(x=t_now, line_color=NAVY, line_width=2, line_dash="dot")
    fig.update_layout(
        height=300,
        margin=dict(t=20, b=0, l=0, r=0),
        xaxis_title="Simulation time (s)",
        yaxis_title="",
        font=dict(size=15, color=SLATE),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _station_detail_fig(state: dict, snapshot: pd.DataFrame, station: int, t_now: float,
                        window_s: float) -> go.Figure:
    health = state["health_history"]
    buffers = state["buffer_history"]
    start_t = max(state["timeline_min"], t_now - window_s)
    health_view = health[(health["station"] == station) & (health["t"] >= start_t) & (health["t"] <= t_now)]
    buf_view = buffers[(buffers["t"] >= start_t) & (buffers["t"] <= t_now)].copy()
    buf_col = f"buffer_{station}"
    buf_out_col = f"buffer_{station + 1}"

    fig = go.Figure()
    if not health_view.empty:
        fig.add_trace(
            go.Scatter(
                x=health_view["t"],
                y=health_view["health_true"],
                mode="lines",
                name="Health",
                line=dict(color=NAVY, width=3),
            )
        )
    if not buf_view.empty and buf_col in buf_view:
        fig.add_trace(
            go.Bar(
                x=buf_view["t"],
                y=buf_view[buf_col],
                name="Inbound buffer",
                marker_color="rgba(245,166,35,0.55)",
                yaxis="y2",
                opacity=0.8,
            )
        )
    if not buf_view.empty and buf_out_col in buf_view:
        fig.add_trace(
            go.Scatter(
                x=buf_view["t"],
                y=buf_view[buf_out_col],
                mode="lines",
                name="Outbound buffer",
                line=dict(color=RED, width=2, dash="dot"),
                yaxis="y2",
            )
        )

    fig.add_vline(x=t_now, line_color=AMBER, line_width=2)
    fig.update_layout(
        height=320,
        margin=dict(t=20, b=0, l=0, r=0),
        barmode="overlay",
        xaxis_title="Simulation time (s)",
        yaxis=dict(title="Health", range=[0, 1.05]),
        yaxis2=dict(title="Buffers", overlaying="y", side="right", range=[0, state["buffer_capacity"] + 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=15, color=SLATE),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _ops_card(label: str, value: str, sub: str) -> str:
    return (
        '<div class="ops-card">'
        f'<div class="ops-label">{label}</div>'
        f'<div class="ops-value">{value}</div>'
        f'<div class="ops-sub">{sub}</div>'
        "</div>"
    )


def _legend_pill(color: str, text: str) -> str:
    return (
        '<div class="legend-pill">'
        f'<span class="legend-dot" style="background:{color};"></span>'
        f"{text}</div>"
    )


def _section_header(title: str, help_text: str, *, level: int = 3) -> None:
    left, right = st.columns([20, 1])
    with left:
        if level == 2:
            st.subheader(title)
        else:
            st.markdown(f"### {title}")
    with right:
        with st.popover("ℹ️"):
            st.markdown(help_text)


def _state_badge(state_name: str) -> str:
    return STATE_COLORS.get(state_name, GREY)


def _station_events(state: dict, station: int, t_now: float, limit: int = 8) -> pd.DataFrame:
    events = state["line_events"]
    if events.empty:
        return pd.DataFrame(columns=["t", "state", "buffer_in", "buffer_out"])
    cols = ["t", "state", "buffer_in", "buffer_out"]
    return (events[(events["station"] == station) & (events["t"] <= t_now)][cols]
            .sort_values("t", ascending=False)
            .head(limit)
            .reset_index(drop=True))


def _station_card_copy(row: pd.Series) -> str:
    if row["state"] == DOWN:
        return "Station is down and needs immediate inspection."
    if row["state"] == BLOCKED:
        return "Downstream congestion is preventing work from leaving this station."
    if row["state"] == STARVED:
        return "Upstream starvation is limiting this station right now."
    if row["pressure"] >= 0.62:
        return "Pressure is building and could turn into the next bottleneck."
    return "Station is flowing normally with no strong operational warning."


def _render_station_cards(snapshot: pd.DataFrame) -> int | None:
    clicked_station = None
    rows = [snapshot.iloc[i:i + 4] for i in range(0, len(snapshot), 4)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                st.markdown(
                    (
                        '<div class="station-card">'
                        f'<div class="station-card-title">S{int(row["index"])} · {row["name"]}</div>'
                        f'<div class="station-card-meta">{row["stage"]} · Tier {row["tier"]} · '
                        f'Pressure {row["pressure"]:.2f}</div>'
                        f'<div class="station-card-state" style="background:{_state_badge(row["state"])}">'
                        f'{row["state"]}</div>'
                        f'<div class="station-card-copy">{_station_card_copy(row)}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"Open S{int(row['index'])}",
                    key=f"open_station_{int(row['index'])}",
                    use_container_width=True,
                ):
                    clicked_station = int(row["index"])
    return clicked_station


def _render_station_clickbar(snapshot: pd.DataFrame) -> int | None:
    clicked_station = None
    st.caption("Click a station below to open its full inspection view.")
    rows = [snapshot.iloc[i:i + 6] for i in range(0, len(snapshot), 6)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                label = f"S{int(row['index'])} · {row['name']}"
                if st.button(label, key=f"station_jump_{int(row['index'])}", use_container_width=True):
                    clicked_station = int(row["index"])
    return clicked_station


def _dismiss_station_dialog() -> None:
    st.session_state.station_modal_station = None


@st.dialog("Station inspection", width="large", on_dismiss=_dismiss_station_dialog)
def _station_detail_dialog(state: dict, snapshot: pd.DataFrame, station: int, t_now: float,
                           window_s: float) -> None:
    row = snapshot.loc[snapshot["index"] == station].iloc[0]
    station_name = row["name"]
    st.markdown(
        f"""
        <div class="ops-banner">
          <h3>S{station} · {station_name}</h3>
          <p>Detailed live inspection at t=<b>{int(t_now)}</b>s. This view expands the selected
          station across the screen so you can narrate health drift, queue pressure, and inferred
          evidence without leaving the main replay.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(_ops_card("Live state", row["state"], f"Tier {row['tier']} · {row['stage']}"),
                unsafe_allow_html=True)
    m2.markdown(_ops_card("Health", f"{row['health_true']:.2f}", "Current health estimate"),
                unsafe_allow_html=True)
    m3.markdown(_ops_card("Inbound queue", str(int(row["buffer_in"])), "Units waiting upstream"),
                unsafe_allow_html=True)
    m4.markdown(_ops_card("Pressure", f"{row['pressure']:.2f}", "Combined bottleneck signal"),
                unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 2])
    with top_left:
        _section_header(
            "Station timeline",
            "This full-width detail chart follows the selected station through recent replay time, "
            "combining health drift with inbound and outbound queue pressure.",
        )
        st.plotly_chart(_station_detail_fig(state, snapshot, station, t_now, window_s), width="stretch")
    with top_right:
        _section_header(
            "Recent station events",
            "These are the latest state changes recorded for the selected station only.",
        )
        station_events = _station_events(state, station, t_now)
        if station_events.empty:
            st.info("No station-specific events have occurred yet at this playback time.")
        else:
            st.dataframe(station_events, width="stretch", hide_index=True)

    if station in _virtual_sensor_station_choices(state):
        vfill = _virtual_sensor_snapshot(state, station, t_now)
        vhist = _virtual_sensor_history(state, station, t_now, max(window_s, 900))
        observed = _virtual_sensor_observations(state, station, t_now, max(window_s, 900))
        lower_left, lower_right = st.columns([3, 2])
        with lower_left:
            _section_header(
                "Virtual sensor fill for this station",
                "This shows the inferred values that are being carried forward when direct readings are missing.",
            )
            if vhist.empty:
                st.info("No virtual-sensor history is available for this station yet.")
            else:
                st.plotly_chart(_virtual_sensor_trend_fig(vhist, t_now), width="stretch")
                st.plotly_chart(_virtual_sensor_fill_flow_fig(vhist, observed, t_now), width="stretch")
        with lower_right:
            _section_header(
                "Inference confidence",
                "Confidence helps decide whether the inferred evidence is strong enough for automation or should be escalated.",
            )
            if vfill.empty:
                st.info("No virtual-sensor snapshot is available yet for this station.")
            else:
                st.plotly_chart(_virtual_sensor_confidence_fig(vhist), width="stretch")
                st.dataframe(
                    vfill[["channel", "method", "estimate", "confidence", "t_global"]].round(3),
                    width="stretch",
                    hide_index=True,
                )

    if st.button("Close station detail", use_container_width=True):
        st.session_state.station_modal_station = None
        st.rerun()


def _render_station_detail_workspace(state: dict, snapshot: pd.DataFrame, station: int,
                                     t_now: float, window_s: float) -> None:
    row = snapshot.loc[snapshot["index"] == station].iloc[0]
    st.markdown(
        f"""
        <div class="ops-banner">
          <h3>Station-wise Detail — S{station} · {row["name"]}</h3>
          <p>Focused inspection view for the selected station at <b>{int(t_now)}</b>s.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(_ops_card("Live state", row["state"], f"Tier {row['tier']} · {row['stage']}"),
                unsafe_allow_html=True)
    m2.markdown(_ops_card("Health", f"{row['health_true']:.2f}", "Current simulated health"),
                unsafe_allow_html=True)
    m3.markdown(_ops_card("Inbound queue", str(int(row["buffer_in"])), "Units waiting upstream"),
                unsafe_allow_html=True)
    m4.markdown(_ops_card("Pressure", f"{row['pressure']:.2f}", "Combined bottleneck signal"),
                unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 2])
    with top_left:
        _section_header(
            "Station timeline",
            "This chart shows health drift and buffer behavior for the chosen station across the selected time window.",
        )
        st.plotly_chart(_station_detail_fig(state, snapshot, station, t_now, window_s), width="stretch")
    with top_right:
        _section_header(
            "Recent station events",
            "These are the latest state changes recorded for the selected station only.",
        )
        station_events = _station_events(state, station, t_now)
        if station_events.empty:
            st.info("No station-specific events have occurred yet at this playback time.")
        else:
            st.dataframe(station_events, width="stretch", hide_index=True)

    if station in _virtual_sensor_station_choices(state):
        _section_header(
            "Virtual sensor evidence for this station",
            "If this station is sensor-poor, the inferred values and confidence are shown here as part of the station inspection.",
        )
        vfill = _virtual_sensor_snapshot(state, station, t_now)
        vhist = _virtual_sensor_history(state, station, t_now, max(window_s, 900))
        observed = _virtual_sensor_observations(state, station, t_now, max(window_s, 900))
        lower_left, lower_right = st.columns([3, 2])
        with lower_left:
            if vhist.empty:
                st.info("No virtual-sensor history is available for this station yet.")
            else:
                st.plotly_chart(_virtual_sensor_trend_fig(vhist, t_now), width="stretch")
                st.plotly_chart(_virtual_sensor_fill_flow_fig(vhist, observed, t_now), width="stretch")
        with lower_right:
            if vfill.empty:
                st.info("No virtual-sensor snapshot is available yet for this station.")
            else:
                st.plotly_chart(_virtual_sensor_confidence_fig(vhist), width="stretch")
                st.dataframe(
                    vfill[["channel", "method", "estimate", "confidence", "t_global"]].round(3),
                    width="stretch",
                    hide_index=True,
                )


def _render_virtual_fill_workspace(state: dict, virtual_station: int, t_now: float, window_s: float) -> None:
    inferred_stations = _virtual_sensor_station_choices(state)
    station_lookup = state["line_stats"].set_index("index")["name"].to_dict()
    st.markdown(
        f"""
        <div class="ops-banner">
          <h3>Virtual Fill Workspace</h3>
          <p>Focused view of inferred sensor behavior for <b>S{virtual_station} · {station_lookup.get(virtual_station, f"Station {virtual_station}")}</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not inferred_stations:
        st.info("No virtual-sensor stations are available in the current replay.")
        return

    vfill = _virtual_sensor_snapshot(state, virtual_station, t_now)
    vhist = _virtual_sensor_history(state, virtual_station, t_now, max(window_s, 900))
    observed = _virtual_sensor_observations(state, virtual_station, t_now, max(window_s, 900))
    control_left, control_right = st.columns([3, 2])
    with control_left:
        st.markdown(
            _ops_card(
                "Focused inferred station",
                f"S{virtual_station} {station_lookup.get(virtual_station, f'Station {virtual_station}')}",
                "Change this from the left navigation controls",
            ),
            unsafe_allow_html=True,
        )
    with control_right:
        method_text = ", ".join(sorted(vfill["method"].dropna().unique().tolist())) if not vfill.empty else "waiting"
        st.markdown(
            _ops_card(
                "Inference mode",
                method_text.title(),
                f"{len(vfill)} channel fills visible at t={int(t_now)}",
            ),
            unsafe_allow_html=True,
        )

    main_left, main_right = st.columns([3, 2])
    with main_left:
        _section_header(
            f"Filled values passed downstream — S{virtual_station}",
            "These are the inferred values that continue downstream whenever direct readings are missing.",
        )
        if vhist.empty:
            st.info("No virtual-sensor history is available for this station at the current playback time.")
        else:
            st.plotly_chart(_virtual_sensor_trend_fig(vhist, t_now), width="stretch")
            st.plotly_chart(_virtual_sensor_fill_flow_fig(vhist, observed, t_now), width="stretch")
    with main_right:
        _section_header(
            "Confidence by channel",
            "This shows how trustworthy each imputed channel is right now.",
        )
        if vfill.empty:
            st.info("No virtual-sensor snapshot is available yet for this station.")
        else:
            st.plotly_chart(_virtual_sensor_confidence_fig(vhist), width="stretch")
            fill_board = vfill[["channel", "method", "estimate", "confidence", "t_global"]].copy()
            fill_board[["estimate", "confidence", "t_global"]] = (
                fill_board[["estimate", "confidence", "t_global"]].round(3)
            )
            st.dataframe(fill_board, width="stretch", hide_index=True)


def _build_state() -> dict:
    features, X, y, train_mask, test_mask = load_production_split()
    model = DefectModel().fit(X[train_mask], y[train_mask])
    importances = pd.Series(model.model.feature_importances_, index=X.columns)

    assessment, risk_thr, _ = assess(model, X[test_mask], importances, trust_thr=0.5)
    meta = features[test_mask][["session_id", "response"]].reset_index()
    assessment = (assessment.reset_index().merge(meta, on="index")
                  .rename(columns={"response": "defect"}))

    demo_bootstrap_error = None
    try:
        _ensure_demo_assets()
    except Exception as exc:  # pragma: no cover - safe fallback for demo bootstrapping
        demo_bootstrap_error = exc

    demo_stream = _load_demo_stream()
    if demo_stream is None:
        line = default_line(seed=DEFAULT_DEMO_SEED).run(until=DEFAULT_DEMO_DURATION)
        source = "in_app_fresh_simulation"
        if demo_bootstrap_error is not None:
            source = f"{source} (bootstrap failed: {type(demo_bootstrap_error).__name__})"
        demo_stream = {
            "brep": bottleneck_report(line),
            "line_stats": pd.DataFrame(line.station_stats()),
            "registry": pd.read_csv(DATA_DIR / "station_registry.csv"),
            "line_events": pd.DataFrame(line.log),
            "health_history": pd.DataFrame(line.health_log),
            "sensor_history": pd.DataFrame(line.sensor_log),
            "buffer_history": _buffer_history_frame(line),
            "timeline_min": float(min((r["t"] for r in line.buffer_log), default=0.0)),
            "timeline_max": float(max((r["t"] for r in line.buffer_log), default=DEFAULT_DEMO_DURATION)),
            "buffer_capacity": int(line.cfg.buffer_capacity),
            "demo_source": source,
            "demo_assessment": pd.DataFrame(),
            "virtual_sensor_events": pd.DataFrame(),
        }

    from sklearn.metrics import roc_auc_score
    try:
        held_out_auc = float(roc_auc_score(assessment["defect"], assessment["risk_score"]))
    except ValueError:
        held_out_auc = float("nan")

    return {
        "assessment": assessment,
        "risk_thr": float(risk_thr),
        "brep": demo_stream["brep"],
        "line_stats": demo_stream["line_stats"],
        "registry": demo_stream["registry"],
        "held_out_auc": held_out_auc,
        "n_defects": int(assessment["defect"].sum()),
        "line_events": demo_stream["line_events"],
        "health_history": demo_stream["health_history"],
        "sensor_history": demo_stream["sensor_history"],
        "buffer_history": demo_stream["buffer_history"],
        "timeline_min": demo_stream["timeline_min"],
        "timeline_max": demo_stream["timeline_max"],
        "buffer_capacity": demo_stream["buffer_capacity"],
        "demo_source": demo_stream["demo_source"],
        "demo_assessment": demo_stream["demo_assessment"],
        "virtual_sensor_events": demo_stream["virtual_sensor_events"],
    }


@st.cache_resource(show_spinner="Training the twin and building the live demo state…")
def get_state() -> dict:
    return _build_state()


def regate(assessment: pd.DataFrame, risk_thr: float, trust_thr: float) -> pd.DataFrame:
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


def render_live_ops(state: dict, t_now: float, selected_station: int, window_s: float) -> tuple[int, float]:
    snapshot = _line_snapshot(state, t_now)
    warnings = _live_warning_board(snapshot, state["buffer_capacity"])
    live_queue = _live_prediction_queue(state, t_now)
    hot = snapshot.sort_values(["pressure", "buffer_in", "health_true"], ascending=[False, False, True]).iloc[0]
    overall = state["brep"]["primary"]
    recent = _recent_events(state, t_now)

    st.markdown(
        f"""
        <div class="ops-banner">
          <h3>Live Twin Playback</h3>
          <p>Replaying a fresh simulated shift at <b>{int(t_now)}</b>s. This view streams
          station states, queue pressure, and health drift from the digital twin in time order.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
        <b>How to read this screen</b><br>
        The top half is the live replay of one unseen demo shift. Colors show station state,
        marker shapes show sensor maturity, buffer size shows congestion, and the warning
        board translates those raw signals into likely failure or bottleneck conditions.
        The lower persona panels then explain what the trained twin would do with similar
        evidence at operator, manager, and leadership level.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="legend-row">'
        + _legend_pill(GREEN, "Green = working normally")
        + _legend_pill(RED, "Red = blocked or severe warning")
        + _legend_pill(AMBER, "Amber = down / medium urgency")
        + _legend_pill(GREY, "Grey = starved or waiting")
        + _legend_pill(NAVY, "Circle = Tier A, diamond = Tier B, square = Tier C")
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_ops_card("Current pressure point", f"S{int(hot['index'])} {hot['name']}",
                          f"{hot['state']} · pressure {hot['pressure']:.2f}"), unsafe_allow_html=True)
    c2.markdown(_ops_card("Live bottleneck anchor", f"S{overall['index']} {overall['name']}",
                          f"{overall['utilisation']*100:.0f}% utilisation over the run"), unsafe_allow_html=True)
    c3.markdown(_ops_card("Current blind spots", str(int((snapshot['tier'] != 'A').sum())),
                          "Stations relying on inferred or sparse signals"), unsafe_allow_html=True)
    c4.markdown(_ops_card("Line health floor", f"{snapshot['health_true'].min():.2f}",
                          "Lowest current station health in the replay"), unsafe_allow_html=True)

    _section_header(
        "Station-to-station live flow",
        "This is the live map of the assembly line. Each node is a station in order. "
        "Color shows live operating state, marker shape shows instrumentation tier, "
        "and node size grows with inbound queue pressure.",
    )
    st.caption("Each node is one station in sequence. Bigger nodes mean more inbound queue.")
    st.plotly_chart(_station_strip_fig(snapshot, selected_station), width="stretch")
    open_station = _render_station_clickbar(snapshot)
    if open_station is None:
        open_station = _render_station_cards(snapshot)
    if open_station is not None:
        st.session_state.pending_station_detail_picker = int(open_station)
        st.session_state.station_modal_station = int(open_station)
        selected_station = int(open_station)
        st.rerun()

    left, right = st.columns([3, 2])
    with left:
        _section_header(
            "Queue pressure over the recent window",
            "This heatmap shows how work accumulates before each station across recent simulated time. "
            "Hotter color means a fuller buffer and a stronger sign of forming congestion.",
        )
        st.caption("Hotter color means work is accumulating in front of that station over time.")
        st.plotly_chart(_buffer_heatmap_fig(state, t_now, window_s), width="stretch")
    with right:
        _section_header(
            "Current station board",
            "This table is the current operational snapshot for all stations: stage, sensor tier, "
            "live state, health estimate, and inbound/outbound buffers.",
        )
        board = snapshot[["index", "name", "stage", "tier", "state", "health_true", "buffer_in", "buffer_out"]].copy()
        board["health_true"] = board["health_true"].round(2)
        board = board.rename(columns={
            "index": "S#",
            "name": "Station",
            "stage": "Stage",
            "tier": "Tier",
            "state": "State",
            "health_true": "Health",
            "buffer_in": "In",
            "buffer_out": "Out",
        })
        st.dataframe(board, width="stretch", hide_index=True)

    risk_left, risk_right = st.columns([3, 2])
    with risk_left:
        _section_header(
            "Potential failures and bottlenecks forming now",
            "This warning chart converts raw live signals into interpretable operational risk. "
            "Higher bars mean the twin sees a stronger chance of breakdown, quality drift, or "
            "queue spillback at that station right now.",
        )
        st.caption("This is the live operational warning layer. It combines degraded health, "
                   "blocking, starvation, and queue buildup into a single visible signal.")
        st.plotly_chart(_warning_bar_fig(warnings), width="stretch")
    with risk_right:
        _section_header(
            "Live prediction queue",
            "This is the trained defect model applied to the demo shift. Each row is a completed unit "
            "seen so far in the replay, with its risk score, trust, and the action policy outcome.",
        )
        if state["demo_assessment"].empty:
            st.info("No model predictions were generated for the current demo shift.")
        elif live_queue.empty:
            st.info("No completed units have reached the live prediction queue at this playback time yet.")
        else:
            board = live_queue.head(8)[[
                "unit_id", "risk_score", "model_confidence", "effective_trust", "action", "latest_t",
            ]].copy()
            board[["risk_score", "model_confidence", "effective_trust", "latest_t"]] = (
                board[["risk_score", "model_confidence", "effective_trust", "latest_t"]].round(2)
            )
            board = board.rename(columns={
                "unit_id": "Unit",
                "risk_score": "Risk",
                "model_confidence": "Model conf",
                "effective_trust": "Eff trust",
                "action": "Action",
                "latest_t": "Seen at t",
            })
            st.dataframe(board, width="stretch", hide_index=True)

    _section_header(
        "Recent line events",
        "This is the event feed for the replay. It records when stations entered working, blocked, "
        "starved, or down states, together with the queue conditions at that moment.",
    )
    if recent.empty:
        st.info("No events have occurred yet at this playback time.")
    else:
        recent_board = recent[["t", "station", "name", "state", "buffer_in", "buffer_out"]].copy()
        recent_board = recent_board.rename(columns={
            "t": "t",
            "station": "S#",
            "name": "Station",
            "state": "State",
            "buffer_in": "In",
            "buffer_out": "Out",
        })
        st.dataframe(recent_board.sort_values("t", ascending=False), width="stretch", hide_index=True)
        st.caption("This feed is useful during the demo to narrate exactly when stations moved "
                   "into blocked, starved, working, or down states.")

    modal_station = st.session_state.get("station_modal_station")
    if modal_station is not None:
        _station_detail_dialog(state, snapshot, int(modal_station), t_now, window_s)
    return int(selected_station), float(window_s)


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
    st.caption("Supervisor view answers: where is the line constraining right now, and which risky units "
               "can be auto-handled versus sent to a human?")

    _section_header(
        "The line — utilisation by station",
        "This chart summarizes which stations consume the most working time over the run. "
        "The primary bottleneck is highlighted, and sensor-poor stations are called out separately.",
    )
    ls = state["line_stats"].copy()
    ls["role"] = np.where(ls["index"] == p["index"], "bottleneck",
                          np.where(ls["tier"] != "A", "sensor-poor", "normal"))
    fig = px.bar(ls, x="name", y="utilisation", color="role",
                 color_discrete_map={"bottleneck": RED, "sensor-poor": AMBER,
                                     "normal": NAVY},
                 labels={"utilisation": "utilisation", "name": ""})
    fig.update_layout(height=320, showlegend=True, legend_title_text="",
                      margin=dict(t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    _section_header(
        "Action queue — where risk meets trust",
        "Each point is a unit assessment from the trained twin. Risk is the model's defect probability, "
        "Effective Trust combines data trust and model confidence, and color shows the action policy outcome.",
    )
    st.caption("Dots above the horizontal threshold are high-risk units. If trust is also high, the system "
               "can auto-act; if trust is low, the same unit is escalated to human verification.")
    left, right = st.columns([3, 2])
    with left:
        fig2 = px.scatter(
            a[a["risk_score"] >= state["risk_thr"] * 0.6], x="effective_trust",
            y="risk_score", color="action", color_discrete_map=ACTION_COLORS,
            hover_data=["input_trust", "model_confidence"],
            labels={"effective_trust": "Effective Trust", "risk_score": "Defect risk"})
        fig2.add_hline(y=state["risk_thr"], line_dash="dot", line_color=GREY)
        fig2.update_layout(height=340, margin=dict(t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width="stretch")
    with right:
        st.caption("High-risk items to verify now (trust too low to auto-act):")
        hv = a[a["action"] == HUMAN_VERIFY].nlargest(8, "risk_score")
        st.dataframe(hv[["risk_score", "input_trust", "model_confidence",
                         "effective_trust"]].round(2), width="stretch")

    blind = registry[registry["tier"] != "A"]
    st.info("Blind spots: " + " · ".join(
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
    st.caption("Manager view answers: is the twin stable across shifts, and how much review workload is it creating?")

    _section_header(
        "Defects and action volume per shift",
        "This compares real defect counts with the amount of automated and human-review work the twin creates per shift.",
    )
    fig = go.Figure()
    fig.add_bar(x=tbl["shift"], y=tbl["auto_act"], name="AUTO-ACT", marker_color=RED)
    fig.add_bar(x=tbl["shift"], y=tbl["human_verify"], name="HUMAN-VERIFY",
                marker_color=AMBER)
    fig.add_trace(go.Scatter(x=tbl["shift"], y=tbl["defects"], name="actual defects",
                             mode="lines+markers", line=dict(color=NAVY, width=3),
                             yaxis="y2"))
    fig.update_layout(barmode="stack", height=360, margin=dict(t=10, b=0),
                      xaxis_title="shift", yaxis_title="flagged units",
                      yaxis2=dict(title="defects", overlaying="y", side="right"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    _section_header(
        "Mean Effective Trust per shift",
        "This trend shows whether the twin is relying on strong direct evidence or increasingly leaning on weak/inferred signals.",
    )
    st.caption("If trust trends down over time, the line may be leaning more heavily on weak or inferred evidence.")
    figt = px.line(tbl, x="shift", y="mean_trust", markers=True)
    figt.update_traces(line_color=AMBER)
    figt.update_layout(height=240, margin=dict(t=10, b=0), yaxis_range=[0, 1],
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
    st.caption("Leadership view answers: does the twin create business value by catching more defects earlier "
               "without over-automating low-trust calls?")

    _section_header(
        "Early-catch lift — defects caught vs. units reviewed",
        "This curve shows the business value of ranking. If you inspect only the riskiest slice of units first, "
        "how many real defects do you catch compared with random review?",
    )
    st.caption("The blue curve should sit above the dashed random line. That gap is the value of ranking the riskiest units first.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["reviewed"] * 100, y=curve["caught"] * 100,
                             mode="lines", line=dict(color=NAVY, width=3),
                             name="the twin"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color=GREY, dash="dash"), name="random"))
    fig.add_vline(x=20, line_dash="dot", line_color=AMBER)
    fig.update_layout(height=340, margin=dict(t=10, b=0),
                      xaxis_title="% of units reviewed (riskiest first)",
                      yaxis_title="% of defects caught",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
        **Business context** *(cited anchor, illustrative — not a claimed saving)*  
        A stopped automotive line runs **~\\$2.3M/hour**. The twin's value is
        catching the **{caught20/total*100:.0f}%** of defects surfaced early in the
        riskiest 20% of units, while the trust gate keeps the shakier calls with a person.
        """
    )


def _init_playback(state: dict) -> None:
    tmin = int(state["timeline_min"])
    tmax = int(state["timeline_max"])
    if "live_t" not in st.session_state:
        st.session_state.live_t = tmin
    st.session_state.live_t = int(np.clip(st.session_state.live_t, tmin, tmax))


def _consume_pending_station_selection() -> None:
    pending_station = st.session_state.pop("pending_station_detail_picker", None)
    if pending_station is not None:
        st.session_state.station_detail_picker = int(pending_station)


def _render_sidebar(state: dict) -> tuple[str, str, float, int, bool, bool]:
    tmin = int(state["timeline_min"])
    tmax = int(state["timeline_max"])
    stations = list(state["line_stats"]["index"])
    inferred_stations = _virtual_sensor_station_choices(state)

    with st.sidebar:
        st.markdown("## Control Deck")
        st.caption("Use the left navigation boxes to switch between the main workspaces.")
        workspace = st.radio(
            "Workspace",
            ["Live View", "Station-wise Detail", "Virtual Fill", "Analytics"],
            key="workspace_nav",
            label_visibility="collapsed",
        )

        with st.expander("Playback Controls", expanded=True):
            autoplay = st.checkbox("Auto-play live replay", value=False, key="autoplay_toggle")
            loop = st.checkbox("Loop at end", value=True, key="loop_toggle")
            playback_step = st.select_slider(
                "Jump per refresh (s)",
                options=[5, 10, 15, 20, 30, 45, 60],
                value=int(st.session_state.get("playback_step_top", 20)),
                key="playback_step_top",
            )
            refresh_ms = st.select_slider(
                "Refresh cadence (ms)",
                options=[500, 800, 1200, 1600, 2200],
                value=int(st.session_state.get("refresh_ms_top", 1200)),
                key="refresh_ms_top",
            )
            live_t = st.slider("Simulation time (s)", tmin, tmax, int(st.session_state.live_t), step=5)
            st.session_state.live_t = live_t
            st.markdown(_ops_card("Replay source", "demo replay", state["demo_source"]), unsafe_allow_html=True)

        with st.expander("Analysis Controls", expanded=True):
            persona = st.radio("Persona", ["Supervisor", "Manager", "Leadership"], key="persona_top")
            trust_thr = st.slider(
                "Auto-act only above this Effective Trust",
                0.0,
                1.0,
                float(st.session_state.get("trust_thr_sidebar", 0.5)),
                0.05,
                key="trust_thr_sidebar",
                help="Lower = more automation. Higher = more escalated to people.",
            )
            selected_station = st.selectbox(
                "Default station focus",
                options=stations,
                index=stations.index(int(st.session_state.get("station_detail_picker", stations[0]))),
                format_func=lambda i: (
                    f"S{i} · "
                    f"{state['line_stats'].loc[state['line_stats']['index'] == i, 'name'].iloc[0]}"
                ),
                key="station_detail_picker",
            )
            window_s = st.select_slider(
                "Detail window (s)",
                options=[120, 240, 480, 900, 1500],
                value=int(st.session_state.get("station_window_picker", 480)),
                key="station_window_picker",
            )
            st.markdown(
                _ops_card("Held-out AUC", f"{state['held_out_auc']:.2f}", f"{state['n_defects']} held-out defects"),
                unsafe_allow_html=True,
            )

        with st.expander("Virtual Sensor Focus", expanded=(workspace == "Virtual Fill")):
            if inferred_stations:
                default_virtual_station = int(st.session_state.get("virtual_station_picker", inferred_stations[0]))
                if default_virtual_station not in inferred_stations:
                    default_virtual_station = inferred_stations[0]
                st.selectbox(
                    "Virtual-sensor station",
                    options=inferred_stations,
                    index=inferred_stations.index(default_virtual_station),
                    format_func=lambda i: (
                        f"S{i} · "
                        f"{state['line_stats'].loc[state['line_stats']['index'] == i, 'name'].iloc[0]}"
                    ),
                    key="virtual_station_picker",
                )
                st.caption("Only stations that truly use imputed values appear here.")
            else:
                st.info("No inferred stations are available in this replay.")

    return (
        str(st.session_state.get("workspace_nav", "Live View")),
        str(st.session_state.get("persona_top", "Supervisor")),
        float(st.session_state.get("trust_thr_sidebar", 0.5)),
        int(st.session_state.get("playback_step_top", 20)),
        bool(st.session_state.get("autoplay_toggle", False)),
        bool(st.session_state.get("loop_toggle", True)),
    )


def main() -> None:
    st.set_page_config(page_title="DigitalTwin.ai", page_icon="🏭", layout="wide")
    _inject_theme()
    st.title("🏭 DigitalTwin.ai — live assembly twin")
    st.caption("Live station playback · defect prediction · Effective Trust gating for high-risk calls.")

    state = get_state()
    _init_playback(state)
    _consume_pending_station_selection()
    tmin = int(state["timeline_min"])
    tmax = int(state["timeline_max"])
    workspace, persona, trust_thr, playback_step, autoplay, loop = _render_sidebar(state)
    refresh_ms = int(st.session_state.get("refresh_ms_top", 1200))

    a = regate(state["assessment"], state["risk_thr"], trust_thr)
    selected_station = int(st.session_state.get("station_detail_picker", int(state["line_stats"]["index"].iloc[0])))
    window_s = float(st.session_state.get("station_window_picker", 480))
    t_now = float(st.session_state.live_t)

    if workspace == "Live View":
        selected_station, window_s = render_live_ops(state, t_now, selected_station, window_s)
    elif workspace == "Station-wise Detail":
        snapshot = _line_snapshot(state, t_now)
        _render_station_detail_workspace(state, snapshot, selected_station, t_now, window_s)
    elif workspace == "Virtual Fill":
        inferred_stations = _virtual_sensor_station_choices(state)
        virtual_station = int(st.session_state.get(
            "virtual_station_picker",
            inferred_stations[0] if inferred_stations else selected_station,
        ))
        if inferred_stations and virtual_station not in inferred_stations:
            virtual_station = inferred_stations[0]
        _render_virtual_fill_workspace(state, virtual_station, t_now, window_s)
    else:
        st.markdown(
            """
            <div class="ops-banner">
              <h3>Analytics Workspace</h3>
              <p>Switch the persona in the left controls to move between operator, manager, and leadership lenses.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if persona == "Supervisor":
            render_supervisor(state, a)
        elif persona == "Manager":
            render_manager(state, a)
        else:
            render_leadership(state, a)

    if autoplay:
        nxt = int(st.session_state.live_t) + int(playback_step)
        if nxt > tmax:
            nxt = tmin if loop else tmax
        time.sleep(refresh_ms / 1000.0)
        st.session_state.live_t = nxt
        st.rerun()


if __name__ == "__main__":
    main()
