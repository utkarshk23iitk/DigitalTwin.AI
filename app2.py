"""Twinly: a time-aware, single-page digital-twin control room.

This dashboard replays data/demo_live as an incoming production stream.  Live
operations only consume rows whose timestamp is at or before the playback
clock; saved validation metadata is kept in a separate analytics section.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "data" / "demo_live"
ARTIFACT_DIR = ROOT / "artifacts"

NAVY = "#0b1728"
INK = "#15233a"
MUTED = "#6d788a"
PAPER = "#f4f1e9"
WHITE = "#fffdf8"
TEAL = "#0e9384"
CYAN = "#2b8fb8"
AMBER = "#e99b26"
RED = "#d45454"
GREEN = "#4d9e73"
STATE_COLORS = {
    "RUNNING": GREEN,
    "IDLE": MUTED,
    "STARVED": "#8b93a1",
    "BLOCKED": AMBER,
    "WARNING": "#d9852e",
    "DEGRADED": "#d46d48",
    "FAULT": RED,
    "STARTUP": "#758196",
}
ACTION_COLORS = {
    "AUTO-ACT": RED,
    "HUMAN-VERIFY": AMBER,
    "HUMAN REVIEW": AMBER,
    "MONITOR": CYAN,
    "PASS": GREEN,
}
EXPECTED_CHANNELS = {"temperature", "torque", "vibration"}
SECTIONS = [
    ("live", "Live Line"),
    ("defects", "Defect Intelligence"),
    ("bottlenecks", "Bottleneck Intelligence"),
    ("health", "Station Health"),
    ("copilot", "AI Copilot"),
    ("analytics", "Model Analytics"),
    ("manager", "Operational Perspectives"),
    ("business", "Business & Scale"),
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
        :root { --navy:#0b1728; --ink:#15233a; --paper:#f4f1e9; --teal:#0e9384; --amber:#e99b26; --red:#d45454; }
        html { scroll-behavior: smooth; }
        body, [class*="st-"] { font-family: "Manrope", sans-serif; color: var(--ink); }
        .stApp { background:
          radial-gradient(circle at 7% 0%, rgba(14,147,132,.12), transparent 25rem),
          radial-gradient(circle at 96% 8%, rgba(233,155,38,.12), transparent 24rem),
          linear-gradient(180deg,#f7f5ef 0%,#f1eee5 100%); }
        .block-container { max-width: 1560px; padding-bottom: 5rem; }
        [data-testid="stMainBlockContainer"] { padding-top:4.5rem !important; }
        h1,h2,h3 { font-family:"Barlow Condensed",sans-serif !important; letter-spacing:.01em; color:var(--navy); }
        h1 { font-size:3.05rem !important; line-height:.95 !important; }
        h2 { font-size:2.2rem !important; margin-top:.25rem !important; }
        h3 { font-size:1.45rem !important; }
        .hero { border-radius:26px; padding:1.25rem 1.45rem; color:#f8f4e9; overflow:hidden;
          background:linear-gradient(120deg,#081523 0%,#102b3b 58%,#0e625d 130%);
          box-shadow:0 24px 60px rgba(11,23,40,.16); position:relative; }
        .hero:after { content:""; position:absolute; width:300px; height:300px; border:1px solid rgba(255,255,255,.1);
          border-radius:50%; right:-80px; top:-180px; box-shadow:0 0 0 36px rgba(255,255,255,.025),0 0 0 72px rgba(255,255,255,.02); }
        .hero-grid { display:flex; justify-content:space-between; align-items:center; gap:1rem; position:relative; z-index:1; }
        .eyebrow { font-family:"Barlow Condensed",sans-serif; font-size:.82rem; font-weight:700; letter-spacing:.17em; text-transform:uppercase; color:#73d2c7; }
        .hero-title { font-family:"Barlow Condensed",sans-serif; font-size:2.25rem; font-weight:700; line-height:1; margin:.15rem 0; }
        .hero-copy { color:rgba(255,255,255,.72); font-size:.9rem; }
        .clock { text-align:right; min-width:170px; }
        .clock strong { font-family:"Barlow Condensed",sans-serif; display:block; font-size:2.2rem; line-height:1; }
        .clock span { color:rgba(255,255,255,.62); font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; }
        [data-testid="stSidebar"] { background:rgba(255,253,248,.97); border-right:1px solid rgba(11,23,40,.08); }
        [data-testid="stSidebar"] .block-container { padding:1.2rem .9rem 2rem; }
        [data-testid="stSidebar"] h3 { margin:.15rem 0 .2rem !important; }
        [data-testid="stSidebar"] .stButton { margin:.18rem 0; }
        [data-testid="stSidebar"] .stButton > button { min-height:2.65rem; border-radius:10px; justify-content:flex-start;
          padding:.55rem .8rem; border:1px solid rgba(11,23,40,.11); background:rgba(255,255,255,.66);
          color:#334158; box-shadow:none; font-family:"Manrope",sans-serif; font-size:.82rem; }
        [data-testid="stSidebar"] .stButton > button:hover { border-color:#0e9384; color:#0b1728; background:#eef8f5; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] { background:#0b1728; color:#fff;
          border-color:#0b1728; box-shadow:0 8px 18px rgba(11,23,40,.16); }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:before,
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:before { content:""; width:4px; height:18px;
          border-radius:4px; background:#51c8ba; margin-right:.35rem; }
        [data-testid="stSidebar"] .stButton > button p { color:inherit !important; }
        .side-nav-copy { color:#6d788a; font-size:.78rem; line-height:1.5; margin:.15rem 0 .8rem; }
        .sidebar-section { color:#7b8797; font-family:"Barlow Condensed",sans-serif; font-size:.72rem; font-weight:700;
          letter-spacing:.13em; text-transform:uppercase; margin:1.2rem 0 .35rem; padding-top:.9rem;
          border-top:1px solid rgba(11,23,40,.09); }
        .playback-readout { border-radius:12px; background:#eaf5f2; color:#24564f; padding:.55rem .7rem;
          font-family:"Barlow Condensed",sans-serif; font-size:1rem; font-weight:700; margin:.25rem 0 .65rem; }
        .section-anchor { scroll-margin-top:8rem; height:1px; }
        .section-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin:2rem 0 .7rem; }
        .section-head > * { min-width:0; }
        .section-head h2 { margin:0 !important; }
        .section-head p { color:#707b8b; max-width:590px; margin:0; text-align:right; font-size:.88rem; overflow-wrap:anywhere; }
        .kpi-card,.panel,.station-node,.bn-card,.trust-card,.persona-card { border:1px solid rgba(11,23,40,.09);
          background:rgba(255,253,248,.86); box-shadow:0 12px 30px rgba(11,23,40,.055); }
        .kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin-top:.85rem; }
        .kpi-card { border-radius:17px; padding:.82rem .9rem; min-height:118px; border-top:3px solid var(--accent,#0e9384); }
        .kpi-label { color:#748093; font-size:.69rem; font-weight:700; text-transform:uppercase; letter-spacing:.09em; }
        .kpi-value { font-family:"Barlow Condensed",sans-serif; font-weight:700; font-size:1.75rem; line-height:1.08;
          color:#0b1728; margin:.22rem 0; white-space:nowrap; word-break:normal; overflow-wrap:normal; }
        .kpi-sub { color:#697587; font-size:.72rem; }
        .panel { border-radius:20px; padding:1rem 1.05rem; }
        .line-stage { color:#7c8797; font-family:"Barlow Condensed",sans-serif; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:.75rem 0 .35rem; }
        .pipeline { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; }
        .station-node { border-radius:18px; padding:.75rem; min-height:168px; position:relative; overflow:hidden; border-top:4px solid var(--state); }
        .station-node:after { content:""; position:absolute; width:80px; height:80px; border:15px solid color-mix(in srgb,var(--state) 10%,transparent); border-radius:50%; right:-35px; bottom:-43px; }
        .station-top { display:flex; justify-content:space-between; gap:.4rem; align-items:flex-start; }
        .station-id { font-family:"Barlow Condensed",sans-serif; font-size:1.32rem; font-weight:700; }
        .station-name { color:#727d8d; font-size:.69rem; min-height:1.9em; }
        .state-pill,.coverage-pill,.severity-pill { display:inline-block; border-radius:999px; color:#fff; font-size:.61rem; font-weight:700; letter-spacing:.05em; padding:.22rem .42rem; }
        .station-unit { margin:.55rem 0 .42rem; font-family:"Barlow Condensed",sans-serif; font-size:1.02rem; color:#1e4b5a; }
        .station-grid { display:grid; grid-template-columns:1fr 1fr; gap:.3rem .5rem; font-size:.65rem; color:#768193; }
        .station-grid strong { display:block; color:#1a2a40; font-size:.76rem; }
        .buffer-row { display:flex; align-items:center; gap:.35rem; margin:.35rem 0 1rem; color:#778294; font-size:.69rem; }
        .buffer-line { flex:1; height:2px; background:linear-gradient(90deg,#0e9384,#e99b26); position:relative; }
        .buffer-line:after { content:""; position:absolute; right:0; top:-3px; border-left:6px solid #e99b26; border-top:4px solid transparent; border-bottom:4px solid transparent; }
        .flow-dot { width:8px;height:8px;border-radius:50%;background:#0e9384;position:absolute;top:-3px;animation:flow 2.2s linear infinite;box-shadow:0 0 10px #0e9384; }
        @keyframes flow { from{left:0} to{left:calc(100% - 8px)} }
        .bn-card { border-radius:20px; padding:1rem 1.05rem; min-height:215px; border-left:5px solid var(--accent); }
        .bn-title { font-family:"Barlow Condensed",sans-serif; font-size:1.55rem; font-weight:700; margin:.18rem 0; }
        .signal-row { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.75rem; }
        .signal { background:#f0eee7; border-radius:9px; padding:.35rem .5rem; font-size:.72rem; color:#526074; }
        .trust-card { border-radius:20px; padding:1rem; }
        .trust-flow { display:flex; align-items:center; justify-content:center; gap:.65rem; flex-wrap:wrap; }
        .trust-step { border-radius:13px; background:#eef1ef; padding:.65rem .8rem; min-width:128px; text-align:center; }
        .trust-step strong { display:block; font-family:"Barlow Condensed",sans-serif; font-size:1.35rem; }
        .trust-arrow { color:#8a95a4; font-size:1.35rem; }
        .llm-flow { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; margin:.2rem 0 1rem; }
        .llm-step { border:1px solid rgba(11,23,40,.09); border-radius:14px; padding:.72rem .8rem;
          background:rgba(255,253,248,.82); color:#667286; font-size:.72rem; line-height:1.45; }
        .llm-step strong { display:block; color:#10233a; font-family:"Barlow Condensed",sans-serif;
          font-size:1.05rem; margin-bottom:.12rem; }
        .alert { display:grid; grid-template-columns:75px 80px 1fr 95px; gap:.6rem; align-items:center; padding:.56rem .7rem;
          border-bottom:1px solid rgba(11,23,40,.07); font-size:.78rem; }
        .alert:last-child { border-bottom:0; }
        .persona-card { border-radius:18px; padding:1rem; height:100%; }
        .persona-card h3 { margin:.2rem 0 .5rem; }
        .empty-honest { border:1px dashed #aeb6c0; border-radius:16px; padding:1rem; color:#687486; background:rgba(255,255,255,.45); }
        div[data-testid="stMetric"] { background:rgba(255,253,248,.85); border:1px solid rgba(11,23,40,.08); border-radius:16px; padding:.75rem; }
        div[data-testid="stPlotlyChart"] { border-radius:18px; overflow:hidden; }
        div[data-testid="stDataFrame"] { border:1px solid rgba(11,23,40,.08); border-radius:15px; overflow:hidden; }
        .stButton>button { border-radius:12px; font-weight:700; border-color:rgba(11,23,40,.15); }
        .stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"] { background:#0b1728; color:#fff; }
        .stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p { color:inherit !important; }
        @media(max-width:900px){
          [data-testid="stMainBlockContainer"] { padding-top:4rem !important; }
          .pipeline,.kpi-grid,.llm-flow{grid-template-columns:repeat(2,minmax(0,1fr))}
          .section-head{display:block}.section-head p{text-align:left;margin-top:.35rem}.hero-title{font-size:1.75rem}.clock{min-width:120px}
        }
        @media(max-width:560px){ .pipeline,.kpi-grid,.llm-flow{grid-template-columns:1fr}.hero-grid{align-items:flex-start}.clock{min-width:105px}.clock strong{font-size:1.65rem} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_csv(name: str) -> pd.DataFrame:
    path = DEMO_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_demo_data(signature: float) -> dict[str, Any]:
    del signature
    names = [
        "line_events.csv", "buffer_history.csv", "health_log.csv", "sensor_log.csv",
        "unit_log.csv", "unit_summary.csv", "station_stats.csv", "station_registry.csv",
        "unit_visit_times.csv", "demo_assessment.csv", "virtual_sensor_events.csv",
    ]
    data = {Path(name).stem: _read_csv(name) for name in names}
    for name in ("manifest", "bottleneck_report"):
        path = DEMO_DIR / f"{name}.json"
        data[name] = json.loads(path.read_text()) if path.exists() else {}
    meta_path = ARTIFACT_DIR / "defect_model_meta.json"
    data["model_meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return data


@st.cache_data(show_spinner=False)
def load_feature_importance(model_mtime: float) -> pd.DataFrame:
    del model_mtime
    path = ARTIFACT_DIR / "defect_model.json"
    if not path.exists():
        return pd.DataFrame(columns=["feature", "gain"])
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(path)
        scores = model.get_booster().get_score(importance_type="gain")
        out = pd.DataFrame(scores.items(), columns=["feature", "gain"])
        return out.sort_values("gain", ascending=False).head(14)
    except Exception:
        return pd.DataFrame(columns=["feature", "gain"])


def data_signature() -> float:
    paths = list(DEMO_DIR.glob("*.csv")) + list(DEMO_DIR.glob("*.json"))
    return max((p.stat().st_mtime for p in paths), default=0.0)


def _window(frame: pd.DataFrame, t_now: float, seconds: float, time_col: str = "t") -> pd.DataFrame:
    if frame.empty or time_col not in frame:
        return frame.iloc[0:0].copy()
    return frame[(frame[time_col] <= t_now) & (frame[time_col] >= t_now - seconds)].copy()


def _latest_by(frame: pd.DataFrame, t_now: float, group: str, time_col: str = "t") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    seen = frame[frame[time_col] <= t_now]
    return seen.sort_values(time_col).groupby(group, as_index=False).tail(1)


def _state_durations(events: pd.DataFrame, stations: list[int], t_now: float, t_min: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station in stations:
        group = events[(events["station"] == station) & (events["t"] <= t_now)].sort_values("t")
        points = [(t_min, "STARTUP")] + list(group[["t", "state"]].itertuples(index=False, name=None))
        totals: dict[str, float] = {}
        for idx, (start, state) in enumerate(points):
            end = points[idx + 1][0] if idx + 1 < len(points) else t_now
            totals[str(state)] = totals.get(str(state), 0.0) + max(0.0, float(end) - float(start))
        horizon = max(1.0, t_now - t_min)
        rows.append({
            "station": station,
            "util_live": totals.get("WORKING", 0.0) / horizon,
            "blocked_live": totals.get("BLOCKED", 0.0) / horizon,
            "starved_live": totals.get("STARVED", 0.0) / horizon,
            "down_live": totals.get("DOWN", 0.0) / horizon,
        })
    return pd.DataFrame(rows)


def _queue_slope(buffers: pd.DataFrame, station: int, t_now: float, seconds: float = 400.0) -> float:
    col = f"buffer_{station}"
    recent = _window(buffers, t_now, seconds)
    if col not in recent or len(recent) < 2 or recent["t"].nunique() < 2:
        return 0.0
    return float(np.polyfit(recent["t"], recent[col], 1)[0] * 100.0)


def _cycle_stats(visits: pd.DataFrame, station: int, t_now: float) -> tuple[float, float]:
    times = visits[(visits["station"] == station) & (visits["t_global"] <= t_now)]["t_global"].sort_values()
    gaps = times.diff().dropna().tail(12)
    if gaps.empty:
        return float("nan"), 0.0
    recent = float(gaps.tail(5).median())
    baseline = float(gaps.median())
    drift = (recent - baseline) / max(baseline, 1e-9)
    return recent, float(drift)


def _coverage_for_station(data: dict[str, Any], station: int, t_now: float) -> dict[str, Any]:
    sensors = data["sensor_log"]
    inferred = data["virtual_sensor_events"]
    registry = data["station_registry"]
    registry_row = registry[registry["station"] == station] if not registry.empty else registry
    tier = str(registry_row.iloc[0].get("tier", "A")) if not registry_row.empty else "A"
    has_physical_sensor = tier == "A"
    observed = sensors[(sensors.get("station") == station) & (sensors.get("t", pd.Series(dtype=float)) <= t_now)] if not sensors.empty and has_physical_sensor else sensors.iloc[0:0]
    channels = set(observed.get("channel", pd.Series(dtype=str)).astype(str)) & EXPECTED_CHANNELS
    virtual = inferred[(inferred.get("station") == station) & (inferred.get("t_global", pd.Series(dtype=float)) <= t_now)] if not inferred.empty else inferred
    virtual_channels = set(virtual.get("channel", pd.Series(dtype=str)).astype(str)) & EXPECTED_CHANNELS
    if channels == EXPECTED_CHANNELS:
        status = "MEASURED"
        coverage = 1.0
        confidence = 1.0
    elif channels:
        status = "PARTIAL"
        coverage = len(channels) / len(EXPECTED_CHANNELS)
        confidence = coverage
    elif virtual_channels:
        status = "INFERRED / VIRTUAL SENSOR"
        coverage = len(virtual_channels) / len(EXPECTED_CHANNELS)
        confidence = float(virtual["confidence"].tail(3).mean()) if "confidence" in virtual else 0.0
    else:
        status = "UNKNOWN"
        coverage = 0.0
        confidence = 0.0
    return {"coverage_status": status, "sensor_coverage": coverage, "coverage_confidence": confidence}


def get_live_state(data: dict[str, Any], t_now: float) -> dict[str, Any]:
    stats = data["station_stats"].copy().sort_values("index")
    stations = [int(v) for v in stats["index"]]
    events = data["line_events"]
    health = data["health_log"]
    buffers = data["buffer_history"]
    visits = data["unit_visit_times"]
    assessments = data["demo_assessment"]
    t_min = float(buffers["t"].min()) if not buffers.empty else 0.0

    latest_event = _latest_by(events, t_now, "station")
    latest_health = _latest_by(health, t_now, "station")
    duration = _state_durations(events, stations, t_now, t_min)
    snapshot = stats.merge(duration, left_on="index", right_on="station", how="left").drop(columns=["station"])
    snapshot = snapshot.merge(latest_event[["station", "state"]], left_on="index", right_on="station", how="left").drop(columns=["station"])
    snapshot = snapshot.merge(latest_health[["station", "health_true"]], left_on="index", right_on="station", how="left").drop(columns=["station"])
    snapshot["state"] = snapshot["state"].fillna("STARTUP")
    snapshot["health_true"] = snapshot["health_true"].fillna(1.0)

    buffer_row = buffers[buffers["t"] <= t_now].tail(1)
    current_visits = visits[visits["t_global"] <= t_now].sort_values("t_global").groupby("station", as_index=False).tail(1) if not visits.empty else visits
    unit_map = dict(zip(current_visits.get("station", []), current_visits.get("unit_id", [])))
    capacity = int(data["manifest"].get("buffer_capacity", 4))
    for idx, row in snapshot.iterrows():
        station = int(row["index"])
        snapshot.loc[idx, "buffer_in"] = int(buffer_row.iloc[0].get(f"buffer_{station}", 0)) if not buffer_row.empty else 0
        snapshot.loc[idx, "queue_slope"] = _queue_slope(buffers, station, t_now)
        cycle, drift = _cycle_stats(visits, station, t_now)
        snapshot.loc[idx, "cycle_time"] = cycle
        snapshot.loc[idx, "cycle_drift"] = drift
        snapshot.loc[idx, "current_unit"] = unit_map.get(station, np.nan)
        coverage = _coverage_for_station(data, station, t_now)
        for key, value in coverage.items():
            snapshot.loc[idx, key] = value

    state_map = {"WORKING": "RUNNING", "BLOCKED": "BLOCKED", "STARVED": "STARVED", "DOWN": "FAULT", "STARTUP": "STARTUP"}
    snapshot["display_state"] = snapshot["state"].map(state_map).fillna("IDLE")
    snapshot.loc[(snapshot["health_true"] < .82) & (snapshot["display_state"] == "RUNNING"), "display_state"] = "DEGRADED"
    snapshot.loc[(snapshot["health_true"] < .9) & (snapshot["display_state"] == "RUNNING"), "display_state"] = "WARNING"
    snapshot["queue_pressure"] = (snapshot["buffer_in"] / max(1, capacity)).clip(0, 1)
    downstream_starve = []
    for _, row in snapshot.iterrows():
        downstream = snapshot[snapshot["index"] > row["index"]]["starved_live"]
        downstream_starve.append(float(downstream.mean()) if len(downstream) else 0.0)
    snapshot["downstream_starve"] = downstream_starve
    # Match the backend's sustained-constraint method. Queue growth remains a
    # leading/emerging signal, avoiding the always-full source buffer artifact.
    snapshot["live_score"] = (
        snapshot["util_live"].fillna(0) + snapshot["blocked_live"].fillna(0)
        - snapshot["starved_live"].fillna(0) + 1.5 * snapshot["downstream_starve"]
        + .12 * (1 - snapshot["health_true"])
    )

    visible_assessments = assessments[assessments["latest_t"] <= t_now].copy() if not assessments.empty else assessments.copy()
    risk_threshold = float(assessments["risk_threshold"].iloc[0]) if not assessments.empty else float(data["model_meta"].get("threshold", .5))
    visible_assessments["high_risk"] = visible_assessments.get("risk_score", 0) >= risk_threshold
    unit_station = visits[visits["t_global"] <= t_now].sort_values("t_global").groupby("unit_id", as_index=False).tail(1) if not visits.empty else visits
    if not visible_assessments.empty and not unit_station.empty:
        visible_assessments = visible_assessments.merge(unit_station[["unit_id", "station", "t_global"]], on="unit_id", how="left")

    ranking = snapshot.sort_values("live_score", ascending=False).reset_index(drop=True)
    current_bn = ranking.iloc[0]
    emerging_candidates = snapshot[(snapshot["queue_slope"] > .02) | (snapshot["cycle_drift"] > .12)].copy()
    if not emerging_candidates.empty:
        emerging_candidates["emerging_score"] = (
            emerging_candidates["queue_slope"].clip(lower=0) +
            emerging_candidates["cycle_drift"].clip(lower=0) +
            .35 * emerging_candidates["queue_pressure"]
        )
        emerging = emerging_candidates.sort_values("emerging_score", ascending=False).iloc[0]
    else:
        emerging = None

    last_station = max(stations)
    completed_ids = set(visits[(visits["station"] == last_station) & (visits["t_global"] <= t_now)]["unit_id"]) if not visits.empty else set()
    started_ids = set(visits[(visits["station"] == min(stations)) & (visits["t_global"] <= t_now)]["unit_id"]) if not visits.empty else set()
    throughput_window = max(300.0, min(900.0, t_now - t_min))
    recent_completed = visits[(visits["station"] == last_station) & (visits["t_global"] <= t_now) & (visits["t_global"] >= t_now - throughput_window)] if not visits.empty else visits
    throughput = len(recent_completed) * 3600.0 / throughput_window
    degraded = int((snapshot["health_true"] < .82).sum())
    high_risk_count = int(visible_assessments.get("high_risk", pd.Series(dtype=bool)).sum())
    current_queue_is_constraint = int(current_bn["index"]) > min(stations) and current_bn["queue_pressure"] >= .9
    line_status = "CRITICAL" if degraded or current_queue_is_constraint else "WATCH" if high_risk_count or emerging is not None else "STABLE"
    kpis = {
        "units_completed": len(completed_ids), "throughput": throughput,
        "wip": len(started_ids - completed_ids), "current_bottleneck": f"S{int(current_bn['index']):02d}",
        "high_risk_units": high_risk_count, "degraded_stations": degraded,
        "average_health": float(snapshot["health_true"].mean()), "line_status": line_status,
    }
    return {
        "timestamp": t_now, "snapshot": snapshot, "assessments": visible_assessments,
        "risk_threshold": risk_threshold, "current_bottleneck": current_bn,
        "emerging_bottleneck": emerging, "kpis": kpis, "capacity": capacity,
        "completed_ids": completed_ids, "started_ids": started_ids,
    }


def _anchor(section_id: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div id="{section_id}" class="section-anchor"></div>'
        f'<div class="section-head"><h2>{title}</h2><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    section_ids = [section_id for section_id, _ in SECTIONS]
    label_to_id = {label: section_id for section_id, label in SECTIONS}
    active = st.session_state.get("active_section", section_ids[0])
    # Migrate sessions created by the previous label-based radio navigation.
    active = label_to_id.get(active, active)
    if active not in section_ids:
        active = section_ids[0]
    st.session_state.active_section = active

    with st.sidebar:
        st.markdown("### Control Deck")
        st.markdown(
            '<div class="side-nav-copy">Choose one workspace. The main canvas updates immediately without scrolling through unrelated sections.</div>',
            unsafe_allow_html=True,
        )
        for section_id, label in SECTIONS:
            if st.button(
                label,
                key=f"nav_{section_id}",
                type="primary" if section_id == active else "secondary",
                width="stretch",
            ):
                st.session_state.active_section = section_id
                st.rerun()
    return active


def _kpi_card(label: str, value: str, sub: str, accent: str = TEAL) -> str:
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def render_kpis(live: dict[str, Any]) -> None:
    k = live["kpis"]
    status_color = GREEN if k["line_status"] == "STABLE" else AMBER if k["line_status"] == "WATCH" else RED
    cards = [
        ("Units completed", str(k["units_completed"]), "Observed at output by now", TEAL),
        ("Throughput", f'{k["throughput"]:.1f}/h', "Rolling output rate", CYAN),
        ("Work in progress", str(k["wip"]), "Started minus completed", AMBER),
        ("Current bottleneck", k["current_bottleneck"], "Highest live constraint score", RED),
        ("High-risk units", str(k["high_risk_units"]), "Model results available by now", RED),
        ("Degraded stations", str(k["degraded_stations"]), "Health below 82%", AMBER),
        ("Average health", f'{k["average_health"]:.0%}', "Across current station states", GREEN),
        ("Line status", k["line_status"], "Live operational posture", status_color),
    ]
    st.markdown(
        f'<div class="kpi-grid">{"".join(_kpi_card(*card) for card in cards)}</div>',
        unsafe_allow_html=True,
    )


def _trend_label(value: float) -> tuple[str, str]:
    if value > .02:
        return "GROWING ↑", RED
    if value < -.02:
        return "FALLING ↓", GREEN
    return "STABLE →", MUTED


def render_live_pipeline(live: dict[str, Any]) -> None:
    snapshot = live["snapshot"]
    capacity = live["capacity"]
    st.markdown('<div class="line-stage">INPUT · BODY & ASSEMBLY</div>', unsafe_allow_html=True)
    for start in range(0, len(snapshot), 4):
        cards: list[str] = []
        for _, row in snapshot.iloc[start:start + 4].iterrows():
            station = int(row["index"])
            state = str(row["display_state"])
            color = STATE_COLORS.get(state, MUTED)
            current = "No active unit" if pd.isna(row["current_unit"]) or state in {"STARTUP", "IDLE"} else f'U{int(row["current_unit"]):03d} in process'
            cycle = "waiting" if pd.isna(row["cycle_time"]) else f'{row["cycle_time"]:.1f}s'
            risk = min(1.0, float(row["live_score"]) / 1.8)
            cards.append(
                f'<div class="station-node" style="--state:{color}"><div class="station-top">'
                f'<div><div class="station-id">S{station:02d}</div><div class="station-name">{row["name"]}</div></div>'
                f'<span class="state-pill" style="background:{color}">{state}</span></div>'
                f'<div class="station-unit">{current}</div><div class="station-grid">'
                f'<span>UTILIZATION<strong>{row["util_live"]:.0%}</strong></span>'
                f'<span>HEALTH<strong>{row["health_true"]:.0%}</strong></span>'
                f'<span>CYCLE<strong>{cycle}</strong></span>'
                f'<span>BUFFER<strong>{int(row["buffer_in"])}/{capacity}</strong></span>'
                f'<span>COVERAGE<strong>{row["sensor_coverage"]:.0%}</strong></span>'
                f'<span>RISK<strong>{risk:.0%}</strong></span></div></div>'
            )
        st.markdown(f'<div class="pipeline">{"".join(cards)}</div>', unsafe_allow_html=True)
        if start + 4 < len(snapshot):
            downstream = int(snapshot.iloc[start + 4]["index"])
            slope = float(snapshot.iloc[start + 4]["queue_slope"])
            trend, trend_color = _trend_label(slope)
            queue = int(snapshot.iloc[start + 4]["buffer_in"])
            stage = "PAINT & CURE" if start == 0 else "FINAL ASSEMBLY & QUALITY"
            st.markdown(
                f'<div class="buffer-row"><b>BUFFER TO S{downstream:02d}</b><div class="buffer-line"><i class="flow-dot"></i></div>'
                f'<span>{queue}/{capacity} occupied</span><b style="color:{trend_color}">{trend}</b></div>'
                f'<div class="line-stage">{stage}</div>', unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="buffer-row"><b>FINAL OUTPUT</b><div class="buffer-line"><i class="flow-dot"></i></div>'
        f'<span>{live["kpis"]["units_completed"]} completed units observed</span></div>',
        unsafe_allow_html=True,
    )


def build_alerts(data: dict[str, Any], live: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    t_now = live["timestamp"]
    alerts: list[dict[str, Any]] = []
    for _, row in live["snapshot"].iterrows():
        station = int(row["index"])
        if row["display_state"] in {"FAULT", "DEGRADED"}:
            alerts.append({"t": t_now, "source": f"S{station:02d}", "message": f'{row["display_state"].title()} station state', "severity": "CRITICAL"})
        elif row["queue_slope"] > .02:
            alerts.append({"t": t_now, "source": f"S{station:02d}", "message": f'Queue rising {row["queue_slope"]:+.2f} units/100s', "severity": "WARNING"})
        if row["coverage_status"] == "UNKNOWN":
            alerts.append({"t": t_now, "source": f"S{station:02d}", "message": "No reliable sensor evidence available", "severity": "INFO"})
    assessment = live["assessments"]
    if not assessment.empty:
        for _, row in assessment[assessment["high_risk"]].nlargest(4, "risk_score").iterrows():
            action = "HUMAN REVIEW" if row["effective_trust"] < .5 else str(row["action"])
            alerts.append({"t": float(row["latest_t"]), "source": f'U{int(row["unit_id"]):03d}', "message": f'Defect risk {row["risk_score"]:.0%} · {action}', "severity": "HIGH"})
    priority = {"CRITICAL": 4, "HIGH": 3, "WARNING": 2, "INFO": 1}
    return sorted(alerts, key=lambda a: (priority.get(a["severity"], 0), a["t"]), reverse=True)[:limit]


def render_alerts(alerts: list[dict[str, Any]]) -> None:
    if not alerts:
        st.markdown('<div class="empty-honest">No alert condition is supported by the data at this playback time.</div>', unsafe_allow_html=True)
        return
    colors = {"CRITICAL": RED, "HIGH": RED, "WARNING": AMBER, "INFO": CYAN}
    rows = []
    for item in alerts:
        stamp = time.strftime("%H:%M:%S", time.gmtime(float(item["t"])))
        color = colors[item["severity"]]
        rows.append(
            f'<div class="alert"><b>{stamp}</b><b>{item["source"]}</b><span>{item["message"]}</span>'
            f'<span class="severity-pill" style="background:{color}">{item["severity"]}</span></div>'
        )
    st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)


def _sensor_baseline(unit_log: pd.DataFrame, t_now: float) -> pd.DataFrame:
    visible = unit_log[(unit_log["t_global"] <= t_now) & unit_log["value_observed"].notna()].copy()
    if visible.empty:
        return pd.DataFrame()
    return visible.groupby(["station", "channel"])["value_observed"].agg(["mean", "std"]).reset_index()


def trace_unit(data: dict[str, Any], live: dict[str, Any], unit_id: int) -> dict[str, Any]:
    t_now = live["timestamp"]
    visits = data["unit_visit_times"]
    path = visits[(visits["unit_id"] == unit_id) & (visits["t_global"] <= t_now)].sort_values("t_global")
    unit_log = data["unit_log"]
    observations = unit_log[(unit_log["unit_id"] == unit_id) & (unit_log["t_global"] <= t_now) & unit_log["value_observed"].notna()].copy()
    abnormal: list[str] = []
    suspect = int(path.iloc[-1]["station"]) if not path.empty else None
    baseline = _sensor_baseline(unit_log, t_now)
    if not observations.empty and not baseline.empty:
        scored = observations.merge(baseline, on=["station", "channel"], how="left")
        scored["z"] = ((scored["value_observed"] - scored["mean"]).abs() / scored["std"].replace(0, np.nan)).fillna(0)
        top = scored.sort_values("z", ascending=False).head(3)
        strong = top[top["z"] >= 1.5]
        if not strong.empty:
            suspect = int(strong.iloc[0]["station"])
            abnormal = [f'S{int(r.station):02d} {r.channel} deviation {r.z:.1f}σ' for r in strong.itertuples()]
    if not abnormal and not path.empty:
        visited = live["snapshot"][live["snapshot"]["index"].isin(path["station"])]
        if not visited.empty:
            weakest = visited.sort_values("health_true").iloc[0]
            suspect = int(weakest["index"])
            abnormal = [f'S{suspect:02d} health is lowest on the observed path ({weakest["health_true"]:.0%})']
    return {"path": path, "suspected_origin": suspect, "abnormal_signals": abnormal}


def _unit_path_html(path: pd.DataFrame, suspect: int | None, total_stations: int) -> str:
    visited = set(path["station"].astype(int)) if not path.empty else set()
    current = max(visited) if visited else -1
    items = []
    for station in range(total_stations):
        if station == suspect:
            marker, color = "⚠", AMBER
        elif station in visited:
            marker, color = "✓", GREEN
        else:
            marker, color = "·", MUTED
        weight = "700" if station == current else "500"
        items.append(f'<span style="color:{color};font-weight:{weight}">S{station:02d} {marker}</span>')
    return '<div style="display:flex;gap:.45rem;flex-wrap:wrap;font-size:.75rem">' + '<span style="color:#a4abb4">→</span>'.join(items) + '</div>'


def render_defects(data: dict[str, Any], live: dict[str, Any]) -> None:
    assessment = live["assessments"]
    if assessment.empty:
        st.markdown('<div class="empty-honest">No unit prediction has become available yet. Predictions appear only after sufficient unit evidence reaches the model.</div>', unsafe_allow_html=True)
        return
    top = assessment.sort_values("risk_score", ascending=False).head(6)
    risk_thr = live["risk_threshold"]
    left, right = st.columns([1.08, .92])
    with left:
        fig = px.scatter(
            top, x="effective_trust", y="risk_score", size="model_confidence", color="action",
            hover_name=top["unit_id"].map(lambda value: f"Unit U{int(value):03d}"),
            color_discrete_map=ACTION_COLORS,
            labels={"effective_trust": "Effective trust", "risk_score": "Defect risk", "model_confidence": "Model confidence"},
        )
        fig.add_hline(
            y=risk_thr, line_dash="dash", line_color=RED,
            annotation_text="High-risk threshold", annotation_position="bottom left",
            annotation_font_color=RED, annotation_font_size=11,
        )
        fig.add_vline(
            x=.5, line_dash="dot", line_color=AMBER,
            annotation_text="Human / auto gate", annotation_position="bottom right",
            annotation_font_color="#9b6414", annotation_font_size=11,
        )
        fig.update_traces(marker=dict(opacity=.82, line=dict(width=1, color="rgba(11,23,40,.25)")))
        fig.update_layout(
            height=390, margin=dict(l=20, r=15, t=70, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,253,248,.65)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        selected = st.selectbox("Inspect a predicted unit", options=list(top["unit_id"].astype(int)), format_func=lambda value: f"U{value:03d}", key="defect_unit_focus")
        row = assessment[assessment["unit_id"] == selected].iloc[0]
        trace = trace_unit(data, live, selected)
        suspect = trace["suspected_origin"]
        station = int(row["station"]) if pd.notna(row.get("station")) else -1
        station_health = live["snapshot"].set_index("index")["health_true"].get(station, np.nan)
        action = "HUMAN REVIEW" if row["model_confidence"] >= .5 and row["input_trust"] < .7 else str(row["action"])
        st.markdown(
            f'<div class="panel"><div class="eyebrow">UNIT INTELLIGENCE · {action}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-end"><h3 style="margin:.2rem 0">U{selected:03d}</h3>'
            f'<b style="font:700 1.65rem Barlow Condensed;color:{RED if row["risk_score"] >= risk_thr else AMBER}">{row["risk_score"]:.1%} risk</b></div>'
            f'{_unit_path_html(trace["path"], suspect, len(live["snapshot"]))}<hr style="border:0;border-top:1px solid #e1ded5">'
            f'<div class="station-grid"><span>CURRENT STATION<strong>S{station:02d}</strong></span>'
            f'<span>SUSPECTED ORIGIN<strong>{"S" + format(suspect,"02d") if suspect is not None else "Insufficient evidence"}</strong></span>'
            f'<span>STATION HEALTH<strong>{station_health:.0%}</strong></span><span>DETECTED AT<strong>t={row["latest_t"]:.0f}s</strong></span>'
            f'<span>MODEL CONFIDENCE<strong>{row["model_confidence"]:.0%}</strong></span><span>INPUT TRUST<strong>{row["input_trust"]:.0%}</strong></span></div></div>',
            unsafe_allow_html=True,
        )
        signals = trace["abnormal_signals"]
        st.caption("Suspected Origin is an evidence-based diagnostic lead, not a proven causal root cause.")
        if signals:
            st.markdown("**Relevant observed signals:** " + " · ".join(signals))
        if suspect is not None and not trace["path"].empty:
            origin_visit = trace["path"][trace["path"]["station"] == suspect]
            if not origin_visit.empty:
                origin_t = float(origin_visit.iloc[0]["t_global"])
                visits = data["unit_visit_times"]
                cohort = visits[
                    (visits["station"] == suspect) & (visits["t_global"] > origin_t)
                    & (visits["t_global"] <= min(live["timestamp"], origin_t + 600))
                    & (visits["unit_id"] != selected)
                ]["unit_id"].drop_duplicates().head(5)
                if len(cohort):
                    labels = ", ".join(f"U{int(value):03d}" for value in cohort)
                    st.markdown(f"**Downstream watch cohort:** {labels}")
                    st.caption("These units followed through the suspected station within 10 simulation minutes. This proximity is a review cue, not proof they are defective.")
    show = top[["unit_id", "risk_score", "model_confidence", "input_trust", "effective_trust", "action", "latest_t"]].copy()
    show.columns = ["Unit", "Defect risk", "Model confidence", "Input trust", "Effective trust", "Action", "Detection time"]
    show["Unit"] = show["Unit"].map(lambda value: f"U{int(value):03d}")
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Defect risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Model confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Input trust": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Effective trust": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })


def render_bottlenecks(data: dict[str, Any], live: dict[str, Any]) -> None:
    current = live["current_bottleneck"]
    emerging = live["emerging_bottleneck"]
    trend, trend_color = _trend_label(float(current["queue_slope"]))
    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="bn-card" style="--accent:{RED}"><div class="eyebrow" style="color:{RED}">CURRENT BOTTLENECK</div>'
            f'<div class="bn-title">S{int(current["index"]):02d} · {current["name"]}</div>'
            f'<div style="color:#657184;font-size:.8rem">Highest live constraint score from observed state duration, queue pressure, cycle drift and health.</div>'
            f'<div class="signal-row"><span class="signal">Utilization {current["util_live"]:.0%}</span>'
            f'<span class="signal">Queue {int(current["buffer_in"])}/{live["capacity"]}</span>'
            f'<span class="signal" style="color:{trend_color}">{trend}</span>'
            f'<span class="signal">Cycle drift {current["cycle_drift"]:+.0%}</span></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        if emerging is None:
            st.markdown(
                f'<div class="bn-card" style="--accent:{GREEN}"><div class="eyebrow" style="color:{GREEN}">EMERGING BOTTLENECK</div>'
                '<div class="bn-title">No clear formation signal</div><div style="color:#657184;font-size:.8rem">'
                'No station currently clears the supported queue-growth or cycle-drift trigger. Twinly will keep watching.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bn-card" style="--accent:{AMBER}"><div class="eyebrow" style="color:{AMBER}">EMERGING BOTTLENECK</div>'
                f'<div class="bn-title">S{int(emerging["index"]):02d} · {emerging["name"]}</div>'
                '<div style="color:#657184;font-size:.8rem">Leading warning from supported recent signals. This is not a precise time-to-failure forecast.</div>'
                f'<div class="signal-row"><span class="signal">Queue {emerging["queue_slope"]:+.2f}/100s</span>'
                f'<span class="signal">Cycle drift {emerging["cycle_drift"]:+.0%}</span>'
                f'<span class="signal">Utilization {emerging["util_live"]:.0%}</span>'
                f'<span class="signal">Pressure {emerging["queue_pressure"]:.0%}</span></div></div>', unsafe_allow_html=True,
            )

    chart_left, chart_right = st.columns([1.35, .65])
    with chart_left:
        buffers = _window(data["buffer_history"], live["timestamp"], 900)
        focus = {int(current["index"])}
        if emerging is not None:
            focus.add(int(emerging["index"]))
        fig = go.Figure()
        for station in sorted(focus):
            col = f"buffer_{station}"
            if col in buffers:
                name = live["snapshot"].set_index("index").loc[station, "name"]
                fig.add_trace(go.Scatter(x=buffers["t"], y=buffers[col], mode="lines", name=f"S{station:02d} {name}", line_shape="hv", line=dict(width=3)))
        fig.update_layout(title="Recent queue evidence", height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Input buffer units", xaxis_title="Simulation time (s)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,253,248,.65)", legend_orientation="h")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with chart_right:
        ranking = live["snapshot"].nlargest(7, "live_score").sort_values("live_score")
        fig = px.bar(ranking, x="live_score", y=ranking.apply(lambda row: f'S{int(row["index"]):02d} {row["name"]}', axis=1), orientation="h", color="queue_pressure", color_continuous_scale=[[0, GREEN], [.55, AMBER], [1, RED]])
        fig.update_layout(title="Live constraint ranking", height=330, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Composite evidence score", yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _station_risk(live_score: float) -> str:
    if live_score >= 1.25:
        return "HIGH"
    if live_score >= .8:
        return "WATCH"
    return "LOW"


def render_station_health(data: dict[str, Any], live: dict[str, Any]) -> None:
    snapshot = live["snapshot"].copy()
    snapshot["risk"] = snapshot["live_score"].map(_station_risk)
    snapshot["station_label"] = snapshot.apply(lambda row: f'S{int(row["index"]):02d} · {row["name"]}', axis=1)
    table = snapshot[["station_label", "display_state", "health_true", "util_live", "queue_pressure", "coverage_status", "sensor_coverage", "risk"]].copy()
    table.columns = ["Station", "State", "Health", "Utilization", "Queue pressure", "Coverage", "Sensor coverage", "Risk"]
    st.dataframe(table, hide_index=True, width="stretch", height=330, column_config={
        "Health": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Utilization": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Queue pressure": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Sensor coverage": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    })

    station = st.selectbox("Station detail", options=list(snapshot["index"].astype(int)), format_func=lambda value: snapshot.set_index("index").loc[value, "station_label"], key="health_station_focus")
    t_now = live["timestamp"]
    window_s = st.select_slider("Detail history window", options=[300, 600, 1200, 2400], value=1200, format_func=lambda value: f"{value // 60} min", key="health_window")
    health = _window(data["health_log"][data["health_log"]["station"] == station], t_now, window_s)
    visits = data["unit_visit_times"]
    station_visits = visits[(visits["station"] == station) & (visits["t_global"] <= t_now) & (visits["t_global"] >= t_now - window_s)].sort_values("t_global").copy()
    station_visits["cycle_time"] = station_visits["t_global"].diff()
    buffers = _window(data["buffer_history"], t_now, window_s)
    sensors = data["sensor_log"]
    sensors = sensors[(sensors["station"] == station) & (sensors["t"] <= t_now) & (sensors["t"] >= t_now - window_s) & sensors["channel"].isin(EXPECTED_CHANNELS)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Health", "Observed cycle interval", "Input buffer", "Measured sensors"))
    if not health.empty:
        fig.add_trace(go.Scatter(x=health["t"], y=health["health_true"], name="Health", line=dict(color=GREEN, width=3)), row=1, col=1)
    if not station_visits.empty:
        fig.add_trace(go.Scatter(x=station_visits["t_global"], y=station_visits["cycle_time"], name="Cycle", mode="lines+markers", line=dict(color=AMBER)), row=1, col=2)
    buffer_col = f"buffer_{station}"
    if buffer_col in buffers:
        fig.add_trace(go.Scatter(x=buffers["t"], y=buffers[buffer_col], name="Buffer", line_shape="hv", fill="tozeroy", line=dict(color=RED)), row=2, col=1)
    for channel, group in sensors.groupby("channel"):
        fig.add_trace(go.Scatter(x=group["t"], y=group["value"], name=channel.title(), mode="lines"), row=2, col=2)
    fig.update_layout(height=590, margin=dict(l=10, r=10, t=55, b=15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,253,248,.65)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_sensor_coverage(data: dict[str, Any], live: dict[str, Any]) -> None:
    snapshot = live["snapshot"].copy()
    coverage_colors = {"MEASURED": GREEN, "PARTIAL": AMBER, "INFERRED / VIRTUAL SENSOR": CYAN, "UNKNOWN": MUTED}
    cols = st.columns(4)
    for position, (_, row) in enumerate(snapshot.iterrows()):
        with cols[position % 4]:
            status = str(row["coverage_status"])
            confidence_copy = f'Inference confidence {row["coverage_confidence"]:.0%}' if status == "INFERRED / VIRTUAL SENSOR" else f'Channel coverage {row["sensor_coverage"]:.0%}'
            st.markdown(
                f'<div class="panel" style="margin-bottom:.65rem;border-top:3px solid {coverage_colors[status]}">'
                f'<div class="station-id">S{int(row["index"]):02d}</div><div class="station-name">{row["name"]}</div>'
                f'<span class="coverage-pill" style="background:{coverage_colors[status]};margin:.5rem 0">{status}</span>'
                f'<div style="font-size:.7rem;color:#6d788a">{confidence_copy}</div></div>', unsafe_allow_html=True,
            )
    virtual = data["virtual_sensor_events"]
    virtual = virtual[virtual["t_global"] <= live["timestamp"]].sort_values("t_global").groupby(["station", "channel"], as_index=False).tail(1) if not virtual.empty else virtual
    if not virtual.empty:
        virtual_show = virtual[["station", "channel", "method", "estimate", "confidence", "t_global"]].copy()
        virtual_show["station"] = virtual_show["station"].map(lambda value: f"S{int(value):02d}")
        virtual_show.columns = ["Station", "Channel", "Method", "Inferred value", "Confidence", "Updated at"]
        st.markdown("#### Virtual sensor values actually passed downstream")
        st.dataframe(virtual_show, hide_index=True, width="stretch", column_config={"Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    else:
        st.markdown('<div class="empty-honest">No virtual-sensor estimate exists at this playback time. Twinly does not fabricate a replacement for display.</div>', unsafe_allow_html=True)

    assessed = live["assessments"]
    if not assessed.empty:
        top = assessed.sort_values("risk_score", ascending=False).iloc[0]
        action = "HUMAN REVIEW" if top["risk_score"] >= live["risk_threshold"] and top["effective_trust"] < .5 else str(top["action"])
        st.markdown(
            f'<div class="trust-card"><div class="eyebrow">TWINLY TRUST LAYER · TOP CURRENT RISK</div><div class="trust-flow">'
            f'<div class="trust-step">Defect risk<strong>{top["risk_score"]:.0%}</strong></div><span class="trust-arrow">+</span>'
            f'<div class="trust-step">Model confidence<strong>{top["model_confidence"]:.0%}</strong></div><span class="trust-arrow">×</span>'
            f'<div class="trust-step">Input trust<strong>{top["input_trust"]:.0%}</strong></div><span class="trust-arrow">→</span>'
            f'<div class="trust-step">Effective trust<strong>{top["effective_trust"]:.0%}</strong></div><span class="trust-arrow">→</span>'
            f'<div class="trust-step" style="background:{ACTION_COLORS.get(action, NAVY)};color:white">Action<strong>{action}</strong></div>'
            '</div><div style="text-align:center;color:#758092;font-size:.72rem;margin-top:.7rem">Prediction + Data Reliability → Effective Trust → Action</div></div>',
            unsafe_allow_html=True,
        )


def build_llm_context(data: dict[str, Any], live: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = live["snapshot"]
    high_risk = live["assessments"].sort_values("risk_score", ascending=False).head(5)
    emerging = live["emerging_bottleneck"]
    context = {
        "timestamp_seconds": round(float(live["timestamp"]), 1),
        "line_status": live["kpis"]["line_status"],
        "live_kpis": {
            key: round(float(value), 3) if isinstance(value, (float, np.floating)) else value
            for key, value in live["kpis"].items()
        },
        "station_states": [
            {
                "station": f'S{int(row["index"]):02d}', "name": row["name"],
                "state": row["display_state"], "health": round(float(row["health_true"]), 3),
                "utilization": round(float(row["util_live"]), 3), "buffer": int(row["buffer_in"]),
                "queue_trend_per_100s": round(float(row["queue_slope"]), 3),
                "coverage": row["coverage_status"],
            }
            for _, row in snapshot.iterrows()
        ],
        "top_bottleneck": {
            "station": f'S{int(live["current_bottleneck"]["index"]):02d}',
            "name": live["current_bottleneck"]["name"],
            "queue": int(live["current_bottleneck"]["buffer_in"]),
            "utilization": round(float(live["current_bottleneck"]["util_live"]), 3),
        },
        "emerging_bottleneck": None if emerging is None else {
            "station": f'S{int(emerging["index"]):02d}', "name": emerging["name"],
            "queue_growth_per_100s": round(float(emerging["queue_slope"]), 3),
            "cycle_drift": round(float(emerging["cycle_drift"]), 3),
        },
        "high_risk_units": [],
        "sensor_coverage": [
            {"station": f'S{int(row["index"]):02d}', "status": row["coverage_status"], "confidence": round(float(row["coverage_confidence"]), 3)}
            for _, row in snapshot.iterrows() if row["coverage_status"] != "MEASURED"
        ],
        "recent_alerts": alerts[:6],
    }
    for _, row in high_risk.iterrows():
        trace = trace_unit(data, live, int(row["unit_id"]))
        suspect = trace["suspected_origin"]
        context["high_risk_units"].append({
            "unit": f'U{int(row["unit_id"]):03d}', "risk": round(float(row["risk_score"]), 4),
            "model_confidence": round(float(row["model_confidence"]), 3),
            "input_trust": round(float(row["input_trust"]), 3),
            "effective_trust": round(float(row["effective_trust"]), 3), "action": row["action"],
            "current_station": None if pd.isna(row.get("station")) else f'S{int(row["station"]):02d}',
            "suspected_origin": None if suspect is None else f"S{suspect:02d}",
            "origin_evidence": trace["abnormal_signals"][:3],
        })
    return context


def _offline_shift_summary(context: dict[str, Any]) -> str:
    kpi = context["live_kpis"]
    bottleneck = context["top_bottleneck"]
    emerging = context["emerging_bottleneck"]
    risks = context["high_risk_units"]
    coverage = context["sensor_coverage"]
    parts = [
        f'The line is currently **{context["line_status"].lower()}** at t={context["timestamp_seconds"]:.0f}s, '
        f'with {kpi["units_completed"]} completed units, {kpi["wip"]} units in progress, and '
        f'an observed rolling throughput of {kpi["throughput"]:.1f} units/hour.',
        f'**{bottleneck["station"]} {bottleneck["name"]}** is the current constraint based on the live evidence available now.',
    ]
    if emerging:
        parts.append(
            f'**{emerging["station"]} {emerging["name"]}** also has an emerging signal: '
            f'queue trend {emerging["queue_growth_per_100s"]:+.2f} units/100s and cycle drift {emerging["cycle_drift"]:+.0%}.'
        )
    else:
        parts.append("No separate emerging bottleneck currently clears the evidence trigger.")
    if risks:
        top = risks[0]
        parts.append(
            f'**{top["unit"]}** has the highest currently available defect risk ({top["risk"]:.1%}). '
            f'Its effective trust is {top["effective_trust"]:.1%}, so the policy action is **{top["action"]}**.'
        )
    else:
        parts.append("No unit-level prediction is available at this playback time.")
    if coverage:
        labels = ", ".join(f'{item["station"]} ({item["status"].lower()})' for item in coverage[:4])
        parts.append(f"Observability is incomplete at {labels}; decisions depending on those inputs should preserve human oversight.")
    return "\n\n".join(parts)


def _offline_answer(question: str, context: dict[str, Any]) -> str:
    q = question.lower().strip()
    bottleneck = context["top_bottleneck"]
    emerging = context["emerging_bottleneck"]
    risks = context["high_risk_units"]
    stations = context["station_states"]
    coverage = context["sensor_coverage"]
    if not q:
        return "Ask a question about the current line state."
    if "bottleneck" in q:
        answer = f'{bottleneck["station"]} {bottleneck["name"]} is the current bottleneck, with {bottleneck["utilization"]:.0%} live utilization and an input queue of {bottleneck["queue"]}.'
        if emerging:
            answer += f' {emerging["station"]} {emerging["name"]} is the leading emerging candidate.'
        return answer
    if any(term in q for term in ("highest defect", "highest risk", "which unit")):
        if not risks:
            return "The current data is insufficient to determine this reliably."
        top = risks[0]
        return f'{top["unit"]} has the highest currently available defect risk at {top["risk"]:.1%}, with {top["model_confidence"]:.1%} model confidence and {top["effective_trust"]:.1%} effective trust. The policy action is {top["action"]}.'
    if "origin" in q or "originate" in q or "root cause" in q:
        if not risks:
            return "The current data is insufficient to determine this reliably."
        selected = next((item for item in risks if item["unit"].lower() in q.replace(" ", "")), risks[0])
        if not selected.get("suspected_origin"):
            return "The current data is insufficient to determine this reliably."
        evidence = "; ".join(selected.get("origin_evidence", [])) or "the weakest observed station evidence on its path"
        return f'{selected["suspected_origin"]} is the suspected origin for {selected["unit"]}, based on {evidence}. This is a diagnostic lead, not a confirmed causal root cause.'
    if "worst health" in q or "lowest health" in q:
        worst = min(stations, key=lambda row: row["health"])
        return f'{worst["station"]} {worst["name"]} has the lowest current health at {worst["health"]:.1%}.'
    if any(term in q for term in ("sensor", "coverage", "incomplete", "inferred")):
        if not coverage:
            return "All stations currently have measured coverage for the required channels."
        return "Stations without full measured coverage: " + "; ".join(f'{item["station"]}: {item["status"]} (confidence {item["confidence"]:.0%})' for item in coverage) + "."
    if "human" in q or "review" in q or "trust" in q:
        if not risks:
            return "Human review is recommended when high defect risk is paired with low effective trust. No current unit evidence is available to apply that policy."
        top = risks[0]
        return f'For {top["unit"]}, model confidence ({top["model_confidence"]:.0%}) is multiplied by input trust ({top["input_trust"]:.0%}) to produce {top["effective_trust"]:.0%} effective trust. A high-risk call below the trust gate is routed to HUMAN REVIEW.'
    if "summary" in q or "supervisor" in q or "manager" in q or "focus" in q:
        return _offline_shift_summary(context)
    for station in stations:
        if station["station"].lower() in q:
            return f'{station["station"]} {station["name"]} is {station["state"]}, health {station["health"]:.0%}, utilization {station["utilization"]:.0%}, buffer {station["buffer"]}, and sensor status {station["coverage"]}.'
    return "The current data is insufficient to determine this reliably."


def _extract_response_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def _classify_llm_http_error(exc: urllib.error.HTTPError) -> tuple[str, str]:
    error_code = ""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        error = payload.get("error", {})
        error_code = str(error.get("code") or error.get("type") or "")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        pass
    if exc.code == 429:
        if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return error_code, "Cloud LLM quota is unavailable (HTTP 429). Check billing/credits and the API project linked to this key."
        return error_code, "Cloud LLM rate limit reached (HTTP 429). Wait briefly or check the API project's request and token limits."
    if exc.code == 401:
        return error_code, "Cloud LLM authentication failed (HTTP 401). Replace the API key with a valid server-side key."
    if exc.code == 403:
        return error_code, "Cloud LLM access was denied (HTTP 403). Check the API project's model permissions."
    return error_code, f"Cloud LLM request failed with HTTP {exc.code}."


def _call_llm(context: dict[str, Any], request_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    prompt = (
        "Use only the JSON context below. Do not calculate new plant metrics, infer causation, or invent facts. "
        "If the context cannot answer the request, respond exactly: The current data is insufficient to determine this reliably.\n\n"
        f"CURRENT_TWIN_CONTEXT={json.dumps(context, separators=(',', ':'), default=str)}\n\nREQUEST={request_text}"
    )
    body = json.dumps({
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "instructions": "You are Twinly's concise industrial operations copilot. Remain grounded in supplied structured data.",
        "input": prompt, "max_output_tokens": 450, "store": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                text = _extract_response_text(json.loads(response.read().decode("utf-8")))
                if not text:
                    raise RuntimeError("The cloud model returned no text.")
                return text
        except urllib.error.HTTPError as exc:
            error_code, message = _classify_llm_http_error(exc)
            quota_error = error_code in {"insufficient_quota", "billing_hard_limit_reached"}
            if exc.code == 429 and not quota_error and attempt == 0:
                retry_after = exc.headers.get("Retry-After", "1")
                try:
                    wait_seconds = min(3.0, max(0.5, float(retry_after)))
                except ValueError:
                    wait_seconds = 1.0
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(message) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("Cloud LLM connection failed. Check network access and try again.") from exc
    raise RuntimeError("Cloud LLM remained unavailable after retrying.")


def generate_shift_summary(context: dict[str, Any], use_llm: bool) -> str:
    return _call_llm(context, "Generate a concise current-shift summary with immediate operator priorities.") if use_llm else _offline_shift_summary(context)


def answer_dashboard_question(question: str, context: dict[str, Any], use_llm: bool) -> str:
    return _call_llm(context, question) if use_llm else _offline_answer(question, context)


def render_ai_copilot(context: dict[str, Any]) -> None:
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    st.markdown(
        '<div class="llm-flow">'
        '<div class="llm-step"><strong>1 · Twin engine</strong>Calculates live metrics, risks and trust locally.</div>'
        '<div class="llm-step"><strong>2 · Grounded context</strong>Packages only the current structured snapshot as JSON.</div>'
        '<div class="llm-step"><strong>3 · LLM explanation</strong>Turns supplied evidence into concise operator language.</div>'
        '<div class="llm-step"><strong>4 · Safe fallback</strong>Uses deterministic local rules if the cloud call fails.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([.65, .35])
    with left:
        st.markdown("#### Structured context → grounded explanation")
        st.caption("The Copilot receives compact calculated outputs, never raw CSVs, and is not allowed to calculate plant metrics.")
    with right:
        if has_key:
            use_llm = st.toggle("Use configured LLM", value=True, key="use_cloud_llm")
            st.caption(f'Cloud model: {os.getenv("OPENAI_MODEL", "gpt-5-mini")}')
        else:
            use_llm = False
            st.info("No `OPENAI_API_KEY` found. Grounded local mode is active; the rest of Twinly works normally.")
    if st.button("Generate Current Shift Summary", type="primary", key="generate_summary"):
        try:
            st.session_state.shift_summary = generate_shift_summary(context, use_llm)
            st.session_state.pop("copilot_cloud_error", None)
        except RuntimeError as exc:
            st.session_state.copilot_cloud_error = str(exc)
            st.session_state.shift_summary = generate_shift_summary(context, False)
    if st.session_state.get("shift_summary"):
        with st.container(border=True):
            st.markdown(st.session_state.shift_summary)

    with st.form("copilot_question_form", clear_on_submit=False):
        question = st.text_input("Ask Twinly about the current dashboard state", placeholder="Why is human review recommended?")
        submitted = st.form_submit_button("Ask Copilot")
    if submitted:
        try:
            st.session_state.copilot_answer = answer_dashboard_question(question, context, use_llm)
            st.session_state.pop("copilot_cloud_error", None)
        except RuntimeError as exc:
            st.session_state.copilot_cloud_error = str(exc)
            st.session_state.copilot_answer = answer_dashboard_question(question, context, False)
    if st.session_state.get("copilot_answer"):
        with st.container(border=True):
            st.markdown("**COPILOT ANSWER**")
            st.markdown(st.session_state.copilot_answer)
    if st.session_state.get("copilot_cloud_error"):
        st.warning(f'{st.session_state.copilot_cloud_error} Twinly automatically used its grounded local fallback; live analytics were unaffected.')
    show_context = st.toggle("Show the exact structured context sent to the LLM", value=False, key="show_copilot_context")
    if show_context:
        st.json(context, expanded=False)


def render_model_analytics(data: dict[str, Any]) -> None:
    meta = data["model_meta"]
    metrics = meta.get("metrics", {})
    if not metrics:
        st.markdown('<div class="empty-honest">Saved validation metadata is unavailable. Live operations remain unaffected.</div>', unsafe_allow_html=True)
        return
    st.info("This section describes held-out model validation, not the current line state.")
    metric_cards = [
        ("ROC AUC", metrics.get("auc", np.nan), "Ranking quality"),
        ("MCC", metrics.get("mcc", np.nan), "Imbalance-aware score"),
        ("Recall", metrics.get("recall", np.nan), "Defects detected"),
        ("Precision", metrics.get("precision", np.nan), "Flag hit rate"),
        ("Threshold", metrics.get("threshold", meta.get("threshold", np.nan)), "Risk operating point"),
        ("Positive tests", metrics.get("n_pos_test", 0), f'of {metrics.get("n_test", 0):,} rows'),
    ]
    for start in range(0, len(metric_cards), 3):
        cols = st.columns(3)
        for col, (label, value, help_text) in zip(cols, metric_cards[start:start + 3]):
            with col:
                display = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                st.metric(label, display, help=help_text)

    left, right = st.columns([1.1, .9])
    with left:
        model_path = ARTIFACT_DIR / "defect_model.json"
        importance = load_feature_importance(model_path.stat().st_mtime if model_path.exists() else 0.0)
        if not importance.empty:
            plot = importance.sort_values("gain")
            fig = px.bar(plot, x="gain", y="feature", orientation="h", color="gain", color_continuous_scale=[[0, CYAN], [1, TEAL]])
            fig.update_layout(title="Saved model feature importance (gain)", height=460, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", xaxis_title="XGBoost gain", yaxis_title=None)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        vals = pd.DataFrame({
            "metric": ["ROC AUC", "Recall", "Precision", "MCC"],
            "value": [metrics.get("auc", 0), metrics.get("recall", 0), metrics.get("precision", 0), max(0, metrics.get("mcc", 0))],
        })
        fig = go.Figure(go.Bar(x=vals["value"], y=vals["metric"], orientation="h", marker_color=[TEAL, CYAN, AMBER, MUTED], text=vals["value"].map(lambda value: f"{value:.3f}"), textposition="outside"))
        fig.update_layout(title="Held-out validation profile", height=320, margin=dict(l=10, r=35, t=50, b=10), xaxis=dict(range=[0, 1]), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="empty-honest"><b>Honest evaluation boundary</b><br>The saved artifact includes aggregate validation metrics but not the individual held-out labels and predictions. '
            'Twinly therefore does not reconstruct a fake confusion matrix or ROC/PR curve. Re-run evaluation only when those curve points are needed.</div>',
            unsafe_allow_html=True,
        )


def render_manager_view(data: dict[str, Any], live: dict[str, Any]) -> None:
    visits = data["unit_visit_times"]
    last_station = int(live["snapshot"]["index"].max())
    completed = visits[(visits["station"] == last_station) & (visits["t_global"] <= live["timestamp"])].copy()
    if not completed.empty:
        completed["window"] = (completed["t_global"] // 600 * 600).astype(int)
        output = completed.groupby("window").size().reset_index(name="units")
    else:
        output = pd.DataFrame(columns=["window", "units"])
    assessment = live["assessments"].copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="persona-card"><div class="eyebrow">SUPERVISOR · NOW</div><h3>Immediate control</h3>'
                    f'<p>Inspect {live["kpis"]["current_bottleneck"]}; {live["kpis"]["high_risk_units"]} high-risk unit(s) and {live["kpis"]["degraded_stations"]} degraded station(s) currently require attention.</p></div>', unsafe_allow_html=True)
    with c2:
        inferred = int((live["snapshot"]["coverage_status"] != "MEASURED").sum())
        st.markdown('<div class="persona-card"><div class="eyebrow">PLANT MANAGER · SHIFT</div><h3>Flow and reliability</h3>'
                    f'<p>Rolling throughput is {live["kpis"]["throughput"]:.1f}/h. {inferred} station(s) currently depend on incomplete or inferred coverage.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="persona-card"><div class="eyebrow">LEADERSHIP · SCALE</div><h3>Governed automation</h3>'
                    '<p>Twinly links each prediction to data reliability before action, preserving human review where observability is weak.</p></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        if not output.empty:
            fig = px.bar(output, x="window", y="units", color_discrete_sequence=[TEAL], labels={"window": "Simulation time (s)", "units": "Output per 10 min"})
            fig.update_layout(title="Observed production trend", height=340, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        state_mix = live["snapshot"]["display_state"].value_counts().rename_axis("state").reset_index(name="stations")
        fig = px.pie(state_mix, values="stations", names="state", hole=.65, color="state", color_discrete_map=STATE_COLORS)
        fig.update_layout(title="Current station-state mix", height=340, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", legend_orientation="h")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if not assessment.empty:
        actions = assessment["action"].value_counts().rename_axis("Action").reset_index(name="Units")
        st.dataframe(actions, hide_index=True, width="stretch")


def render_business_view(data: dict[str, Any], live: dict[str, Any]) -> None:
    snapshot = live["snapshot"]
    measured = float((snapshot["coverage_status"] == "MEASURED").mean())
    inferred = int((snapshot["coverage_status"] == "INFERRED / VIRTUAL SENSOR").sum())
    assessment = live["assessments"]
    auto = int((assessment.get("action", pd.Series(dtype=str)) == "AUTO-ACT").sum())
    human = int((assessment.get("action", pd.Series(dtype=str)) == "HUMAN-VERIFY").sum())
    cols = st.columns(4)
    business = [
        ("Production exposure", live["kpis"]["line_status"], "Current simulated line posture"),
        ("Measured observability", f"{measured:.0%}", "Required channels directly measured"),
        ("Virtual coverage", str(inferred), "Stations supported by inference"),
        ("Governed high-risk actions", str(auto + human), f"{auto} auto · {human} human verify"),
    ]
    for col, (label, value, sub) in zip(cols, business):
        with col:
            st.markdown(_kpi_card(label, value, sub, TEAL), unsafe_allow_html=True)
    st.markdown(
        '<div class="panel" style="margin-top:.8rem"><div class="eyebrow">ILLUSTRATIVE BUSINESS CASE · NOT A CLAIMED SAVING</div>'
        '<h3>Value without false precision</h3><p>Twinly can reduce operational exposure by surfacing congestion earlier, prioritizing likely defects before final quality control, '
        'and automating only calls that clear the Effective Trust gate. A defensible currency ROI requires plant-specific scrap cost, throughput, labor, downtime and intervention efficacy; '
        'those inputs are intentionally not fabricated in this prototype.</p></div>', unsafe_allow_html=True,
    )


def _restart_playback(t_min: int) -> None:
    st.session_state.live_t = t_min
    st.session_state.sync_timeline = True
    st.session_state.autoplay = False


def _toggle_playback() -> None:
    st.session_state.autoplay = not st.session_state.get("autoplay", False)


def _timeline_changed() -> None:
    st.session_state.live_t = int(st.session_state.timeline_slider)
    st.session_state.shift_summary = ""
    st.session_state.copilot_answer = ""


def render_playback_controls(t_min: int, t_max: int) -> tuple[int, int, int, bool]:
    if "live_t" not in st.session_state:
        st.session_state.live_t = t_min
    if st.session_state.pop("sync_timeline", False) or "timeline_slider" not in st.session_state:
        st.session_state.timeline_slider = int(st.session_state.live_t)
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    with st.sidebar:
        st.markdown('<div class="sidebar-section">Live playback</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="playback-readout">Simulation time&nbsp;&nbsp;{int(st.session_state.live_t):,} s</div>',
            unsafe_allow_html=True,
        )
        row = st.columns(2)
        with row[0]:
            st.button("Pause" if st.session_state.autoplay else "Play", on_click=_toggle_playback, width="stretch")
        with row[1]:
            st.button("Restart", on_click=_restart_playback, args=(t_min,), width="stretch")
        autoplay = st.toggle("Auto-play", key="autoplay")
        loop = st.toggle("Loop playback", value=True, key="loop_playback")
        st.slider("Simulation timeline", t_min, t_max, step=5, key="timeline_slider", on_change=_timeline_changed)
        speed = st.selectbox("Playback speed", [1, 2, 4, 8], index=2, format_func=lambda value: f"{value}×")
    st.session_state.live_t = int(st.session_state.timeline_slider)
    return int(st.session_state.live_t), int(speed), 20, bool(loop)


def main() -> None:
    st.set_page_config(page_title="Twinly · Digital Twin Control Room", page_icon="T", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    if not DEMO_DIR.exists():
        st.error(f"Demo stream not found at {DEMO_DIR}. Generate it before launching Twinly.")
        st.stop()
    data = load_demo_data(data_signature())
    buffers = data["buffer_history"]
    if buffers.empty:
        st.error("The demo stream has no buffer timeline, so live playback cannot start.")
        st.stop()
    t_min = int(buffers["t"].min())
    t_max = int(buffers["t"].max())
    active_section = render_navigation()
    t_now, speed, step, loop_enabled = render_playback_controls(t_min, t_max)
    live = get_live_state(data, float(t_now))

    st.markdown(
        f'<div class="hero"><div class="hero-grid"><div><div class="eyebrow">TWINLY · TRUST-AWARE PRODUCTION INTELLIGENCE</div>'
        '<div class="hero-title">See the line. Understand the risk. Act with confidence.</div>'
        f'<div class="hero-copy">Simulated live stream · {len(live["snapshot"])} stations · current-time evidence only</div></div>'
        f'<div class="clock"><strong>{time.strftime("%H:%M:%S", time.gmtime(t_now))}</strong><span>Simulation clock · t={t_now}s</span></div></div></div>',
        unsafe_allow_html=True,
    )
    render_kpis(live)
    alerts = build_alerts(data, live)
    context = build_llm_context(data, live, alerts)
    if active_section == "live":
        _anchor("live-line", "Live Digital Twin", "Units, station conditions, buffers and trust-aware risk reconstructed only from events available at the current clock.")
        render_live_pipeline(live)
        st.markdown("### Live alert feed")
        render_alerts(alerts)
    elif active_section == "defects":
        _anchor("defects", "Defect Intelligence", "Unit-level model output, path evidence, suspected origin and the confidence × input-trust decision gate.")
        render_defects(data, live)
    elif active_section == "bottlenecks":
        _anchor("bottlenecks", "Bottleneck Intelligence", "The sustained current constraint is separated from leading queue-growth and cycle-drift evidence.")
        render_bottlenecks(data, live)
    elif active_section == "health":
        _anchor("health", "Station Health", "Compare every station, then inspect health, cycle, buffer and sensor history without leaking future observations.")
        render_station_health(data, live)
        st.markdown("### Sensor coverage & trust")
        render_sensor_coverage(data, live)
    elif active_section == "copilot":
        _anchor("copilot", "AI Copilot", "A lightweight explanation layer over compact, already-calculated dashboard outputs. It never becomes the source of plant metrics.")
        render_ai_copilot(context)
    elif active_section == "analytics":
        _anchor("analytics", "Model Analytics", "Historical held-out evaluation is isolated here so validation quality is never confused with current plant conditions.")
        render_model_analytics(data)
    elif active_section == "manager":
        _anchor("manager", "Operational Perspectives", "One twin, organized for the supervisor's next action, the manager's shift view, and leadership's governance lens.")
        render_manager_view(data, live)
    elif active_section == "business":
        _anchor("business", "Business & Scale", "Operational impact framing stays transparent: current simulated evidence is shown, while plant-specific ROI remains an explicit input requirement.")
        render_business_view(data, live)

    st.caption(f'Playback source: {DEMO_DIR} · Showing data through t={t_now}s only · Live and validation sections are intentionally separated.')

    if st.session_state.autoplay:
        next_t = t_now + step * speed
        if next_t > t_max:
            next_t = t_min if loop_enabled else t_max
            if not loop_enabled:
                st.session_state.autoplay = False
        time.sleep(.8)
        st.session_state.live_t = int(next_t)
        st.session_state.sync_timeline = True
        st.rerun()


if __name__ == "__main__":
    main()
