"""
generate_demo_data.py — create a separate simulated "live" shift for the demo.

This is intentionally distinct from the offline training corpus in
data/simulated/. The idea is:

  - train/tune models offline on historical sessions
  - generate one fresh unseen shift here
  - replay that shift in the dashboard as if data were arriving live

In production, the files written here would be replaced by actual streaming
plant telemetry or batch snapshots from PLC / SCADA / MES systems.

Usage:
    python data/generate_demo_data.py
    python data/generate_demo_data.py --duration 9000 --seed 999 --out data/demo_live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bottleneck_detect import bottleneck_report  # noqa: E402
from line_sim import DECOY_CHANNELS, HEALTH_TICK, REAL_CHANNELS, default_line  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DATA_DIR / "demo_live"


def _station_registry(line) -> pd.DataFrame:
    rows = []
    for s in line.cfg.stations:
        rows.append({
            "station": s.index,
            "name": s.name,
            "tier": s.tier,
            "is_inspection": s.is_inspection,
            "mean_cycle": s.mean_cycle,
            "failure_rate": s.failure_rate,
            "base_defect_rate": s.base_defect_rate,
        })
    return pd.DataFrame(rows)


def _channel_registry() -> pd.DataFrame:
    rows = [{"channel": ch, "is_decoy": False} for ch in REAL_CHANNELS]
    rows += [{"channel": ch, "is_decoy": True} for ch in DECOY_CHANNELS]
    return pd.DataFrame(rows)


def _buffer_history_frame(line) -> pd.DataFrame:
    rows = []
    n = len(line.cfg.stations)
    for rec in line.buffer_log:
        row = {"t": float(rec["t"])}
        for i in range(n):
            row[f"buffer_{i}"] = int(rec["levels"][i])
        rows.append(row)
    return pd.DataFrame(rows)


def _pivot_unit_log(unit_log: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if unit_log.empty:
        return pd.DataFrame(columns=["session_id", "unit_id"])
    wide = unit_log.pivot_table(index=["session_id", "unit_id"],
                                columns=["station", "channel"], values=value_col)
    wide.columns = [f"S{st}_{ch}" for st, ch in wide.columns]
    return wide.reset_index()


def generate_demo(duration: float, seed: int, out_dir: Path) -> dict:
    print(f"[sim] live demo shift  seed={seed}  duration={duration:.0f}s")
    line = default_line(seed=seed).run(until=duration)
    registry = _station_registry(line)
    brep = bottleneck_report(line)
    station_stats = pd.DataFrame(line.station_stats())
    line_events = pd.DataFrame(line.log)
    if not line_events.empty:
        line_events.insert(0, "session_id", 0)
        line_events["t_global"] = line_events["t"]
    buffer_history = _buffer_history_frame(line)
    if not buffer_history.empty:
        buffer_history.insert(0, "session_id", 0)
    health_log = pd.DataFrame(line.health_log)
    if not health_log.empty:
        health_log.insert(0, "session_id", 0)
        health_log["t_global"] = health_log["t"]
    sensor_log = pd.DataFrame(line.sensor_log)
    if not sensor_log.empty:
        sensor_log.insert(0, "session_id", 0)
        sensor_log["t_global"] = sensor_log["t"]
    unit_log = pd.DataFrame(line.unit_log)
    if not unit_log.empty:
        unit_log.insert(0, "session_id", 0)
        unit_log["t_global"] = unit_log["t"]
    unit_summary = pd.DataFrame(line.unit_summary)
    if not unit_summary.empty:
        unit_summary.insert(0, "session_id", 0)
    unit_obs = _pivot_unit_log(unit_log, "value_observed")
    unit_true = _pivot_unit_log(unit_log, "value_true")
    unit_features = unit_summary.merge(unit_obs, on=["session_id", "unit_id"], how="left")
    unit_features_true = unit_summary[["session_id", "unit_id", "response"]].merge(
        unit_true, on=["session_id", "unit_id"], how="left")
    unit_visit_times = (unit_log[["session_id", "unit_id", "station", "t_global"]]
                        .drop_duplicates() if not unit_log.empty
                        else pd.DataFrame(columns=["session_id", "unit_id", "station", "t_global"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out_dir / "station_registry.csv", index=False)
    _channel_registry().to_csv(out_dir / "channel_registry.csv", index=False)
    line_events.to_csv(out_dir / "line_events.csv", index=False)
    buffer_history.to_csv(out_dir / "buffer_history.csv", index=False)
    health_log.to_csv(out_dir / "health_log.csv", index=False)
    sensor_log.to_csv(out_dir / "sensor_log.csv", index=False)
    unit_log.to_csv(out_dir / "unit_log.csv", index=False)
    unit_summary.to_csv(out_dir / "unit_summary.csv", index=False)
    unit_features.to_csv(out_dir / "unit_features.csv", index=False)
    unit_features_true.to_csv(out_dir / "unit_features_true.csv", index=False)
    unit_visit_times.to_csv(out_dir / "unit_visit_times.csv", index=False)
    station_stats.to_csv(out_dir / "station_stats.csv", index=False)
    with open(out_dir / "bottleneck_report.json", "w") as fh:
        json.dump(brep, fh, indent=2)

    manifest = {
        "kind": "demo_live_shift",
        "duration_s": duration,
        "seed": seed,
        "health_tick_s": HEALTH_TICK,
        "buffer_capacity": int(line.cfg.buffer_capacity),
        "stations": len(line.cfg.stations),
        "totals": {
            "event_rows": len(line.log),
            "buffer_rows": len(line.buffer_log),
            "health_rows": len(line.health_log),
            "sensor_rows": len(line.sensor_log),
            "unit_rows": len(line.unit_log),
            "finished_units": len(line.unit_summary),
            "defects": int(sum(u["response"] for u in line.unit_summary)),
            "defect_rate_pct": round(
                float(pd.DataFrame(line.unit_summary)["response"].mean() * 100), 4
            ) if line.unit_summary else 0.0,
        },
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[write] {out_dir}/")
    print(json.dumps(manifest["totals"], indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=8000.0,
                    help="sim-seconds for the live demo shift (default: 8000)")
    ap.add_argument("--seed", type=int, default=999,
                    help="random seed for the live demo shift (default: 999)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="output directory (default: data/demo_live)")
    args = ap.parse_args()
    generate_demo(args.duration, args.seed, args.out)


if __name__ == "__main__":
    main()
