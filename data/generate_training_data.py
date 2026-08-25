"""
generate_training_data.py — Persist historical simulated sessions for offline
training.

This is the OFFLINE half of the train-offline / infer-live pattern: run the
health-driven line simulation for many independent sessions, harvest enough
degradation episodes and defects to actually train on, and write the result
to data/simulated/ as CSVs. Models are trained against these files; the
*live* demo later runs a fresh, separate simulation and applies the already-
trained models to it — never the same run, so nothing trains on the event
it is simultaneously being asked to predict.

Sessions are run back-to-back in *simulated* time (session i's rows get
t_global = i * duration + t), so a plain chronological split — earlier
sessions for training, the last session held out — is well-defined and
never splits within a single health episode.

Output (data/simulated/):
    station_registry.csv   — static per-station config (one row each, tier,
                              is_inspection, baseline cycle time, etc.)
    health_log.csv          — ground-truth health_true per station over time.
                              NEVER a model input — kept only to score
                              detection lead time / false-alarm rate offline.
    sensor_log.csv          — observable channel readings (dense for tier A,
                              sparse for tier C, absent for tier B)
    unit_features.csv       — per-unit modelling table: response label +
                              OBSERVED S{station}_{channel} columns only
                              (NaN where the tier hides it) -- what
                              defect_model.py should actually train on
    unit_features_true.csv  — per-unit TRUE S{station}_{channel} columns,
                              for scoring virtual-sensor imputation accuracy
                              only. Never join this into model training.
    manifest.json           — generation parameters, for reproducibility

Usage:
    python data/generate_training_data.py                    # defaults
    python data/generate_training_data.py --sessions 8 --duration 80000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from line_sim import HEALTH_TICK, default_line  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "simulated"


def _station_registry(line) -> pd.DataFrame:
    rows = []
    for s in line.cfg.stations:
        rows.append({
            "station": s.index, "name": s.name, "tier": s.tier,
            "is_inspection": s.is_inspection, "mean_cycle": s.mean_cycle,
            "failure_rate": s.failure_rate, "base_defect_rate": s.base_defect_rate,
        })
    return pd.DataFrame(rows)


def _pivot_unit_log(unit_log: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if unit_log.empty:
        return pd.DataFrame(columns=["session_id", "unit_id"])
    wide = unit_log.pivot_table(index=["session_id", "unit_id"],
                                columns=["station", "channel"], values=value_col)
    wide.columns = [f"S{st}_{ch}" for st, ch in wide.columns]
    return wide.reset_index()


def generate(n_sessions: int, duration: float, base_seed: int, n_test_sessions: int = 1) -> dict:
    health_frames, sensor_frames, unit_log_frames, summary_frames = [], [], [], []
    registry = None

    for sess in range(n_sessions):
        seed = base_seed + sess
        print(f"[sim]   session {sess}/{n_sessions - 1}  seed={seed}  "
              f"duration={duration:.0f}s")
        line = default_line(seed=seed).run(until=duration)
        if registry is None:
            registry = _station_registry(line)

        offset = sess * duration
        for name, log, frames in (
            ("health", line.health_log, health_frames),
            ("sensor", line.sensor_log, sensor_frames),
            ("unit", line.unit_log, unit_log_frames),
        ):
            df = pd.DataFrame(log)
            if not df.empty:
                df.insert(0, "session_id", sess)
                df["t_global"] = df["t"] + offset
            frames.append(df)

        summ = pd.DataFrame(line.unit_summary)
        if not summ.empty:
            summ.insert(0, "session_id", sess)
        summary_frames.append(summ)

        n_units = len(line.unit_summary)
        n_def = sum(u["response"] for u in line.unit_summary)
        print(f"        units={n_units}  defects={n_def}  "
              f"({n_def / max(1, n_units) * 100:.3f}%)")

    health_log = pd.concat(health_frames, ignore_index=True)
    sensor_log = pd.concat(sensor_frames, ignore_index=True)
    unit_log = pd.concat(unit_log_frames, ignore_index=True)
    unit_summary = pd.concat(summary_frames, ignore_index=True)

    unit_obs = _pivot_unit_log(unit_log, "value_observed")
    unit_true = _pivot_unit_log(unit_log, "value_true")
    unit_features = unit_summary.merge(unit_obs, on=["session_id", "unit_id"], how="left")
    unit_features_true = unit_summary[["session_id", "unit_id", "response"]].merge(
        unit_true, on=["session_id", "unit_id"], how="left")
    # When each unit was actually AT each station -- needed so trend features
    # can be computed "as of" that moment, never leaking a later reading in.
    unit_visit_times = (unit_log[["session_id", "unit_id", "station", "t_global"]]
                        .drop_duplicates())

    OUT_DIR.mkdir(exist_ok=True)
    registry.to_csv(OUT_DIR / "station_registry.csv", index=False)
    health_log.to_csv(OUT_DIR / "health_log.csv", index=False)
    sensor_log.to_csv(OUT_DIR / "sensor_log.csv", index=False)
    unit_features.to_csv(OUT_DIR / "unit_features.csv", index=False)
    unit_features_true.to_csv(OUT_DIR / "unit_features_true.csv", index=False)
    unit_visit_times.to_csv(OUT_DIR / "unit_visit_times.csv", index=False)

    manifest = {
        "n_sessions": n_sessions,
        "duration_per_session_s": duration,
        "base_seed": base_seed,
        "health_tick_s": HEALTH_TICK,
        "train_sessions": list(range(n_sessions - n_test_sessions)),
        "test_sessions": list(range(n_sessions - n_test_sessions, n_sessions)),
        "split_note": (f"Chronological: train on the first {n_sessions - n_test_sessions} "
                       f"sessions, evaluate on the last {n_test_sessions} held-out "
                       "sessions. Never shuffle rows across sessions -- that would "
                       "leak within a single health episode. Multiple test sessions "
                       "(not just one) are used so the held-out defect count is "
                       "large enough for a stable evaluation."),
        "totals": {
            "health_log_rows": len(health_log),
            "sensor_log_rows": len(sensor_log),
            "units": len(unit_summary),
            "defects": int(unit_summary["response"].sum()) if len(unit_summary) else 0,
            "defect_rate_pct": round(
                float(unit_summary["response"].mean() * 100), 4) if len(unit_summary) else 0.0,
        },
    }
    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=5,
                    help="number of independent simulated sessions (default: 5)")
    ap.add_argument("--duration", type=float, default=50_000.0,
                    help="sim-seconds per session (default: 50000)")
    ap.add_argument("--seed", type=int, default=100,
                    help="base seed; session i uses seed + i (default: 100)")
    ap.add_argument("--test-sessions", type=int, default=1,
                    help="number of trailing sessions held out for evaluation (default: 1)")
    args = ap.parse_args()

    manifest = generate(args.sessions, args.duration, args.seed, args.test_sessions)
    print(f"\n[write] {OUT_DIR}/")
    print(json.dumps(manifest["totals"], indent=2))
    print(f"\nTrain sessions: {manifest['train_sessions']}  "
          f"Test session: {manifest['test_sessions']}")


if __name__ == "__main__":
    main()
