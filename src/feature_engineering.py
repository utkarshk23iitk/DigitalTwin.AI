"""
feature_engineering.py — turn the raw persisted simulation logs into a
trend-aware, tier-complete per-unit feature table for defect_model.py.

Two problems this solves, both flagged before building it:

  1. unit_features.csv only has an INSTANTANEOUS snapshot per station visit.
     A model trained on that learns "is this one reading unusual," not "has
     this been drifting" -- exactly the naive-threshold behaviour we wanted
     to avoid. Fix: compute rolling mean/std/slope from sensor_log as of
     each unit's actual visit time (never using a later reading), and use
     THOSE as the model's inputs instead of the raw snapshot.

  2. Tier B stations have ZERO columns in unit_features.csv (never directly
     observed, by design) and Tier C is ~99.5% empty -- so the defect model
     currently can't use them at all. Fix: apply virtual_sensor.py's already-
     validated methods (spatial regression for Tier B, Kalman for Tier C) to
     produce a dense estimate + confidence for every unit, computed only from
     what would genuinely be available live (smoothed neighbour readings /
     the filter's running state) -- this is also what closes the loop between
     virtual_sensor.py and the defect model.

Output: data/simulated/model_features.csv -- one row per unit, engineered
trend/imputed features only (never the raw unit_features_true.csv columns,
which stay strictly for offline validation).

Usage:
    python src/feature_engineering.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from virtual_sensor import (CHANNELS, HEALTH_TICK, SpatialVirtualSensor,  # noqa: E402
                            TemporalVirtualSensor, fit_virtual_sensors, load_all,
                            load_tuned_virtual_sensor_params)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"

DEFAULT_FEATURE_PARAMS = {
    "tier_a_window_ticks": 6,
    "tier_a_max_staleness": 60.0,
    "tier_c_window_checks": 3,
    "tier_c_max_staleness": 3600.0,
}


def _rolling_trend(sensor_log: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add roll_mean/roll_std/roll_slope to each sensor_log row, computed
    from that row's own trailing window within its (session, station,
    channel) series -- never looking forward."""
    out = []
    for (_, _, _), g in sensor_log.groupby(["session_id", "station", "channel"]):
        g = g.sort_values("t_global").copy()
        g["roll_mean"] = g["value"].rolling(window, min_periods=2).mean()
        g["roll_std"] = g["value"].rolling(window, min_periods=2).std()
        g["roll_slope"] = (g["value"] - g["value"].shift(window)) / window
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else sensor_log.assign(
        roll_mean=np.nan, roll_std=np.nan, roll_slope=np.nan)


def _asof_join_trend(visits: pd.DataFrame, trend: pd.DataFrame, tolerance: float
                     ) -> pd.DataFrame:
    """For each unit's visit time, pick up the most recent trend snapshot at
    or before that moment, within `tolerance` seconds -- otherwise NaN."""
    v = visits.sort_values("t_global")
    t = trend.sort_values("t_global")
    merged = pd.merge_asof(v, t, on="t_global", by="session_id",
                           direction="backward", tolerance=tolerance,
                           suffixes=("", "_trend"))
    return merged


def build_tier_a_c_features(unit_visit_times: pd.DataFrame, sensor_log: pd.DataFrame,
                            registry: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Rolling trend features for every station whose own channel is ever
    directly observed (tier A dense, tier C sparse)."""
    cfg = {**DEFAULT_FEATURE_PARAMS, **(params or {})}
    frames = []
    observed_stations = sensor_log["station"].unique()
    for station in observed_stations:
        tier = registry.loc[registry["station"] == station, "tier"].iloc[0]
        window = int(cfg["tier_a_window_ticks"]) if tier == "A" else int(cfg["tier_c_window_checks"])
        staleness = float(cfg["tier_a_max_staleness"]) if tier == "A" else float(cfg["tier_c_max_staleness"])

        s_log = sensor_log[sensor_log["station"] == station]
        trend = _rolling_trend(s_log, window)
        visits = unit_visit_times[unit_visit_times["station"] == station][
            ["session_id", "unit_id", "t_global"]]

        # Iterate whatever channels this station actually emits -- real AND
        # decoy. Decoys become trend features exactly like real ones (the
        # model is never told which is which); train_defect_model.py later
        # checks that importance ranks them low. Tier-B/C imputation below
        # still runs on real CHANNELS only -- we don't reconstruct noise.
        for ch in sorted(s_log["channel"].unique()):
            t_ch = trend[trend["channel"] == ch][
                ["session_id", "t_global", "roll_mean", "roll_std", "roll_slope"]]
            merged = _asof_join_trend(visits, t_ch, tolerance=staleness)
            merged = merged.rename(columns={
                "roll_mean": f"S{station}_{ch}_mean",
                "roll_std": f"S{station}_{ch}_std",
                "roll_slope": f"S{station}_{ch}_slope",
            })[["session_id", "unit_id",
               f"S{station}_{ch}_mean", f"S{station}_{ch}_std", f"S{station}_{ch}_slope"]]
            frames.append(merged.set_index(["session_id", "unit_id"]))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.reset_index()


def build_tier_b_features(sensors: dict, unit_visit_times: pd.DataFrame,
                          tier_ac_features: pd.DataFrame, registry: pd.DataFrame
                          ) -> pd.DataFrame:
    """Apply each Tier-B station's fitted spatial regressor to the SMOOTHED
    (rolling-mean) predictor readings -- not raw instantaneous ones, since
    that's what actually carries the correlation signal (see virtual_sensor's
    validation notes)."""
    b_stations = registry[registry["tier"] == "B"]["station"].tolist()
    if not b_stations:
        return pd.DataFrame()

    frames = []
    for station in b_stations:
        for ch in CHANNELS:
            info = sensors.get((station, ch))
            if not info or info["method"] != "spatial":
                continue
            sensor: SpatialVirtualSensor = info["sensor"]
            # Use the *_mean rolling feature of each predictor as the input,
            # falling back to nothing (NaN -> unrecoverable for that row) if
            # the predictor's own trend wasn't available at that moment.
            pred_mean_cols = [c.replace(f"_{ch}", f"_{ch}_mean") for c in sensor.predictor_cols]
            missing = [c for c in pred_mean_cols if c not in tier_ac_features.columns]
            if missing:
                continue
            sub = tier_ac_features[["session_id", "unit_id"] + pred_mean_cols].copy()
            values, confs = [], []
            for _, row in sub.iterrows():
                x = {orig: row[mean_col] for orig, mean_col in
                     zip(sensor.predictor_cols, pred_mean_cols)}
                v, c = sensor.estimate(x)
                values.append(v)
                confs.append(c)
            sub[f"S{station}_{ch}_est"] = values
            sub[f"S{station}_{ch}_conf"] = confs
            frames.append(sub.set_index(["session_id", "unit_id"])[
                [f"S{station}_{ch}_est", f"S{station}_{ch}_conf"]])

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.reset_index()


def build_tier_c_kalman_features(sensors: dict, unit_visit_times: pd.DataFrame,
                                 sensor_log: pd.DataFrame, registry: pd.DataFrame
                                 ) -> pd.DataFrame:
    """Replay each Tier-C station's Kalman filter forward through time (fit
    on train-session noise parameters, but the recursive predict/update walk
    itself is causal and touches no future data), producing a dense estimate
    + confidence for every unit -- not just the ~0.5% that land on a real
    check."""
    c_stations = registry[registry["tier"] == "C"]["station"].tolist()
    frames = []
    for station in c_stations:
        visits = unit_visit_times[unit_visit_times["station"] == station]
        for ch in CHANNELS:
            info = sensors.get((station, ch))
            if not info or info["method"] != "temporal":
                continue
            base: TemporalVirtualSensor = info["sensor"]
            checks = (sensor_log[(sensor_log["station"] == station)
                                 & (sensor_log["channel"] == ch)]
                     .sort_values("t_global"))
            if checks.empty:
                continue

            rows = []
            for session_id, sess_checks in checks.groupby("session_id"):
                sess_checks = sess_checks.sort_values("t_global")
                kf = TemporalVirtualSensor(station, ch, q=base.kf.Q[0, 0] or 1e-6,
                                           r=base.kf.R[0, 0],
                                           x0=float(sess_checks["value"].iloc[0]))
                kf.set_q_per_tick(getattr(base, "_q_per_tick", 1e-6))
                kf.last_t = float(sess_checks["t_global"].iloc[0])
                check_iter = sess_checks.itertuples()
                next_check = next(check_iter, None)

                sess_visits = visits[visits["session_id"] == session_id].sort_values("t_global")
                for v in sess_visits.itertuples():
                    while next_check is not None and next_check.t_global <= v.t_global:
                        kf.update(next_check.t_global, next_check.value)
                        next_check = next(check_iter, None)
                    est, conf = kf.predict_to(v.t_global)
                    rows.append({"session_id": session_id, "unit_id": v.unit_id,
                                f"S{station}_{ch}_est": est, f"S{station}_{ch}_conf": conf})
            if rows:
                frames.append(pd.DataFrame(rows).set_index(["session_id", "unit_id"]))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.reset_index()


def build_features_from_frames(obs: pd.DataFrame, registry: pd.DataFrame,
                               sensor_log: pd.DataFrame, unit_visit_times: pd.DataFrame,
                               sensors: dict, verbose: bool = True,
                               params: dict | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_FEATURE_PARAMS, **(params or {})}
    if verbose:
        print("[2/4] tier A/C rolling trend features...")
    tier_ac = build_tier_a_c_features(unit_visit_times, sensor_log, registry, cfg)

    if verbose:
        print("[3/4] tier B spatial-imputed features...")
    tier_b = build_tier_b_features(sensors, unit_visit_times, tier_ac, registry)

    if verbose:
        print("[4/4] tier C Kalman-imputed features...")
    tier_c = build_tier_c_kalman_features(sensors, unit_visit_times, sensor_log, registry)

    base = obs[["session_id", "unit_id", "response", "defect_occurred_at", "defect_caught_at"]]
    result = base
    for extra in (tier_ac, tier_b, tier_c):
        if not extra.empty:
            result = result.merge(extra, on=["session_id", "unit_id"], how="left")

    expected_virtual = {
        f"S{station}_{channel}_{suffix}"
        for (station, channel), info in sensors.items()
        if info.get("method") in {"spatial", "temporal"}
        for suffix in ("est", "conf")
    }
    missing_virtual = sorted(expected_virtual - set(result.columns))
    if missing_virtual:
        raise RuntimeError(
            "virtual-sensor feature generation is incomplete: "
            + ", ".join(missing_virtual)
        )

    if verbose:
        n_feat = result.shape[1] - 5
        print(f"\n[done] {len(result)} units x {n_feat} engineered features")
        print(f"       defect rate: {result['response'].mean() * 100:.3f}%")
    return result


def build_features(verbose: bool = True, params: dict | None = None) -> pd.DataFrame:
    cfg = {**DEFAULT_FEATURE_PARAMS, **(params or {})}
    true, obs, registry, sensor_log, manifest = load_all()
    unit_visit_times = pd.read_csv(DATA_DIR / "unit_visit_times.csv")

    if verbose:
        print("[1/4] fitting virtual sensors (needed for tier B/C features)...")
    sensors = fit_virtual_sensors(verbose=False, params=cfg)

    return build_features_from_frames(
        obs=obs,
        registry=registry,
        sensor_log=sensor_log,
        unit_visit_times=unit_visit_times,
        sensors=sensors,
        verbose=verbose,
        params=cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-tuned", action="store_true",
                        help="use validation-selected virtual-sensor parameters")
    args = parser.parse_args()
    selected_params = load_tuned_virtual_sensor_params() if args.use_tuned else {}
    features = build_features(params=selected_params)
    out_path = DATA_DIR / "model_features.csv"
    features.to_csv(out_path, index=False)
    print(f"[write] {out_path}")
