"""
virtual_sensor.py — infer sensor-poor station state + confidence.

The technique used for a given station is chosen from MEASURED correlation
against the historical training data, not from a hard-coded tier label:

    1. Compute this station's correlation with every other station's channel,
       using TRUE values (data/simulated/unit_features_true.csv), on the
       train sessions only.
    2. If the strongest correlation clears CORR_THRESHOLD -> SPATIAL:
       regress this station's value on the correlated stations' OBSERVED
       readings (what would actually be available live).
    3. Else, if this station has any of its own historical observations
       (sparse checks in sensor_log.csv) -> TEMPORAL: a Kalman filter that
       predicts forward from its own past and corrects on real checks.
    4. Else: UNRECOVERABLE. No technique can fill this gap honestly -- it
       must surface as a hard blind spot to effective_trust.py, not a
       confident-looking guess.

This is deliberately NOT wired to the tier labels in line_sim.py (A/B/C) --
those exist to shape the *simulation*, but the *inference* strategy here is
re-derived from data alone, which is also a check on the design: if a tier-B
station is discovered to need TEMPORAL treatment (or vice versa), that is a
real finding worth reporting, not a bug to hide.

Usage:
    python src/virtual_sensor.py     # fit + validate against the held-out
                                      # test session, print a report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "simulated"
CHANNELS = ("torque", "vibration", "temperature")

# A station clears the "spatially predictable" bar if some other station's
# channel correlates at least this strongly with it, measured on TRUE values
# over the training sessions. Below this, spatial regression would be fitting
# noise -- fall back to a temporal filter instead. Chosen, not measured; a
# real deployment would tune this against how much a wrong regression could
# cost versus a wider Kalman confidence interval.
CORR_THRESHOLD = 0.30
MIN_PAIR_ROWS = 30          # minimum overlapping rows to trust a correlation
HEALTH_TICK = 10.0           # must match line_sim.HEALTH_TICK


def _station_of(col: str) -> int:
    return int(col.split("_", 1)[0][1:])


def _channel_of(col: str) -> str:
    return col.split("_", 1)[1]


def load_all():
    true = pd.read_csv(DATA_DIR / "unit_features_true.csv")
    obs = pd.read_csv(DATA_DIR / "unit_features.csv")
    registry = pd.read_csv(DATA_DIR / "station_registry.csv")
    sensor_log = pd.read_csv(DATA_DIR / "sensor_log.csv")
    with open(DATA_DIR / "manifest.json") as fh:
        manifest = json.load(fh)
    return true, obs, registry, sensor_log, manifest


def _train(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    return df[df["session_id"].isin(manifest["train_sessions"])]


def _test(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    return df[df["session_id"].isin(manifest["test_sessions"])]


def rank_predictors(true_train: pd.DataFrame, target_col: str) -> list[tuple[str, float]]:
    """Correlate target_col against every OTHER station's same channel,
    using true values so a tier-B station (never observed) can still be
    assessed for predictability during offline calibration."""
    if target_col not in true_train.columns:
        return []
    ch = _channel_of(target_col)
    target_station = _station_of(target_col)
    candidates = [c for c in true_train.columns
                  if c.endswith(f"_{ch}") and _station_of(c) != target_station]
    ranked = []
    for c in candidates:
        m = true_train[[target_col, c]].dropna()
        if len(m) >= MIN_PAIR_ROWS:
            r = m[target_col].corr(m[c])
            if pd.notna(r):
                ranked.append((c, float(r)))
    ranked.sort(key=lambda kv: -abs(kv[1]))
    return ranked


class SpatialVirtualSensor:
    """Regress a station's value on OTHER stations' observed readings."""

    def __init__(self, station: int, channel: str, predictor_cols: list[str],
                model: LinearRegression, residual_std: float):
        self.station = station
        self.channel = channel
        self.predictor_cols = predictor_cols
        self.model = model
        # Residual std from training -> a fixed-width confidence band. A
        # richer version would predict per-row interval width (e.g. via
        # quantile regression); this is the honest minimum for a POC.
        self.residual_std = residual_std

    def estimate(self, observed_row: dict) -> tuple[float | None, float]:
        x = [observed_row.get(c) for c in self.predictor_cols]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in x):
            return None, 0.0
        x_df = pd.DataFrame([x], columns=self.predictor_cols)
        value = float(self.model.predict(x_df)[0])
        # confidence: 1 / (1 + normalised residual spread) -> in (0, 1],
        # shrinking as the regression's historical error grows.
        confidence = float(1.0 / (1.0 + self.residual_std))
        return value, confidence


class TemporalVirtualSensor:
    """1-D Kalman filter tracking a station's own value between sparse checks."""

    def __init__(self, station: int, channel: str, q: float, r: float, x0: float):
        self.station = station
        self.channel = channel
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.x = np.array([[x0]])
        self.kf.F = np.array([[1.0]])
        self.kf.H = np.array([[1.0]])
        self.kf.P = np.array([[r]])
        self.kf.Q = np.array([[q]])
        self.kf.R = np.array([[r]])
        self.last_t = 0.0

    def predict_to(self, t: float) -> tuple[float, float]:
        """Advance the filter to time t with no new observation and return
        (estimate, confidence). Confidence decays as ticks-since-check grows."""
        ticks = max(0.0, (t - self.last_t) / HEALTH_TICK)
        # scale process noise by elapsed ticks: longer gap -> more drift
        self.kf.Q = np.array([[ticks * self._q_per_tick]]) if ticks > 0 else self.kf.Q
        self.kf.predict()
        var = float(self.kf.P[0, 0])
        confidence = float(1.0 / (1.0 + var))
        return float(self.kf.x[0, 0]), confidence

    def update(self, t: float, value: float):
        self.kf.update(np.array([[value]]))
        self.last_t = t

    def set_q_per_tick(self, q_per_tick: float):
        self._q_per_tick = q_per_tick


def _calibrate_kalman(sensor_log_train: pd.DataFrame, station: int, channel: str
                      ) -> TemporalVirtualSensor | None:
    """Estimate process noise per tick from the gaps between real checks,
    and measurement noise from short-gap consecutive readings. Uses only the
    station's own sparse OBSERVED history -- no hidden ground truth."""
    s = (sensor_log_train[(sensor_log_train["station"] == station)
                          & (sensor_log_train["channel"] == channel)]
         .sort_values("t_global"))
    if len(s) < 5:
        return None
    vals = s["value"].to_numpy()
    ts = s["t_global"].to_numpy()
    gaps_ticks = np.diff(ts) / HEALTH_TICK
    diffs = np.diff(vals)
    gaps_ticks = np.maximum(gaps_ticks, 1.0)
    # method-of-moments: Var(diff) ~= q_per_tick * ticks + 2r (a jump plus
    # the noise in each of the two endpoint readings)
    per_tick_sq = (diffs ** 2) / gaps_ticks
    q_per_tick = float(np.median(per_tick_sq)) * 0.5
    r = max(1e-6, float(np.var(diffs[gaps_ticks <= 1.5])) / 2.0) if np.any(gaps_ticks <= 1.5) \
        else max(1e-6, q_per_tick)
    sensor = TemporalVirtualSensor(station, channel, q=q_per_tick, r=r, x0=float(vals[-1]))
    sensor.set_q_per_tick(q_per_tick)
    sensor.last_t = float(ts[-1])
    return sensor


def fit_virtual_sensors(verbose: bool = True) -> dict:
    true, obs, registry, sensor_log, manifest = load_all()
    true_train = _train(true, manifest)
    sensor_log_train = _train(sensor_log, manifest)

    sensors: dict[tuple[int, str], dict] = {}
    needs_inference = registry[registry["tier"] != "A"]["station"].tolist()

    for station in needs_inference:
        tier = registry.loc[registry["station"] == station, "tier"].iloc[0]
        for ch in CHANNELS:
            target_col = f"S{station}_{ch}"
            ranked = rank_predictors(true_train, target_col)
            best = ranked[0] if ranked else None

            if best and abs(best[1]) >= CORR_THRESHOLD:
                predictor_cols = [c for c, r in ranked if abs(r) >= CORR_THRESHOLD][:3]
                # Predictors come from the OBSERVED table (what's actually
                # available live); the target comes from the TRUE table,
                # since a tier-B target has no observed column at all. Join
                # explicitly rather than assume matching row order.
                keys = ["session_id", "unit_id"]
                obs_side = _train(obs, manifest)[keys + predictor_cols]
                true_side = _train(true, manifest)[keys + [target_col]]
                train_rows = obs_side.merge(true_side, on=keys, how="inner").drop(columns=keys)
                train_rows = train_rows.dropna()
                if len(train_rows) < MIN_PAIR_ROWS:
                    method, detail = "unrecoverable", {"reason": "too few complete rows to fit"}
                else:
                    model = LinearRegression().fit(
                        train_rows[predictor_cols], train_rows[target_col])
                    resid = train_rows[target_col] - model.predict(train_rows[predictor_cols])
                    method = "spatial"
                    detail = {"predictors": predictor_cols,
                             "top_corr": round(best[1], 3),
                             "sensor": SpatialVirtualSensor(
                                 station, ch, predictor_cols, model, float(resid.std()))}
            else:
                kalman = _calibrate_kalman(sensor_log_train, station, ch)
                if kalman is not None:
                    method = "temporal"
                    detail = {"sensor": kalman,
                             "best_corr": round(best[1], 3) if best else None}
                else:
                    method = "unrecoverable"
                    detail = {"reason": "no correlated station and no own observations"}

            sensors[(station, ch)] = {"tier_label": tier, "method": method, **detail}
            if verbose:
                name = registry.loc[registry["station"] == station, "name"].iloc[0]
                print(f"  S{station} {name:<14} {ch:<12} tier={tier}  -> {method:<13} "
                      f"{detail.get('predictors', detail.get('best_corr', ''))}")
    return sensors


def validate(sensors: dict) -> pd.DataFrame:
    """Score each sensor's imputation accuracy on the held-out test session,
    against the hidden true values -- never used for fitting."""
    true, obs, registry, sensor_log, manifest = load_all()
    true_test = _test(true, manifest)
    obs_test = _test(obs, manifest)
    sensor_log_test = _test(sensor_log, manifest)

    true_train = _train(true, manifest)

    rows = []
    for (station, ch), info in sensors.items():
        target_col = f"S{station}_{ch}"
        if info["method"] == "spatial":
            sensor: SpatialVirtualSensor = info["sensor"]
            preds, truths = [], []
            for _, r in obs_test.iterrows():
                v, _ = sensor.estimate(r.to_dict())
                if v is not None:
                    t = true_test.loc[true_test["unit_id"] == r["unit_id"], target_col]
                    if len(t) and pd.notna(t.iloc[0]):
                        preds.append(v)
                        truths.append(float(t.iloc[0]))
            if preds:
                preds_a, truths_a = np.array(preds), np.array(truths)
                mae = float(np.mean(np.abs(preds_a - truths_a)))
                # Fair baseline: predicting the training-set mean, i.e. "no
                # virtual sensor at all". 99% of rows are healthy noise by
                # design, so an aggregate MAE win over this can be thin --
                # the real test is the drifted subset below.
                base_mean = float(true_train[target_col].mean())
                base_mae = float(np.mean(np.abs(truths_a - base_mean)))
                corr = float(np.corrcoef(preds_a, truths_a)[0, 1]) if len(preds_a) > 5 else None
                # "drifted" rows: true value sits in the tails of its own
                # training distribution -- i.e. a plausible active episode,
                # not routine noise. This is where a virtual sensor earns
                # its keep; averaging over mostly-quiet rows hides that.
                train_std = float(true_train[target_col].std())
                drifted = np.abs(truths_a - base_mean) > 1.5 * train_std
                drift_mae = (float(np.mean(np.abs(preds_a[drifted] - truths_a[drifted])))
                            if drifted.sum() >= 5 else None)
                drift_base_mae = (float(np.mean(np.abs(base_mean - truths_a[drifted])))
                                 if drifted.sum() >= 5 else None)
                rows.append({"station": station, "channel": ch, "method": "spatial",
                            "n": len(preds), "mae": round(mae, 3),
                            "baseline_mae": round(base_mae, 3),
                            "corr_pred_true": round(corr, 3) if corr is not None else None,
                            "n_drifted": int(drifted.sum()),
                            "drift_mae": round(drift_mae, 3) if drift_mae is not None else None,
                            "drift_baseline_mae": (round(drift_base_mae, 3)
                                                   if drift_base_mae is not None else None)})
        elif info["method"] == "temporal":
            s = (sensor_log_test[(sensor_log_test["station"] == station)
                                 & (sensor_log_test["channel"] == ch)]
                 .sort_values("t_global"))
            if len(s) < 2:
                continue
            kf = TemporalVirtualSensor(station, ch, q=info["sensor"].kf.Q[0, 0] or 1e-6,
                                       r=info["sensor"].kf.R[0, 0], x0=float(s["value"].iloc[0]))
            kf.set_q_per_tick(getattr(info["sensor"], "_q_per_tick", 1e-6))
            kf.last_t = float(s["t_global"].iloc[0])
            errs, ff_errs = [], []
            last_value = float(s["value"].iloc[0])
            for _, row in s.iloc[1:].iterrows():
                pred, _ = kf.predict_to(row["t_global"])
                errs.append(abs(pred - row["value"]))
                # naive forward-fill baseline: "just repeat the last real
                # reading" -- what you'd get without a filter at all
                ff_errs.append(abs(last_value - row["value"]))
                kf.update(row["t_global"], row["value"])
                last_value = row["value"]
            if errs:
                rows.append({"station": station, "channel": ch, "method": "temporal",
                            "n": len(errs), "mae": round(float(np.mean(errs)), 3),
                            "baseline_mae": round(float(np.mean(ff_errs)), 3),
                            "corr_pred_true": None, "n_drifted": None,
                            "drift_mae": None, "drift_baseline_mae": None})
        else:
            rows.append({"station": station, "channel": ch, "method": "unrecoverable",
                        "n": 0, "mae": None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Fitting virtual sensors (method chosen from measured correlation, "
          "not the tier label):")
    sensors = fit_virtual_sensors()

    print("\nValidating against the held-out test session (never used for fitting):")
    report = validate(sensors)
    print(report.to_string(index=False))
