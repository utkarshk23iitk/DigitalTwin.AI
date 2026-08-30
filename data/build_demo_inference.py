"""
build_demo_inference.py — run the trained virtual-sensor + defect-risk pipeline
over data/demo_live/.

Outputs:
    data/demo_live/model_features.csv
    data/demo_live/demo_assessment.csv
    data/demo_live/virtual_sensor_events.csv

Usage:
    python data/build_demo_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from defect_model import DefectModel  # noqa: E402
from effective_trust import assess, compute_input_trust, load_production_split  # noqa: E402
from feature_engineering import build_features_from_frames  # noqa: E402
from train_defect_model import _channel_of_feature, _load_decoy_channels  # noqa: E402
from virtual_sensor import fit_virtual_sensors, load_tuned_virtual_sensor_params  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
DEMO_DIR = DATA_DIR / "demo_live"
MODEL_META_PATH = DATA_DIR.parent / "artifacts" / "defect_model_meta.json"


def _load_demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(DEMO_DIR / "unit_features.csv")
    registry = pd.read_csv(DEMO_DIR / "station_registry.csv")
    sensor_log = pd.read_csv(DEMO_DIR / "sensor_log.csv")
    unit_visit_times = pd.read_csv(DEMO_DIR / "unit_visit_times.csv")
    return obs, registry, sensor_log, unit_visit_times


def _real_only(X: pd.DataFrame) -> pd.DataFrame:
    decoys = _load_decoy_channels()
    cols = [c for c in X.columns if _channel_of_feature(c) not in decoys]
    return X[cols]


def _virtual_sensor_events(features: pd.DataFrame, unit_visit_times: pd.DataFrame,
                           sensors: dict, model_columns: set[str]) -> pd.DataFrame:
    rows = []
    for (station, channel), info in sensors.items():
        if info["method"] not in {"spatial", "temporal"}:
            continue
        est_col = f"S{station}_{channel}_est"
        conf_col = f"S{station}_{channel}_conf"
        if ({est_col, conf_col} - set(features.columns)) or ({est_col, conf_col} - model_columns):
            continue
        visits = unit_visit_times[unit_visit_times["station"] == station][
            ["session_id", "unit_id", "t_global"]
        ]
        merged = visits.merge(
            features[["session_id", "unit_id", est_col, conf_col]],
            on=["session_id", "unit_id"],
            how="left",
        )
        for _, row in merged.iterrows():
            rows.append({
                "session_id": int(row["session_id"]),
                "unit_id": int(row["unit_id"]),
                "station": station,
                "channel": channel,
                "t_global": float(row["t_global"]),
                "method": info["method"],
                "estimate": row[est_col],
                "confidence": row[conf_col],
            })
    return pd.DataFrame(rows)


def build_demo_inference(verbose: bool = True, use_tuned: bool = False) -> dict[str, Path]:
    params = load_tuned_virtual_sensor_params() if use_tuned else {}
    if verbose:
        print("[1/4] fitting virtual sensors on training history...")
    sensors = fit_virtual_sensors(verbose=False, params=params)

    obs, registry, sensor_log, unit_visit_times = _load_demo_frames()
    if verbose:
        print("[2/4] building demo_live engineered features...")
    features = build_features_from_frames(
        obs=obs,
        registry=registry,
        sensor_log=sensor_log,
        unit_visit_times=unit_visit_times,
        sensors=sensors,
        verbose=verbose,
        params=params,
    )
    features_path = DEMO_DIR / "model_features.csv"
    features.to_csv(features_path, index=False)
    if verbose:
        print(f"[write] {features_path}")

    if verbose:
        print("[3/4] training production defect model and scoring demo units...")
    _, X_train, y_train, train_mask, _ = load_production_split()
    model_config = json.loads(MODEL_META_PATH.read_text()) if MODEL_META_PATH.exists() else {}
    model = DefectModel(
        xgb_params=model_config.get("xgb_params"),
        threshold_params=model_config.get("threshold_params"),
        auto_load_tuned=False,
    ).fit(X_train[train_mask], y_train[train_mask])
    importances = pd.Series(model.model.feature_importances_, index=X_train.columns)

    non_feature = ["session_id", "unit_id", "response", "defect_occurred_at", "defect_caught_at"]
    X_demo = _real_only(features.drop(columns=non_feature))
    generated_virtual = {col for col in X_demo.columns if col.endswith(("_est", "_conf"))}
    missing_from_model = sorted(generated_virtual - set(X_train.columns))
    if missing_from_model:
        raise RuntimeError(
            "the trained feature table is stale and would drop generated virtual sensors: "
            + ", ".join(missing_from_model)
            + ". Re-run python src/feature_engineering.py and retrain the model."
        )
    X_demo = X_demo.reindex(columns=X_train.columns, fill_value=float("nan"))
    assessment, risk_thr, trust_thr = assess(model, X_demo, importances, trust_thr=0.5)
    latest_visit = (unit_visit_times.groupby(["session_id", "unit_id"])["t_global"]
                    .max().reset_index().rename(columns={"t_global": "latest_t"}))
    assessment = pd.concat([features[non_feature].reset_index(drop=True), assessment.reset_index(drop=True)], axis=1)
    assessment = assessment.merge(latest_visit, on=["session_id", "unit_id"], how="left")
    assessment["risk_threshold"] = risk_thr
    assessment["trust_threshold"] = trust_thr
    assessment_path = DEMO_DIR / "demo_assessment.csv"
    assessment.to_csv(assessment_path, index=False)
    if verbose:
        print(f"[write] {assessment_path}")

    if verbose:
        print("[4/4] writing virtual-sensor fill events for the dashboard...")
    virtual_events = _virtual_sensor_events(
        features, unit_visit_times, sensors, model_columns=set(X_train.columns)
    )
    virtual_path = DEMO_DIR / "virtual_sensor_events.csv"
    virtual_events.to_csv(virtual_path, index=False)
    if verbose:
        print(f"[write] {virtual_path}")

    return {
        "features": features_path,
        "assessment": assessment_path,
        "virtual_events": virtual_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-tuned", action="store_true",
                    help="use validation-selected virtual-sensor parameters")
    args = ap.parse_args()
    build_demo_inference(use_tuned=args.use_tuned)


if __name__ == "__main__":
    main()
