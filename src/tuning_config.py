from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
TUNING_DIR = ARTIFACT_DIR / "tuning"
CHECKPOINT_DIR = TUNING_DIR / "checkpoints"

DEFECT_PARAMS_PATH = TUNING_DIR / "defect_model_best.json"
VIRTUAL_SENSOR_PARAMS_PATH = TUNING_DIR / "virtual_sensor_best.json"
STUDY_DB_PATH = TUNING_DIR / "optuna_studies.db"
TRAINED_MODEL_PATH = ARTIFACT_DIR / "defect_model.json"
TRAINED_MODEL_META_PATH = ARTIFACT_DIR / "defect_model_meta.json"


def ensure_artifact_dirs() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    TUNING_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path) as fh:
        return json.load(fh)


def save_json(path: Path, payload: Any) -> None:
    ensure_artifact_dirs()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def optuna_storage_uri() -> str:
    ensure_artifact_dirs()
    return f"sqlite:///{STUDY_DB_PATH}"


def best_payload(study_name: str, metric_name: str, metric_value: float,
                 params: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "study_name": study_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "params": params,
        "saved_at_utc": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    return payload


def checkpoint_path(study_name: str) -> Path:
    safe = study_name.replace("/", "_")
    return CHECKPOINT_DIR / f"{safe}.json"
