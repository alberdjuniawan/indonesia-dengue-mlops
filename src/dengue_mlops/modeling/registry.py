from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import BaseModel, load_model_from_dict

REGISTRY_DIR = Path("models/registry")
CHECKPOINTS_DIR = Path("models/checkpoints")
CHAMPION_MODEL_FILE = REGISTRY_DIR / "champion_model.json"
CHAMPION_METADATA_FILE = REGISTRY_DIR / "model_metadata.json"


def save_checkpoint(
    model: BaseModel, version: str, metrics: dict[str, float], feature_names: list[str]
) -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINTS_DIR / f"model_v{version}.json"
    payload = {
        "version": version,
        "saved_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "feature_names": feature_names,
        "model": model.to_dict(),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return checkpoint_file


def register_champion(
    model: BaseModel,
    version: str,
    metrics: dict[str, float],
    feature_names: list[str],
    description: str = "Production Dengue Forecast Model",
) -> tuple[Path, Path]:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # Save model weights/architecture
    model_payload = {"version": version, "feature_names": feature_names, "model": model.to_dict()}
    CHAMPION_MODEL_FILE.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    # Save metadata
    metadata = {
        "model_name": getattr(model, "name", "dengue_forecast_model"),
        "version": version,
        "promoted_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "feature_count": len(feature_names),
        "features": feature_names,
        "description": description,
        "status": "CHAMPION",
    }
    CHAMPION_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Also save as checkpoint
    save_checkpoint(model, version, metrics, feature_names)

    return CHAMPION_MODEL_FILE, CHAMPION_METADATA_FILE


def load_champion() -> tuple[BaseModel, dict[str, Any]]:
    if not CHAMPION_MODEL_FILE.exists() or not CHAMPION_METADATA_FILE.exists():
        raise FileNotFoundError("Champion model is not registered yet.")

    model_data = json.loads(CHAMPION_MODEL_FILE.read_text(encoding="utf-8"))
    metadata = json.loads(CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))

    model = load_model_from_dict(model_data["model"])
    return model, metadata
