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
STAGE_TRANSITION_LOG_FILE = REGISTRY_DIR / "stage_transition_log.json"

# Production Quality Gate SLA Thresholds (LK-01 & MLOps Standards)
DEFAULT_MIN_R2: float = 0.80
DEFAULT_MAX_MAE: float = 25.0
DEFAULT_MAX_RMSE: float = 45.0
DEFAULT_MAX_WAPE: float = 25.0


def save_checkpoint(
    model: BaseModel,
    version: str,
    metrics: dict[str, float],
    feature_names: list[str],
    checkpoints_dir: Path = CHECKPOINTS_DIR,
) -> Path:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoints_dir / f"model_v{version}.json"
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
    registry_dir: Path = REGISTRY_DIR,
) -> tuple[Path, Path]:
    registry_dir.mkdir(parents=True, exist_ok=True)
    champion_model_file = registry_dir / "champion_model.json"
    champion_metadata_file = registry_dir / "model_metadata.json"

    # Save model weights/architecture
    model_payload = {"version": version, "feature_names": feature_names, "model": model.to_dict()}
    champion_model_file.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

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
    champion_metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Also save as checkpoint
    save_checkpoint(model, version, metrics, feature_names)

    return champion_model_file, champion_metadata_file


def load_champion(registry_dir: Path = REGISTRY_DIR) -> tuple[BaseModel, dict[str, Any]]:
    champion_model_file = registry_dir / "champion_model.json"
    champion_metadata_file = registry_dir / "model_metadata.json"

    if not champion_model_file.exists() or not champion_metadata_file.exists():
        raise FileNotFoundError("Champion model is not registered yet.")

    model_data = json.loads(champion_model_file.read_text(encoding="utf-8"))
    metadata = json.loads(champion_metadata_file.read_text(encoding="utf-8"))

    model = load_model_from_dict(model_data["model"])
    return model, metadata


def validate_model_metrics(
    metrics: dict[str, float],
    min_r2: float = DEFAULT_MIN_R2,
    max_mae: float = DEFAULT_MAX_MAE,
    max_rmse: float = DEFAULT_MAX_RMSE,
    max_wape: float = DEFAULT_MAX_WAPE,
) -> dict[str, Any]:
    """Validate candidate model metrics against production SLA performance thresholds.

    Evaluates R2, MAE, RMSE, and WAPE to verify that newly trained models meet
    rigorous quality standards before promotion to MLflow Model Registry stages.
    """
    r2_val = float(metrics.get("r2", 0.0))
    mae_val = float(metrics.get("mae", float("inf")))
    rmse_val = float(metrics.get("rmse", float("inf")))
    wape_val = float(metrics.get("wape", float("inf")))

    r2_passed = r2_val >= min_r2
    mae_passed = mae_val <= max_mae
    rmse_passed = rmse_val <= max_rmse
    wape_passed = wape_val <= max_wape

    failure_reasons: list[str] = []
    if not r2_passed:
        failure_reasons.append(f"R² ({r2_val:.4f}) is below minimum threshold ({min_r2:.4f})")
    if not mae_passed:
        failure_reasons.append(f"MAE ({mae_val:.2f}) exceeds maximum threshold ({max_mae:.2f})")
    if not rmse_passed:
        failure_reasons.append(f"RMSE ({rmse_val:.2f}) exceeds maximum threshold ({max_rmse:.2f})")
    if not wape_passed:
        failure_reasons.append(
            f"WAPE ({wape_val:.2f}%) exceeds maximum threshold ({max_wape:.2f}%)"
        )

    all_passed = len(failure_reasons) == 0

    return {
        "status": "PASSED" if all_passed else "FAILED",
        "passed": all_passed,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "checks": {
            "r2": {
                "threshold": min_r2,
                "actual": round(r2_val, 4),
                "operator": ">=",
                "passed": r2_passed,
            },
            "mae": {
                "threshold": max_mae,
                "actual": round(mae_val, 4),
                "operator": "<=",
                "passed": mae_passed,
            },
            "rmse": {
                "threshold": max_rmse,
                "actual": round(rmse_val, 4),
                "operator": "<=",
                "passed": rmse_passed,
            },
            "wape": {
                "threshold": max_wape,
                "actual": round(wape_val, 4),
                "operator": "<=",
                "passed": wape_passed,
            },
        },
        "failure_reasons": failure_reasons,
    }
