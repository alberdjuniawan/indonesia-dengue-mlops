from __future__ import annotations

import json
import warnings
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


def promote_to_staging(
    model_name: str,
    version: str | int,
    tracking_uri: str = "sqlite:///mlflow.db",
    reason: str = "Promoted to Staging via automated quality gate validation (LK-08)",
    actor: str = "GitHub Actions (mlops-automation.yaml) on Host beto",
    metrics: dict[str, float] | None = None,
    log_file: Path = STAGE_TRANSITION_LOG_FILE,
) -> dict[str, Any]:
    """Transition a validated model version to 'Staging' in MLflow Model Registry.

    Applies the industrial dual-registration pattern:
    1. Transitions model stage to 'Staging' via transition_model_version_stage.
    2. Sets registered model alias '@staging' pointing directly to this version.
    3. Attaches quality gate audit metadata and verification tags.
    4. Records the automated audit event in models/registry/stage_transition_log.json.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    target_version_str = str(version)

    # 1. Inspect existing version state
    mv_before = client.get_model_version(name=model_name, version=target_version_str)
    from_stage = getattr(mv_before, "current_stage", "None")

    # 2. Transition stage to 'Staging'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mv_after = client.transition_model_version_stage(
            name=model_name,
            version=target_version_str,
            stage="Staging",
            archive_existing_versions=False,
        )

    # 3. Assign '@staging' alias
    client.set_registered_model_alias(name=model_name, alias="staging", version=target_version_str)

    # 4. Attach validation tags
    now_iso = datetime.now(UTC).isoformat()
    client.set_model_version_tag(model_name, target_version_str, "stage", "Staging")
    client.set_model_version_tag(model_name, target_version_str, "quality_gate", "PASSED")
    client.set_model_version_tag(model_name, target_version_str, "promoted_to_staging_at", now_iso)
    client.set_model_version_tag(model_name, target_version_str, "promoted_by", actor)
    if metrics:
        client.set_model_version_tag(
            model_name, target_version_str, "validation_mae", str(round(metrics.get("mae", 0.0), 3))
        )
        client.set_model_version_tag(
            model_name, target_version_str, "validation_r2", str(round(metrics.get("r2", 0.0), 4))
        )

    # 5. Record transition log entry
    transition_entry = {
        "timestamp": now_iso,
        "model_name": model_name,
        "version": target_version_str,
        "from_stage": from_stage,
        "to_stage": getattr(mv_after, "current_stage", "Staging"),
        "archive_existing": False,
        "reason": reason,
        "actor": actor,
    }

    log_data: dict[str, Any] = {}
    if log_file.exists():
        try:
            log_data = json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log_data = {}

    log_data.setdefault("generated_at", now_iso)
    log_data["latest_staging_version"] = target_version_str
    log_data["last_staging_promotion"] = now_iso
    transitions_list = log_data.setdefault("transitions", [])
    transitions_list.append(transition_entry)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    return {
        "model_name": model_name,
        "promoted_version": target_version_str,
        "previous_stage": from_stage,
        "target_stage": "Staging",
        "alias": "staging",
        "transition_timestamp": now_iso,
        "actor": actor,
        "reason": reason,
    }
