from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dengue_mlops.modeling.registry import (
    DEFAULT_MAX_MAE,
    DEFAULT_MAX_RMSE,
    DEFAULT_MAX_WAPE,
    DEFAULT_MIN_R2,
    promote_to_staging,
    validate_model_metrics,
)


def test_validate_model_metrics_passing() -> None:
    """Verify that candidate model meeting all SLA thresholds passes quality gate."""
    metrics = {"rmse": 28.50, "mae": 13.80, "r2": 0.9650, "wape": 14.10}
    result = validate_model_metrics(metrics)

    assert result["passed"] is True
    assert result["status"] == "PASSED"
    assert len(result["failure_reasons"]) == 0
    assert result["checks"]["r2"]["passed"] is True
    assert result["checks"]["mae"]["passed"] is True
    assert result["checks"]["rmse"]["passed"] is True
    assert result["checks"]["wape"]["passed"] is True
    assert result["checks"]["r2"]["actual"] == 0.9650
    assert result["checks"]["mae"]["actual"] == 13.80


def test_validate_model_metrics_failing_r2() -> None:
    """Verify that model with R2 below SLA threshold fails quality gate."""
    metrics = {
        "rmse": 35.0,
        "mae": 18.0,
        "r2": 0.7450,  # Below minimum 0.80
        "wape": 17.5,
    }
    result = validate_model_metrics(metrics, min_r2=0.80)

    assert result["passed"] is False
    assert result["status"] == "FAILED"
    assert result["checks"]["r2"]["passed"] is False
    assert result["checks"]["mae"]["passed"] is True
    assert any("R²" in reason for reason in result["failure_reasons"])


def test_validate_model_metrics_failing_mae() -> None:
    """Verify that model with MAE exceeding SLA threshold fails quality gate."""
    metrics = {
        "rmse": 40.0,
        "mae": 28.50,  # Exceeds maximum 25.0
        "r2": 0.8850,
        "wape": 22.0,
    }
    result = validate_model_metrics(metrics, max_mae=25.0)

    assert result["passed"] is False
    assert result["status"] == "FAILED"
    assert result["checks"]["mae"]["passed"] is False
    assert result["checks"]["r2"]["passed"] is True
    assert any("MAE" in reason for reason in result["failure_reasons"])


def test_validate_model_metrics_multiple_failures() -> None:
    """Verify that all failing checks are captured when multiple thresholds are breached."""
    metrics = {
        "rmse": 52.0,  # Exceeds 45.0
        "mae": 32.0,  # Exceeds 25.0
        "r2": 0.65,  # Below 0.80
        "wape": 31.0,  # Exceeds 25.0%
    }
    result = validate_model_metrics(metrics)

    assert result["passed"] is False
    assert len(result["failure_reasons"]) == 4
    for key in ("r2", "mae", "rmse", "wape"):
        assert result["checks"][key]["passed"] is False


def test_validate_model_metrics_custom_thresholds() -> None:
    """Verify that custom SLA threshold parameters are respected."""
    metrics = {"rmse": 20.0, "mae": 10.0, "r2": 0.92, "wape": 9.5}

    # Strict thresholds: min_r2=0.95 should fail
    strict_res = validate_model_metrics(metrics, min_r2=0.95, max_mae=12.0)
    assert strict_res["passed"] is False
    assert strict_res["checks"]["r2"]["passed"] is False

    # Relaxed thresholds: min_r2=0.90 should pass
    relaxed_res = validate_model_metrics(metrics, min_r2=0.90, max_mae=12.0)
    assert relaxed_res["passed"] is True


def test_promote_to_staging_workflow(tmp_path: Path) -> None:
    """Verify that promote_to_staging transitions stage, assigns alias, and writes audit log."""
    mock_client = MagicMock()
    mock_mv_before = MagicMock()
    mock_mv_before.current_stage = "None"
    mock_client.get_model_version.return_value = mock_mv_before

    mock_mv_after = MagicMock()
    mock_mv_after.current_stage = "Staging"
    mock_client.transition_model_version_stage.return_value = mock_mv_after

    test_log_file = tmp_path / "stage_transition_log.json"
    metrics = {"r2": 0.965, "mae": 14.2}

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        res = promote_to_staging(
            model_name="DengueOutbreakForecaster",
            version="6",
            tracking_uri="sqlite:///test.db",
            reason="CI test promotion",
            actor="pytest_runner",
            metrics=metrics,
            log_file=test_log_file,
        )

    # 1. Verify MlflowClient interactions
    mock_client.get_model_version.assert_called_once_with(
        name="DengueOutbreakForecaster", version="6"
    )
    mock_client.transition_model_version_stage.assert_called_once_with(
        name="DengueOutbreakForecaster",
        version="6",
        stage="Staging",
        archive_existing_versions=False,
    )
    mock_client.set_registered_model_alias.assert_called_once_with(
        name="DengueOutbreakForecaster", alias="staging", version="6"
    )

    # 2. Verify returned summary
    assert res["model_name"] == "DengueOutbreakForecaster"
    assert res["promoted_version"] == "6"
    assert res["previous_stage"] == "None"
    assert res["target_stage"] == "Staging"
    assert res["alias"] == "staging"
    assert res["actor"] == "pytest_runner"

    # 3. Verify log file written
    assert test_log_file.exists()
    logged = json.loads(test_log_file.read_text(encoding="utf-8"))
    assert logged["latest_staging_version"] == "6"
    assert len(logged["transitions"]) == 1
    assert logged["transitions"][0]["to_stage"] == "Staging"
    assert logged["transitions"][0]["reason"] == "CI test promotion"


def test_quality_gate_defaults() -> None:
    """Verify that default quality gate parameters align with LK-01 specifications."""
    assert DEFAULT_MIN_R2 == 0.80
    assert DEFAULT_MAX_MAE == 25.0
    assert DEFAULT_MAX_RMSE == 45.0
    assert DEFAULT_MAX_WAPE == 25.0
