from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import load_dataset, split_features
from .metrics import calculate_metrics
from .models import (
    BaselineLagModel,
    BaseModel,
    EnsembleGradientRegressor,
    PyTorchTemporalRegressor,
    RidgeRegressionModel,
)
from .registry import register_champion
from .tracking import tracker


def run_training_pipeline(
    features_path: Path = Path("data/features/features.csv"),
    model_version: str = "1.0.0",
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    records = load_dataset(features_path)
    split = split_features(records)

    print(
        f"Dataset split: train={len(split.train_x)}, val={len(split.val_x)}, test={len(split.test_x)}"
    )

    candidates: list[BaseModel] = [
        BaselineLagModel(lag_idx=0),
        RidgeRegressionModel(alpha=1.0, lr=0.005, epochs=500),
        EnsembleGradientRegressor(ridge_weight=0.70, lag_weight=0.30),
        PyTorchTemporalRegressor(hidden_dim=64, lr=0.005, epochs=150, device=device),
    ]

    best_model: BaseModel | None = None
    best_val_mae = float("inf")
    candidate_results: list[dict[str, Any]] = []

    for model in candidates:
        model_name = getattr(model, "name", type(model).__name__)
        print(f"\n--- Training {model_name} ---")

        run_id = tracker.start_run(run_name=f"train_{model_name}")
        tracker.log_param("model_name", model_name)
        tracker.log_param("seed", seed)
        tracker.log_param("version", model_version)

        # Fit model
        model.fit(split.train_x, split.train_y)

        # Train & Val evaluation
        train_preds = model.predict(split.train_x)
        val_preds = model.predict(split.val_x)

        train_metrics = calculate_metrics(split.train_y, train_preds)
        val_metrics = calculate_metrics(split.val_y, val_preds)

        print(f"  Train metrics: {train_metrics}")
        print(f"  Val metrics  : {val_metrics}")

        tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        tracker.end_run()

        result = {
            "model_name": model_name,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "run_id": run_id,
        }
        candidate_results.append(result)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model = model

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    champion_name = getattr(best_model, "name", type(best_model).__name__)
    print(f"\nSelected Champion Model: {champion_name} (Val MAE: {best_val_mae})")

    # Evaluate champion on holdout test set
    test_preds = best_model.predict(split.test_x)
    test_metrics = calculate_metrics(split.test_y, test_preds)
    print(f"Holdout Test Metrics: {test_metrics}")

    # Register champion model
    champion_file, metadata_file = register_champion(
        model=best_model,
        version=model_version,
        metrics=test_metrics,
        feature_names=split.feature_names,
        description=f"Champion {champion_name} trained on historical dengue data",
    )

    report = {
        "status": "SUCCESS",
        "champion_model": champion_name,
        "version": model_version,
        "test_metrics": test_metrics,
        "candidates": candidate_results,
        "model_file": str(champion_file),
        "metadata_file": str(metadata_file),
    }

    # Save training report
    report_file = Path("models/training_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
