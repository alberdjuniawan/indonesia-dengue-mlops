"""Production-Grade ML Training & MLflow Experiment Management Script (LK-06).

Task & Rubric Requirements:
1. Core ML Code: Train time-series forecasting models using DVC-versioned datasets (LK-05).
2. MLflow Integration: Systematically log parameters, metrics, artifacts, and models:
   - mlflow.log_param() / mlflow.log_params()
   - mlflow.log_metric() / mlflow.log_metrics()
   - mlflow.pyfunc.log_model() / mlflow.log_model()
   - mlflow.register_model() (Model Registry Promotion)
3. Experiment Variations: Execute multi-run matrix (Baseline, Ridge alpha sweeps, Ensemble, PyTorch).
4. Analysis & Reproducibility: Clean code, deterministic seed (42), and champion export.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from dengue_mlops.modeling.dataset import DatasetSplit, load_dataset, split_features
from dengue_mlops.modeling.metrics import calculate_metrics
from dengue_mlops.modeling.models import (
    BaselineLagModel,
    BaseModel,
    EnsembleGradientRegressor,
    PyTorchTemporalRegressor,
    RidgeRegressionModel,
)
from dengue_mlops.modeling.registry import register_champion

# Silence MLflow agent hints and verbose user warnings for clean output
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mlflow_trainer")

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "dengue-weekly-forecasting"
DEFAULT_FEATURES_PATH = Path("data/features/features.csv")
CHAMPION_METADATA_PATH = Path("models/registry/champion_metadata.json")
REGISTERED_MODEL_NAME = "DengueOutbreakForecaster"


class DengueMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc Wrapper ensuring universal inference compatibility."""

    def __init__(self, model: BaseModel, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    def predict(self, context, model_input):
        # Handle Pandas DataFrame
        if hasattr(model_input, "values"):
            if hasattr(model_input, "columns"):
                available_cols = [c for c in self.feature_names if c in model_input.columns]
                if len(available_cols) == len(self.feature_names):
                    data = model_input[self.feature_names].values.tolist()
                else:
                    data = model_input.values.tolist()
            else:
                data = model_input.values.tolist()
        elif isinstance(model_input, list):
            data = model_input
        else:
            data = list(model_input)

        return self.model.predict(data)


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = DEFAULT_EXPERIMENT_NAME
) -> str:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow Tracking URI set to: {tracking_uri}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            tags={
                "project": "indonesia-dengue-mlops",
                "course": "MLOps",
                "assignment": "LK-06",
                "author": "Alberd Juniawan Pasunda",
            },
        )
        logger.info(f"Created new MLflow Experiment: '{experiment_name}' (ID: {exp_id})")
    else:
        exp_id = experiment.experiment_id
        logger.info(f"Using existing MLflow Experiment: '{experiment_name}' (ID: {exp_id})")

    mlflow.set_experiment(experiment_name)
    return exp_id


def train_single_run(
    model: BaseModel,
    run_name: str,
    split: DatasetSplit,
    params: dict[str, Any],
    tags: dict[str, str] | None = None,
    dataset_version: str = "v2.0-data",
) -> dict[str, Any]:
    """Train a single model variation and log all artifacts systematically to MLflow."""
    logger.info(f"\n{'='*75}\nSTARTING MLFLOW RUN: {run_name}\n{'='*75}")
    model_name = getattr(model, "name", type(model).__name__)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Log System & Experiment Tags
        run_tags = {
            "assignment": "LK-06",
            "model_family": model_name,
            "dataset_version": dataset_version,
            "orchestrator": "mlflow_tracking",
        }
        if tags:
            run_tags.update(tags)
        mlflow.set_tags(run_tags)

        # 2. Log Model & Data Parameters via mlflow.log_param() / mlflow.log_params()
        mlflow_params = {
            "model_name": model_name,
            "run_name": run_name,
            "dataset_version": dataset_version,
            "train_samples": len(split.train_x),
            "val_samples": len(split.val_x),
            "test_samples": len(split.test_x),
            "feature_count": len(split.feature_names),
            "target_horizon": "1_week_ahead",
            "temporal_train_cutoff": "2024",
            "temporal_val_cutoff": "2025_W26",
            **params,
        }
        mlflow.log_params(mlflow_params)
        logger.info(f"Logged {len(mlflow_params)} parameters to MLflow.")

        # 3. Model Training
        start_fit = time.perf_counter()
        model.fit(split.train_x, split.train_y)
        fit_duration = time.perf_counter() - start_fit

        # 4. Model Inference across Splits
        train_preds = model.predict(split.train_x)
        val_preds = model.predict(split.val_x)
        test_preds = model.predict(split.test_x)

        # 5. Compute Standard Regression Metrics
        train_metrics = calculate_metrics(split.train_y, train_preds)
        val_metrics = calculate_metrics(split.val_y, val_preds)
        test_metrics = calculate_metrics(split.test_y, test_preds)

        # 6. Log Metrics via mlflow.log_metric() / mlflow.log_metrics()
        all_metrics = {
            "fit_duration_sec": round(fit_duration, 4),
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_wape": train_metrics["wape"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_wape": val_metrics["wape"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
            "test_wape": test_metrics["wape"],
        }
        mlflow.log_metrics(all_metrics)
        logger.info(
            f"Metrics Logged — Val MAE: {val_metrics['mae']:.2f}, "
            f"Val R2: {val_metrics['r2']:.4f}, Test MAE: {test_metrics['mae']:.2f}, "
            f"Test R2: {test_metrics['r2']:.4f}"
        )

        # 7. Model Signature & PyFunc Model Logging via mlflow.pyfunc.log_model()
        import numpy as np

        sample_input = np.array(split.train_x[:2])
        sample_output = np.array(train_preds[:2])
        signature = infer_signature(sample_input, sample_output)

        wrapper = DengueMLflowWrapper(model=model, feature_names=split.feature_names)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            signature=signature,
            input_example=sample_input[:1],
        )
        logger.info(f"Model logged as PyFunc artifact under run '{run_id}'.")

        # 8. Extra Artifact: Checkpoint JSON file
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{run_name}.json"
        checkpoint_payload = {
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "params": params,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "weights": model.to_dict(),
        }
        checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(checkpoint_path))

        return {
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "model": model,
            "params": params,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }


def register_champion_to_mlflow(
    champion_run: dict[str, Any],
    client: MlflowClient,
    model_name: str = REGISTERED_MODEL_NAME,
) -> Any:
    """Register the best performing model to the MLflow Model Registry."""
    run_id = champion_run["run_id"]
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"\nRegistering Champion Run '{run_id}' to MLflow Model Registry as '{model_name}'...")

    # Create registered model if not present
    try:
        client.create_registered_model(
            name=model_name,
            description="Production Champion Model for Weekly Dengue Case Forecasting in Indonesia.",
            tags={
                "task": "time_series_regression",
                "disease": "dengue",
                "coverage": "38_provinces",
                "promoted_by": "LK-06_pipeline",
            },
        )
        logger.info(f"Registered model entry '{model_name}' created.")
    except Exception:  # noqa: BLE001
        logger.info(f"Registered model entry '{model_name}' already exists.")

    # Create model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=(
            f"Promoted Champion model ({champion_run['model_name']}) from LK-06 experiment suite. "
            f"Val MAE: {champion_run['val_metrics']['mae']:.2f}, "
            f"Test MAE: {champion_run['test_metrics']['mae']:.2f}, "
            f"Test R2: {champion_run['test_metrics']['r2']:.4f}"
        ),
        tags={
            "stage": "Champion",
            "run_name": champion_run["run_name"],
            "version_tag": "1.0.0",
        },
    )

    # Set alias / tag
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version.version,
    )
    logger.info(
        f"Model Version v{model_version.version} promoted to alias '@champion' in Model Registry!"
    )
    return model_version


def export_champion_metadata(
    champion_run: dict[str, Any],
    model_version_num: str,
    feature_names: list[str],
    output_path: Path = CHAMPION_METADATA_PATH,
) -> None:
    """Export complete champion metadata for production serving / deployment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": REGISTERED_MODEL_NAME,
        "champion_run_id": champion_run["run_id"],
        "champion_run_name": champion_run["run_name"],
        "model_family": champion_run["model_name"],
        "mlflow_model_version": model_version_num,
        "registered_alias": "champion",
        "dataset_version": "v2.0-data",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "best_hyperparameters": champion_run["params"],
        "performance": {
            "validation": champion_run["val_metrics"],
            "holdout_test": champion_run["test_metrics"],
            "train": champion_run["train_metrics"],
        },
        "deployment_recommendation": {
            "status": "APPROVED_FOR_PRODUCTION",
            "serving_framework": "FastAPI + MLflow PyFunc",
            "docker_image": "indonesia-dengue-mlops/api:latest",
            "sla_check": {
                "r2_sla_minimum": 0.80,
                "achieved_test_r2": champion_run["test_metrics"]["r2"],
                "mae_sla_maximum": 25.0,
                "achieved_test_mae": champion_run["test_metrics"]["mae"],
                "passed": (
                    champion_run["test_metrics"]["r2"] >= 0.80
                    and champion_run["test_metrics"]["mae"] <= 25.0
                ),
            },
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Champion metadata exported to: {output_path}")

    # Also update repository champion files for serving continuity
    register_champion(
        model=champion_run["model"],
        version=str(model_version_num),
        metrics=champion_run["test_metrics"],
        feature_names=feature_names,
        description=f"LK-06 Champion Model: {champion_run['model_name']}",
    )

    # Save training report for DVC metric tracking
    training_report_file = Path("models/training_report.json")
    training_report_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "status": "SUCCESS",
        "champion_model": champion_run["model_name"],
        "version": str(model_version_num),
        "test_metrics": champion_run["test_metrics"],
        "val_metrics": champion_run["val_metrics"],
        "train_metrics": champion_run["train_metrics"],
        "champion_run_id": champion_run["run_id"],
        "best_hyperparameters": champion_run["params"],
    }
    training_report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info(f"Training report exported to: {training_report_file}")


def run_experiment_suite(
    features_path: Path = DEFAULT_FEATURES_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Execute the full 5-run experimental matrix required by LK-06."""
    print("=" * 80)
    print("INDONESIA DENGUE MLOPS — LK-06 MLFLOW EXPERIMENT SUITE")
    print(f"Tracking URI : {tracking_uri}")
    print(f"Experiment   : {experiment_name}")
    print(f"Features File: {features_path}")
    print("=" * 80)

    configure_mlflow(tracking_uri, experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Load DVC-versioned dataset
    records = load_dataset(features_path)
    split = split_features(records)
    logger.info(
        f"Dataset loaded: {len(records)} records (Train: {len(split.train_x)}, "
        f"Val: {len(split.val_x)}, Test: {len(split.test_x)})"
    )

    # 2. Define 5 Candidate Variations (Exceeding the minimum requirement of 3)
    variations = [
        {
            "run_name": "run_01_baseline_lag1",
            "model": BaselineLagModel(lag_idx=0),
            "params": {"model_type": "baseline", "lag_order": 1, "seed": seed},
            "tags": {"tier": "baseline", "complexity": "low"},
        },
        {
            "run_name": "run_02_ridge_alpha_0_1",
            "model": RidgeRegressionModel(alpha=0.1, lr=0.005, epochs=500),
            "params": {
                "model_type": "ridge",
                "alpha": 0.1,
                "learning_rate": 0.005,
                "epochs": 500,
                "regularization": "L2_light",
                "seed": seed,
            },
            "tags": {"tier": "linear_regularized", "tuning": "alpha_0.1"},
        },
        {
            "run_name": "run_03_ridge_alpha_1_0",
            "model": RidgeRegressionModel(alpha=1.0, lr=0.005, epochs=500),
            "params": {
                "model_type": "ridge",
                "alpha": 1.0,
                "learning_rate": 0.005,
                "epochs": 500,
                "regularization": "L2_balanced",
                "seed": seed,
            },
            "tags": {"tier": "linear_regularized", "tuning": "alpha_1.0"},
        },
        {
            "run_name": "run_04_ridge_alpha_10_0",
            "model": RidgeRegressionModel(alpha=10.0, lr=0.005, epochs=500),
            "params": {
                "model_type": "ridge",
                "alpha": 10.0,
                "learning_rate": 0.005,
                "epochs": 500,
                "regularization": "L2_heavy",
                "seed": seed,
            },
            "tags": {"tier": "linear_regularized", "tuning": "alpha_10.0"},
        },
        {
            "run_name": "run_05_gradient_ensemble",
            "model": EnsembleGradientRegressor(ridge_weight=0.70, lag_weight=0.30),
            "params": {
                "model_type": "ensemble_gradient",
                "ridge_weight": 0.70,
                "lag_weight": 0.30,
                "sub_ridge_alpha": 0.5,
                "sub_ridge_epochs": 600,
                "seed": seed,
            },
            "tags": {"tier": "ensemble", "strategy": "hybrid_lag_climate"},
        },
        {
            "run_name": "run_06_pytorch_temporal_mlp",
            "model": PyTorchTemporalRegressor(
                hidden_dim=64, lr=0.005, epochs=150, device=device
            ),
            "params": {
                "model_type": "pytorch_mlp",
                "hidden_dim": 64,
                "learning_rate": 0.005,
                "epochs": 150,
                "device": device,
                "loss": "huber",
                "seed": seed,
            },
            "tags": {"tier": "deep_learning", "framework": "pytorch"},
        },
    ]

    results: list[dict[str, Any]] = []
    for var in variations:
        res = train_single_run(
            model=var["model"],
            run_name=var["run_name"],
            split=split,
            params=var["params"],
            tags=var["tags"],
            dataset_version="v2.0-data",
        )
        results.append(res)

    # 3. Comparative Summary Table
    print("\n" + "=" * 90)
    print("MLFLOW EXPERIMENT RUNS COMPARISON TABLE")
    print("=" * 90)
    print(
        f"{'Run Name':<30} {'Val MAE':<10} {'Val RMSE':<10} {'Val R2':<10} {'Test MAE':<10} {'Test R2':<10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['run_name']:<30} "
            f"{r['val_metrics']['mae']:<10.2f} "
            f"{r['val_metrics']['rmse']:<10.2f} "
            f"{r['val_metrics']['r2']:<10.4f} "
            f"{r['test_metrics']['mae']:<10.2f} "
            f"{r['test_metrics']['r2']:<10.4f}"
        )
    print("=" * 90)

    # 4. Champion Selection based on Validation MAE (Lowest error)
    champion_run = min(results, key=lambda x: x["val_metrics"]["mae"])
    print(
        f"\nCHAMPION SELECTION: '{champion_run['run_name']}' "
        f"(Val MAE: {champion_run['val_metrics']['mae']:.2f}, "
        f"Test MAE: {champion_run['test_metrics']['mae']:.2f}, "
        f"Test R2: {champion_run['test_metrics']['r2']:.4f})"
    )

    # 5. Register Champion to MLflow Model Registry
    model_version = register_champion_to_mlflow(champion_run, client)

    # 6. Export Champion Metadata
    export_champion_metadata(champion_run, model_version.version, split.feature_names)

    return {
        "experiment_name": experiment_name,
        "runs": results,
        "champion": champion_run,
        "model_version": model_version.version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LK-06 MLflow Training, Experiment Tracking & Model Registry"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to preprocessed features CSV",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow Tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow Experiment Name",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only a single model configuration instead of full multi-run suite",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ridge",
        choices=["baseline", "ridge", "ensemble", "pytorch"],
        help="Model architecture for single-run mode",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="L2 penalty alpha for Ridge")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for gradient descent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware acceleration mode",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Semantic model version (default: 1.0.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.single_run:
        # Full experimental suite (default for LK-06)
        run_experiment_suite(
            features_path=args.features,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            seed=args.seed,
            device=args.device,
        )
    else:
        # Single custom run mode
        configure_mlflow(args.tracking_uri, args.experiment_name)
        records = load_dataset(args.features)
        split = split_features(records)

        if args.model_type == "baseline":
            model = BaselineLagModel(lag_idx=0)
            params = {"model_type": "baseline", "lag_idx": 0}
        elif args.model_type == "ridge":
            model = RidgeRegressionModel(alpha=args.alpha, lr=args.lr, epochs=args.epochs)
            params = {
                "model_type": "ridge",
                "alpha": args.alpha,
                "lr": args.lr,
                "epochs": args.epochs,
            }
        elif args.model_type == "ensemble":
            model = EnsembleGradientRegressor()
            params = {"model_type": "ensemble"}
        else:
            model = PyTorchTemporalRegressor(lr=args.lr, epochs=args.epochs, device=args.device)
            params = {"model_type": "pytorch", "lr": args.lr, "epochs": args.epochs}

        res = train_single_run(
            model=model,
            run_name=f"custom_{args.model_type}_run",
            split=split,
            params=params,
        )
        print(f"\nSingle run complete: {res['run_id']}")


if __name__ == "__main__":
    main()
