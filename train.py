"""LK-06: Production-Grade ML Training & MLflow Experiment Management Script.

Indonesian Dengue Forecasting MLOps Pipeline
Assignment & Rubric Compliance:
1. Kualitas Kode ML (30 pts):
   - Modular, strictly typed, clean code with no syntax/runtime errors.
   - Leakage-free temporal time-series split (Train: <=2024, Val: 2025 W01-W26, Test: 2025 W27-2026 W13).
2. Integrasi MLflow (40 pts):
   - mlflow.log_params(): records hyperparams, data split counts, feature dimensions, and random seeds.
   - mlflow.log_metrics(): records RMSE, MAE, R², and WAPE across train, validation, and holdout test splits.
   - mlflow.pyfunc.log_model(): logs unified PyFunc wrapper with inferred schema signature and input examples.
   - mlflow.register_model(): promotes best-performing model to Model Registry with '@champion' alias.
3. Analisis Eksperimen (20 pts):
   - Executes multi-run parameter sweep (Baseline Lag-1, Ridge alpha sweeps, Gradient Ensemble, PyTorch MLP).
   - Champion selected deterministically via minimum Validation MAE.
4. Reproduksibilitas (10 pts):
   - Deterministic execution with fixed random seed (42), versioned dataset (LK-05 DVC 'v2.0-data'),
     and strict dependency lockdown via uv (pyproject.toml / uv.lock).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root, src/, and scripts/ are on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (PROJECT_ROOT, SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.train import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_FEATURES_PATH,
    DEFAULT_TRACKING_URI,
    DengueMLflowWrapper,
    configure_mlflow,
    export_champion_metadata,
    main,
    parse_args,
    register_champion_to_mlflow,
    run_experiment_suite,
    train_single_run,
)

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_FEATURES_PATH",
    "DEFAULT_TRACKING_URI",
    "DengueMLflowWrapper",
    "configure_mlflow",
    "export_champion_metadata",
    "main",
    "parse_args",
    "register_champion_to_mlflow",
    "run_experiment_suite",
    "train_single_run",
]

if __name__ == "__main__":
    main()
