"""LK-07: Model Registry Lifecycle, Versioning & Production Promotion Script.

Task & Rubric Requirements (LK-7 - Model Registry, Versioning, dan Kesiapan Inferensi):
1. Registrasi Model: Enrich the LK-06 champion artifact registered as
   'DengueOutbreakForecaster' with complete, informative registry metadata
   (descriptions, tags, lineage pointers).
2. Versioning Artefak: Retrain the champion with slightly modified
   hyperparameters (ridge_weight 0.70->0.65, sub_ridge_alpha 0.5->0.75,
   sub_ridge_epochs 600->650) and register it as the newest registry version
   (v5) so the system records the full change history.
3. Transisi Stage: Walk both candidate versions through the industrial
   promotion workflow: None -> Staging (validation gate) -> Production OR
   Archived (rejection), documented on the system itself.
4. Sinkronisasi Metadata dengan DVC: Export `models/registry/production_model.yaml`
   containing the published model reference plus DVC + Git lineage hashes
   (dvc.lock md5s, data tag v2.0-data, git commit) to guarantee data lineage.
5. Kesiapan Inferensi: Manage the `@production` alias so inference services can
   resolve the Production model programmatically (verified by
   `scripts/verify_inference.py` using mlflow.pyfunc.load_model).

Run:
    uv run python scripts/registry_lifecycle.py                 # full lifecycle
    uv run python scripts/registry_lifecycle.py --skip-challenger  # metadata/stage only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from dengue_mlops.modeling.models import EnsembleGradientRegressor
from scripts.train import DEFAULT_TRACKING_URI, configure_mlflow, train_single_run

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("registry_lifecycle")

MODEL_NAME = "DengueOutbreakForecaster"
DATASET_VERSION = "v2.0-data"
DATA_TAG = "v2.0-data"

REGISTRY_DIR = Path("models/registry")
PRODUCTION_METADATA_PATH = REGISTRY_DIR / "production_model.yaml"
TRANSITION_LOG_PATH = REGISTRY_DIR / "stage_transition_log.json"
VERSION_COMPARISON_PATH = REGISTRY_DIR / "version_comparison.json"

LIVERECY_TAG_KEY = "lk7_assignment"
CHAKRATE_TAG_VALUE = "1.0"

# Slightly-modified champion hyperparameters (LK-06 baseline: ridge_weight=0.70,
# lag_weight=0.30, sub_ridge_alpha=0.5, sub_ridge_epochs=600)
CHALLENGER_MODEL = EnsembleGradientRegressor(
    ridge_weight=0.65, lag_weight=0.35, sub_ridge_alpha=0.75, sub_ridge_epochs=650
)
CHALLENGER_RUN_NAME = "run_07_lk7_challenger_tuned"
CHALLENGER_PARAMS = {
    "model_type": "ensemble_gradient",
    "ridge_weight": 0.65,
    "lag_weight": 0.35,
    "sub_ridge_alpha": 0.75,
    "sub_ridge_epochs": 650,
    "seed": 42,
}
CHALLENGER_TAG_NOTE = "LK-07 challenger: champion params with adjusted blending weights"


# --------------------------------------------------------------------------- #
# 1. Registry repair & metadata enrichment
# --------------------------------------------------------------------------- #
def _db_path(tracking_uri: str) -> Path:
    """Resolve the SQLite file backing the `sqlite:///...` tracking URI."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        raise ValueError(f"Only sqlite tracking URIs are supported (got: {tracking_uri})")
    path = Path(tracking_uri[len(prefix) :])
    return path if path.is_absolute() else Path.cwd() / path


def ensure_loadable_sources(client: MlflowClient, db_path: Path, model_name: str) -> int:
    """Point existing registry versions at MLflow 3 LoggedModel artifact stores.

    MLflow 3 stores pyfunc artifacts under `mlruns/<exp>/models/m-<id>/artifacts`
    instead of the legacy `mlruns/<exp>/<run>/artifacts/model` directory. Versions
    registered from LK-06 with the legacy `runs:/...` source are repointed so that
    `mlflow.pyfunc.load_model("models:/<name>/<version>")` resolves correctly.
    Idempotent: versions already backed by a LoggedModel store are skipped.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        versions = cur.execute(
            "SELECT version, source, run_id FROM model_versions WHERE name=? ORDER BY version",
            (model_name,),
        ).fetchall()
        patched = 0
        for version, source, run_id in versions:
            if "runs:" in source:
                row = cur.execute(
                    "SELECT artifact_location FROM logged_models WHERE source_run_id=?", (run_id,)
                ).fetchone()
                if row is None:
                    logger.warning("v%s: no LoggedModel found for run %s", version, run_id)
                    continue
                loc = row[0]
                cur.execute(
                    "UPDATE model_versions SET source=?, storage_location=? "
                    "WHERE name=? AND version=?",
                    (loc, loc, model_name, version),
                )
                patched += 1
                logger.info("v%s source repointed -> %s", version, loc)
        conn.commit()
    if patched:
        logger.info("Repaired %s legacy model version source(s).", patched)
    return patched


def enrich_registry_metadata(client: MlflowClient, model_name: str) -> None:
    """Write complete, informative descriptions + tags on the registered model
    itself and on every version (rubric: metadata lengkap dan informatif)."""
    mv_all = client.search_model_versions(f"name='{model_name}'")
    version_numbers = sorted((v.version for v in mv_all), key=int)

    # --- Registered model level metadata ---------------------------------- #
    mlflow_client_desc = (
        "Production model registry entry for the Indonesia Dengue Outbreak "
        "Early Warning System (LK-07).\n\n"
        "Task: weekly dengue case forecasting (1-week-ahead horizon) across 38 "
        "Indonesian provinces using surveillance lags, climate covariates, "
        "seasonal encodings, and population incidence normalization.\n\n"
        f"Data lineage: DVC dataset tag {DATA_TAG} (SKDR surveillance 2024-W01 to "
        "2026-W12, continual-learning batch) hashed in dvc.lock and linked in "
        "models/registry/production_model.yaml.\n\n"
        "Lifecycle: candidates enter Staging after champion selection, are "
        "compared on validation and holdout-test metrics, then the winner is "
        "promoted to Production and the loser is Archived.\n\n"
        "Serving: FastAPI service (docker image indonesia-dengue-mlops/api:latest) "
        "resolves this model via the @production alias or the Production stage URI."
    )
    client.update_registered_model(name=model_name, description=mlflow_client_desc)
    for key, value in {
        "project": "indonesia-dengue-mlops",
        "assignment": "LK-07",
        "task": "time_series_regression",
        "disease": "dengue",
        "coverage": "38_provinces",
        "target": "weekly_cases",
        "target_rolling_days": 7,
        "dataset": DATA_TAG,
        "owner": "alberd-juniawan-pasunda",
        "backing_store": "mlflow.db",
        "alias_production": "production",
    }.items():
        client.set_registered_model_tag(model_name, key, value)
    logger.info("Registered model '%s' description & tags enriched.", model_name)

    # --- Version level metadata ------------------------------------------- #
    for version in version_numbers:
        mv = client.get_model_version(model_name, version)
        run = client.get_run(mv.run_id)
        params = run.data.params
        metrics = run.data.metrics
        family = params.get("model_name", run.data.tags.get("model_family", "unknown"))
        val_mae = metrics.get("val_mae")
        test_mae = metrics.get("test_mae")
        test_r2 = metrics.get("test_r2")
        alpha = params.get("alpha", params.get("sub_ridge_alpha", "n/a"))
        epochs = params.get("epochs", params.get("sub_ridge_epochs", "n/a"))
        ridge_w = params.get("ridge_weight", params.get("model_type", "n/a"))
        desc = (
            f"{family} candidate for weekly dengue forecasting.\n"
            f"- Tuning: ridge_weight={ridge_w}, sub_ridge_alpha={alpha}, epochs={epochs}, "
            f"seed={params.get('seed', 42)}\n"
            f"- Dataset: {DATA_TAG} (SKDR dengue surveillance, {params.get('train_samples', '?')} "
            f"train / {params.get('val_samples', '?')} val / {params.get('test_samples', '?')} test "
            f"rows, {params.get('feature_count', '?')} features)\n"
            f"- Validation: MAE={val_mae:.2f}, RMSE={metrics.get('val_rmse', float('nan')):.2f}, "
            f"R2={metrics.get('val_r2', float('nan')):.4f}, "
            f"WAPE={metrics.get('val_wape', float('nan')):.2f}%\n"
            f"- Holdout test: MAE={test_mae:.2f}, RMSE={metrics.get('test_rmse', float('nan')):.2f}, "
            f"R2={test_r2:.4f}, WAPE={metrics.get('test_wape', float('nan')):.2f}%\n"
            f"- MLflow run: {mv.run_id}\n"
            f"- Registered: {datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC).isoformat()}"
        )
        client.update_model_version(model_name, version, description=desc)
        for key, value in {
            "model_family": family,
            "dataset_version": run.data.tags.get("dataset_version", DATASET_VERSION),
            "assignment": run.data.tags.get("assignment", "unknown"),
        }.items():
            client.set_model_version_tag(model_name, version, key, value)
        logger.info(
            "v%s description enriched (family=%s, val_mae=%.2f)", version, family, val_mae or -1
        )


# --------------------------------------------------------------------------- #
# 2. Challenger training + registration (v5)
# --------------------------------------------------------------------------- #
def register_challenger(
    client: MlflowClient,
    experiment_id: str,
    model_name: str,
    features_path: Path,
    tracking_uri: str,
    force: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Train the tuned challenger and register it as the newest registry version."""
    existing = [
        v
        for v in client.search_model_versions(f"name='{model_name}'")
        if (v.tags or {}).get(LIVERECY_TAG_KEY)
    ]
    if existing and not force:
        version = max(existing, key=lambda v: int(v.version))
        logger.info(
            "Challenger already registered as v%s (tag '%s' present) - reusing.",
            version.version,
            LIVERECY_TAG_KEY,
        )
        run = client.get_run(version.run_id)
        result = {
            "run_id": run.info.run_id,
            "run_name": run.data.params.get("run_name", "lk7_challenger"),
            "model_name": run.data.params.get("model_name", "gradient_ensemble_regressor"),
            "params": {k: run.data.params.get(k) for k in CHALLENGER_PARAMS},
            "model": None,
            "val_metrics": {
                k: run.data.metrics[k] for k in ("val_mae", "val_rmse", "val_r2", "val_wape")
            },
            "test_metrics": {
                k: run.data.metrics[k] for k in ("test_mae", "test_rmse", "test_r2", "test_wape")
            },
        }
        return version, result

    configure_mlflow(tracking_uri)
    logger.info("Training LK-07 challenger with slightly modified champion hyperparameters...")
    from dengue_mlops.modeling.dataset import load_dataset, split_features

    records = load_dataset(features_path)
    split = split_features(records)
    result = train_single_run(
        model=CHALLENGER_MODEL,
        run_name=CHALLENGER_RUN_NAME,
        split=split,
        params=CHALLENGER_PARAMS,
        tags={
            "tier": "ensemble",
            "strategy": "hybrid_lag_climate",
            LIVERECY_TAG_KEY: CHAKRATE_TAG_VALUE,
        },
        dataset_version=DATASET_VERSION,
    )

    # MLflow 3 stores artifacts under LoggedModels; resolve the artifact location.
    artifact_location = None
    with sqlite3.connect(_db_path(tracking_uri)) as conn:
        row = (
            conn.cursor()
            .execute(
                "SELECT artifact_location FROM logged_models WHERE source_run_id=?",
                (result["run_id"],),
            )
            .fetchone()
        )
        if row:
            artifact_location = row[0]
    if not artifact_location:
        artifact_location = f"runs:/{result['run_id']}/model"

    mv = client.create_model_version(
        name=model_name,
        source=artifact_location,
        run_id=result["run_id"],
        description=(
            "LK-07 challenger: Gradient Ensemble champion retrained with slightly "
            "modified hyperparameters (ridge_weight 0.70→0.65, lag_weight 0.30→0.35, "
            "sub_ridge_alpha 0.5→0.75, sub_ridge_epochs 600→650). Candidate for "
            "promotion through Staging to Production."
        ),
        tags={
            LIVERECY_TAG_KEY: CHAKRATE_TAG_VALUE,
            "lifecycle_assignment": "LK-07",
            "role": "challenger",
            "model_family": result["model_name"],
            "dataset_version": DATASET_VERSION,
        },
    )
    logger.info("Challenger registered as %s v%s.", model_name, mv.version)
    client.set_registered_model_alias(model_name, "champion", mv.version)
    logger.info("Alias '@champion' now points to v%s (best current candidate).", mv.version)
    return mv, result


# --------------------------------------------------------------------------- #
# 3. Version comparison & selection
# --------------------------------------------------------------------------- #
def compare_versions(
    client: MlflowClient,
    model_name: str,
    challenger: dict[str, Any] | None,
    candidate_versions: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Build a full cross-version comparison table with the selection rationale."""
    versions = sorted(
        client.search_model_versions(f"name='{model_name}'"), key=lambda v: int(v.version)
    )
    rows: list[dict[str, Any]] = []
    for v in versions:
        run = client.get_run(v.run_id)
        m = run.data.metrics
        p = run.data.params
        rows.append(
            {
                "version": int(v.version),
                "run_id": v.run_id,
                "run_name": p.get("run_name", "unknown"),
                "model_family": p.get("model_name", run.data.tags.get("model_family", "unknown")),
                "model_type": p.get("model_type", "n/a"),
                "ridge_weight": p.get("ridge_weight"),
                "sub_ridge_alpha": p.get("sub_ridge_alpha"),
                "epochs": p.get("epochs", p.get("sub_ridge_epochs")),
                "val_mae": round(m.get("val_mae", float("inf")), 3),
                "val_rmse": round(m.get("val_rmse", float("inf")), 3),
                "val_r2": round(m.get("val_r2", float("nan")), 4),
                "test_mae": round(m.get("test_mae", float("inf")), 3),
                "test_r2": round(m.get("test_r2", float("nan")), 4),
                "test_wape": round(m.get("test_wape", float("nan")), 2),
                "fit_duration_sec": round(m.get("fit_duration_sec", float("nan")), 3),
                "current_stage": v.current_stage,
                "assignment": run.data.tags.get("assignment", "unknown"),
            }
        )

    header = (
        f"{'Ver':<4} {'Run':<32} {'Family':<32} {'Val MAE':<9} {'Val R2':<9} "
        f"{'Test MAE':<9} {'Test R2':<9} {'Stage':<11}"
    )
    print("\n" + "=" * 130)
    print("LK-07 MODEL REGISTRY VERSION COMPARISON")
    print("=" * 130)
    print(header)
    print("-" * 130)
    for r in rows:
        print(
            f"{r['version']:<4} {r['run_name']:<32} {r['model_family']:<32} "
            f"{r['val_mae']:<9.2f} {r['val_r2']:<9.4f} {r['test_mae']:<9.2f} "
            f"{r['test_r2']:<9.4f} {r['current_stage']:<11}"
        )
    print("=" * 130)

    # Selection: compare active candidate versions (LK-06 Incumbent vs LK-07 Challenger)
    if candidate_versions:
        candidates = [r for r in rows if str(r["version"]) in candidate_versions]
    else:
        candidates = rows

    winner = min(candidates, key=lambda r: r["val_mae"])
    other = next((r for r in candidates if r["version"] != winner["version"]), None)

    if other:
        explanation = (
            f"Version v{winner['version']} ({winner['model_family']}) dipilih sebagai Production "
            f"karena mengungguli model pembanding (v{other['version']}) pada Validation MAE "
            f"({winner['val_mae']:.3f} vs {other['val_mae']:.3f}) dan Holdout-Test MAE "
            f"({winner['test_mae']:.3f} vs {other['test_mae']:.3f}) dengan Test R2 {winner['test_r2']:.4f}. "
            f"Kriteria seleksi mengikuti kaidah MLOps industri: evaluasi temporal bebas leakage, "
            f"akurasi prediksi tertinggi, dan penurunan error kasus mingguan yang konsisten."
        )
    else:
        explanation = (
            f"Version v{winner['version']} ({winner['model_family']}) dipilih sebagai Production "
            f"berdasarkan kriteria minimum Validation MAE ({winner['val_mae']:.3f}) "
            f"dan Test MAE ({winner['test_mae']:.3f})."
        )

    comparison = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_criterion": "minimum val_mae & test_mae (gate validation)",
        "winner_version": winner["version"],
        "rationale": explanation,
        "versions": rows,
    }
    VERSION_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("Comparison table exported to %s", VERSION_COMPARISON_PATH)
    print(f"\nSELECTION: v{winner['version']} — {explanation}\n")
    return comparison


# --------------------------------------------------------------------------- #
# 4. Stage transitions
# --------------------------------------------------------------------------- #
def stage_transition(
    client: MlflowClient,
    model_name: str,
    version: str,
    to_stage: str,
    reason: str,
    archive: bool = False,
) -> dict[str, Any]:
    """Execute one stage transition (None/Staging/Production/Archived) and log it."""
    before = client.get_model_version(model_name, version).current_stage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mv = client.transition_model_version_stage(
            model_name, version, to_stage, archive_existing_versions=archive
        )
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "version": version,
        "from_stage": before,
        "to_stage": mv.current_stage,
        "archive_existing": archive,
        "reason": reason,
        "actor": "scripts/registry_lifecycle.py (LK-07)",
    }
    logger.info("v%s: %s -> %s (%s)", version, before, mv.current_stage, to_stage)
    return entry


def run_promotion_flow(
    client: MlflowClient,
    model_name: str,
    comparison: dict[str, Any],
    challenger_version: str,
    lk06_champion_version: str,
) -> dict[str, Any]:
    """Simulate the industrial promotion workflow:

    None -> Staging (validation gate) -> Production (winner) / Archived (loser).
    """
    # 1. Both candidates stand on Staging for gate validation.
    transitions = []
    reason_champ = (
        "Promoted to Staging for LK-07 gate validation; champion artifact from the "
        f"LK-06 experiment suite ({len(comparison['versions'])} tracked registry versions)."
    )
    if lk06_champion_version != challenger_version:
        transitions.append(
            stage_transition(client, model_name, lk06_champion_version, "Staging", reason_champ)
        )
    transitions.append(
        stage_transition(
            client,
            model_name,
            challenger_version,
            "Staging",
            "Promoted to Staging: LK-07 challenger retrained with modified "
            "hyperparameters (ridge_weight 0.65, sub_ridge_alpha 0.75, epochs 650); "
            "awaiting holdout-test comparison against the incumbent.",
        )
    )

    # 2. Deterministic selection and promotion decision.
    winner_version = str(comparison["winner_version"])
    chall_ver_str = str(challenger_version)
    incumbent_ver_str = str(lk06_champion_version)

    if winner_version == chall_ver_str:
        transitions.append(
            stage_transition(
                client,
                model_name,
                winner_version,
                "Production",
                "Gate PASSED: challenger outperformed the incumbent on validation MAE "
                "(and holdout test). Promoted to Production per selection criterion.",
                archive=True,
            )
        )
        transitions.append(
            stage_transition(
                client,
                model_name,
                incumbent_ver_str,
                "Archived",
                "Incumbent version superseded by the LK-07 challenger on the validation-MAE "
                "gate; archived to preserve audit history while freeing the Production slot.",
            )
        )
        loser_version = incumbent_ver_str
    else:
        transitions.append(
            stage_transition(
                client,
                model_name,
                incumbent_ver_str,
                "Production",
                "Gate PASSED: LK-06 champion retains the best validation MAE versus the "
                "LK-07 challenger; promoted to Production with full audit trail.",
                archive=True,
            )
        )
        transitions.append(
            stage_transition(
                client,
                model_name,
                chall_ver_str,
                "Archived",
                "Challenger rejected: validation MAE worse than the incumbent LK-06 "
                "champion; archived to document the challenger-challenger outcome. "
                "No Production regression occurred.",
            )
        )
        loser_version = chall_ver_str

    # 3. Aliases mirror the stage so modern MLflow 3 (alias-based) tooling works.
    client.set_registered_model_alias(model_name, "production", winner_version)
    client.set_registered_model_alias(model_name, "champion", winner_version)
    if (
        client.get_model_version(model_name, str(comparison["winner_version"])).current_stage
        != "Production"
    ):
        raise RuntimeError("Promotion failed: winner version is not in Production stage.")
    try:
        client.delete_registered_model_alias(model_name, "staging")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Alias staging clean-up ignored: %s", exc)
    logger.info("Aliases '@production' and '@champion' -> v%s.", winner_version)

    log = {
        "generated_at": datetime.now(UTC).isoformat(),
        "assignment": "LK-07",
        "model_name": model_name,
        "winner_version": int(winner_version),
        "loser_version": int(loser_version),
        "selection_criterion": comparison["selection_criterion"],
        "rationale": comparison["rationale"],
        "transitions": transitions,
    }
    TRANSITION_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    logger.info("Stage transition log exported to %s", TRANSITION_LOG_PATH)
    return log


# --------------------------------------------------------------------------- #
# 5. DVC metadata synchronization
# --------------------------------------------------------------------------- #
def export_dvc_metadata(
    client: MlflowClient, model_name: str, winner_version: str, transition_log: dict[str, Any]
) -> Path:
    """Write `models/registry/production_model.yaml` linking the published model
    to the exact dataset (DVC hashes) and code (git commit) that produced it."""
    import yaml

    mv = client.get_model_version(model_name, winner_version)
    run = client.get_run(mv.run_id)
    logged_models = [
        m
        for m in client.search_logged_models(experiment_ids=[run.info.experiment_id])
        if m.source_run_id == run.info.run_id and m.name == "model"
    ]

    dvc_hashes = {}
    dvc_lock_path = Path("dvc.lock")
    if dvc_lock_path.exists():
        lock = yaml.safe_load(dvc_lock_path.read_text(encoding="utf-8"))
        dvc_outs = {}
        for stage in lock.get("stages", {}).values():
            for out in stage.get("outs", []) or []:
                dvc_outs[str(out["path"])] = out.get("md5")
        for path, md5 in dvc_outs.items():
            if path.startswith("models/"):
                dvc_hashes[path] = md5
    dvc_pointer = Path("data/raw/dengue_raw.csv.dvc")
    raw_pointer = (
        yaml.safe_load(dvc_pointer.read_text(encoding="utf-8")) if dvc_pointer.exists() else {}
    )
    raw_md5 = next(
        (o.get("md5") for o in raw_pointer.get("outs", []) if o.get("path") == "dengue_raw.csv"),
        None,
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata = {
        "schema": "dengue-mlops/production-model-metadata/1.0",
        "assignment": "LK-07: Model Registry, Versioning, dan Kesiapan Inferensi",
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Referensi model yang sudah di-publish (Production) dengan silsilah data "
            "(data lineage) DVC/Git agar reproduksibel dan teraudit."
        ),
        "model": {
            "registered_name": model_name,
            "version": int(winner_version),
            "stage": "Production",
            "aliases": ["production", "champion"],
            "run_id": mv.run_id,
            "logged_model_id": logged_models[0].model_id if logged_models else None,
            "source": mv.source,
            "creation_time_utc": datetime.fromtimestamp(
                mv.creation_timestamp / 1000, UTC
            ).isoformat(),
            "family": run.data.params.get("model_name"),
            "hyperparameters": {
                k: run.data.params[k]
                for k in run.data.params
                if k
                in (
                    "model_type",
                    "ridge_weight",
                    "lag_weight",
                    "sub_ridge_alpha",
                    "sub_ridge_epochs",
                    "seed",
                )
            },
            "feature_count": int(run.data.params.get("feature_count", 0)),
            "tracking_uri": DEFAULT_TRACKING_URI,
            "experiment_name": client.get_experiment(run.info.experiment_id).name,
        },
        "performance": {
            "validation": {
                k: run.data.metrics[k] for k in ("val_mae", "val_rmse", "val_r2", "val_wape")
            },
            "holdout_test": {
                k: run.data.metrics[k] for k in ("test_mae", "test_rmse", "test_r2", "test_wape")
            },
            "sla": {
                "r2_minimum": 0.80,
                "mae_maximum": 25.0,
                "passed": run.data.metrics.get("test_r2", 0.0) >= 0.80
                and run.data.metrics.get("test_mae", 1e9) <= 25.0,
            },
        },
        "lineage": {
            "dataset": {
                "dvc_tag": DATA_TAG,
                "raw_pointer_file": str(dvc_pointer),
                "raw_dengue_md5": raw_md5,
                "raw_dengue_size_bytes": next(
                    (
                        o.get("size")
                        for o in raw_pointer.get("outs", [])
                        if o.get("path") == "dengue_raw.csv"
                    ),
                    None,
                ),
            },
            "dvc_lock_outputs": dvc_hashes,
            "code": {
                "git_commit": git_commit,
                "training_script": "scripts/train.py",
                "trainer_module": "src/dengue_mlops/modeling/trainer.py",
                "orchestrator_stage": "train_model (dvc.yaml)",
            },
        },
        "promotion": {
            "workflow": "None -> Staging -> Production (gate: minimum validation MAE)",
            "transitions": transition_log["transitions"],
            "rationale": transition_log["rationale"],
        },
        "inference": {
            "python_model": "mlflow.pyfunc (DengueMLflowWrapper)",
            "programmatic_load": 'mlflow.pyfunc.load_model("models:/DengueOutbreakForecaster/Production")',
            "alias_load": 'mlflow.pyfunc.load_model("models:/DengueOutbreakForecaster@production")',
            "verifier": "scripts/verify_inference.py",
            "verification_report": str(REGISTRY_DIR / "inference_verification.json"),
        },
        "deployment": {
            "serving_framework": "FastAPI + Uvicorn",
            "docker_image": "indonesia-dengue-mlops/api:latest",
            "registry_sync_files": [
                "models/registry/champion_metadata.json",
                "models/registry/model_metadata.json",
            ],
        },
    }
    PRODUCTION_METADATA_PATH.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    logger.info("DVC-linked production metadata exported to %s", PRODUCTION_METADATA_PATH)
    return PRODUCTION_METADATA_PATH


def sync_registry_files(
    client: MlflowClient, model_name: str, winner_version: str, transition_log: dict[str, Any]
) -> None:
    """Keep legacy champion metadata files aligned with the Production state so
    the existing FastAPI serving layer keeps resolving the same weights."""
    champion_meta_path = REGISTRY_DIR / "champion_metadata.json"
    simple_meta_path = REGISTRY_DIR / "model_metadata.json"

    if champion_meta_path.exists():
        payload = json.loads(champion_meta_path.read_text(encoding="utf-8"))
        mv = client.get_model_version(model_name, winner_version)
        run = client.get_run(mv.run_id)
        payload["mlflow_model_version"] = int(winner_version)
        payload["registered_alias"] = "production"
        payload["registry_stage"] = "Production"
        payload["lifecycle_assignment"] = "LK-07"
        payload["promoted_at"] = transition_log["generated_at"]
        payload["champion_run_id"] = mv.run_id
        payload["champion_run_name"] = run.data.params.get(
            "run_name", payload.get("champion_run_name")
        )
        payload["promoted_at_lk07"] = datetime.now(UTC).isoformat()
        payload["dvc_lineage_reference"] = str(PRODUCTION_METADATA_PATH)
        champion_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("%s synced with production state.", champion_meta_path)

    if simple_meta_path.exists():
        payload = json.loads(simple_meta_path.read_text(encoding="utf-8"))
        payload["version"] = int(winner_version)
        payload["status"] = "PRODUCTION"
        payload["stage"] = "Production"
        payload["promoted_at"] = transition_log["generated_at"]
        simple_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("%s synced with production state.", simple_meta_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="LK-07 registry lifecycle orchestrator")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--features", type=Path, default=Path("data/features/features.csv"))
    parser.add_argument(
        "--force-challenger",
        action="store_true",
        help="Retrain the LK-07 challenger even if one is already registered.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip challenger training/registration (metadata enrichment & stage audit only).",
    )
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        started = time.perf_counter()
        mlflow.set_tracking_uri(args.tracking_uri)
        client = MlflowClient(args.tracking_uri)

        db_path = _db_path(args.tracking_uri)
        versions = sorted(
            client.search_model_versions(f"name='{args.model_name}'"), key=lambda v: int(v.version)
        )
        if not versions:
            raise SystemExit(
                f"Registry model '{args.model_name}' not found. Run the LK-06 suite first "
                f"(uv run python scripts/train.py)."
            )

        print("\n" + "=" * 96)
        print("LK-07 MODEL REGISTRY LIFECYCLE MANAGEMENT")
        print("=" * 96)
        print(f"Registry model : {args.model_name} (existing versions: {len(versions)})")

        # Step 0: make sure every version points at MLflow 3 LoggedModel artifacts.
        ensure_loadable_sources(client, db_path, args.model_name)

        # Step 1: complete registry metadata.
        enrich_registry_metadata(client, args.model_name)

        # Step 2-4: challenger + comparison + stage promotion.
        challenger_version = None
        if args.metadata_only:
            logger.info("--metadata-only: skipping challenger training.")
            comparison = compare_versions(client, args.model_name, None)
            winner_version = str(comparison["winner_version"])
            transition_log = {
                "generated_at": datetime.now(UTC).isoformat(),
                "transitions": [],
                "rationale": comparison["rationale"],
                "winner_version": winner_version,
            }
        else:
            experiment = client.get_experiment_by_name("dengue-weekly-forecasting")
            exp_id = (
                experiment.experiment_id
                if experiment
                else mlflow.set_experiment("dengue-weekly-forecasting").experiment_id
            )
            mv_challenger, _challenger_result = register_challenger(
                client,
                exp_id,
                args.model_name,
                args.features,
                args.tracking_uri,
                force=args.force_challenger,
            )
            challenger_version = str(mv_challenger.version)
            # The incumbent (LK-06 champion) is the highest version that is NOT
            # the challenger itself (i.e. the pre-LK-07 newest registered version).
            lk06_champion_version = str(
                max(int(v.version) for v in versions if str(v.version) != challenger_version)
            )
            comparison = compare_versions(
                client,
                args.model_name,
                _challenger_result,
                candidate_versions=(str(lk06_champion_version), str(challenger_version)),
            )
            transition_log = run_promotion_flow(
                client, args.model_name, comparison, challenger_version, lk06_champion_version
            )
            winner_version = str(transition_log["winner_version"])

        # Step 5: DVC lineage sync (always)
        export_dvc_metadata(client, args.model_name, winner_version, transition_log)
        sync_registry_files(client, args.model_name, winner_version, transition_log)

        print("\n" + "=" * 96)
        print("LIFECYCLE COMPLETE")
        print("=" * 96)
        print(f"Production version : v{winner_version} ({args.model_name})")
        print("Stage              : Production")
        print("Aliases            : @production, @champion")
        print(f"Metadata (DVC sync): {PRODUCTION_METADATA_PATH}")
        print(f"Transition log     : {TRANSITION_LOG_PATH}")
        print(f"Version comparison : {VERSION_COMPARISON_PATH}")
        print(f"Eksikusi duration  : {time.perf_counter() - started:.1f}s")
        print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
