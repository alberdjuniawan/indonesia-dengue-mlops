from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dengue_mlops.modeling.dataset import load_dataset, split_features
from dengue_mlops.modeling.metrics import calculate_metrics
from dengue_mlops.modeling.registry import (
    DEFAULT_MAX_MAE,
    DEFAULT_MAX_RMSE,
    DEFAULT_MAX_WAPE,
    DEFAULT_MIN_R2,
    load_champion,
    promote_to_staging,
    validate_model_metrics,
)

DEFAULT_EVALUATION_REPORT = Path("models/evaluation_report.json")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_MODEL_NAME = "DengueOutbreakForecaster"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate champion dengue model against SLA performance thresholds (LK-08)"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/features.csv"),
        help="Path to features CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVALUATION_REPORT,
        help="Path to output evaluation report JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate model metrics against SLA performance thresholds (default: True)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Disable SLA threshold validation check",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=DEFAULT_MIN_R2,
        help=f"Minimum acceptable R2 score (default: {DEFAULT_MIN_R2})",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=DEFAULT_MAX_MAE,
        help=f"Maximum acceptable MAE (default: {DEFAULT_MAX_MAE})",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=DEFAULT_MAX_RMSE,
        help=f"Maximum acceptable RMSE (default: {DEFAULT_MAX_RMSE})",
    )
    parser.add_argument(
        "--max-wape",
        type=float,
        default=DEFAULT_MAX_WAPE,
        help=f"Maximum acceptable WAPE percentage (default: {DEFAULT_MAX_WAPE}%%)",
    )
    parser.add_argument(
        "--promote-staging",
        action="store_true",
        default=False,
        help="Automatically promote validated model to MLflow Model Registry 'Staging' stage",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow Tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"MLflow Registered Model Name (default: {DEFAULT_MODEL_NAME})",
    )
    args = parser.parse_args()

    model, metadata = load_champion()
    print(f"Loaded Champion Model: {metadata['model_name']} (v{metadata['version']})")

    records = load_dataset(args.features)
    split = split_features(records, feature_names=metadata["features"])

    test_preds = model.predict(split.test_x)
    metrics = calculate_metrics(split.test_y, test_preds)

    # Per-province error analysis
    by_province: dict[str, dict[str, list[float]]] = {}
    for prov, t, p in zip(split.provinces_test, split.test_y, test_preds, strict=False):
        by_province.setdefault(prov, {"y_true": [], "y_pred": []})
        by_province[prov]["y_true"].append(t)
        by_province[prov]["y_pred"].append(p)

    province_metrics = {
        prov: calculate_metrics(data["y_true"], data["y_pred"])
        for prov, data in by_province.items()
    }

    report: dict[str, object] = {
        "model_name": metadata["model_name"],
        "version": metadata["version"],
        "test_sample_count": len(split.test_x),
        "overall_metrics": metrics,
        "province_metrics": province_metrics,
    }

    print("\n" + "=" * 65)
    print("MODEL EVALUATION SUMMARY (HOLDOUT TEST SPLIT)")
    print("=" * 65)
    print(f"Overall RMSE : {metrics['rmse']:.3f}")
    print(f"Overall MAE  : {metrics['mae']:.3f}")
    print(f"Overall R2   : {metrics['r2']:.4f}")
    print(f"Overall WAPE : {metrics['wape']:.2f}%")
    print("=" * 65)

    validation_result = None
    if args.validate:
        validation_result = validate_model_metrics(
            metrics=metrics,
            min_r2=args.min_r2,
            max_mae=args.max_mae,
            max_rmse=args.max_rmse,
            max_wape=args.max_wape,
        )
        report["quality_gate"] = validation_result

        print("\n" + "=" * 65)
        print("AUTOMATED QUALITY GATE VALIDATION (LK-08)")
        print("=" * 65)
        print(f"{'Metric':<12} {'Actual':<10} {'Threshold':<12} {'Status':<10}")
        print("-" * 65)
        for metric_name, check in validation_result["checks"].items():
            op = check["operator"]
            th = check["threshold"]
            act = check["actual"]
            status = "PASSED" if check["passed"] else "FAILED"
            print(f"{metric_name.upper():<12} {act:<10} {op} {th:<10} {status:<10}")
        print("-" * 65)
        print(f"Overall Quality Gate: {validation_result['status']}")
        print("=" * 65)

        if not validation_result["passed"]:
            print("\n❌ Quality Gate Validation FAILED:")
            for reason in validation_result["failure_reasons"]:
                print(f"  - {reason}")

    # Tahap 4: Auto-Registry Promotion to Staging
    if args.promote_staging:
        if args.validate and validation_result and not validation_result["passed"]:
            print("\n❌ Quality gate validation failed. Staging promotion ABORTED.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            sys.exit(1)

        print("\n" + "=" * 65)
        print("MLFLOW MODEL REGISTRY PROMOTION TO STAGING (TAHAP 4)")
        print("=" * 65)

        # Determine target version: read champion_metadata if exists or fallback to metadata version
        target_version = metadata.get("version", "1")
        champion_meta_path = Path("models/registry/champion_metadata.json")
        if champion_meta_path.exists():
            try:
                cm = json.loads(champion_meta_path.read_text(encoding="utf-8"))
                target_version = cm.get("mlflow_model_version", target_version)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                target_version = metadata.get("version", "1")

        try:
            promotion_result = promote_to_staging(
                model_name=args.model_name,
                version=target_version,
                tracking_uri=args.tracking_uri,
                metrics=metrics,
                reason=(
                    f"Automated Quality Gate PASSED: R2={metrics['r2']:.4f} >= {args.min_r2}, "
                    f"MAE={metrics['mae']:.2f} <= {args.max_mae}. Promoted to Staging via CI/CD (LK-08)."
                ),
            )
            report["registry_promotion"] = promotion_result
            print(f"Model Name       : {promotion_result['model_name']}")
            print(f"Promoted Version : v{promotion_result['promoted_version']}")
            print(f"Previous Stage   : {promotion_result['previous_stage']}")
            print(f"Target Stage     : {promotion_result['target_stage']}")
            print(f"Registry Alias   : @{promotion_result['alias']}")
            print(f"Promoted By      : {promotion_result['actor']}")
            print("=" * 65)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Error promoting model to MLflow Model Registry: {e}")
            report["registry_promotion_error"] = str(e)
            if args.validate and validation_result and validation_result["passed"]:
                print("Notice: Proceeding with evaluation report output despite registry error.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nEvaluation report saved to: {args.output}")

    if args.validate and validation_result and not validation_result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
