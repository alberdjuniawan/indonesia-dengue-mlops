from __future__ import annotations

import argparse
import json
from pathlib import Path

from dengue_mlops.features.pipeline import build_and_save_features
from dengue_mlops.modeling.dataset import load_dataset, split_features
from dengue_mlops.modeling.registry import load_champion
from dengue_mlops.modeling.trainer import run_training_pipeline
from dengue_mlops.monitoring.drift import DriftDetector


def check_and_retrain(force: bool = False, psi_threshold: float = 0.20) -> dict:
    print("Checking model health and data drift...")
    features_path = Path("data/features/features.csv")

    try:
        champion_model, metadata = load_champion()
        records = load_dataset(features_path)
        split = split_features(records, feature_names=metadata["features"])

        detector = DriftDetector(psi_threshold=psi_threshold)
        drift_report = detector.compute_feature_drift(
            reference_data=split.train_x,
            current_data=split.test_x,
            feature_names=split.feature_names,
        )
        print(
            f"Drift check completed. Max PSI: {drift_report['max_psi']} (Threshold: {psi_threshold})"
        )
        needs_retrain = force or drift_report["drift_detected"]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Notice: Could not compare against previous model ({e}). Proceeding to retrain.")
        needs_retrain = True
        drift_report = {"drift_detected": True, "reason": str(e)}

    if not needs_retrain:
        print("Model is stable. No retraining required.")
        return {"action": "NOOP", "reason": "No drift detected"}

    print("\nDrift detected or force flag supplied. Triggering retraining workflow...")
    # 1. Update features
    build_and_save_features()

    # 2. Retrain model with incremented version
    current_ver = metadata.get("version", "1.0.0") if "metadata" in locals() else "1.0.0"
    parts = current_ver.split(".")
    new_ver = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else "1.1.0"

    training_report = run_training_pipeline(features_path=features_path, model_version=new_ver)
    print(f"\nModel maintenance complete. Promoted version {new_ver} as new Champion.")

    result = {
        "action": "RETRAINED",
        "previous_version": current_ver,
        "new_version": new_ver,
        "drift_report": drift_report,
        "training_report": training_report,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger model maintenance / retraining")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    parser.add_argument("--threshold", type=float, default=0.20, help="PSI drift threshold")
    args = parser.parse_args()

    result = check_and_retrain(force=args.force, psi_threshold=args.threshold)
    print("\nResult:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
