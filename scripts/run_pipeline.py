from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dengue_mlops.features.pipeline import build_and_save_features
from dengue_mlops.modeling.dataset import load_dataset, split_features
from dengue_mlops.modeling.registry import load_champion
from dengue_mlops.modeling.trainer import run_training_pipeline
from dengue_mlops.monitoring.drift import DriftDetector


def run_integrated_pipeline(
    version: str = "1.0.0", force_data: bool = False, device: str = "auto"
) -> None:
    start_time = time.perf_counter()
    print("=" * 80)
    print("INDONESIA DENGUE MLOPS - INTEGRATED PIPELINE ORCHESTRATION")
    print(f"Hardware Acceleration Mode: {device}")
    print("=" * 80)

    # 1. Ingestion Verification
    print("\n[Stage 1/5] Verifying Ingestion & Datasets...")
    boundary_path = Path("data/raw/boundaries/indonesia_provinces.geojson")
    pop_path = Path("data/processed/population.csv")
    dengue_path = Path("data/processed/dengue.csv")
    weather_path = Path("data/processed/weather_weekly.csv")
    interim_path = Path("data/interim/merged_master_interim.csv")

    if not boundary_path.exists():
        print("  - Boundaries missing. Please ensure indonesia_provinces.geojson exists.")
    else:
        print(f"  - Boundaries present: {boundary_path}")

    if not pop_path.exists():
        print("  - Population data missing.")
        sys.exit(1)
    else:
        print(f"  - Population dataset verified: {pop_path}")

    if not dengue_path.exists() or force_data:
        print("  - Generating/refreshing dengue surveillance data...")
        from scripts.generate_dengue_data import main as gen_dengue

        gen_dengue()
    else:
        print(f"  - Dengue surveillance dataset verified: {dengue_path}")

    if not weather_path.exists() or force_data:
        print("  - Generating/refreshing weather dataset...")
        from scripts.generate_weather_data import main as gen_weather

        gen_weather()
    else:
        print(f"  - Weather dataset verified: {weather_path}")

    if not interim_path.exists() or force_data:
        print("  - Generating/refreshing raw & interim datasets...")
        from scripts.populate_full_raw_and_interim import main as gen_interim

        gen_interim()
    else:
        print(f"  - Interim dataset verified: {interim_path}")

    # 2. Feature Engineering
    print("\n[Stage 2/5] Running Feature Engineering Pipeline...")
    features_path = build_and_save_features()
    print(f"  - Features generated and stored at: {features_path}")

    # 3. Model Training & Experiment Tracking
    print(f"\n[Stage 3/5] Model Training & Champion Selection (v{version}, device={device})...")
    train_report = run_training_pipeline(
        features_path=features_path, model_version=version, device=device
    )
    print(f"  - Champion Model: {train_report['champion_model']}")
    print(f"  - Validation Metrics: {train_report['test_metrics']}")

    # 4. Model Evaluation & Audit
    print("\n[Stage 4/5] Model Evaluation & Health Audit...")
    champion, metadata = load_champion()
    records = load_dataset(features_path)
    split = split_features(records, feature_names=metadata["features"])

    test_preds = champion.predict(split.test_x)
    print(f"  - Processed {len(test_preds)} holdout evaluation cases.")

    # 5. Monitoring & Drift Assessment
    print("\n[Stage 5/5] Baseline Observability & Drift Verification...")
    detector = DriftDetector(psi_threshold=0.20)
    drift = detector.compute_feature_drift(
        reference_data=split.train_x, current_data=split.test_x, feature_names=split.feature_names
    )
    print(f"  - Max Feature PSI Score: {drift['max_psi']}")
    print(f"  - Drift Detected       : {drift['drift_detected']}")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    print(f"MLOPS PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
    print(f"Production Champion Model: {metadata['model_name']} (v{metadata['version']})")
    print(f"Artifact: {train_report['model_file']}")
    print("System is Ready for Production Serving & Monitoring!")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate end-to-end Indonesia Dengue MLOps pipeline"
    )
    parser.add_argument("--version", type=str, default="1.0.0", help="Model semantic version")
    parser.add_argument("--force-data", action="store_true", help="Force regenerate datasets")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware acceleration device",
    )
    args = parser.parse_args()

    run_integrated_pipeline(version=args.version, force_data=args.force_data, device=args.device)


if __name__ == "__main__":
    main()
