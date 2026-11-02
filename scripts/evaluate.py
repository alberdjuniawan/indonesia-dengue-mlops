from __future__ import annotations

import argparse
import json
from pathlib import Path

from dengue_mlops.modeling.dataset import load_dataset, split_features
from dengue_mlops.modeling.metrics import calculate_metrics
from dengue_mlops.modeling.registry import load_champion

DEFAULT_EVALUATION_REPORT = Path("models/evaluation_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate champion dengue model")
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

    report = {
        "model_name": metadata["model_name"],
        "version": metadata["version"],
        "test_sample_count": len(split.test_x),
        "overall_metrics": metrics,
        "province_metrics": province_metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEvaluation Summary:")
    print(f"Overall RMSE : {metrics['rmse']}")
    print(f"Overall MAE  : {metrics['mae']}")
    print(f"Overall R2   : {metrics['r2']}")
    print(f"Overall WAPE : {metrics['wape']}%")
    print(f"Report saved : {args.output}")


if __name__ == "__main__":
    main()
