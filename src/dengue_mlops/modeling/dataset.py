from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dengue_mlops.features.pipeline import FEATURE_COLUMNS

DEFAULT_FEATURES_PATH = Path("data/features/features.csv")


@dataclass
class DatasetSplit:
    train_x: list[list[float]]
    train_y: list[float]
    val_x: list[list[float]]
    val_y: list[float]
    test_x: list[list[float]]
    test_y: list[float]
    feature_names: list[str]
    provinces_train: list[str]
    provinces_val: list[str]
    provinces_test: list[str]


def load_dataset(path: Path = DEFAULT_FEATURES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Features file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def split_features(
    records: list[dict[str, Any]],
    feature_names: list[str] | None = None,
    train_cutoff_year: int = 2024,
    val_cutoff_week: int = 26,
    val_year: int = 2025,
) -> DatasetSplit:
    if feature_names is None:
        # Filter feature names to only numeric ones that exist in records
        sample = records[0]
        feature_names = [col for col in FEATURE_COLUMNS if col in sample]

    train_x, train_y, train_prov = [], [], []
    val_x, val_y, val_prov = [], [], []
    test_x, test_y, test_prov = [], [], []

    for r in records:
        yr = int(r["year"])
        wk = int(r["week"])
        y_val = float(r["target_cases"])
        x_row = [float(r[col]) for col in feature_names]
        prov = r["province"]

        # Temporal splitting logic
        if yr <= train_cutoff_year:
            train_x.append(x_row)
            train_y.append(y_val)
            train_prov.append(prov)
        elif yr == val_year and wk <= val_cutoff_week:
            val_x.append(x_row)
            val_y.append(y_val)
            val_prov.append(prov)
        else:
            test_x.append(x_row)
            test_y.append(y_val)
            test_prov.append(prov)

    return DatasetSplit(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        feature_names=feature_names,
        provinces_train=train_prov,
        provinces_val=val_prov,
        provinces_test=test_prov,
    )
