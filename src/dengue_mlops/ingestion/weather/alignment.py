from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["province", "year", "week"]

WEATHER_COLUMNS = [
    "temperature_mean_c",
    "humidity_mean",
    "pressure_mean_pa",
    "wind_speed_mean",
    "precipitation_sum_mm",
]


def load_skdr_periods(dengue_path: str | Path) -> pd.DataFrame:
    dengue_path = Path(dengue_path)

    if not dengue_path.exists():
        raise FileNotFoundError(dengue_path)

    dengue = pd.read_csv(dengue_path)

    required = set(KEY_COLUMNS)

    missing = required.difference(dengue.columns)

    if missing:
        raise ValueError("Missing SKDR columns: " + ", ".join(sorted(missing)))

    return dengue[KEY_COLUMNS].drop_duplicates().sort_values(KEY_COLUMNS).reset_index(drop=True)


def validate_weather_schema(weather_weekly: pd.DataFrame) -> None:
    required = set(KEY_COLUMNS + WEATHER_COLUMNS)

    missing = required.difference(weather_weekly.columns)

    if missing:
        raise ValueError("Missing weather columns: " + ", ".join(sorted(missing)))


def align_weather_with_skdr(weather_weekly: pd.DataFrame, dengue_path: str | Path) -> pd.DataFrame:
    validate_weather_schema(weather_weekly)

    periods = load_skdr_periods(dengue_path)

    result = periods.merge(
        weather_weekly[KEY_COLUMNS + WEATHER_COLUMNS], on=KEY_COLUMNS, how="left"
    )

    result["weather_complete"] = result[WEATHER_COLUMNS].notna().all(axis=1)

    result = result.sort_values(KEY_COLUMNS).reset_index(drop=True)

    return result
