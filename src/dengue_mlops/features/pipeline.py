from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POPULATION_PATH = Path("data/processed/population.csv")
DENGUE_PATH = Path("data/processed/dengue.csv")
WEATHER_PATH = Path("data/processed/weather_weekly.csv")
OUTPUT_CSV_PATH = Path("data/features/features.csv")


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 4)
    rolling_window: int = 4
    target_horizon: int = 1
    target_column: str = "target_cases"


FEATURE_COLUMNS = [
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "cases_roll_mean_4w",
    "cases_roll_std_4w",
    "incidence_lag_1",
    "incidence_lag_2",
    "temp_mean_lag_1",
    "temp_mean_lag_2",
    "temp_roll_mean_4w",
    "humidity_mean_lag_1",
    "humidity_mean_lag_2",
    "humidity_roll_mean_4w",
    "rainfall_sum_lag_1",
    "rainfall_sum_lag_2",
    "rainfall_roll_sum_4w",
    "wind_speed_lag_1",
    "sin_week",
    "cos_week",
    "population",
]


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def engineer_features(
    dengue_records: list[dict[str, Any]],
    population_records: list[dict[str, Any]],
    weather_records: list[dict[str, Any]],
    config: FeatureConfig | None = None,
) -> list[dict[str, Any]]:
    if config is None:
        config = FeatureConfig()
    # Map population by province (latest year or default)
    pop_map: dict[str, float] = {}
    for r in population_records:
        pop_map[r["province"]] = float(r["population"])

    # Map weather by (province, year, week)
    weather_map: dict[tuple[str, int, int], dict[str, float]] = {}
    for r in weather_records:
        key = (r["province"], int(r["year"]), int(r["week"]))
        weather_map[key] = {
            "temp": float(r["temperature_mean_c"]),
            "humidity": float(r["humidity_mean"]),
            "pressure": float(r["pressure_mean_pa"]),
            "wind": float(r["wind_speed_mean"]),
            "rainfall": float(r["precipitation_sum_mm"]),
        }

    # Group dengue records by province
    by_province: dict[str, list[dict[str, Any]]] = {}
    for r in dengue_records:
        p = r["province"]
        by_province.setdefault(p, []).append(
            {"year": int(r["year"]), "week": int(r["week"]), "cases": int(r["cases"])}
        )

    featured_rows: list[dict[str, Any]] = []

    for province, records in by_province.items():
        # Sort chronologically
        records.sort(key=lambda x: (x["year"], x["week"]))
        pop = pop_map.get(province, 1_000_000.0)

        # Precompute per-timestep values
        series: list[dict[str, Any]] = []
        for r in records:
            yr = r["year"]
            wk = r["week"]
            cases = r["cases"]
            w = weather_map.get(
                (province, yr, wk),
                {
                    "temp": 27.0,
                    "humidity": 78.0,
                    "pressure": 101000.0,
                    "wind": 2.0,
                    "rainfall": 30.0,
                },
            )
            incidence = (cases / pop) * 100_000.0
            series.append({"year": yr, "week": wk, "cases": cases, "incidence": incidence, **w})

        n = len(series)
        # We need at least 4 past weeks for lags and 1 future week for target
        for i in range(4, n - config.target_horizon):
            curr = series[i]
            target = series[i + config.target_horizon]["cases"]

            # Lags
            c_lags = [series[i - l]["cases"] for l in (1, 2, 3, 4)]
            inc_lags = [series[i - l]["incidence"] for l in (1, 2)]

            # Rolling cases (last 4 weeks: i-3 to i)
            recent_cases = [series[i - k]["cases"] for k in range(4)]
            mean_c = sum(recent_cases) / 4.0
            var_c = sum((c - mean_c) ** 2 for c in recent_cases) / 4.0
            std_c = math.sqrt(var_c)

            # Weather lags & rolling
            t_lag1 = series[i - 1]["temp"]
            t_lag2 = series[i - 2]["temp"]
            t_roll4 = sum(series[i - k]["temp"] for k in range(4)) / 4.0

            h_lag1 = series[i - 1]["humidity"]
            h_lag2 = series[i - 2]["humidity"]
            h_roll4 = sum(series[i - k]["humidity"] for k in range(4)) / 4.0

            r_lag1 = series[i - 1]["rainfall"]
            r_lag2 = series[i - 2]["rainfall"]
            r_roll4 = sum(series[i - k]["rainfall"] for k in range(4))

            w_lag1 = series[i - 1]["wind"]

            # Cyclical calendar features
            wk_rad = 2.0 * math.pi * curr["week"] / 52.0
            sin_w = math.sin(wk_rad)
            cos_w = math.cos(wk_rad)

            featured_rows.append(
                {
                    "province": province,
                    "year": curr["year"],
                    "week": curr["week"],
                    "cases_current": curr["cases"],
                    "cases_lag_1": c_lags[0],
                    "cases_lag_2": c_lags[1],
                    "cases_lag_3": c_lags[2],
                    "cases_lag_4": c_lags[3],
                    "cases_roll_mean_4w": round(mean_c, 2),
                    "cases_roll_std_4w": round(std_c, 2),
                    "incidence_lag_1": round(inc_lags[0], 4),
                    "incidence_lag_2": round(inc_lags[1], 4),
                    "temp_mean_lag_1": t_lag1,
                    "temp_mean_lag_2": t_lag2,
                    "temp_roll_mean_4w": round(t_roll4, 2),
                    "humidity_mean_lag_1": h_lag1,
                    "humidity_mean_lag_2": h_lag2,
                    "humidity_roll_mean_4w": round(h_roll4, 2),
                    "rainfall_sum_lag_1": r_lag1,
                    "rainfall_sum_lag_2": r_lag2,
                    "rainfall_roll_sum_4w": round(r_roll4, 2),
                    "wind_speed_lag_1": w_lag1,
                    "sin_week": round(sin_w, 4),
                    "cos_week": round(cos_w, 4),
                    "population": pop,
                    "target_cases": target,
                }
            )

    # Sort deterministically
    featured_rows.sort(key=lambda r: (r["year"], r["week"], r["province"]))
    return featured_rows


def build_and_save_features(
    dengue_path: Path = DENGUE_PATH,
    population_path: Path = POPULATION_PATH,
    weather_path: Path = WEATHER_PATH,
    output_path: Path = OUTPUT_CSV_PATH,
) -> Path:
    dengue = load_csv(dengue_path)
    pop = load_csv(population_path)
    weather = load_csv(weather_path)

    features = engineer_features(dengue, pop, weather)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(features[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)

    return output_path
