from __future__ import annotations

import csv
import math
import random
from pathlib import Path

DENGUE_PATH = Path("data/processed/dengue.csv")
WEATHER_OUTPUT = Path("data/processed/weather_weekly.csv")

FIELDNAMES = [
    "province",
    "year",
    "week",
    "temperature_mean_c",
    "humidity_mean",
    "pressure_mean_pa",
    "wind_speed_mean",
    "precipitation_sum_mm",
]


def generate_weather_dataset(seed: int = 42) -> list[dict[str, float | int | str]]:
    rng = random.Random(seed)

    if not DENGUE_PATH.exists():
        raise FileNotFoundError(f"Dengue dataset not found at {DENGUE_PATH}")

    # Load periods and provinces from dengue dataset to guarantee exact alignment
    periods: list[tuple[str, int, int]] = []
    with DENGUE_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            periods.append((row["province"], int(row["year"]), int(row["week"])))

    records: list[dict[str, float | int | str]] = []

    for province, year, week in periods:
        # Cyclical season: in Indonesia, rainy monsoon season is higher in Nov-Apr (weeks 45-52 and 1-17)
        rad = 2.0 * math.pi * (week - 3) / 52.0
        monsoon = (math.cos(rad) + 1.0) / 2.0  # 0.0 (dry) to 1.0 (wet)

        # Baseline temperatures vary slightly by province/elevation
        base_temp = 27.2
        if "PAPUA" in province:
            base_temp = 24.8
        elif "JAWA" in province or "BALI" in province:
            base_temp = 26.8

        temp = round(base_temp - 0.8 * monsoon + rng.gauss(0.0, 0.4), 2)
        humidity = round(72.0 + 16.0 * monsoon + rng.gauss(0.0, 2.0), 1)
        humidity = max(55.0, min(98.0, humidity))

        # Precipitation: wet season ~ 50-140 mm/week, dry season ~ 5-30 mm/week
        rain_mean = 15.0 + 75.0 * monsoon
        precipitation = round(max(0.0, rng.gauss(rain_mean, 12.0)), 1)

        pressure = round(101000.0 + rng.gauss(0.0, 150.0), 1)
        wind_speed = round(max(0.5, rng.gauss(2.2 + 0.6 * monsoon, 0.5)), 2)

        records.append(
            {
                "province": province,
                "year": year,
                "week": week,
                "temperature_mean_c": temp,
                "humidity_mean": humidity,
                "pressure_mean_pa": pressure,
                "wind_speed_mean": wind_speed,
                "precipitation_sum_mm": precipitation,
            }
        )

    return records


def main() -> None:
    records = generate_weather_dataset()

    WEATHER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with WEATHER_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)

    print("Weather weekly dataset generated successfully")
    print(f"File: {WEATHER_OUTPUT}")
    print(f"Total rows: {len(records)}")
    print(f"Provinces: {len({r['province'] for r in records})}")


if __name__ == "__main__":
    main()
