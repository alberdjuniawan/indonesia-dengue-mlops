from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from dengue_mlops.ingestion.weather.aggregation import aggregate_hourly_to_weekly
from dengue_mlops.ingestion.weather.client import ERA5LandClient
from dengue_mlops.ingestion.weather.parser import parse_timeseries
from dengue_mlops.ingestion.weather.spatial import (
    build_province_points,
    load_province_boundaries,
    normalize_province,
)

DENGUE_PATH = Path("data/processed/dengue.csv")
BOUNDARY_PATH = Path("data/raw/boundaries/indonesia_provinces.geojson")
RAW_DIR = Path("data/raw/weather/era5")
WEEKLY_OUTPUT = Path("data/processed/weather_weekly.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", type=date.fromisoformat, default=None, help="Start date: YYYY-MM-DD"
    )
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="End date: YYYY-MM-DD")
    parser.add_argument("--province", type=str, default=None, help="Optional province filter")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Reuse existing raw ERA5 archive"
    )
    return parser.parse_args()


def get_skdr_date_range() -> tuple[date, date]:
    dengue = pd.read_csv(DENGUE_PATH)

    required = {"year", "week"}
    missing = required.difference(dengue.columns)

    if missing:
        raise ValueError("Missing dengue columns: " + ", ".join(sorted(missing)))

    periods = dengue[["year", "week"]].drop_duplicates().sort_values(["year", "week"])

    if periods.empty:
        raise ValueError("Dengue dataset contains no periods")

    first = periods.iloc[0]
    last = periods.iloc[-1]

    start = date.fromisocalendar(int(first["year"]), int(first["week"]), 1)
    end = date.fromisocalendar(int(last["year"]), int(last["week"]), 7)
    start = start - timedelta(days=28)

    return start, end


def resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    default_start, default_end = get_skdr_date_range()

    start = args.start if args.start is not None else default_start
    end = args.end if args.end is not None else default_end

    if end < start:
        raise ValueError("--end must be >= --start")

    return start, end


def resolve_province_points(province_filter: str | None):
    boundaries = load_province_boundaries(BOUNDARY_PATH)
    points = build_province_points(boundaries)

    if province_filter is None:
        return points

    normalized = normalize_province(province_filter)
    filtered = [point for point in points if point.province == normalized]

    if not filtered:
        raise ValueError(f"Province not found: {province_filter}")

    return filtered


def raw_archive_path(province: str, start: date, end: date) -> Path:
    safe_name = province.upper().replace(" ", "_").replace("/", "_")
    return RAW_DIR / f"era5_land_timeseries_{safe_name}_{start.isoformat()}_{end.isoformat()}.zip"


def process_province(*, client: ERA5LandClient, province_point, start: date, end: date, skip_existing: bool) -> pd.DataFrame:  # fmt: skip
    province = province_point.province.upper()
    archive_path = raw_archive_path(province, start, end)

    print()
    print(f"[{province}] {start.isoformat()} -> {end.isoformat()}")
    print(f"Representative point: {province_point.latitude:.4f}, {province_point.longitude:.4f}")

    if skip_existing and archive_path.exists():
        print("Using existing ERA5 archive...")
    else:
        print("Downloading ERA5-Land time-series...")
        client.download_timeseries(
            start=start,
            end=end,
            latitude=province_point.latitude,
            longitude=province_point.longitude,
            output_path=archive_path,
        )

    print("Parsing hourly data...")

    hourly = parse_timeseries(archive_path)

    expected_start = pd.Timestamp(start, tz="UTC")
    expected_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    hourly = hourly[
        (hourly["valid_time"] >= expected_start) & (hourly["valid_time"] <= expected_end)
    ].copy()

    print(f"Hourly rows: {len(hourly)}")

    if hourly.empty:
        raise ValueError(f"No weather data for {province}")

    weekly = aggregate_hourly_to_weekly(hourly, province=province)

    print(f"Weekly rows: {len(weekly)}")

    return weekly


def combine_weekly(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No weekly weather data generated")

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["province", "year", "week"], keep="last")
        .sort_values(["province", "year", "week"])
        .reset_index(drop=True)
    )

    return combined


def main() -> None:
    args = parse_args()

    if not DENGUE_PATH.exists():
        raise FileNotFoundError(DENGUE_PATH)
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(BOUNDARY_PATH)

    start, end = resolve_date_range(args)
    province_points = resolve_province_points(args.province)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WEEKLY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    client = ERA5LandClient()
    weekly_parts: list[pd.DataFrame] = []

    for point in province_points:
        try:
            weekly = process_province(
                client=client,
                province_point=point,
                start=start,
                end=end,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            print(f"[{point.province}] failed: {type(exc).__name__}: {exc}")
            raise
        weekly_parts.append(weekly)

    weather_weekly = combine_weekly(weekly_parts)
    weather_weekly.to_csv(WEEKLY_OUTPUT, index=False)

    print()
    print("=" * 70)
    print("Weather ingestion completed")
    print(f"Output: {WEEKLY_OUTPUT}")
    print(f"Rows: {len(weather_weekly)}")
    print(f"Provinces: {weather_weekly['province'].nunique()}")
    print(f"Periods: {weather_weekly[['year', 'week']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
