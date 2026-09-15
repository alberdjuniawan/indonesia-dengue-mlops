from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "valid_time",
    "temperature_c",
    "humidity",
    "pressure_pa",
    "wind_speed",
    "precipitation_m",
}


def aggregate_hourly_to_weekly(hourly: pd.DataFrame, *, province: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(hourly.columns)

    if missing:
        raise ValueError("Missing weather columns: " + ", ".join(sorted(missing)))

    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "province",
                "year",
                "week",
                "temperature_mean_c",
                "humidity_mean",
                "pressure_mean_pa",
                "wind_speed_mean",
                "precipitation_sum_mm",
                "hour_count",
            ]
        )

    frame = hourly.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)

    iso = frame["valid_time"].dt.isocalendar()
    frame["year"] = iso.year.astype(int)
    frame["week"] = iso.week.astype(int)

    weekly = frame.groupby(["year", "week"], as_index=False).agg(
        temperature_mean_c=("temperature_c", "mean"),
        humidity_mean=("humidity", "mean"),
        pressure_mean_pa=("pressure_pa", "mean"),
        wind_speed_mean=("wind_speed", "mean"),
        precipitation_sum_m=("precipitation_m", "sum"),
        hour_count=("valid_time", "count"),
    )

    weekly["precipitation_sum_mm"] = weekly["precipitation_sum_m"] * 1000.0
    weekly = weekly.drop(columns=["precipitation_sum_m"])
    weekly.insert(0, "province", province.upper())
    weekly = weekly.sort_values(["year", "week"]).reset_index(drop=True)

    return weekly
