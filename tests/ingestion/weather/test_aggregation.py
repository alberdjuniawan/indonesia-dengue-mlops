import pandas as pd
import pytest

from dengue_mlops.ingestion.weather.aggregation import aggregate_hourly_to_weekly


def test_aggregate_hourly_to_weekly():
    hourly = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 06:00:00",
                    "2026-01-01 12:00:00",
                    "2026-01-01 18:00:00",
                    "2026-01-02 00:00:00",
                    "2026-01-02 06:00:00",
                    "2026-01-02 12:00:00",
                    "2026-01-02 18:00:00",
                    "2026-01-03 00:00:00",
                ],
                utc=True,
            ),
            "temperature_c": [24.0, 26.0, 28.0, 30.0, 25.0, 27.0, 29.0, 31.0, 26.0],
            "humidity": [80.0, 70.0, 60.0, 50.0, 75.0, 65.0, 55.0, 45.0, 70.0],
            "pressure_pa": [
                100000.0,
                100100.0,
                100200.0,
                100300.0,
                100400.0,
                100500.0,
                100600.0,
                100700.0,
                100800.0,
            ],
            "wind_speed": [2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0, 4.0],
            "precipitation_m": [0.010, 0.005, 0.000, 0.015, 0.020, 0.000, 0.005, 0.010, 0.025],
        }
    )

    result = aggregate_hourly_to_weekly(hourly, province="JAWA TIMUR")

    assert not result.empty
    assert len(result) == 1

    row = result.iloc[0]

    assert row["province"] == "JAWA TIMUR"
    assert row["year"] == 2026
    assert row["week"] == 1

    assert row["temperature_mean_c"] == pytest.approx(hourly["temperature_c"].mean())

    assert row["humidity_mean"] == pytest.approx(hourly["humidity"].mean())

    assert row["pressure_mean_pa"] == pytest.approx(hourly["pressure_pa"].mean())

    assert row["wind_speed_mean"] == pytest.approx(hourly["wind_speed"].mean())

    assert row["precipitation_sum_mm"] == pytest.approx(hourly["precipitation_m"].sum() * 1000.0)

    assert row["hour_count"] == 9


def test_aggregate_hourly_to_weekly_separates_iso_weeks():
    hourly = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                [
                    "2025-12-28 23:00:00",
                    "2025-12-29 00:00:00",
                    "2026-01-04 23:00:00",
                    "2026-01-05 00:00:00",
                ],
                utc=True,
            ),
            "temperature_c": [20.0, 21.0, 22.0, 23.0],
            "humidity": [80.0, 81.0, 82.0, 83.0],
            "pressure_pa": [95000.0, 95100.0, 95200.0, 95300.0],
            "wind_speed": [1.0, 2.0, 3.0, 4.0],
            "precipitation_m": [0.001, 0.002, 0.003, 0.004],
        }
    )

    result = aggregate_hourly_to_weekly(hourly, province="JAWA TIMUR")

    assert len(result) == 3

    first = result.iloc[0]
    second = result.iloc[1]
    third = result.iloc[2]

    # 2025-12-28 belongs to ISO week 52 of 2025.
    assert first["year"] == 2025
    assert first["week"] == 52
    assert first["hour_count"] == 1

    # 2025-12-29 through 2026-01-04 is ISO week 1 of 2026.
    assert second["year"] == 2026
    assert second["week"] == 1
    assert second["hour_count"] == 2

    # 2026-01-05 starts ISO week 2.
    assert third["year"] == 2026
    assert third["week"] == 2
    assert third["hour_count"] == 1
