import pandas as pd

from dengue_mlops.ingestion.weather.alignment import align_weather_with_skdr


def test_align_weather_with_skdr(tmp_path):
    dengue_path = tmp_path / "dengue.csv"

    pd.DataFrame(
        [
            {"province": "JAWA TIMUR", "year": 2026, "week": 1, "cases": 100},
            {"province": "JAWA TIMUR", "year": 2026, "week": 2, "cases": 110},
        ]
    ).to_csv(dengue_path, index=False)

    weather = pd.DataFrame(
        [
            {
                "province": "JAWA TIMUR",
                "year": 2026,
                "week": 1,
                "temperature_mean_c": 24.5,
                "humidity_mean": 85.0,
                "pressure_mean_pa": 96700.0,
                "wind_speed_mean": 1.2,
                "precipitation_sum_mm": 40.0,
            }
        ]
    )

    result = align_weather_with_skdr(weather, dengue_path)

    assert len(result) == 2

    week_1 = result.loc[result["week"] == 1].iloc[0]

    assert week_1["weather_complete"]

    assert week_1["temperature_mean_c"] == 24.5

    assert week_1["precipitation_sum_mm"] == 40.0

    week_2 = result.loc[result["week"] == 2].iloc[0]

    assert not week_2["weather_complete"]


def test_align_weather_with_skdr_detects_missing_columns(tmp_path):
    dengue_path = tmp_path / "dengue.csv"

    pd.DataFrame([{"province": "JAWA TIMUR", "year": 2026, "week": 1}]).to_csv(
        dengue_path, index=False
    )

    weather = pd.DataFrame(
        [{"province": "JAWA TIMUR", "year": 2026, "week": 1, "temperature_mean_c": 24.5}]
    )

    try:
        align_weather_with_skdr(weather, dengue_path)
    except ValueError as exc:
        assert "Missing weather columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
