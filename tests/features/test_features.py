from __future__ import annotations

from dengue_mlops.features.pipeline import FeatureConfig, engineer_features


def test_engineer_features() -> None:
    dengue = [
        {"province": "JAWA TIMUR", "year": 2025, "week": w, "cases": 100 + w * 2}
        for w in range(1, 10)
    ]
    population = [{"province": "JAWA TIMUR", "year": 2025, "population": 40_000_000}]
    weather = [
        {
            "province": "JAWA TIMUR",
            "year": 2025,
            "week": w,
            "temperature_mean_c": 27.0,
            "humidity_mean": 80.0,
            "pressure_mean_pa": 101000.0,
            "wind_speed_mean": 2.0,
            "precipitation_sum_mm": 50.0,
        }
        for w in range(1, 10)
    ]

    config = FeatureConfig(lags=(1, 2, 3, 4), target_horizon=1)
    rows = engineer_features(dengue, population, weather, config=config)

    assert len(rows) > 0
    first = rows[0]
    assert "cases_lag_1" in first
    assert "cases_roll_mean_4w" in first
    assert "incidence_lag_1" in first
    assert "target_cases" in first
    assert first["target_cases"] > 0
