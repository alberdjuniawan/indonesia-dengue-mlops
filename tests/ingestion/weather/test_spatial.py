import json

import pytest

from dengue_mlops.ingestion.weather.spatial import (
    build_province_points,
    load_province_boundaries,
    normalize_province,
)


def test_normalize_province():
    assert normalize_province("East Java") == "EAST JAVA"
    assert normalize_province(" Jawa Timur ") == "JAWA TIMUR"
    assert normalize_province("Daerah Istimewa Yogyakarta") == "DI YOGYAKARTA"


def test_load_boundaries_and_points(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"shapeName": "Jawa Timur"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[112.0, -8.0], [113.0, -8.0], [113.0, -7.0], [112.0, -7.0], [112.0, -8.0]]
                    ],
                },
            }
        ],
    }

    path = tmp_path / "boundary.geojson"

    path.write_text(json.dumps(geojson), encoding="utf-8")

    boundaries = load_province_boundaries(path)

    assert len(boundaries) == 1
    assert list(boundaries.columns) == ["province", "geometry"]

    assert boundaries["province"].iloc[0] == "JAWA TIMUR"

    points = build_province_points(boundaries)

    assert len(points) == 1

    point = points[0]

    assert point.province == "JAWA TIMUR"
    assert point.latitude == pytest.approx(-7.5)
    assert point.longitude == pytest.approx(112.5)
