import pandas as pd

from dengue_mlops.ingestion.skdr.ocr_parser import (
    build_row_centers,
    match_province,
    merge_number_candidates,
)


def test_match_province():
    province, score = match_province("JAWA TENGAH")

    assert province == "JAWA TENGAH"
    assert score >= 0.60


def test_merge_number_candidates():
    df = pd.DataFrame(
        [
            {
                "text": "421",
                "value": 421,
                "confidence": 0.90,
                "center_x": 102.0,
                "center_y": 52.0,
                "x_left": 92.0,
                "x_right": 112.0,
                "y_top": 47.0,
                "y_bottom": 57.0,
                "preprocess": "otsu",
            },
            {
                "text": "421",
                "value": 421,
                "confidence": 0.80,
                "center_x": 100.0,
                "center_y": 50.0,
                "x_left": 90.0,
                "x_right": 110.0,
                "y_top": 45.0,
                "y_bottom": 55.0,
                "preprocess": "threshold_40",
            },
        ]
    )

    result = merge_number_candidates(df)

    assert len(result) == 2
    assert result.iloc[0]["center_y"] == 50.0
    assert result.iloc[1]["center_y"] == 52.0


def test_build_row_centers():
    df = pd.DataFrame([{"center_y": 100.0}, {"center_y": 103.0}, {"center_y": 200.0}])

    centers = build_row_centers(df)

    assert len(centers) == 2
