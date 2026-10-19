from __future__ import annotations

from dengue_mlops.modeling.models import BaselineLagModel
from dengue_mlops.modeling.registry import load_champion, register_champion
from dengue_mlops.modeling.trainer import run_training_pipeline
from dengue_mlops.serving.app import (
    get_metrics_exposition,
    health_check,
    predict_batch,
    predict_single,
    submit_feedback,
)
from dengue_mlops.serving.schemas import (
    BatchPredictionRequest,
    DenguePredictionRequest,
    FeedbackRequest,
)


def _setup_champion() -> None:
    model = BaselineLagModel(lag_idx=0)
    register_champion(
        model=model,
        version="1.0.0",
        metrics={"rmse": 10.0, "mae": 8.0, "r2": 0.95},
        feature_names=["cases_lag_1"],
    )
    import dengue_mlops.serving.app as app_mod

    app_mod.champion_model, app_mod.champion_metadata = load_champion()


def test_health_check_endpoint() -> None:
    _setup_champion()
    resp = health_check()
    assert resp.status in ("healthy", "degraded")
    assert resp.timestamp != ""


def test_predict_single_endpoint() -> None:
    _setup_champion()
    req = DenguePredictionRequest(province="JAWA BARAT", year=2026, week=10, cases_lag_1=500.0)

    resp = predict_single(req)
    assert resp.province == "JAWA BARAT"
    assert resp.year == 2026
    assert resp.week == 10
    assert resp.predicted_cases == 500.0
    assert resp.inference_time_ms >= 0.0


def test_predict_batch_endpoint() -> None:
    _setup_champion()
    batch = BatchPredictionRequest(
        items=[
            DenguePredictionRequest(province="JAWA BARAT", year=2026, week=1, cases_lag_1=300.0),
            DenguePredictionRequest(province="JAWA TIMUR", year=2026, week=1, cases_lag_1=250.0),
        ]
    )
    resp = predict_batch(batch)
    assert resp.total_processed == 2
    assert len(resp.predictions) == 2
    assert resp.predictions[0].predicted_cases == 300.0
    assert resp.predictions[1].predicted_cases == 250.0


def test_feedback_and_metrics_endpoints() -> None:
    fb = FeedbackRequest(
        province="JAWA BARAT", year=2026, week=1, predicted_cases=300.0, actual_cases=310.0
    )
    fb_resp = submit_feedback(fb)
    assert fb_resp["status"] == "RECORDED"

    metrics_resp = get_metrics_exposition()
    assert metrics_resp.status_code == 200
    assert b"dengue_prediction_requests_total" in metrics_resp.body

    # Restore production champion
    run_training_pipeline(model_version="1.0.0")
