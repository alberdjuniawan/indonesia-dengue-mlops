from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from dengue_mlops.modeling.models import BaseModel
from dengue_mlops.modeling.registry import load_champion
from dengue_mlops.monitoring.drift import DriftDetector
from dengue_mlops.monitoring.metrics_exporter import metrics
from dengue_mlops.monitoring.performance import PerformanceMonitor

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DenguePredictionRequest,
    DenguePredictionResponse,
    FeedbackRequest,
    HealthResponse,
    ModelInfoResponse,
)

champion_model: BaseModel | None = None
champion_metadata: dict[str, Any] = {}
performance_monitor = PerformanceMonitor()
drift_detector = DriftDetector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global champion_model, champion_metadata
    try:
        champion_model, champion_metadata = load_champion()
        print(f"Loaded Champion Model {champion_metadata.get('model_name')}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Notice: Model could not be loaded at startup ({e})")
        champion_model = None
        champion_metadata = {}
    yield


app = FastAPI(
    title="Indonesia Dengue Forecast API",
    description="Production ML Serving System for Dengue Epidemiological Forecasting",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_feature_vector(req: DenguePredictionRequest, feature_names: list[str]) -> list[float]:
    req_dict = req.model_dump()
    vec = []
    for f in feature_names:
        vec.append(float(req_dict.get(f, 0.0)))
    return vec


def _ensure_model_loaded() -> None:
    global champion_model, champion_metadata
    if champion_model is None:
        try:
            champion_model, champion_metadata = load_champion()
        except (FileNotFoundError, ValueError, KeyError):
            raise HTTPException(status_code=503, detail="Champion model is not available.")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        _ensure_model_loaded()
    except HTTPException:
        pass
    return HealthResponse(
        status="healthy" if champion_model is not None else "degraded",
        model_loaded=champion_model is not None,
        model_version=champion_metadata.get("version", "unknown"),
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/v1/model/info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    _ensure_model_loaded()
    return ModelInfoResponse(
        model_name=champion_metadata.get("model_name", "dengue_model"),
        version=champion_metadata.get("version", "1.0.0"),
        status=champion_metadata.get("status", "CHAMPION"),
        features=champion_metadata.get("features", []),
        metrics=champion_metadata.get("metrics", {}),
        promoted_at=champion_metadata.get("promoted_at", ""),
    )


@app.post("/v1/predict", response_model=DenguePredictionResponse)
def predict_single(request: DenguePredictionRequest) -> DenguePredictionResponse:
    _ensure_model_loaded()

    start_t = time.perf_counter()
    feature_names = champion_metadata.get("features", [])
    features_vec = [_extract_feature_vector(request, feature_names)]

    preds = champion_model.predict(features_vec)
    pred_value = preds[0]
    duration = time.perf_counter() - start_t

    metrics.record_request(duration=duration, predicted_cases=pred_value)

    return DenguePredictionResponse(
        province=request.province,
        year=request.year,
        week=request.week,
        predicted_cases=pred_value,
        model_version=champion_metadata.get("version", "1.0.0"),
        inference_time_ms=round(duration * 1000.0, 3),
    )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch_req: BatchPredictionRequest) -> BatchPredictionResponse:
    _ensure_model_loaded()

    start_t = time.perf_counter()
    feature_names = champion_metadata.get("features", [])
    matrix = [_extract_feature_vector(item, feature_names) for item in batch_req.items]

    preds = champion_model.predict(matrix)
    total_duration = time.perf_counter() - start_t

    version = champion_metadata.get("version", "1.0.0")
    per_item_ms = (
        round((total_duration / len(batch_req.items)) * 1000.0, 3) if batch_req.items else 0.0
    )

    responses = [
        DenguePredictionResponse(
            province=item.province,
            year=item.year,
            week=item.week,
            predicted_cases=pred_val,
            model_version=version,
            inference_time_ms=per_item_ms,
        )
        for item, pred_val in zip(batch_req.items, preds, strict=False)
    ]

    for p in preds:
        metrics.record_request(duration=total_duration / max(1, len(preds)), predicted_cases=p)

    return BatchPredictionResponse(
        predictions=responses,
        total_processed=len(responses),
        batch_latency_ms=round(total_duration * 1000.0, 3),
    )


@app.post("/v1/feedback")
def submit_feedback(feedback: FeedbackRequest) -> dict[str, Any]:
    performance_monitor.record_feedback(
        actual=feedback.actual_cases, prediction=feedback.predicted_cases
    )
    current_metrics = performance_monitor.get_metrics()
    metrics.update_mae(current_metrics["rolling_mae"])
    return {"status": "RECORDED", "performance": current_metrics}


@app.get("/v1/monitoring/performance")
def get_performance_stats() -> dict[str, Any]:
    return performance_monitor.get_metrics()


@app.get("/metrics")
def get_metrics_exposition() -> Response:
    content = metrics.generate_metrics_text()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
