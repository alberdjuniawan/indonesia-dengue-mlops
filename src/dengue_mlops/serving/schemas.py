from __future__ import annotations

from pydantic import BaseModel, Field


class DenguePredictionRequest(BaseModel):
    province: str = Field(..., example="JAWA BARAT")
    year: int = Field(..., ge=2020, le=2030, example=2026)
    week: int = Field(..., ge=1, le=53, example=13)
    cases_lag_1: float = Field(..., ge=0.0, example=650.0)
    cases_lag_2: float = Field(default=640.0, ge=0.0)
    cases_lag_3: float = Field(default=630.0, ge=0.0)
    cases_lag_4: float = Field(default=620.0, ge=0.0)
    cases_roll_mean_4w: float = Field(default=635.0, ge=0.0)
    cases_roll_std_4w: float = Field(default=12.0, ge=0.0)
    incidence_lag_1: float = Field(default=1.29, ge=0.0)
    incidence_lag_2: float = Field(default=1.27, ge=0.0)
    temp_mean_lag_1: float = Field(default=26.8, example=27.0)
    temp_mean_lag_2: float = Field(default=26.9)
    temp_roll_mean_4w: float = Field(default=26.8)
    humidity_mean_lag_1: float = Field(default=82.0)
    humidity_mean_lag_2: float = Field(default=81.5)
    humidity_roll_mean_4w: float = Field(default=82.0)
    rainfall_sum_lag_1: float = Field(default=65.0)
    rainfall_sum_lag_2: float = Field(default=70.0)
    rainfall_roll_sum_4w: float = Field(default=260.0)
    wind_speed_lag_1: float = Field(default=2.1)
    sin_week: float = Field(default=1.0)
    cos_week: float = Field(default=0.0)
    population: float = Field(default=50340000.0)


class DenguePredictionResponse(BaseModel):
    province: str
    year: int
    week: int
    predicted_cases: float
    model_version: str
    inference_time_ms: float


class BatchPredictionRequest(BaseModel):
    items: list[DenguePredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[DenguePredictionResponse]
    total_processed: int
    batch_latency_ms: float


class FeedbackRequest(BaseModel):
    province: str
    year: int
    week: int
    predicted_cases: float
    actual_cases: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    status: str
    features: list[str]
    metrics: dict[str, float]
    promoted_at: str
