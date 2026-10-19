from __future__ import annotations

import math
from collections import deque
from typing import Any


class PerformanceMonitor:
    def __init__(self, window_size: int = 100, error_threshold: float = 0.20) -> None:
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.actuals: deque[float] = deque(maxlen=window_size)
        self.predictions: deque[float] = deque(maxlen=window_size)
        self.baseline_mae: float = 16.0

    def record_feedback(self, actual: float, prediction: float) -> None:
        self.actuals.append(float(actual))
        self.predictions.append(float(prediction))

    def get_metrics(self) -> dict[str, Any]:
        if not self.actuals:
            return {
                "sample_count": 0,
                "rolling_mae": 0.0,
                "rolling_rmse": 0.0,
                "degradation_ratio": 0.0,
                "alert_triggered": False,
            }

        n = len(self.actuals)
        errors = [abs(a - p) for a, p in zip(self.actuals, self.predictions, strict=False)]
        mae = sum(errors) / n
        mse = sum((a - p) ** 2 for a, p in zip(self.actuals, self.predictions, strict=False)) / n
        rmse = math.sqrt(mse)

        degradation_ratio = (
            (mae - self.baseline_mae) / self.baseline_mae if self.baseline_mae > 0 else 0.0
        )
        alert = degradation_ratio > self.error_threshold

        return {
            "sample_count": n,
            "rolling_mae": round(mae, 3),
            "rolling_rmse": round(rmse, 3),
            "degradation_ratio": round(degradation_ratio, 4),
            "error_threshold": self.error_threshold,
            "alert_triggered": alert,
        }
