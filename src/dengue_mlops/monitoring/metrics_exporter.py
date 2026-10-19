from __future__ import annotations


class PrometheusMetrics:
    def __init__(self) -> None:
        self.request_count = 0
        self.request_latencies: list[float] = []
        self.drift_score = 0.0
        self.rolling_mae = 0.0
        self.last_predicted_cases = 0.0

    def record_request(self, duration: float, predicted_cases: float) -> None:
        self.request_count += 1
        self.request_latencies.append(duration)
        if len(self.request_latencies) > 500:
            self.request_latencies.pop(0)
        self.last_predicted_cases = predicted_cases

    def update_drift(self, psi_score: float) -> None:
        self.drift_score = psi_score

    def update_mae(self, mae_score: float) -> None:
        self.rolling_mae = mae_score

    def generate_metrics_text(self) -> str:
        avg_latency = (
            sum(self.request_latencies) / len(self.request_latencies)
            if self.request_latencies
            else 0.0
        )
        lines = [
            "# HELP dengue_prediction_requests_total Total number of prediction requests processed",
            "# TYPE dengue_prediction_requests_total counter",
            f"dengue_prediction_requests_total {self.request_count}",
            "",
            "# HELP dengue_prediction_duration_seconds Average prediction latency in seconds",
            "# TYPE dengue_prediction_duration_seconds gauge",
            f"dengue_prediction_duration_seconds {round(avg_latency, 6)}",
            "",
            "# HELP dengue_feature_drift_psi Current maximum Population Stability Index drift score",
            "# TYPE dengue_feature_drift_psi gauge",
            f"dengue_feature_drift_psi {self.drift_score}",
            "",
            "# HELP dengue_model_mae_rolling Current rolling MAE against ground truth",
            "# TYPE dengue_model_mae_rolling gauge",
            f"dengue_model_mae_rolling {self.rolling_mae}",
            "",
            "# HELP dengue_last_prediction_cases Most recent forecasted dengue caseload",
            "# TYPE dengue_last_prediction_cases gauge",
            f"dengue_last_prediction_cases {self.last_predicted_cases}",
            "",
        ]
        return "\n".join(lines)


# Global singleton
metrics = PrometheusMetrics()
