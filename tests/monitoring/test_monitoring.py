from __future__ import annotations

from dengue_mlops.monitoring.drift import DriftDetector, calculate_psi
from dengue_mlops.monitoring.metrics_exporter import PrometheusMetrics
from dengue_mlops.monitoring.performance import PerformanceMonitor


def test_calculate_psi_identical_distributions() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    score = calculate_psi(data, data)
    assert score == 0.0


def test_drift_detector() -> None:
    ref = [[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]]
    cur = [[100.0, 200.0], [150.0, 250.0], [200.0, 300.0]]

    detector = DriftDetector(psi_threshold=0.20)
    report = detector.compute_feature_drift(ref, cur, ["f1", "f2"])

    assert report["status"] == "COMPLETED"
    assert report["drift_detected"] is True
    assert "f1" in report["drifted_features"]


def test_performance_monitor_alert() -> None:
    monitor = PerformanceMonitor(window_size=10, error_threshold=0.20)
    monitor.baseline_mae = 10.0

    # Low error
    for _ in range(5):
        monitor.record_feedback(actual=100.0, prediction=102.0)
    m = monitor.get_metrics()
    assert m["alert_triggered"] is False

    # High error exceeding threshold
    for _ in range(10):
        monitor.record_feedback(actual=100.0, prediction=150.0)
    m2 = monitor.get_metrics()
    assert m2["alert_triggered"] is True


def test_prometheus_metrics() -> None:
    prom = PrometheusMetrics()
    prom.record_request(duration=0.012, predicted_cases=250.0)
    prom.update_drift(0.04)
    prom.update_mae(12.5)

    text = prom.generate_metrics_text()
    assert "dengue_prediction_requests_total 1" in text
    assert "dengue_feature_drift_psi 0.04" in text
    assert "dengue_model_mae_rolling 12.5" in text
