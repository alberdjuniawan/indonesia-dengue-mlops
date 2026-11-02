from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def calculate_psi(
    expected: Sequence[float], actual: Sequence[float], num_bins: int = 10, epsilon: float = 1e-4
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distribution.

    PSI < 0.10 : Little or no shift
    0.10 <= PSI < 0.25 : Moderate shift
    PSI >= 0.25 : Significant shift / Drift detected
    """
    if not expected or not actual:
        return 0.0

    # Determine bin edges based on expected (reference) distribution
    sorted_exp = sorted(expected)
    n_exp = len(sorted_exp)
    n_act = len(actual)

    # Generate quantile breakpoints
    breakpoints = []
    for i in range(1, num_bins):
        idx = int(i * n_exp / num_bins)
        breakpoints.append(sorted_exp[idx])
    breakpoints = sorted(set(breakpoints))

    if not breakpoints:
        return 0.0

    # Count occurrences in each bin
    def bin_counts(data: Sequence[float], breaks: list[float]) -> list[int]:
        counts = [0] * (len(breaks) + 1)
        for val in data:
            placed = False
            for b_idx, b_val in enumerate(breaks):
                if val <= b_val:
                    counts[b_idx] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
        return counts

    exp_counts = bin_counts(expected, breakpoints)
    act_counts = bin_counts(actual, breakpoints)

    psi_total = 0.0
    for exp_c, act_c in zip(exp_counts, act_counts, strict=False):
        exp_pct = max(exp_c / n_exp, epsilon)
        act_pct = max(act_c / n_act, epsilon)
        bin_psi = (act_pct - exp_pct) * math.log(act_pct / exp_pct)
        psi_total += bin_psi

    return round(psi_total, 4)


class DriftDetector:
    def __init__(self, psi_threshold: float = 0.20) -> None:
        self.psi_threshold = psi_threshold

    def compute_feature_drift(
        self,
        reference_data: Sequence[Sequence[float]],
        current_data: Sequence[Sequence[float]],
        feature_names: Sequence[str],
    ) -> dict[str, Any]:
        if not reference_data or not current_data:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False, "psi_scores": {}}

        num_features = len(feature_names)
        psi_scores: dict[str, float] = {}
        drifted_features: list[str] = []

        for j in range(num_features):
            col_name = feature_names[j]
            ref_col = [row[j] for row in reference_data]
            cur_col = [row[j] for row in current_data]

            score = calculate_psi(ref_col, cur_col)
            psi_scores[col_name] = score
            if score >= self.psi_threshold:
                drifted_features.append(col_name)

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        drift_detected = len(drifted_features) > 0

        return {
            "status": "COMPLETED",
            "drift_detected": drift_detected,
            "drift_method": "psi",
            "threshold": self.psi_threshold,
            "max_psi": max_psi,
            "drifted_features": drifted_features,
            "psi_scores": psi_scores,
        }

    def compute_prediction_drift(
        self, reference_predictions: Sequence[float], current_predictions: Sequence[float]
    ) -> dict[str, Any]:
        score = calculate_psi(reference_predictions, current_predictions)
        return {
            "prediction_psi": score,
            "drift_detected": score >= self.psi_threshold,
            "threshold": self.psi_threshold,
        }
