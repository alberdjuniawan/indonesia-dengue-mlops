from __future__ import annotations

import math
from collections.abc import Sequence


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred, strict=False)) / len(y_true)
    return round(math.sqrt(mse), 3)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    val = sum(abs(t - p) for t, p in zip(y_true, y_pred, strict=False)) / len(y_true)
    return round(val, 3)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    if ss_tot == 0.0:
        return 1.0
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred, strict=False))
    return round(1.0 - (ss_res / ss_tot), 4)


def wape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    tot_true = sum(y_true)
    if tot_true == 0.0:
        return 0.0
    tot_err = sum(abs(t - p) for t, p in zip(y_true, y_pred, strict=False))
    return round((tot_err / tot_true) * 100.0, 2)


def calculate_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
