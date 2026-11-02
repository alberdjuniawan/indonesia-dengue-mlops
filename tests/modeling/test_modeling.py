from __future__ import annotations

from pathlib import Path

from dengue_mlops.modeling.metrics import calculate_metrics
from dengue_mlops.modeling.models import (
    BaselineLagModel,
    EnsembleGradientRegressor,
    RidgeRegressionModel,
    load_model_from_dict,
)
from dengue_mlops.modeling.registry import load_champion, register_champion


def test_metrics_calculation() -> None:
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 310.0]

    res = calculate_metrics(y_true, y_pred)
    assert res["rmse"] == 10.0
    assert res["mae"] == 10.0
    assert res["r2"] > 0.95
    assert res["wape"] < 10.0


def test_baseline_and_ridge_models() -> None:
    X = [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]]
    y = [11.0, 21.0, 31.0, 41.0]

    # Baseline
    base = BaselineLagModel(lag_idx=0)
    base_preds = base.predict(X)
    assert len(base_preds) == 4
    assert base_preds[0] == 10.0

    # Ridge
    ridge = RidgeRegressionModel(alpha=0.1, lr=0.01, epochs=200)
    ridge.fit(X, y)
    ridge_preds = ridge.predict(X)
    assert len(ridge_preds) == 4
    assert abs(ridge_preds[0] - 11.0) < 5.0

    # Ensemble
    ens = EnsembleGradientRegressor()
    ens.fit(X, y)
    ens_preds = ens.predict(X)
    assert len(ens_preds) == 4

    # Serialization roundtrip
    d = ens.to_dict()
    restored = load_model_from_dict(d)
    restored_preds = restored.predict(X)
    assert restored_preds == ens_preds

    # PyTorch Temporal Regressor
    from dengue_mlops.modeling.models import PyTorchTemporalRegressor

    nn_mod = PyTorchTemporalRegressor(hidden_dim=32, lr=0.01, epochs=100, device="auto")
    nn_mod.fit(X, y)
    nn_preds = nn_mod.predict(X)
    assert len(nn_preds) == 4
    assert abs(nn_preds[0] - 11.0) < 6.0

    # PyTorch serialization roundtrip
    nn_dict = nn_mod.to_dict()
    assert nn_dict["type"] == "PyTorchTemporalRegressor"
    restored_nn = load_model_from_dict(nn_dict)
    assert len(restored_nn.predict(X)) == 4

    # Separate Province Models
    from dengue_mlops.modeling.models import SeparateProvinceModels

    provinces = ["JAWA BARAT", "BALI", "JAWA BARAT", "BALI"]
    sep = SeparateProvinceModels(alpha=0.1, lr=0.01, epochs=100)
    sep.fit_with_provinces(X, y, provinces)
    sep_preds = sep.predict_with_provinces(X, provinces)
    assert len(sep_preds) == 4

    # Separate models serialization roundtrip
    sep_dict = sep.to_dict()
    assert sep_dict["type"] == "SeparateProvinceModels"
    restored_sep = load_model_from_dict(sep_dict)
    assert len(restored_sep.predict(X)) == 4


def test_registry_champion_roundtrip(tmp_path: Path) -> None:
    model = RidgeRegressionModel()
    metrics = {"rmse": 12.5, "mae": 8.2, "r2": 0.95, "wape": 10.0}
    features = ["cases_lag_1", "temp_lag_1"]

    register_champion(model, version="0.9.9", metrics=metrics, feature_names=features)
    _loaded_model, metadata = load_champion()

    assert metadata["version"] == "0.9.9"
    assert metadata["features"] == features
    assert metadata["metrics"]["r2"] == 0.95
