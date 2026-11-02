from __future__ import annotations

from pathlib import Path

from dengue_mlops.modeling.dataset import load_dataset, split_features
from dengue_mlops.modeling.metrics import calculate_metrics
from dengue_mlops.modeling.models import (
    BaselineLagModel,
    EnsembleGradientRegressor,
    PyTorchTemporalRegressor,
    RidgeRegressionModel,
)

FEATURES_PATH = Path("data/features/features.csv")


class SeparateProvinceModels:
    """Trains 38 distinct local estimators tailored to each province's unique endemic scale."""

    def __init__(self, alpha: float = 0.5, lr: float = 0.005, epochs: int = 500) -> None:
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.name = "separate_province_models (38 local estimators)"
        self.province_models: dict[str, RidgeRegressionModel] = {}
        self.fallback = RidgeRegressionModel(alpha=alpha, lr=lr, epochs=epochs)

    def fit(self, X: list[list[float]], y: list[float], provinces: list[str]) -> None:
        by_prov: dict[str, tuple[list[list[float]], list[float]]] = {}
        for x_row, y_val, prov in zip(X, y, provinces, strict=False):
            by_prov.setdefault(prov, ([], []))
            by_prov[prov][0].append(x_row)
            by_prov[prov][1].append(y_val)

        for prov, (x_p, y_p) in by_prov.items():
            m = RidgeRegressionModel(alpha=self.alpha, lr=self.lr, epochs=self.epochs)
            m.fit(x_p, y_p)
            self.province_models[prov] = m

        self.fallback.fit(X, y)

    def predict(self, X: list[list[float]], provinces: list[str]) -> list[float]:
        preds = []
        for x_row, prov in zip(X, provinces, strict=False):
            m = self.province_models.get(prov, self.fallback)
            preds.extend(m.predict([x_row]))
        return preds


def run_benchmark() -> None:
    records = load_dataset(FEATURES_PATH)
    split = split_features(records)

    print("=" * 85)
    print("BENCHMARK EKSPERIMEN: GLOBAL POOLED MODEL VS SEPARATE PER-PROVINCE MODELS")
    print("=" * 85)
    print(
        f"Dataset Split: Train={len(split.train_x)} | Val={len(split.val_x)} | Test={len(split.test_x)}"
    )
    print(f"Jumlah Provinsi: {len(set(split.provinces_train))} provinsi\n")

    # 1. Baseline Lag-1
    base = BaselineLagModel(lag_idx=0)
    base_val_preds = base.predict(split.val_x)
    base_test_preds = base.predict(split.test_x)
    m_base_val = calculate_metrics(split.val_y, base_val_preds)
    m_base_test = calculate_metrics(split.test_y, base_test_preds)

    # 2. Global Ridge Regression (1 model untuk 38 provinsi)
    ridge = RidgeRegressionModel(alpha=1.0, lr=0.005, epochs=500)
    ridge.fit(split.train_x, split.train_y)
    ridge_val_preds = ridge.predict(split.val_x)
    ridge_test_preds = ridge.predict(split.test_x)
    m_ridge_val = calculate_metrics(split.val_y, ridge_val_preds)
    m_ridge_test = calculate_metrics(split.test_y, ridge_test_preds)

    # 3. Global Ensemble Gradient Regressor (1 model untuk 38 provinsi)
    ens = EnsembleGradientRegressor(ridge_weight=0.70, lag_weight=0.30)
    ens.fit(split.train_x, split.train_y)
    ens_val_preds = ens.predict(split.val_x)
    ens_test_preds = ens.predict(split.test_x)
    m_ens_val = calculate_metrics(split.val_y, ens_val_preds)
    m_ens_test = calculate_metrics(split.test_y, ens_test_preds)

    # 4. Separate Per-Province Models (38 model individual)
    sep = SeparateProvinceModels(alpha=0.5, lr=0.005, epochs=500)
    sep.fit(split.train_x, split.train_y, split.provinces_train)
    sep_val_preds = sep.predict(split.val_x, split.provinces_val)
    sep_test_preds = sep.predict(split.test_x, split.provinces_test)
    m_sep_val = calculate_metrics(split.val_y, sep_val_preds)
    m_sep_test = calculate_metrics(split.test_y, sep_test_preds)

    # 5. PyTorch Deep Neural Network (GPU accelerated)
    pytorch_mod = PyTorchTemporalRegressor(hidden_dim=64, lr=0.005, epochs=150, device="auto")
    pytorch_mod.fit(split.train_x, split.train_y)
    py_val_preds = pytorch_mod.predict(split.val_x)
    py_test_preds = pytorch_mod.predict(split.test_x)
    m_py_val = calculate_metrics(split.val_y, py_val_preds)
    m_py_test = calculate_metrics(split.test_y, py_test_preds)

    results = [
        ("Baseline (Lag-1 Persistence)", m_base_val, m_base_test),
        ("Global Ridge Regression", m_ridge_val, m_ridge_test),
        ("Global Ensemble Gradient Regressor", m_ens_val, m_ens_test),
        ("Separate Models (38 Local Estimators)", m_sep_val, m_sep_test),
        ("PyTorch Deep Neural Network (GPU)", m_py_val, m_py_test),
    ]

    print("+" + "-" * 38 + "+" + "-" * 22 + "+" + "-" * 22 + "+")
    print(
        f"| {'Arsitektur Model':<36} | {'Validation (2025 W27-52)':<20} | {'Test Holdout (2026 W01-12)':<20} |"
    )
    print(f"| {'':<36} | {'MAE':>6} {'RMSE':>7} {'R2':>6} | {'MAE':>6} {'RMSE':>7} {'R2':>6} |")
    print("+" + "-" * 38 + "+" + "-" * 22 + "+" + "-" * 22 + "+")

    for name, v, t in results:
        print(
            f"| {name:<36} | {v['mae']:>6.2f} {v['rmse']:>7.2f} {v['r2']:>6.4f} | {t['mae']:>6.2f} {t['rmse']:>7.2f} {t['r2']:>6.4f} |"
        )
    print("+" + "-" * 38 + "+" + "-" * 22 + "+" + "-" * 22 + "+")

    # Analisis per-provinsi: Global vs Separate
    print("\n--- PERBANDINGAN PER-PROVINSI: GLOBAL ENSEMBLE VS SEPARATE MODELS ---")
    print(
        f"{'Provinsi':<22} | {'Aktual Rata2':<12} | {'MAE Global':<12} | {'MAE Separate':<12} | {'Model Terbaik':<15}"
    )
    print("-" * 85)

    by_prov_test: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for prov, act, p_glob, p_sep in zip(
        split.provinces_test, split.test_y, ens_test_preds, sep_test_preds, strict=False
    ):
        by_prov_test.setdefault(prov, ([], [], []))
        by_prov_test[prov][0].append(act)
        by_prov_test[prov][1].append(p_glob)
        by_prov_test[prov][2].append(p_sep)

    top_provinces = [
        "JAWA BARAT",
        "JAWA TIMUR",
        "JAKARTA",
        "BALI",
        "DI YOGYAKARTA",
        "KALIMANTAN TIMUR",
        "PAPUA",
        "MALUKU",
        "SULAWESI SELATAN",
    ]
    for prov in top_provinces:
        if prov in by_prov_test:
            acts, globs, seps = by_prov_test[prov]
            avg_act = sum(acts) / len(acts)
            mae_g = sum(abs(a - g) for a, g in zip(acts, globs)) / len(acts)
            mae_s = sum(abs(a - s) for a, s in zip(acts, seps)) / len(acts)
            winner = "Separate" if mae_s < mae_g else "Global"
            print(
                f"{prov:<22} | {avg_act:>10.1f}   | {mae_g:>10.2f}   | {mae_s:>10.2f}   | {winner:<15}"
            )


if __name__ == "__main__":
    run_benchmark()
