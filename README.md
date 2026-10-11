# Indonesia Dengue MLOps

An MLOps project for developing a dengue outbreak early warning system in Indonesia.

The system combines several data sources, including dengue surveillance data, weather data, and population data, to support dengue risk analysis and prediction.

## SKDR Data Pipeline

SKDR dengue surveillance reports are processed from raw PDF files into structured CSV data through the following pipeline:

```text
SKDR PDF -> Download -> Page Discovery -> Render & Layout Detection -> PDF Text / OCR -> Province & Case Matching -> Validation -> dengue.csv
```

* **Source:** `https://surveilans.kemkes.go.id` via `discovery.py:build_pdf_url` -> `downloader.py:download_pdf` -> `data/raw/skdr/<year>/Mxx_<year>.pdf`.
* **Discovery:** `page_locator.py:find_dengue_pages` keyword `Kasus Suspek Dengue Berdasarkan Provinsi` (`page_locator.py:6`).
* **Layout:** `ocr.py:render_page` (RENDER_SCALE 2.0) -> `detect_layout` (PaddleOCR) -> `select_target_chart` -> `crop_target_chart` -> `mask_non_chart_regions` (excluded bboxes for text/OCR).
* **Text layer:** `parser.py:parse_dengue_page` with `chart_bbox` + `excluded_bboxes`, `NUMBER_PATTERN`, `MAX_ROW_Y_DISTANCE=8`, `match_province` (0.55).
* **OCR fallback:** `ocr_parser.py` + `ocr.py:extract_province_candidates` / `extract_number_candidates` / `recover_missing_province_candidates` -> `build_rows_from_candidates` (38 provinces, trigram similarity).
* **Validation:** `validator.py:validate_dengue_rows` checks 38 provinces, duplicates, invalid rows -> `VALID`/`PARTIAL`/`INVALID`/`IMAGE_ONLY`.

Run:

```bash
uv run python scripts/download_skdr.py
uv run python scripts/build_dengue_dataset.py
# outputs: data/processed/dengue.csv, data/processed/dengue_ingestion_report.csv
```

## Weather Data Pipeline

ERA5-Land reanalysis from Copernicus CDS is ingested per province and aligned to SKDR ISO weeks:

```text
Province Boundaries -> Representative Points -> CDS ERA5-Land (hourly, NetCDF) -> Parse & Derive (humidity, wind_speed) -> Weekly Aggregation (ISO week, UTC) -> Alignment with SKDR -> weather_weekly.csv
```

* **Source:** `reanalysis-era5-land-timeseries` via `cdsapi` (`2m_temperature`, `2m_dewpoint_temperature`, `surface_pressure`, `10m_u/v_component_of_wind`, `total_precipitation`).
* **Spatial:** `data/raw/boundaries/indonesia_provinces.geojson` -> `load_province_boundaries` -> `build_province_points` (representative point per province, EPSG:4326).
* **Parsing:** `src/dengue_mlops/ingestion/weather/parser.py` extracts CDS ZIP, merges NetCDF frames (`outer` join), derives `humidity` (Magnus) and `wind_speed`.
* **Aggregation:** `aggregation.py` groups hourly `valid_time` by ISO `year`/`week` -> `temperature_mean_c`, `humidity_mean`, `pressure_mean_pa`, `wind_speed_mean`, `precipitation_sum_mm` + `hour_count`.
* **Alignment:** `alignment.py` left-joins weekly weather to SKDR periods from `data/processed/dengue.csv` and flags `weather_complete`.

Run:

```bash
uv run python scripts/download_boundaries.py
uv run python scripts/fetch_weather.py --help
uv run python scripts/fetch_weather.py --skip-existing
# output: data/processed/weather_weekly.csv
# raw archives: data/raw/weather/era5/era5_land_timeseries_<PROVINCE>_<start>_<end>.zip
```

## Feature Engineering Pipeline

Surveillance, weather, and population tables are merged into a temporal feature matrix:

```text
dengue.csv + weather_weekly.csv + population.csv -> Lags (1..4) + Rolling Windows + Cyclical Week Encodings -> features.csv
```

* **Autoregressive lags:** `features/pipeline.py:engineer_features` calculates `cases_lag_1` to `cases_lag_4` and rolling window statistics (`mean`, `std`) per province.
* **Climate lags:** Rolling 4-week rainfall sums, lag-1 temperature, and humidity interactions.
* **Seasonality encoding:** Trigonometric transformation (`week_sin`, `week_cos`) to capture annual monsoon cycles.
* **Incidence normalization:** Cases per 100,000 population derived from BPS census figures.

Run:

```bash
uv run python scripts/build_features.py
# output: data/features/features.csv
```

## Modeling & Evaluation Pipeline

Models are trained using chronological train/validation/test splits without temporal data leakage:

```text
features.csv -> Chronological Split -> Baseline / Ridge / Ensemble / Separate / PyTorch -> Champion Registration -> champion.json
```

* **Global Ensemble Regressor:** Combines regularized gradient Ridge regression with autoregressive lag weighting (`models.py:EnsembleGradientRegressor`).
* **Separate Province Models:** 38 local estimators tailored to each province's endemic scale (`models.py:SeparateProvinceModels`). Low-incidence regions (Papua, Maluku) achieve >50% lower MAE compared to global pooling.
* **Deep Neural Regressor:** PyTorch MLP with dropout and Huber loss (`models.py:PyTorchTemporalRegressor`), supporting CUDA/GPU (`--device cuda|auto|cpu`).
* **Registry & Evaluation:** Champion-challenger evaluation gate based on test holdout MAE and $R^2$ (`registry.py:register_champion`).

Run:

```bash
# train and register champion model
uv run python scripts/train.py --device auto

# run head-to-head benchmark (Global vs Separate Models)
uv run python scripts/benchmark_separate_vs_global.py

# evaluate model performance against holdout
uv run python scripts/evaluate.py
# outputs: models/champion.json, models/evaluation_report.json
```

## Serving & API

The trained model is served as a production FastAPI service:

```bash
# run locally
uv run uvicorn dengue_mlops.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
* `GET /health` - Service health and active model version.
* `POST /predict` - Single province weekly dengue case forecast.
* `POST /predict/batch` - Multi-province batch forecasting.
* `POST /feedback` - Ingest ground truth observations for online evaluation.
* `GET /metrics` - Prometheus metrics exporter.

Example single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "province": "JAWA BARAT",
    "year": 2026,
    "week": 14,
    "cases_lag_1": 650.0,
    "cases_lag_2": 620.0,
    "cases_lag_3": 580.0,
    "cases_lag_4": 510.0,
    "precipitation_sum_mm": 110.5,
    "temp_mean_lag_1": 26.8,
    "humidity_mean_lag_1": 82.0
  }'
```

## Monitoring & Observability

Observability tracks data drift, prediction latency, and model accuracy degradation:

* **Data Drift Detection:** Population Stability Index (`drift.py:calculate_psi`) monitors distribution shifts in incoming feature payloads.
* **Prometheus Metrics:** Tracks inference count, latency histograms, error counters, and drift status.
* **Automated Retraining:** Triggered when PSI exceeds 0.20 or when ground truth MAE degrades beyond threshold (`scripts/trigger_retraining.py`).

Run:

```bash
uv run python scripts/trigger_retraining.py
```

## Containerization

Build and run using Docker:

```bash
# build container image
docker build -t indonesia-dengue-mlops/api:latest .

# run container
docker run -p 8000:8000 indonesia-dengue-mlops/api:latest
```

Or deploy with docker-compose:

```bash
docker compose up -d
```

## Experimentation Notebooks

Structured notebooks documenting exploration, ingestion trial-and-error, and model benchmarks:

* `notebooks/experiments/01_skdr_ingestion_experiments.ipynb` - Step-by-step development of layout detection, OCR noise filtering, and 38-province matching.
* `notebooks/eda/01_exploratory_data_analysis.ipynb` - Spatial distribution and climate correlation analysis.
* `notebooks/experiments/02_data_preprocessing_and_interim.ipynb` - Missing surveillance gap audits and interim reconstruction.
* `notebooks/experiments/03_feature_engineering_pipeline.ipynb` - Lag features, rolling windows, and cyclical calendar transforms.
* `notebooks/experiments/04_model_training_and_gpu_benchmarks.ipynb` - GPU training benchmarks and Global vs Separate model evaluations.

## Project Structure

```text
indonesia-dengue-mlops/
├── .devcontainer/
├── .dvc/
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
│   └── base.yaml
├── data/
│   ├── raw/
│   │   ├── skdr/
│   │   │   ├── 2025/
│   │   │   ├── 2026/
│   │   │   └── manifest.csv
│   │   ├── boundaries/
│   │   │   └── indonesia_provinces.geojson
│   │   └── weather/
│   │       └── era5/
│   ├── interim/
│   ├── processed/
│   │   ├── dengue.csv
│   │   └── weather_weekly.csv
│   └── features/
│       └── features.csv
├── models/
│   └── champion.json
├── notebooks/
│   ├── eda/
│   └── experiments/
├── scripts/
│   ├── download_skdr.py
│   ├── build_dengue_dataset.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark_separate_vs_global.py
│   └── trigger_retraining.py
├── src/
│   └── dengue_mlops/
│       ├── ingestion/
│       ├── features/
│       ├── modeling/
│       ├── serving/
│       └── monitoring/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Data Version Control (DVC) & Manajemen Versi Data

Proyek ini mengintegrasikan **DVC (Data Version Control)** untuk melacak silsilah dataset (*data lineage*) dan memfasilitasi *continual learning* tanpa membebani repositori Git dengan file berukuran besar.

```text
[Sumber Dinamis / Kemenkes SKDR]
             │
             ▼
┌───────────────────────────┐
│   scripts/ingest_data.py  │  ◄── Resilient HTTP Client (Retry, Backoff, Timeout)
└─────────────┬─────────────┘
              │
              ├───► [Pembersihan Sistematis] ──► data/processed/dengue.csv (Modeling Ready)
              │
              ▼
   data/raw/dengue_raw.csv (Data Mentah)
              │
              ├───► DVC Cache (.dvc/cache) ──► Remote Storage (/tmp/dvc_remote, MinIO/S3)
              │
              ▼
   data/raw/dengue_raw.csv.dvc (Pointer Hash ~100 bytes)
              │
              ▼
         Git Commits & Tags (v1.0-data, v2.0-data)
```

### Mengapa Versioning Dataset Krusial untuk Reproduksibilitas Model?

1. **Persamaan Fundamental Machine Learning:**
   Dalam software engineering konvensional, perilaku aplikasi ditentukan oleh kode: $\text{App} = f(\text{Code})$. Namun pada Machine Learning, model merupakan fungsi dari kode, data, dan konfigurasi:
   $$\text{Model} = f(\text{Data}, \text{Code}, \text{Hyperparameters})$$
   Jika hanya kode yang dilacak oleh Git sementara data dibiarkan berubah di filesystem, eksperimen model menjadi **tidak dapat direproduksi (*unreproducible*)**. Melacak hash data melalui DVC yang di-commit bersama hash commit Git menjamin bahwa siapapun dapat merekonstruksi kondisi pelatihan yang 100% identik di masa mendatang.

2. **Dinamika Non-Stasioner & Concept/Data Drift:**
   Surveilans penyakit demam berdarah (*dengue*) di Indonesia sangat dipengaruhi oleh anomali iklim tahunan (El Niño, IOD) dan musim hujan. Distribusi kasus berubah seiring waktu (*temporal distribution shift*). Model v1.0 yang dilatih dengan data hingga 2026-W08 memiliki karakteristik performa yang berbeda dengan model v2.0 (hingga 2026-W12). Dengan DVC, kita memiliki rekaman silsilah (*data lineage*) yang jelas antara versi model tertentu dengan distribusi data spesifik yang menghasilkannya.

3. **Efisiensi Repositori & Pemisahan Tanggung Jawab:**
   Git didesain untuk pelacakan teks berbasis baris (*delta compression*). Menyimpan file CSV/PDF/binary berukuran besar secara langsung di Git akan menyebabkan pembengkakan ukuran repositori (*repository bloat*), memperlambat operasi *clone/pull*, dan melanggar batas ukuran remote host (GitHub 100MB limit). DVC menyelesaikan ini dengan memisahkan metadata kecil (pointer file `.dvc` berukuran ~100 byte yang dikelola Git) dari isi data sebenarnya (dikelola oleh DVC di Content-Addressable Storage dan di-push ke cloud/object storage seperti MinIO, AWS S3, atau GCP).

4. **Continual Learning & Audit Kualitas Data:**
   Dalam skenario produksi, data surveilans mingguan baru terus masuk. Jika suatu batch data baru mengalami korupsi atau inkonsistensi pelaporan dari faskes, DVC memungkinkan *instant time travel* / *rollback* ke versi dataset stabil sebelumnya dengan satu perintah `dvc checkout`.

---

### Alur Kerja & Standardisasi Instruksi DVC

#### 1. Inisialisasi DVC & Konfigurasi Remote
```bash
# Inisialisasi DVC pada repositori Git
uv run dvc init

# Konfigurasi remote storage lokal (mensimulasikan external storage mount)
uv run dvc remote add -d local_storage /tmp/indonesia_dengue_dvc_storage

# Konfigurasi remote storage MinIO / S3 (opsional untuk lingkungan cloud)
uv run dvc remote add minio s3://dengue-data-registry/dvc-storage
uv run dvc remote modify minio endpointurl http://localhost:9000

# Commit konfigurasi DVC ke Git
git add .dvc .dvcignore .gitignore
git commit -m "chore(dvc): initialize DVC repository and remote storage configuration"
```

#### 2. Tracking Dataset Awal (v1.0 Baseline)
Jalankan skrip ingesti untuk mengambil dataset surveilans baseline (2024-W01 s.d. 2026-W08):
```bash
uv run python scripts/ingest_data.py --mode initial
# Output: data/raw/dengue_raw.csv (4,256 baris)
# Clean:  data/processed/dengue.csv (4,256 baris)

# Lacak dataset mentah menggunakan DVC
uv run dvc add data/raw/dengue_raw.csv

# Push data ke remote storage
uv run dvc push -r local_storage data/raw/dengue_raw.csv.dvc

# Simpan metadata pointer ke Git dan beri tag versi
git add data/raw/dengue_raw.csv.dvc scripts/ingest_data.py
git commit -m "feat(dvc): track initial raw dengue surveillance dataset v1.0 with DVC"
git tag -a v1.0-data -m "Initial baseline dengue raw dataset (2024-W01 to 2026-W08)"
```
Hash pointer awal yang dihasilkan: `15c69a33b875b5217a48a7ddd372b85d` (247,065 bytes).

#### 3. Simulasi Continual Learning (Penambahan Data Baru)
Simulasikan penerimaan batch data surveilans terbaru (2026-W09 s.d. 2026-W12):
```bash
uv run python scripts/ingest_data.py --mode continual
# Menambahkan 152 baris data baru -> Total 4,408 baris

# Verifikasi deteksi perubahan oleh DVC
uv run dvc status data/raw/dengue_raw.csv.dvc
# Output:
# data/raw/dengue_raw.csv.dvc:
#     changed outs:
#         modified: data/raw/dengue_raw.csv
```

#### 4. Versioning Versi Baru (v2.0 Update)
Lacak dataset yang telah diperbarui dan amati pembaruan hash MD5:
```bash
# Perbarui pelacakan DVC
uv run dvc add data/raw/dengue_raw.csv

# Push versi baru ke remote storage
uv run dvc push -r local_storage data/raw/dengue_raw.csv.dvc

# Commit pointer DVC baru ke Git dan beri tag versi
git add data/raw/dengue_raw.csv.dvc
git commit -m "feat(dvc): update raw dengue dataset with continual learning batch v2.0"
git tag -a v2.0-data -m "Continual learning batch update (2024-W01 to 2026-W12)"
```
Hash pointer baru yang dihasilkan: `d2e8b41f59d2488c8a3e4f2eaec63e54` (255,907 bytes).

#### 5. Audit & Diff Perbandingan Antar Versi Data
Bandingkan perbedaan metadata data antar dua tag/commit:
```bash
# Periksa perbedaan metadata antar versi dataset
uv run dvc diff v1.0-data v2.0-data --show-hash

# Tampilkan dalam format tabel Markdown
uv run dvc diff v1.0-data v2.0-data --show-hash --md
```
Output tabel perbandingan:
| Status | Hash | Path |
| :--- | :--- | :--- |
| **modified** | `15c69a33..d2e8b41f` | `data/raw/dengue_raw.csv` |

#### 6. Time Travel & Verifikasi Reproduksibilitas
Untuk mengembalikan data ke versi 1.0 kapan saja:
```bash
git checkout v1.0-data data/raw/dengue_raw.csv.dvc
uv run dvc checkout data/raw/dengue_raw.csv.dvc
wc -l data/raw/dengue_raw.csv  # Kembali tepat 4,257 baris

# Kembali ke versi terbaru (v2.0):
git checkout main data/raw/dengue_raw.csv.dvc
uv run dvc checkout data/raw/dengue_raw.csv.dvc
wc -l data/raw/dengue_raw.csv  # Tepat 4,409 baris
```

---

### Ingesti Dinamis & Pembersihan Data Sistematis (`ingest_data.py`)

Skrip `scripts/ingest_data.py` dirancang untuk memenuhi standar ketat *Production-grade Data Engineering*:
* **Pengambilan Data Dinamis Otomatis:** Mendukung penarikan data mingguan dinamis melalui parameter `--mode` (`initial`, `continual`, `periodic`) dengan sinkronisasi kalender ISO epidemiologi.
* **Resiliensi Jaringan & Error Handling:** Dilengkapi `ResilientHTTPClient` yang menerapkan *Exponential Backoff* (faktor 0.5s), *Connection Timeout* (5.0s), *Automatic Retry* untuk status HTTP 429/5xx, serta penanganan *exception* koneksi tanpa menghentikan pipeline.
* **Pembersihan Data Sistematis:**
  1. *Normalisasi Entitas Wilayah:* Memetakan variasi nama provinsi ke 38 provinsi kanonikal Kemenkes/BPS (misal `DKI JAKARTA` -> `JAKARTA`).
  2. *Imputasi Nilai Hilang:* Deteksi celah pelaporan surveilans dan imputasi *temporal rolling forward-fill*.
  3. *Penanganan Outlier & Integritas Nilai:* Membatasi *negative values* ($cases \ge 0$) dan meredam lonjakan anomali ekstrem.
  4. *Audit Trail & Data Health:* Menghasilkan laporan audit kualitas data di `data/interim/data_cleaning_audit.json` dengan skor integritas 100%.

## Tech Stack

- Python 3.12
- uv
- pandas, scikit-learn, PyTorch
- PyMuPDF, OpenCV, PaddleOCR, EasyOCR
- cdsapi, shapely
- FastAPI, Uvicorn, Pydantic
- Prometheus Client
- Docker, Docker Compose
- pytest, Ruff
- GitHub Actions

## Development

Set up the environment with `uv`:

```bash
uv sync --dev
```

Run test suite and linters:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## License

MIT License
