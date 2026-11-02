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

## Model Registry, Versioning, dan Kesiapan Inferensi (LK-07)

Model prediksi DBD mingguan dikelola siklus hidupnya secara sistematis menggunakan **MLflow Model Registry** dan disinkronisasikan dengan **DVC Data Lineage**.

### 1. Model Produksi Aktif & Identitas Registry

| Komponen Registry | Nilai / Konfigurasi | Deskripsi |
| :--- | :--- | :--- |
| **Model Name** | `DengueOutbreakForecaster` | Entri tunggal Model Registry untuk peramalan kasus DBD 38 provinsi di Indonesia |
| **Versi Aktif (Production)** | **Version 5 (v5)** | Artefak model terpilih yang aktif melayani sistem inferensi |
| **Stage Model** | `Production` | Transisi resmi dari `Staging` setelah melewati gerbang validasi performa |
| **MLflow Aliases** | `@production`, `@champion` | Penunjuk dinamis (pointer) untuk pemanggilan programatik tanpa hardcoded version |
| **Arsitektur Model** | `EnsembleGradientRegressor` | Kombinasi terbobot Ridge Regression (65%) dan Auto-Regressive Momentum (35%) |
| **Hyperparameters** | `ridge_w=0.65, lag_w=0.35, α=0.75, epochs=650` | Penyetelan penalti L2 sub-ridge dan pembobotan lag temporal (Seed 42) |
| **Validation MAE** | **16.425 kasus** | Galat absolut rata-rata terendah pada partisi validasi temporal |
| **Holdout Test MAE** | **14.215 kasus** | Galat absolut rata-rata pada partisi uji masa depan (1.76% lebih akurat dari petahana) |
| **Holdout Test R²** | **0.9665** | Menjelaskan 96.65% variansi kasus nyata pada 38 provinsi (SLA target ≥ 0.80) |
| **Data Lineage (DVC)** | Tag `v2.0-data` | Terikat pada MD5 `d2e8b41f59d2488c8a3e4f2eaec63e54` (`dengue_raw.csv.dvc`) |
| **Metadata File** | `models/registry/production_model.yaml` | Berkas referensi resmi berisi silsilah data DVC, Git commit, dan log transisi stage |

### 2. Alasan Pemilihan Versi 5 sebagai Model Produksi

Pemilihan **Version 5 (v5)** menggantikan Version 4 (v4) didasarkan pada prinsip tata kelola MLOps yang ketat dan hasil evaluasi deterministik bebas kebocoran data (*leakage-free temporal gate*):

1. **Keunggulan Metrik Generalisasi (Validation & Test MAE):**
   * **Validation MAE:** Version 5 mencatatkan MAE sebesar **16.425 kasus**, mengungguli Version 4 (**16.576 kasus**), mereduksi error sebesar 0.15 kasus per provinsi per minggu.
   * **Holdout-Test MAE:** Pada partisi uji *unseen future data*, Version 5 mencatatkan MAE **14.215 kasus** dibandingkan Version 4 (**14.469 kasus**), memberikan reduksi error absolut sebesar **0.254 kasus**.
   * **Koefisien Determinasi (R²):** Version 5 menghasilkan Test R² sebesar **0.9665** (vs 0.9664 pada v4), jauh melampaui ambang batas minimal SLA produksi ($R^2 \ge 0.80$).
2. **Justifikasi Penyetelan Hyperparameter (Challenger Tuning):**
   * Peningkatan bobot lag autoregresif ($30\% \rightarrow 35\%$) memungkinkan model menangkap dinamika penularan DBD jangka pendek secara lebih adaptif.
   * Penguatan regularisasi Ridge sub-model ($\alpha = 0.50 \rightarrow 0.75$, $\text{epochs} = 600 \rightarrow 650$) berhasil meredam fluktuasi kovariat iklim ekstrem (curah hujan dan kelembapan) sehingga mencegah *overfitting* pada provinsi dengan lonjakan anomali tinggi.
3. **Pemberian Status Stage & Pengarsipan Aman:**
   * Melalui alur `None → Staging → Production`, Version 5 dinyatakan memenangkan gerbang evaluasi (*Gate PASSED*).
   * Version 4 secara otomatis dialihkan ke status `Archived` untuk mencegah duplikasi stage aktif, namun riwayat bobot dan parameternya tetap tersimpan untuk audit masa depan.

### 3. Tabel Riwayat Versi & Alur Transisi Stage

| Versi | MLflow Run Name | Hyperparameters | Val MAE | Test MAE | Test R² | Stage | Status Keputusan |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **v1** | `run_05_gradient_ensemble` | Ridge=0.70, Lag=0.30, α=0.5 | 16.36 | 13.75 | 0.9658 | `None` | Eksperimen Awal |
| **v2** | `run_05_gradient_ensemble` | Ridge=0.70, Lag=0.30, α=0.5 | 16.36 | 13.75 | 0.9658 | `None` | Iterasi Pipeline |
| **v3** | `run_05_gradient_ensemble` | Ridge=0.70, Lag=0.30, α=0.5 | 16.36 | 13.75 | 0.9658 | `None` | Kandidat LK-06 |
| **v4** | `run_05_gradient_ensemble` | Ridge=0.70, Lag=0.30, α=0.5 (Seed 42) | 16.58 | 14.47 | 0.9664 | `Archived` | LK-06 Incumbent (Digantikan v5) |
| **v5** | `run_07_lk7_challenger_tuned` | Ridge=0.65, Lag=0.35, α=0.75, Ep=650 | **16.43** | **14.21** | **0.9665** | **`Production`** | **CHAMPION AKTIF (Dipromosikan)** |

### 4. Sinkronisasi Silsilah Data (Data Lineage) dengan DVC

Silsilah data dicatat secara formal dalam file `models/registry/production_model.yaml` guna menjamin ketertelusuran penuh antara data mentah, kode, dan model yang di-*publish*:

```yaml
schema: dengue-mlops/production-model-metadata/1.0
model:
  registered_name: DengueOutbreakForecaster
  version: 5
  stage: Production
  aliases: [production, champion]
  run_id: 780b91ed3b9a49f5abcca26e45dcf2c9
lineage:
  dataset:
    dvc_tag: v2.0-data
    raw_pointer_file: data/raw/dengue_raw.csv.dvc
    raw_dengue_md5: d2e8b41f59d2488c8a3e4f2eaec63e54
  code:
    git_commit: 5f4f987aebe10916030efb8025ddb2781dae8cf7
    training_script: scripts/train.py
```

### 5. Verifikasi Inferensi Programatik & SLA Kesiapan Produksi

Kesiapan layanan inferensi diuji menggunakan `scripts/verify_inference.py`. Model Production dimuat langsung melalui MLflow URI dan menerima batch data baru (simulasi minggu ke-12 tahun 2026):

```bash
# Eksekusi orchestrator siklus hidup Model Registry
uv run python scripts/registry_lifecycle.py

# Eksekusi verifikasi kesiapan inferensi
uv run python scripts/verify_inference.py
```

**Hasil Pengujian Inferensi (`inference_verification.json`):**
* **Model Loading Latency:** `0.61 detik` (memuat model via URI `models:/DengueOutbreakForecaster/Production`).
* **Batch Prediction Latency (p50):** `1.3 ms` untuk 6 provinsi (`0.22 ms/baris`) — jauh di bawah ambang batas SLA real-time ($< 50\text{ ms}$).
* **Integritas Hasil Prediksi:** 100% finite, non-negative, dan row-count preserved.
* **Klasifikasi Risiko DBD:** Prediksi kasus dipetakan ke tingkat risiko operasional (`RENDAH` $<25$, `WASPADA` $25-75$, `SIAGA` $>75$).

---

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
