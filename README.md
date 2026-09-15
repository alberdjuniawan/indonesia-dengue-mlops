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
python scripts/download_skdr.py
python scripts/build_dengue_dataset.py
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
python scripts/download_boundaries.py
python scripts/fetch_weather.py --help
python scripts/fetch_weather.py --skip-existing
# output: data/processed/weather_weekly.csv
# raw archives: data/raw/weather/era5/era5_land_timeseries_<PROVINCE>_<start>_<end>.zip
```

## Project Structure

```text
indonesia-dengue-mlops/
├── .devcontainer/
├── .github/
├── config/
├── data/
│   ├── raw/
│   │   ├── skdr/
│   │   │   ├── 2025/
│   │   │   └── 2026/
│   │   ├── boundaries/
│   │   │   └── indonesia_provinces.geojson
│   │   └── weather/
│   │       └── era5/
│   ├── interim/
│   ├── processed/
│   └── features/
├── models/
├── notebooks/
│   ├── eda/
│   ├── experiments/
│   └── reports/
├── scripts/
├── src/
│   └── dengue_mlops/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

Raw datasets and generated data are not stored in Git. `.gitkeep` files are used to preserve the directory structure.

## Tech Stack

- Python 3.12
- uv
- pandas, geopandas, xarray, h5netcdf
- PyMuPDF, OpenCV, PaddleOCR, EasyOCR, PyTorch
- cdsapi (Copernicus CDS), shapely
- pytest, Ruff
- GitHub Codespaces, GitHub Actions

## GitHub Codespaces

The development environment is configured using `.devcontainer/devcontainer.json`.

Create a Codespace from the repository and run:

```bash
uv sync
```

Then verify the environment:

```bash
python --version
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

### Code Formatting

Formatting is enforced by `ruff` (`line-length = 100`). Install hooks once:

```bash
uv run pre-commit install
```

## License

MIT License
