# Indonesia Dengue MLOps

An MLOps project for developing a dengue outbreak early warning system in Indonesia.

The system combines several data sources, including dengue surveillance data, weather data, and population data, to support dengue risk analysis and prediction.

## SKDR Data Pipeline

SKDR dengue surveillance reports are processed from raw PDF files into structured CSV data through the following pipeline:

```text
SKDR PDF -> Page Discovery -> Dengue Page Extraction -> PDF Text / OCR -> Province & Case Matching -> Validation -> Processed CSV
```

PDF text extraction is used when possible, with OCR as a fallback for reports that cannot be extracted completely.

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
│   │   ├── bmkg/
│   │   └── population/
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
- pandas
- PyMuPDF
- OpenCV
- PaddleOCR
- EasyOCR
- PyTorch
- pytest
- Ruff
- GitHub Codespaces
- GitHub Actions

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
