from pathlib import Path
from unittest.mock import patch

from scripts.ingest_data import (
    ResilientHTTPClient,
    clean_and_validate_dataset,
    run_ingestion,
    synthesize_period_data,
)


def test_synthesize_period_data():
    records = synthesize_period_data(2025, 10)
    assert len(records) == 38
    for r in records:
        assert r["year"] == 2025
        assert r["week"] == 10
        assert r["cases"] >= 0
        assert isinstance(r["province"], str)


def test_resilient_http_client_timeout():
    import requests

    client = ResilientHTTPClient(max_retries=1, timeout=0.01)
    # Using an unroutable IP/port or mock
    with patch.object(
        client.session, "get", side_effect=requests.exceptions.ConnectionError("Connection error")
    ):
        resp = client.fetch_url("http://non-existent-domain.test/file.pdf")
        assert resp is None


def test_clean_and_validate_dataset():
    raw_sample = [
        {"year": 2026, "week": 1, "province": "DKI JAKARTA", "cases": 120},
        {"year": 2026, "week": 1, "province": "DKI JAKARTA", "cases": 120},  # duplicate
        {"year": 2026, "week": 1, "province": "JAWA BARAT", "cases": -5},  # negative
        {"year": 2026, "week": 1, "province": "JAWA TIMUR", "cases": 999999},  # outlier
        {"year": 2026, "week": 1, "province": "UNKNOWN PROVINCE", "cases": 10},  # invalid
    ]
    cleaned, audit = clean_and_validate_dataset(raw_sample)

    assert audit["total_raw_rows"] == 5
    assert audit["deduplicated_rows"] == 1
    assert audit["clipped_outliers"] == 2
    assert audit["normalized_provinces"] == 2  # DKI JAKARTA normalized twice

    prov_names = [r["province"] for r in cleaned]
    assert "JAKARTA" in prov_names
    assert "UNKNOWN PROVINCE" not in prov_names

    cases_map = {r["province"]: r["cases"] for r in cleaned}
    assert cases_map["JAWA BARAT"] == 0
    assert cases_map["JAWA TIMUR"] == 5000


def test_run_ingestion_initial_and_continual(tmp_path: Path):
    raw_csv = tmp_path / "raw.csv"
    processed_csv = tmp_path / "clean.csv"
    audit_json = tmp_path / "audit.json"

    # Test initial ingestion
    with patch("scripts.ingest_data.fetch_remote_surveillance_record", return_value=None):
        run_ingestion(
            mode="initial", raw_path=raw_csv, processed_path=processed_csv, audit_path=audit_json
        )

    assert raw_csv.exists()
    assert processed_csv.exists()
    assert audit_json.exists()

    with open(raw_csv) as f:
        initial_lines = len(f.readlines())
    # 2024 (52) + 2025 (52) + 2026 (8) = 112 weeks * 38 provinces = 4256 + 1 header = 4257
    assert initial_lines == 4257

    # Test continual learning update
    with patch("scripts.ingest_data.fetch_remote_surveillance_record", return_value=None):
        run_ingestion(
            mode="continual", raw_path=raw_csv, processed_path=processed_csv, audit_path=audit_json
        )

    with open(raw_csv) as f:
        updated_lines = len(f.readlines())
    # 4 weeks * 38 = 152 additional rows -> 4257 + 152 = 4409
    assert updated_lines == 4409
