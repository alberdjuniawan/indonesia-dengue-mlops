from dengue_mlops.ingestion.population.models import PopulationRow
from dengue_mlops.ingestion.population.validator import (
    EXPECTED_PROVINCES,
    normalize_rows,
    validate_population_rows,
)


def _build_rows() -> list[PopulationRow]:
    return [
        PopulationRow(province=province, year=2025, population=1_000_000)
        for province in sorted(EXPECTED_PROVINCES)
    ]


def test_expected_province_count() -> None:
    assert len(EXPECTED_PROVINCES) == 38


def test_normalize_bps_aliases() -> None:
    rows = [
        PopulationRow("DKI JAKARTA", 2025, 10_000_000),
        PopulationRow("KEP. BANGKA BELITUNG", 2025, 1_000_000),
    ]

    normalized = normalize_rows(rows)

    assert normalized[0].province == "JAKARTA"
    assert normalized[1].province == "BANGKA BELITUNG"


def test_validate_exact_38_provinces() -> None:
    rows = _build_rows()

    validated = validate_population_rows(rows, expected_year=2025)

    assert len(validated) == 38
    assert {row.province for row in validated} == EXPECTED_PROVINCES


def test_validate_rejects_missing_province() -> None:
    rows = _build_rows()[:-1]

    try:
        validate_population_rows(rows, expected_year=2025)
    except ValueError as exc:
        assert "Province set mismatch" in str(exc)
    else:
        raise AssertionError("Expected province-set validation to fail")


def test_validate_rejects_invalid_population() -> None:
    rows = _build_rows()
    rows[0] = PopulationRow(rows[0].province, rows[0].year, 0)

    try:
        validate_population_rows(rows, expected_year=2025)
    except ValueError as exc:
        assert "Population must be positive" in str(exc)
    else:
        raise AssertionError("Expected population validation to fail")


def test_normalize_kepulauan_riau_alias() -> None:
    rows = [PopulationRow("KEP. RIAU", 2025, 2_000_000)]

    normalized = normalize_rows(rows)

    assert normalized[0].province == "KEPULAUAN RIAU"
