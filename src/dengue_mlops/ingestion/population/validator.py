from __future__ import annotations

from collections.abc import Iterable

from dengue_mlops.ingestion.population.models import PopulationRow

EXPECTED_PROVINCES = frozenset(
    {
        "ACEH",
        "BALI",
        "BANGKA BELITUNG",
        "BANTEN",
        "BENGKULU",
        "DI YOGYAKARTA",
        "GORONTALO",
        "JAKARTA",
        "JAMBI",
        "JAWA BARAT",
        "JAWA TENGAH",
        "JAWA TIMUR",
        "KALIMANTAN BARAT",
        "KALIMANTAN SELATAN",
        "KALIMANTAN TENGAH",
        "KALIMANTAN TIMUR",
        "KALIMANTAN UTARA",
        "KEPULAUAN RIAU",
        "LAMPUNG",
        "MALUKU",
        "MALUKU UTARA",
        "NUSA TENGGARA BARAT",
        "NUSA TENGGARA TIMUR",
        "PAPUA",
        "PAPUA BARAT",
        "PAPUA BARAT DAYA",
        "PAPUA PEGUNUNGAN",
        "PAPUA SELATAN",
        "PAPUA TENGAH",
        "RIAU",
        "SULAWESI BARAT",
        "SULAWESI SELATAN",
        "SULAWESI TENGAH",
        "SULAWESI TENGGARA",
        "SULAWESI UTARA",
        "SUMATERA BARAT",
        "SUMATERA SELATAN",
        "SUMATERA UTARA",
    }
)

PROVINCE_ALIASES = {
    "KEP. BANGKA BELITUNG": "BANGKA BELITUNG",
    "KEP. RIAU": "KEPULAUAN RIAU",
    "DKI JAKARTA": "JAKARTA",
}


def normalize_province(name: str) -> str:
    normalized = " ".join(name.upper().split())
    return PROVINCE_ALIASES.get(normalized, normalized)


def normalize_rows(rows: Iterable[PopulationRow]) -> list[PopulationRow]:
    return [
        PopulationRow(
            province=normalize_province(row.province), year=row.year, population=row.population
        )
        for row in rows
    ]


def validate_population_rows(
    rows: Iterable[PopulationRow], *, expected_year: int | None = None
) -> list[PopulationRow]:
    normalized_rows = normalize_rows(rows)

    if not normalized_rows:
        raise ValueError("Population dataset is empty")

    provinces = [row.province for row in normalized_rows]

    if len(provinces) != len(set(provinces)):
        raise ValueError("Duplicate province rows detected")

    actual_provinces = frozenset(provinces)

    if actual_provinces != EXPECTED_PROVINCES:
        missing = sorted(EXPECTED_PROVINCES - actual_provinces)
        unexpected = sorted(actual_provinces - EXPECTED_PROVINCES)

        raise ValueError(f"Province set mismatch; missing={missing}, unexpected={unexpected}")

    if expected_year is not None:
        years = {row.year for row in normalized_rows}

        if years != {expected_year}:
            raise ValueError(f"Expected year {expected_year}, got {sorted(years)}")

    invalid_population = [row.province for row in normalized_rows if row.population <= 0]

    if invalid_population:
        raise ValueError(
            "Population must be positive; invalid provinces: " + ", ".join(invalid_population)
        )

    return sorted(normalized_rows, key=lambda row: row.province)
