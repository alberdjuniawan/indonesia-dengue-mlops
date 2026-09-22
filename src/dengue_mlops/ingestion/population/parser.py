from __future__ import annotations

from typing import Any

from dengue_mlops.ingestion.population.models import PopulationRow

TOTAL_TURVAR_LABEL = "jumlah"
INDONESIA_CODE = 9999


def _find_total_turvar(payload: dict[str, Any]) -> int:
    for item in payload.get("turvar", []):
        label = str(item.get("label", "")).strip().lower()

        if label == TOTAL_TURVAR_LABEL:
            return int(item["val"])

    raise ValueError("BPS table does not contain a 'Jumlah' turvar")


def _extract_var_code(payload: dict[str, Any]) -> int:
    variables = payload.get("var", [])

    if len(variables) != 1:
        raise ValueError(
            f"Expected exactly one variable in BPS detail response, got {len(variables)}"
        )

    return int(variables[0]["val"])


def _extract_year(payload: dict[str, Any]) -> tuple[int, int]:
    years = payload.get("tahun", [])

    if len(years) != 1:
        raise ValueError(f"Expected exactly one year in BPS detail response, got {len(years)}")

    year = years[0]

    return int(year["val"]), int(year["label"])


def _extract_year_unit_code(payload: dict[str, Any]) -> int:
    year_units = payload.get("turtahun", [])

    if len(year_units) != 1:
        raise ValueError(
            f"Expected exactly one turtahun entry in BPS detail response, got {len(year_units)}"
        )

    return int(year_units[0]["val"])


def _extract_population_value(
    datacontent: dict[str, Any],
    *,
    province_code: int,
    var_code: int,
    total_turvar: int,
    year_code: int,
    year_unit_code: int,
) -> int:
    expected_key = f"{province_code}{var_code}{total_turvar}{year_code}{year_unit_code}"

    if expected_key not in datacontent:
        raise ValueError(
            "Could not find BPS population value for "
            f"province_code={province_code}, "
            f"var={var_code}, "
            f"turvar={total_turvar}, "
            f"year={year_code}, "
            f"turtahun={year_unit_code}; "
            f"expected key={expected_key}"
        )

    return int(datacontent[expected_key])


def parse_population(payload: dict[str, Any]) -> list[PopulationRow]:
    var_code = _extract_var_code(payload)
    total_turvar = _find_total_turvar(payload)
    year_code, year = _extract_year(payload)
    year_unit_code = _extract_year_unit_code(payload)

    datacontent = payload.get("datacontent", {})

    if not datacontent:
        raise ValueError("BPS table contains no datacontent")

    rows: list[PopulationRow] = []

    for province in payload.get("vervar", []):
        province_code = int(province["val"])
        province_name = str(province["label"]).strip()

        if province_code == INDONESIA_CODE:
            continue

        population = _extract_population_value(
            datacontent,
            province_code=province_code,
            var_code=var_code,
            total_turvar=total_turvar,
            year_code=year_code,
            year_unit_code=year_unit_code,
        )

        rows.append(PopulationRow(province=province_name, year=year, population=population))

    return rows
