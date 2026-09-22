from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dengue_mlops.ingestion.population.client import BPSClient
from dengue_mlops.ingestion.population.discovery import candidate_score, discover_population_tables
from dengue_mlops.ingestion.population.models import PopulationRow, PopulationTable
from dengue_mlops.ingestion.population.parser import parse_population
from dengue_mlops.ingestion.population.validator import validate_population_rows

RAW_DIR = Path("data/raw/population/bps")
PROCESSED_OUTPUT = Path("data/processed/population.csv")
REPORT_OUTPUT = Path("data/processed/population_ingestion_report.json")


ValidCandidate = tuple[PopulationTable, dict[str, Any], list[PopulationRow], int]


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _resolve_table(
    candidate: PopulationTable, payload: dict[str, Any], rows: list[PopulationRow]
) -> tuple[PopulationTable, list[PopulationRow]]:
    available_years = tuple(sorted(int(year) for year in payload.get("available_years", [])))

    if not available_years:
        raise ValueError("Table has no available years")

    latest_year = max(available_years)

    validated_rows = validate_population_rows(rows, expected_year=latest_year)

    if len(validated_rows) != 38:
        raise ValueError(f"Expected 38 provinces after validation, got {len(validated_rows)}")

    table = PopulationTable(
        table_id=candidate.table_id,
        title=candidate.title,
        subject_id=candidate.subject_id,
        latest_period=latest_year,
        last_update=candidate.last_update,
        available_years=available_years,
    )

    return table, validated_rows


def _select_best_candidate(candidates: list[ValidCandidate]) -> ValidCandidate:
    if not candidates:
        raise ValueError("No valid BPS population candidates available")

    return max(
        candidates,
        key=lambda item: (
            item[0].latest_period if item[0].latest_period is not None else -1,
            item[0].last_update,
            item[3],
            item[0].table_id,
        ),
    )


def _select_candidate(
    client: BPSClient,
) -> tuple[PopulationTable, dict[str, Any], list[PopulationRow]]:
    candidates = discover_population_tables(client)

    if not candidates:
        raise ValueError("No BPS population table candidates found")

    valid_candidates: list[ValidCandidate] = []
    errors: list[str] = []

    for candidate in candidates:
        try:
            payload = client.view_table_statistics(candidate.table_id)

            rows = parse_population(payload)

            resolved, validated_rows = _resolve_table(candidate, payload, rows)

            valid_candidates.append(
                (resolved, payload, validated_rows, candidate_score({"title": resolved.title}))
            )

        except (ValueError, RuntimeError, KeyError) as exc:
            errors.append(f"{candidate.table_id} ({candidate.title}): {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - fallback for unexpected BPS payload errors
            errors.append(f"{candidate.table_id} ({candidate.title}): {type(exc).__name__}: {exc}")

    if not valid_candidates:
        raise ValueError("No BPS population candidate passed validation:\n" + "\n".join(errors))

    selected = _select_best_candidate(valid_candidates)

    return (selected[0], selected[1], selected[2])


def _snapshot_path(table: PopulationTable) -> Path:
    if table.latest_period is None:
        raise ValueError("Cannot create snapshot path without a year")

    year = table.latest_period
    table_tag = _safe_filename(table.table_id)
    update_tag = _safe_filename(table.last_update or "unknown_update")

    return RAW_DIR / str(year) / (f"population_{year}_{table_tag}_{update_tag}.json")


def _save_raw_snapshot(
    *, table: PopulationTable, payload: dict[str, Any], overwrite: bool = False
) -> tuple[Path, bool]:
    output_path = _snapshot_path(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existed = output_path.exists()

    if not existed or overwrite:
        snapshot = {
            "fetched_at": datetime.now(UTC).isoformat(),
            "table": {
                "id": table.table_id,
                "title": table.title,
                "subject_id": table.subject_id,
                "latest_period": table.latest_period,
                "last_update": table.last_update,
                "available_years": list(table.available_years),
            },
            "response": payload,
        }

        output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path, existed


def _load_snapshot_rows(path: Path) -> tuple[PopulationTable, list[PopulationRow]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))

    table_data = snapshot["table"]
    payload = snapshot["response"]

    table = PopulationTable(
        table_id=str(table_data["id"]),
        title=str(table_data["title"]),
        subject_id=int(table_data["subject_id"]),
        latest_period=int(table_data["latest_period"]),
        last_update=str(table_data["last_update"]),
        available_years=tuple(int(year) for year in table_data["available_years"]),
    )

    rows = validate_population_rows(parse_population(payload), expected_year=table.latest_period)

    return table, rows


def _rebuild_processed_dataset(
    *, current_table: PopulationTable, current_rows: list[PopulationRow]
) -> None:
    current_rows = validate_population_rows(current_rows, expected_year=current_table.latest_period)

    latest_rows: dict[tuple[str, int], tuple[str, PopulationRow]] = {}

    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.glob("*/*.json")):
            try:
                table, rows = _load_snapshot_rows(path)
            except Exception:  # noqa: BLE001, S112 - skip corrupted snapshots
                continue

            for row in rows:
                key = (row.province, row.year)

                previous = latest_rows.get(key)

                if previous is None or table.last_update > previous[0]:
                    latest_rows[key] = (table.last_update, row)

    for row in current_rows:
        key = (row.province, row.year)

        previous = latest_rows.get(key)

        if previous is None or current_table.last_update >= previous[0]:
            latest_rows[key] = (current_table.last_update, row)

    frame = pd.DataFrame(
        [
            {"province": row.province, "year": row.year, "population": row.population}
            for _, row in latest_rows.values()
        ]
    )

    if frame.empty:
        raise ValueError("No population rows available for processed dataset")

    frame = frame.sort_values(["year", "province"]).reset_index(drop=True)

    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(PROCESSED_OUTPUT, index=False)


def ingest_population(*, client: BPSClient, force: bool = False) -> dict[str, Any]:
    (table, payload, rows) = _select_candidate(client)

    raw_path, already_exists = _save_raw_snapshot(table=table, payload=payload, overwrite=force)

    changed = force or not already_exists

    if changed:
        _rebuild_processed_dataset(current_table=table, current_rows=rows)

    report = {
        "status": ("updated" if changed else "unchanged"),
        "table_id": table.table_id,
        "title": table.title,
        "year": table.latest_period,
        "last_update": table.last_update,
        "available_years": list(table.available_years),
        "province_count": len(rows),
        "raw_snapshot": str(raw_path),
        "processed_output": str(PROCESSED_OUTPUT),
    }

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    REPORT_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
