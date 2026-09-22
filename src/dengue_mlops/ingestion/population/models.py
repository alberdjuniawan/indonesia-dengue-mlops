from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationTable:
    table_id: str
    title: str
    subject_id: int
    latest_period: int | None
    last_update: str
    available_years: tuple[int, ...]


@dataclass(frozen=True)
class PopulationRow:
    province: str
    year: int
    population: int


@dataclass(frozen=True)
class PopulationSnapshot:
    table: PopulationTable
    rows: tuple[PopulationRow, ...]
