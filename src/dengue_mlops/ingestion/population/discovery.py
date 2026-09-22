from __future__ import annotations

from typing import Any

from dengue_mlops.ingestion.population.client import BPSClient
from dengue_mlops.ingestion.population.models import PopulationTable

POPULATION_SUBJECT_ID = 519

REQUIRED_TITLE_TERMS = ("jumlah penduduk", "menurut provinsi", "jenis kelamin")

EXCLUDED_TITLE_TERMS = (
    "proyeksi",
    "kabupaten",
    "kota",
    "kelompok umur",
    "umur",
    "generasi",
    "status perkawinan",
    "komuter",
)


def _normalized_title(title: str) -> str:
    return " ".join(title.lower().split())


def is_population_candidate(item: dict[str, Any]) -> bool:
    try:
        subject_id = int(item.get("id_subject", -1))
    except (TypeError, ValueError):
        return False

    if subject_id != POPULATION_SUBJECT_ID:
        return False

    title = _normalized_title(item.get("title", ""))

    if not all(term in title for term in REQUIRED_TITLE_TERMS):
        return False

    return not any(term in title for term in EXCLUDED_TITLE_TERMS)


def candidate_score(item: dict[str, Any]) -> int:
    title = _normalized_title(item.get("title", ""))
    score = 0

    if title == "jumlah penduduk menurut provinsi dan jenis kelamin":
        score += 100

    if "jenis kelamin" in title:
        score += 30

    if "jumlah penduduk" in title:
        score += 20

    if "menurut provinsi" in title:
        score += 20

    return score


def _build_table(item: dict[str, Any]) -> PopulationTable:
    latest_period = item.get("latest_period")

    return PopulationTable(
        table_id=str(item["id"]),
        title=str(item["title"]),
        subject_id=int(item["id_subject"]),
        latest_period=(int(latest_period) if latest_period is not None else None),
        last_update=str(item.get("last_update", "")),
        available_years=(),
    )


def discover_population_tables(client: BPSClient) -> list[PopulationTable]:
    first_page = client.list_table_statistics(page=1, perpage=100)

    metadata = first_page["data"][0]
    total_pages = int(metadata["pages"])

    candidates: list[dict[str, Any]] = list(first_page["data"][1])

    for page in range(2, total_pages + 1):
        payload = client.list_table_statistics(page=page, perpage=100)

        candidates.extend(payload["data"][1])

    matching = [item for item in candidates if is_population_candidate(item)]

    matching.sort(key=candidate_score, reverse=True)

    return [_build_table(item) for item in matching]
