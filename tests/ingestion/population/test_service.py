from dengue_mlops.ingestion.population.models import PopulationTable
from dengue_mlops.ingestion.population.service import _select_best_candidate


def _candidate(*, table_id: str, year: int, last_update: str, score: int = 150):
    table = PopulationTable(
        table_id=table_id,
        title=("Jumlah Penduduk menurut Provinsi dan Jenis Kelamin"),
        subject_id=519,
        latest_period=year,
        last_update=last_update,
        available_years=(year,),
    )

    return (table, {}, [], score)


def test_selection_prefers_latest_year() -> None:
    older = _candidate(table_id="older", year=2025, last_update="2026-07-01 00:00:00")

    newer = _candidate(table_id="newer", year=2026, last_update="2027-01-01 00:00:00")

    selected = _select_best_candidate([older, newer])

    assert selected[0].table_id == "newer"


def test_selection_prefers_latest_update_for_same_year() -> None:
    older = _candidate(table_id="older", year=2026, last_update="2027-01-01 00:00:00")

    newer = _candidate(table_id="newer", year=2026, last_update="2027-03-01 00:00:00")

    selected = _select_best_candidate([older, newer])

    assert selected[0].table_id == "newer"


def test_selection_uses_score_when_year_and_update_match() -> None:
    lower_score = _candidate(
        table_id="lower-score", year=2026, last_update="2027-03-01 00:00:00", score=100
    )

    higher_score = _candidate(
        table_id="higher-score", year=2026, last_update="2027-03-01 00:00:00", score=150
    )

    selected = _select_best_candidate([lower_score, higher_score])

    assert selected[0].table_id == "higher-score"
