from pathlib import Path

from dengue_mlops.ingestion.skdr.parser import parse_dengue_page

PDF_PATH = Path("data/raw/skdr/2026/M34_2026.pdf")


def test_parse_dengue_page():
    rows = parse_dengue_page(PDF_PATH, 42)

    assert isinstance(rows, list)

    for row in rows:
        assert "province" in row
        assert "cases" in row
