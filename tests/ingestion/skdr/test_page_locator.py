from pathlib import Path

from dengue_mlops.ingestion.skdr.page_locator import find_pages_by_keyword

PDF_PATH = Path("data/raw/skdr/2026/M34_2026.pdf")


def test_find_pages_by_keyword():
    pages = find_pages_by_keyword(PDF_PATH, "Suspek Dengue")

    assert pages
