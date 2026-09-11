from pathlib import Path

from dengue_mlops.ingestion.skdr.page_locator import find_pages_by_keyword

PDF_PATH = Path("tests/fixtures/sample_skdr.pdf")


def test_find_pages_by_keyword():
    pages = find_pages_by_keyword(PDF_PATH, "Suspek Dengue")

    assert pages
