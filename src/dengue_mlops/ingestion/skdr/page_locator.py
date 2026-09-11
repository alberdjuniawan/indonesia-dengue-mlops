import re
from pathlib import Path

import pymupdf

DENGUE_PAGE_KEYWORD = "Kasus Suspek Dengue Berdasarkan Provinsi"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_pages_by_keyword(pdf_path: Path, keyword: str) -> list[int]:
    pages: list[int] = []

    normalized_keyword = normalize_text(keyword).casefold()

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = normalize_text(page.get_text("text"))

            if normalized_keyword in text.casefold():
                pages.append(page_number)

    return pages


def find_dengue_pages(pdf_path: Path) -> list[int]:
    return find_pages_by_keyword(pdf_path, DENGUE_PAGE_KEYWORD)
