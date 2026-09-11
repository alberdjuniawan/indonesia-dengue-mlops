from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from .ocr_parser import match_province

NUMBER_PATTERN = re.compile(r"^\d+(?:[.,]\d{3})*$")
MAX_ROW_Y_DISTANCE = 8
CHART_BOUNDARY_TOLERANCE = 3


def word_center(word):
    return ((float(word[0]) + float(word[2])) / 2, (float(word[1]) + float(word[3])) / 2)


def word_inside_bbox(word, chart_bbox) -> bool:
    if chart_bbox is None:
        return True

    cx, cy = word_center(word)

    x1, y1, x2, y2 = map(float, chart_bbox)

    return (
        x1 - CHART_BOUNDARY_TOLERANCE <= cx <= x2 + CHART_BOUNDARY_TOLERANCE
        and y1 - CHART_BOUNDARY_TOLERANCE <= cy <= y2 + CHART_BOUNDARY_TOLERANCE
    )


def word_inside_any_bbox(word, bboxes) -> bool:
    if not bboxes:
        return False

    cx, cy = word_center(word)

    for bbox in bboxes:
        x1, y1, x2, y2 = map(float, bbox)

        if (
            x1 - CHART_BOUNDARY_TOLERANCE <= cx <= x2 + CHART_BOUNDARY_TOLERANCE
            and y1 - CHART_BOUNDARY_TOLERANCE <= cy <= y2 + CHART_BOUNDARY_TOLERANCE
        ):
            return True

    return False


def parse_number(text: str):
    text = str(text).strip().replace(" ", "")

    if not NUMBER_PATTERN.fullmatch(text):
        return None

    normalized = text.replace(",", "").replace(".", "")

    try:
        return int(normalized)
    except ValueError:
        return None


def group_words_by_y(words):
    if not words:
        return []

    line_groups = {}

    for word in words:
        key = (int(word[5]), int(word[6]))

        line_groups.setdefault(key, []).append(word)

    groups = list(line_groups.values())

    groups.sort(key=_group_center_y)

    merged = []

    for group in groups:
        cy = _group_center_y(group)

        best_group = None
        best_distance = float("inf")

        for existing in merged:
            existing_y = _group_center_y(existing)

            distance = abs(cy - existing_y)

            if distance <= MAX_ROW_Y_DISTANCE and distance < best_distance:
                best_group = existing
                best_distance = distance

        if best_group is None:
            merged.append(list(group))
        else:
            best_group.extend(group)

    return merged


def _group_center_y(group) -> float:
    if not group:
        return 0.0

    return sum((float(word[1]) + float(word[3])) / 2 for word in group) / len(group)


def parse_dengue_page(
    pdf_path: Path, page_number: int, chart_bbox=None, excluded_bboxes=None
) -> list[dict]:
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Invalid page {page_number} for {pdf_path}. Total pages: {len(document)}"
            )

        page = document[page_number - 1]

        words = page.get_text("words")

    if not words:
        return []

    numeric_words = []
    text_words = []

    for word in words:
        text = str(word[4]).strip()

        if not text:
            continue

        if chart_bbox is not None and not word_inside_bbox(word, chart_bbox):
            continue

        if word_inside_any_bbox(word, excluded_bboxes):
            continue

        number = parse_number(text)

        if number is not None:
            numeric_words.append(
                {
                    "value": number,
                    "text": text,
                    "x1": float(word[0]),
                    "y1": float(word[1]),
                    "x2": float(word[2]),
                    "y2": float(word[3]),
                    "center_x": (float(word[0]) + float(word[2])) / 2,
                    "center_y": (float(word[1]) + float(word[3])) / 2,
                }
            )

        else:
            text_words.append(word)

    if not text_words:
        return []

    province_rows = group_words_by_y(text_words)

    results = []
    seen_provinces = set()

    for province_row in province_rows:
        province_row = sorted(province_row, key=lambda word: float(word[0]))

        row_text = " ".join(str(word[4]).strip() for word in province_row if str(word[4]).strip())

        if not row_text:
            continue

        province, _score = match_province(row_text, threshold=0.55, margin=0.0)

        if province is None:
            continue

        if province in seen_provinces:
            continue

        province_y = sum((float(word[1]) + float(word[3])) / 2 for word in province_row) / len(
            province_row
        )

        province_right = max(float(word[2]) for word in province_row)

        candidates = []

        for number_index, number in enumerate(numeric_words):
            y_distance = abs(number["center_y"] - province_y)

            if y_distance > MAX_ROW_Y_DISTANCE:
                continue

            if number["center_x"] < province_right:
                continue

            x_distance = abs(number["center_x"] - province_right)

            candidates.append(
                {
                    "index": number_index,
                    "number": number,
                    "y_distance": y_distance,
                    "x_distance": x_distance,
                }
            )

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item["y_distance"], item["x_distance"]))

        best = candidates[0]

        number = best["number"]

        results.append({"province": province, "cases": int(number["value"])})

        seen_provinces.add(province)

    return results
