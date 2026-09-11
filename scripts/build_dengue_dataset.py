from __future__ import annotations

import contextlib
import csv
import io
from pathlib import Path

from dengue_mlops.ingestion.skdr.ocr import extract_dengue_ocr_table
from dengue_mlops.ingestion.skdr.page_locator import find_dengue_pages
from dengue_mlops.ingestion.skdr.validator import validate_dengue_rows

RAW_DIR = Path("data/raw/skdr")

OUTPUT_PATH = Path("data/processed/dengue.csv")

REPORT_PATH = Path("data/processed/dengue_ingestion_report.csv")


def _compact(value) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)

    return str(value)


def _extract_method(log: str) -> str:
    if "Extraction method: PDF text layer" in log:
        return "PDF_TEXT"

    if "Text layer incomplete; fallback to OCR" in log:
        return "OCR"

    return "UNKNOWN"


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _invalid_provinces(validation: dict) -> str:
    return _compact([item.get("province", "?") for item in validation.get("invalid_rows", [])])


def _discover_reports():
    reports = []

    if not RAW_DIR.exists():
        raise RuntimeError(f"RAW_DIR does not exist: {RAW_DIR}")

    for year_dir in sorted(RAW_DIR.iterdir()):
        if not year_dir.is_dir():
            continue

        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for pdf_path in sorted(year_dir.glob("*.pdf")):
            stem = pdf_path.stem.upper()

            if not stem.startswith("M"):
                continue

            try:
                week = int(stem.split("_")[0][1:])
            except (IndexError, ValueError):
                continue

            reports.append((pdf_path, year, week))

    return reports


def process_report(pdf_path: Path, year: int, week: int):
    dengue_pages = find_dengue_pages(pdf_path)

    if not dengue_pages:
        validation = {
            "status": "NO_PAGE",
            "row_count": 0,
            "missing_provinces": [],
            "duplicate_provinces": [],
            "invalid_rows": [],
        }

        return ([], validation, "UNKNOWN", "No dengue page found", None)

    page_number = int(dengue_pages[0])

    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            rows = extract_dengue_ocr_table(pdf_path=pdf_path, page_number=page_number)

    except Exception as exc:
        log = stdout_buffer.getvalue()

        validation = {
            "status": "ERROR",
            "row_count": 0,
            "missing_provinces": [],
            "duplicate_provinces": [],
            "invalid_rows": [],
        }

        return ([], validation, _extract_method(log), _error_message(exc), page_number)

    log = stdout_buffer.getvalue()

    validation = validate_dengue_rows(rows)

    method = _extract_method(log)

    if validation["status"] != "VALID":
        return ([], validation, method, "Validation failed", page_number)

    result = []

    for row in rows:
        result.append(
            {"year": year, "week": week, "province": row["province"], "cases": row["cases"]}
        )

    return (result, validation, method, "", page_number)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reports = _discover_reports()

    if not reports:
        raise RuntimeError("No SKDR PDF reports found.")

    all_rows: list[dict] = []
    report_rows: list[dict] = []

    valid_count = 0
    failed_count = 0

    total = len(reports)

    for index, (pdf_path, year, week) in enumerate(reports, start=1):
        label = f"{year} M{week:02d}"

        (rows, validation, method, message, page_number) = process_report(pdf_path, year, week)

        status = validation.get("status", "ERROR")

        is_valid = status == "VALID"

        missing = _compact(validation.get("missing_provinces", []))

        invalid = _invalid_provinces(validation)

        duplicate = _compact(validation.get("duplicate_provinces", []))

        if is_valid:
            valid_count += 1

            all_rows.extend(rows)

            print(f"[{index:>2}/{total}] {label:<10} OK   {len(rows):>2} rows  {method}")

        else:
            failed_count += 1

            if missing:
                detail = f"missing={missing}"

            elif invalid:
                detail = f"invalid={invalid}"

            elif duplicate:
                detail = f"duplicate={duplicate}"

            else:
                detail = message or "unknown error"

            print(
                f"[{index:>2}/{total}] "
                f"{label:<10} "
                f"FAIL "
                f"{status:<8} "
                f"rows="
                f"{validation.get('row_count', 0):>2} "
                f"{detail}"
            )

        report_rows.append(
            {
                "year": year,
                "week": week,
                "report": pdf_path.name,
                "page": (page_number if page_number is not None else ""),
                "extraction_method": method,
                "status": status,
                "row_count": validation.get("row_count", 0),
                "missing_provinces": missing,
                "duplicate_provinces": duplicate,
                "invalid_provinces": invalid,
                "message": message,
            }
        )

    if not all_rows:
        raise RuntimeError("No valid dengue data extracted.")

    all_rows.sort(key=lambda row: (int(row["year"]), int(row["week"]), str(row["province"])))

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["year", "week", "province", "cases"])

        writer.writeheader()

        writer.writerows(all_rows)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "year",
            "week",
            "report",
            "page",
            "extraction_method",
            "status",
            "row_count",
            "missing_provinces",
            "duplicate_provinces",
            "invalid_provinces",
            "message",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(report_rows)

    print()

    print("=" * 70)

    print("DATASET BUILD SUMMARY")

    print("=" * 70)

    print(f"Reports processed : {total}")

    print(f"Valid reports     : {valid_count}")

    print(f"Failed reports    : {failed_count}")

    print(f"Valid rows        : {len(all_rows)}")

    print(f"Dataset           : {OUTPUT_PATH}")

    print(f"Ingestion report  : {REPORT_PATH}")

    pdf_text_count = sum(
        1
        for item in report_rows
        if item["status"] == "VALID" and item["extraction_method"] == "PDF_TEXT"
    )

    ocr_count = sum(
        1
        for item in report_rows
        if item["status"] == "VALID" and item["extraction_method"] == "OCR"
    )

    print()

    print("EXTRACTION METHODS")

    print(f"PDF text layer   : {pdf_text_count}")

    print(f"OCR fallback     : {ocr_count}")

    if failed_count:
        print()

        print("FAILED REPORTS")

        for item in report_rows:
            if item["status"] == "VALID":
                continue

            detail = (
                item["missing_provinces"]
                or item["invalid_provinces"]
                or item["duplicate_provinces"]
                or item["message"]
                or "see ingestion report"
            )

            print(f"- {item['year']} M{int(item['week']):02d}: {item['status']} | {detail}")

    print("=" * 70)


if __name__ == "__main__":
    main()
