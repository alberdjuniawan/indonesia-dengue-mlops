import csv
from datetime import UTC, datetime
from pathlib import Path

from dengue_mlops.ingestion.skdr.discovery import build_pdf_url
from dengue_mlops.ingestion.skdr.downloader import download_pdf

RAW_DIR = Path("data/raw/skdr")
MANIFEST_PATH = Path("data/raw/skdr/manifest.csv")

FIELDNAMES = ["year", "week", "url", "status", "path", "error", "downloaded_at"]


def load_manifest() -> dict[tuple[int, int], dict]:
    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        return {(int(row["year"]), int(row["week"])): row for row in reader}


def save_manifest(records: dict[tuple[int, int], dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        writer.writeheader()

        for key in sorted(records):
            writer.writerow(records[key])


def download_reports(year: int, weeks: range, records: dict[tuple[int, int], dict]) -> None:
    for week in weeks:
        url = build_pdf_url(year, week)

        output_path = RAW_DIR / str(year) / f"M{week:02d}_{year}.pdf"

        print(f"[{year}-W{week:02d}]")

        result = download_pdf(url=url, output_path=output_path, year=year, week=week)

        print(f"status: {result.status}")

        records[(year, week)] = {
            "year": result.year,
            "week": result.week,
            "url": result.url,
            "status": result.status,
            "path": str(result.path) if result.path else "",
            "error": result.error or "",
            "downloaded_at": datetime.now(UTC).isoformat(),
        }


def main() -> None:
    records = load_manifest()

    download_reports(year=2025, weeks=range(1, 53), records=records)

    download_reports(year=2026, weeks=range(1, 53), records=records)

    save_manifest(records)

    print(f"\nManifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
