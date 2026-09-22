from __future__ import annotations

import argparse

from dengue_mlops.ingestion.population.client import BPSClient
from dengue_mlops.ingestion.population.service import ingest_population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the processed population dataset even if the snapshot already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with BPSClient() as client:
        report = ingest_population(client=client, force=args.force)

    print()
    print("=" * 70)
    print("Population ingestion completed")
    print(f"Status: {report['status']}")
    print(f"Table: {report['title']}")
    print(f"Table ID: {report['table_id']}")
    print(f"Year: {report['year']}")
    print(f"Last update: {report['last_update']}")
    print(f"Provinces: {report['province_count']}")
    print(f"Output: {report['processed_output']}")


if __name__ == "__main__":
    main()
