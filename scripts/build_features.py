from __future__ import annotations

import argparse
from pathlib import Path

from dengue_mlops.features.pipeline import build_and_save_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features for dengue ML models")
    parser.add_argument(
        "--dengue",
        type=Path,
        default=Path("data/processed/dengue.csv"),
        help="Path to processed dengue CSV",
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("data/processed/population.csv"),
        help="Path to processed population CSV",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/weather_weekly.csv"),
        help="Path to processed weather CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/features.csv"),
        help="Path to output features CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Building dengue ML features...")
    out = build_and_save_features(
        dengue_path=args.dengue,
        population_path=args.population,
        weather_path=args.weather,
        output_path=args.output,
    )
    print(f"Features successfully generated and saved to: {out}")


if __name__ == "__main__":
    main()
