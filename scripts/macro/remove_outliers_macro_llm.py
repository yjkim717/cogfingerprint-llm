#!/usr/bin/env python3
"""
Batch outlier removal for macro LLM combined feature files.

For each (model, level, domain) combination, this script loads
`macro_dataset/process/LLM/{model}/{level}/{domain}/combined.csv`,
detects outliers per-feature (grouped by model) using the existing
IQR-based routine, and writes a cleaned copy with outliers replaced by NaN.

Usage example:
    python remove_outliers_macro_llm.py --levels LV3 --models DS G12B LMK
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from remove_outliers_from_combined_merged import remove_outliers_from_combined_merged


DEFAULT_MODELS = ["DS", "G4B", "G12B", "LMK"]
DEFAULT_LEVELS = ["LV3"]
DEFAULT_DOMAINS = ["news", "blogs", "academic"]
BASE_DIR = Path("macro_dataset/process/LLM")


def normalize_tokens(values: Iterable[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return defaults
    return [value.upper() for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove per-feature outliers from macro LLM combined.csv files."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to process (default: DS G4B G12B LMK).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        help="Levels to process (default: LV3).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domains to process (default: news blogs academic).",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR factor for outlier detection (default: 1.5).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="combined_outliers_removed.csv",
        help="Output filename (default: combined_outliers_removed.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace original combined.csv after removal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = normalize_tokens(args.models, DEFAULT_MODELS)
    levels = normalize_tokens(args.levels, DEFAULT_LEVELS)
    domains = [d.lower() for d in (args.domains or DEFAULT_DOMAINS)]

    total_tasks = 0
    completed = 0
    skipped = 0

    for model in models:
        for level in levels:
            for domain in domains:
                total_tasks += 1
                input_path = BASE_DIR / model / level / domain / "combined.csv"
                if not input_path.exists():
                    print(f"[SKIP] Missing file: {input_path}")
                    skipped += 1
                    continue

                output_path = input_path.parent / args.suffix
                print(
                    f"\n=== Removing outliers: model={model}, level={level}, domain={domain} ==="
                )
                _, stats = remove_outliers_from_combined_merged(
                    input_file=input_path,
                    output_file=output_path,
                    iqr_factor=args.iqr_factor,
                    feature_columns=None,
                )
                completed += 1

                if args.overwrite:
                    backup_path = input_path.with_suffix(".csv.bak")
                    if backup_path.exists():
                        backup_path.unlink()
                    input_path.rename(backup_path)
                    output_path.rename(input_path)
                    print(f"  Original file replaced. Backup saved to {backup_path}")

                print(
                    f"  Removed {stats['total_outliers']} outliers "
                    f"out of {stats['total_values']} values "
                    f"({(stats['total_outliers'] / stats['total_values'] * 100) if stats['total_values'] else 0:.2f}%)."
                )

    print("\n=== Summary ===")
    print(f"Total combinations: {total_tasks}")
    print(f"Processed: {completed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()

