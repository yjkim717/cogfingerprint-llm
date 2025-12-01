#!/usr/bin/env python3
"""
Extract SBERT vectors (all-MiniLM-L6-v2) for macro datasets.

This script extracts SBERT features for macro dataset samples:
- Human: 15000 samples (academic, blogs, news, 5000 each)
- LLM: 4 models at LV3 (DS, G12B, G4B, LMK), ~5000 samples per domain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.file_utils import read_text as read_utf8_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "macro_dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVEL = "LV3"
MAX_DOC_CHARS = 8_000
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_FILENAME = "sbert_vectors.csv"


def load_text(rel_path: str) -> str:
    """Read UTF-8 text from project-relative path with truncation."""
    abs_path = PROJECT_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"Missing text file: {abs_path}")
    text = read_utf8_text(str(abs_path))
    return text[:MAX_DOC_CHARS]


def collect_entries(domain: str) -> List[Dict]:
    """Collect all samples for a domain (human + each provider/level)."""
    entries: List[Dict] = []

    def _load_csv(csv_path: Path, label: str, provider: str, level: str) -> None:
        if not csv_path.exists():
            print(f"⚠️  File not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rel_path = row.get("path")
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue

            try:
                text = load_text(rel_path)
            except FileNotFoundError:
                continue

            metadata = {
                "filename": row.get("filename"),
                "path": rel_path,
                "label": label,
                "domain": row.get("domain", domain),
                "field": row.get("field"),
                "author_id": row.get("author_id"),
                "provider": provider,
                "level": level,
                "model": row.get("model"),
            }

            out_path = (
                DATA_ROOT / "human" / domain / OUTPUT_FILENAME
                if label == "human"
                else DATA_ROOT / "LLM" / provider / level / domain / OUTPUT_FILENAME
            )

            metadata["output_path"] = str(out_path)
            entries.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

    # Human split
    human_csv = DATA_ROOT / "human" / domain / "combined.csv"
    _load_csv(human_csv, label="human", provider="human", level="LV0")

    # LLM splits (only LV3 for macro dataset)
    for provider in PROVIDERS:
        csv_path = DATA_ROOT / "LLM" / provider / LEVEL / domain / "combined.csv"
        _load_csv(csv_path, label="llm", provider=provider, level=LEVEL)

    return entries


def write_outputs(entries: List[Dict], vectors) -> None:
    """Write SBERT vectors grouped by output path."""
    vectors_arr = np.asarray(vectors)
    if vectors_arr.ndim != 2:
        raise ValueError("Expected 2D array of SBERT vectors.")

    dim = vectors_arr.shape[1]
    vector_columns = [f"sbert_{i+1}" for i in range(dim)]
    vectors_df = pd.DataFrame(vectors_arr, columns=vector_columns)
    meta_df = pd.DataFrame([entry["metadata"] for entry in entries])
    combined_df = pd.concat([meta_df, vectors_df], axis=1)

    for output_path, group in combined_df.groupby("output_path"):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["output_path"]).to_csv(output_file, index=False)
        print(f"✅ {output_file}: {len(group)} samples")


def process_domain(domain: str) -> None:
    entries = collect_entries(domain)
    if not entries:
        print(f"⚠️  No entries found for domain '{domain}'. Skipping.")
        return

    print(f"Processing {len(entries)} samples for {domain}...")
    texts = [entry["text"] for entry in entries]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)

    write_outputs(entries, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SBERT vectors for macro datasets.")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for domain in args.domains:
        print(f"\n=== SBERT vectors for {domain} (macro dataset) ===")
        process_domain(domain)


if __name__ == "__main__":
    main()

