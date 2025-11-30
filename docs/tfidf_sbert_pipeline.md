# TF-IDF & SBERT Regeneration Log

This document records how we rebuilt the TF-IDF and SBERT representations for every micro dataset split (human + each LLM provider/level) using the current cleaned corpus.

## Goals

1. Align embedding features with the exact samples used in the new micro pipeline (2020–2024, two samples per year for blogs, etc.).
2. Produce reusable per-sample vector CSVs colocated with `combined_merged.csv`.
3. Aggregate these vectors into per-author per-year representations for trajectory modeling.

## Scripts & Locations

| Purpose | Script | Output |
| --- | --- | --- |
| Extract per-sample TF-IDF vectors | `scripts/micro/tfidf/extract_tfidf_vectors.py` | `dataset/process/.../tfidf_vectors.csv` |
| Extract per-sample SBERT vectors | `scripts/micro/sbert/extract_sbert_vectors.py` | `dataset/process/.../sbert_vectors.csv` |
| Average TF-IDF vectors per author-year | `scripts/micro/tfidf/aggregate_tfidf_yearly.py` | `tfidf_author_yearly_vectors.csv` |
| Average SBERT vectors per author-year | `scripts/micro/sbert/aggregate_sbert_yearly.py` | `sbert_author_yearly_vectors.csv` |

## Key Configuration

- **TF-IDF**:
  - `max_features = 20_000`
  - English stopwords, scikit-learn vectorizer
  - `TruncatedSVD(n_components=10, random_state=42)`
  - Text truncated to 8,000 chars to stay consistent with Jenny’s pipeline.

- **SBERT**:
  - Model: `sentence-transformers` `all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Same 8,000-char truncation for consistency.

- **Metadata handling**:
  - Reads `combined_merged.csv` for each split to capture `filename`, `path`, `label`, `domain`, `field`, `author_id`, `provider`, `level`, `model`.
  - Text loaded via the original human/LLM paths relative to repo root.

## Running the Pipelines

```bash
# From repo root
python scripts/micro/tfidf/extract_tfidf_vectors.py
python scripts/micro/sbert/extract_sbert_vectors.py

# Aggregate to per-author per-year
python scripts/micro/tfidf/aggregate_tfidf_yearly.py
python scripts/micro/sbert/aggregate_sbert_yearly.py
```

- Domains can be restricted via `--domains academic blogs` if needed.
- The scripts automatically iterate through human splits and all LLM providers (`DS`, `G4B`, `G12B`, `LMK`) across `LV1–LV3`.

## Sample Counts (per split)

- **Academic**: 500 samples → 500 yearly rows (one per author-year).
- **Blogs**: 1,901 samples → 972 yearly rows (5 years × ~194 authors, 2 samples/year averaged).
- **News**: 3,685 samples → 837 yearly rows (varied yearly counts).
- DeepSeek news LV1/LV2 reported 3,684 samples because one source file was missing; yearly rows match the human count (837). Once the missing text is restored, re-run the scripts for that domain to fill the gap.

## Usage Notes

- The resulting yearly vector files (`tfidf_author_yearly_vectors.csv`, `sbert_author_yearly_vectors.csv`) are the canonical inputs for trajectory feature extraction.
- Keep these CSVs under version control to avoid rerunning the expensive embedding passes unless the corpus changes.


