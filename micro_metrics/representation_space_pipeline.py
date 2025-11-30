#!/usr/bin/env python3
"""
representation_space_pipeline.py
------------------------------------------------------
Embedding & TF-IDF Temporal Distance Analysis (Euclidean Only)

Author IDs are unique per (domain, field, author_id) to prevent merging
of authors across different fields (e.g., CS_01 and Physics_01 remain separate).

Computed metrics:
  - mean_yearly_distance
  - std_yearly_distance
  - path_length
  - net_displacement
  - tortuosity
------------------------------------------------------
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------
# Input / Output Directories
# ------------------------------------------------------
DATA_ROOT = "dataset/process"
OUTPUT_ROOT = os.path.join(DATA_ROOT, "representation_pipeline")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ------------------------------------------------------
# Regex patterns for parsing filenames
# ------------------------------------------------------
LLM_RE = re.compile(
    r"(?P<domain>[A-Za-z]+)_(?P<field>[A-Za-z0-9]+)_(?P<author>\d{2,})_"
    r"(?P<year>\d{4})_(?P<index>\d{2,})_(?P<provider>[A-Za-z0-9]+)_LV(?P<level>\d+)\.txt$"
)

HUMAN_RE = re.compile(
    r"(?P<domain>[A-Za-z]+)_(?P<field>[A-Za-z0-9]+)_(?P<author>\d{2,})_"
    r"(?P<year>\d{4})_(?P<index>\d{2,})\.txt$"
)


# ------------------------------------------------------
# Metadata extraction
# ------------------------------------------------------
def extract_metadata(path):
    fname = os.path.basename(str(path))

    # LLM format
    m = LLM_RE.match(fname)
    if m:
        g = m.groupdict()

        author_key = f"{g['provider']}_LV{g['level']}_{g['domain']}_{g['field']}_{g['author']}"

        return dict(
            label="llm",
            year=int(g["year"]),
            provider=g["provider"],
            level=int(g["level"]),
            author=author_key,
        )

    # HUMAN format
    m = HUMAN_RE.match(fname)
    if m:
        g = m.groupdict()

        author_key = f"human_{g['domain']}_{g['field']}_{g['author']}"

        return dict(
            label="human",
            year=int(g["year"]),
            provider="human",
            level=0,
            author=author_key,
        )

    return None


# ------------------------------------------------------
def read_text(path, limit=None):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            t = f.read()
        return t[:limit] if limit else t
    except Exception:
        return ""


# ------------------------------------------------------
def euclidean_distance_matrix(vectors):
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# ------------------------------------------------------
# Compute temporal distances with path metrics
# ------------------------------------------------------
def compute_temporal_distances(df_vectors):
    rows = []

    for author, group in df_vectors.groupby("author"):
        g = group.sort_values("year")

        if len(g) < 2:
            continue

        vectors = np.vstack(g["vector"].values)
        years = g["year"].values

        # full matrix
        dist_matrix = euclidean_distance_matrix(vectors)

        # yearly adjacent distances
        yearly_dists = [dist_matrix[i, i+1] for i in range(len(years)-1)]

        # Added metrics
        path_length = float(np.sum(yearly_dists))
        net_displacement = float(np.linalg.norm(vectors[-1] - vectors[0]))
        tortuosity = path_length / net_displacement if net_displacement > 0 else np.nan

        rows.append({
            "author": author,
            "label": g["label"].iloc[0],
            "provider": g["provider"].iloc[0],
            "level": g["level"].iloc[0],
            "mean_yearly_distance": float(np.mean(yearly_dists)),
            "std_yearly_distance": float(np.std(yearly_dists)),
            "path_length": path_length,
            "net_displacement": net_displacement,
            "tortuosity": tortuosity,
            "n_years": len(years)
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------
def build_tfidf_distance(df, domain):
    print(f"\n=== TF-IDF (Euclidean) for {domain} ===")

    texts, meta = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = read_text(row["path"], limit=8000)
        m = extract_metadata(row["filename"])
        if not m or not text.strip():
            continue

        texts.append(text)
        meta.append(m)

    if len(texts) < 3:
        print("Not enough samples for TF-IDF.")
        return

    meta_df = pd.DataFrame(meta)

    tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
    X = tfidf.fit_transform(texts).toarray()

    svd = TruncatedSVD(n_components=10, random_state=42)
    X_red = svd.fit_transform(X)
    meta_df["vector"] = list(X_red)

    dist_df = compute_temporal_distances(meta_df)

    out_path = os.path.join(OUTPUT_ROOT, f"{domain}_tfidf_euclid.csv")
    dist_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ------------------------------------------------------
def build_embedding_distance(df, domain):
    print(f"\n=== Embedding (Euclidean) for {domain} ===")

    texts, meta = [], []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = read_text(row["path"], limit=8000)
        m = extract_metadata(row["filename"])
        if not m or not text.strip():
            continue

        texts.append(text)
        meta.append(m)

    if len(texts) < 3:
        print("Not enough samples for embeddings.")
        return

    meta_df = pd.DataFrame(meta)
    embs = model.encode(texts, show_progress_bar=True)
    meta_df["vector"] = list(embs)

    dist_df = compute_temporal_distances(meta_df)

    out_path = os.path.join(OUTPUT_ROOT, f"{domain}_embedding_euclid.csv")
    dist_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# ------------------------------------------------------
def process_domain(domain):
    print(f"\nProcessing domain: {domain}")

    input_csv = os.path.join(DATA_ROOT, f"combined_{domain}.csv")
    if not os.path.exists(input_csv):
        print(f"Missing input file: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(subset=["path"])

    build_tfidf_distance(df, domain)
    build_embedding_distance(df, domain)


# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Euclidean drift")
    parser.add_argument("--academic", action="store_true")
    parser.add_argument("--blogs", action="store_true")
    parser.add_argument("--news", action="store_true")

    args = parser.parse_args()

    if args.academic:
        domains = ["academic"]
    elif args.blogs:
        domains = ["blogs"]
    elif args.news:
        domains = ["news"]
    else:
        domains = ["academic", "blogs", "news"]

    for d in domains:
        process_domain(d)

    print("\nEuclidean representation pipeline complete.")
