#!/usr/bin/env python3
"""
Compute trajectory features for macro dataset, grouped by domain+year.

For RQ2: Treat each domain as a "big author", with 1000 samples per year.
Computes the same 75 unified trajectory features as RQ1:
- CE variability (CV, RMSSD_norm, MASD_norm) for 20 CE features
- Geometry features (mean_distance, std_distance, net_displacement, path_length, tortuosity)
  for CE, TFIDF, and SBERT spaces

Usage:
    python scripts/macro/trajectory/compute_macro_trajectory_features.py \
        --level LV3
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

MACRO_DATA_ROOT = PROJECT_ROOT / "macro_dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
LLM_MODELS = ("DS", "G4B", "G12B", "LMK")
LEVEL = "LV3"

# 20 CE features
CE_FEATURES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    "word_diversity",
    "flesch_reading_ease",
    "gunning_fog",
    "average_word_length",
    "num_words",
    "avg_sentence_length",
    "verb_ratio",
    "function_word_ratio",
    "content_word_ratio",
]

VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
)


def extract_year_from_filename(filename: str) -> int | None:
    """Extract 4-digit year from filename."""
    match = re.search(r"(\d{4})", str(filename))
    return int(match.group(1)) if match else None


def compute_variability_features(group_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """Compute CV, RMSSD_norm, MASD_norm for each feature."""
    features = {}
    
    for feat in feature_cols:
        values = group_df[feat].dropna().values
        if len(values) < 2:
            features[f"{feat}_cv"] = 0.0
            features[f"{feat}_rmssd_norm"] = 0.0
            features[f"{feat}_masd_norm"] = 0.0
            continue
        
        # Coefficient of Variation (CV)
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val != 0 else 0.0
        features[f"{feat}_cv"] = float(cv)
        
        # RMSSD (Root Mean Square of Successive Differences) - normalized
        if len(values) > 1:
            diffs = np.diff(values)
            rmssd = np.sqrt(np.mean(diffs ** 2))
            rmssd_norm = rmssd / mean_val if mean_val != 0 else 0.0
            features[f"{feat}_rmssd_norm"] = float(rmssd_norm)
        else:
            features[f"{feat}_rmssd_norm"] = 0.0
        
        # MASD (Mean Absolute Successive Difference) - normalized
        if len(values) > 1:
            masd = np.mean(np.abs(diffs))
            masd_norm = masd / mean_val if mean_val != 0 else 0.0
            features[f"{feat}_masd_norm"] = float(masd_norm)
        else:
            features[f"{feat}_masd_norm"] = 0.0
    
    return features


def compute_yearly_metrics(vectors: np.ndarray) -> Dict[str, float]:
    """Compute geometry metrics from yearly vectors."""
    if len(vectors) < 2:
        return {
            "mean_distance": math.nan,
            "std_distance": math.nan,
            "net_displacement": math.nan,
            "path_length": math.nan,
            "tortuosity": math.nan,
            "n_years": len(vectors),
        }
    
    diffs = vectors[1:] - vectors[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    mean_distance = float(np.mean(dists))
    std_distance = float(np.std(dists))
    
    net_displacement = float(np.linalg.norm(vectors[-1] - vectors[0]))
    path_length = float(np.sum(dists))
    tortuosity = float(path_length / net_displacement) if net_displacement > 0 else 1.0
    
    return {
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "net_displacement": net_displacement,
        "path_length": path_length,
        "tortuosity": tortuosity,
        "n_years": len(vectors),
    }


def process_domain_trajectory_features(
    csv_path: Path, domain: str, label: str, provider: str | None = None, level: str | None = None
) -> pd.DataFrame:
    """Process trajectory features for a domain, grouped by domain+year."""
    if not csv_path.exists():
        print(f"⚠ File not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    
    if df.empty:
        return pd.DataFrame()
    
    # Extract year from filename
    df["year"] = df["filename"].apply(extract_year_from_filename)
    df = df.dropna(subset=["year"])
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Set metadata
    df["domain"] = domain
    df["label"] = label
    if provider:
        df["provider"] = provider
    if level:
        df["level"] = level
    
    # Get CE feature columns and convert to numeric
    ce_cols = [c for c in CE_FEATURES if c in df.columns]
    if not ce_cols:
        print(f"⚠ No CE features found in {csv_path}")
        return pd.DataFrame()
    
    # Convert CE columns to numeric (handle any string values)
    for col in ce_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get TFIDF and SBERT vector columns
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_") and c.replace("tfidf_", "").isdigit()]
    sbert_cols = [c for c in df.columns if c.startswith("sbert_") and c.replace("sbert_", "").isdigit()]
    
    # Convert TFIDF and SBERT columns to numeric
    for col in tfidf_cols + sbert_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Group by domain (treating domain as "big author")
    # Compute trajectory features across all years for this domain
    rows: List[Dict] = []
    
    for domain_val, domain_group in df.groupby("domain"):
        # Get yearly means (each year has ~1000 samples)
        years_sorted = sorted(domain_group["year"].unique())
        
        if len(years_sorted) < 2:
            print(f"  ⚠ {domain_val}: Only {len(years_sorted)} year(s), skipping")
            continue
        
        # Compute yearly means for CE features
        yearly_ce_means = []
        yearly_tfidf_means = []
        yearly_sbert_means = []
        
        for year in years_sorted:
            year_group = domain_group[domain_group["year"] == year]
            yearly_ce_means.append(year_group[ce_cols].mean().values)
            if tfidf_cols:
                yearly_tfidf_means.append(year_group[tfidf_cols].mean().values)
            if sbert_cols:
                yearly_sbert_means.append(year_group[sbert_cols].mean().values)
        
        # Convert to arrays
        ce_trajectory = np.array(yearly_ce_means)  # Shape: (n_years, n_ce_features)
        
        # Compute CE variability features from trajectory of yearly means
        variability_features = {}
        for i, feat in enumerate(ce_cols):
            feat_trajectory = ce_trajectory[:, i]
            if len(feat_trajectory) >= 2:
                mean_val = np.mean(feat_trajectory)
                std_val = np.std(feat_trajectory)
                cv = std_val / mean_val if mean_val != 0 else 0.0
                variability_features[f"{feat}_cv"] = float(cv)
                
                diffs = np.diff(feat_trajectory)
                rmssd = np.sqrt(np.mean(diffs ** 2))
                rmssd_norm = rmssd / mean_val if mean_val != 0 else 0.0
                variability_features[f"{feat}_rmssd_norm"] = float(rmssd_norm)
                
                masd = np.mean(np.abs(diffs))
                masd_norm = masd / mean_val if mean_val != 0 else 0.0
                variability_features[f"{feat}_masd_norm"] = float(masd_norm)
            else:
                variability_features[f"{feat}_cv"] = 0.0
                variability_features[f"{feat}_rmssd_norm"] = 0.0
                variability_features[f"{feat}_masd_norm"] = 0.0
        
        # Compute CE geometry (from trajectory of yearly means)
        ce_geometry = compute_yearly_metrics(ce_trajectory)
        ce_geometry = {f"ce_{k}": v for k, v in ce_geometry.items() if k != "n_years"}
        
        # Compute TFIDF geometry
        if tfidf_cols and len(yearly_tfidf_means) >= 2:
            tfidf_trajectory = np.array(yearly_tfidf_means)
            tfidf_geometry = compute_yearly_metrics(tfidf_trajectory)
            tfidf_geometry = {f"tfidf_{k}": v for k, v in tfidf_geometry.items() if k != "n_years"}
        else:
            tfidf_geometry = {f"tfidf_{k}": math.nan for k in GEOMETRY_METRICS}
        
        # Compute SBERT geometry
        if sbert_cols and len(yearly_sbert_means) >= 2:
            sbert_trajectory = np.array(yearly_sbert_means)
            sbert_geometry = compute_yearly_metrics(sbert_trajectory)
            sbert_geometry = {f"sbert_{k}": v for k, v in sbert_geometry.items() if k != "n_years"}
        else:
            sbert_geometry = {f"sbert_{k}": math.nan for k in GEOMETRY_METRICS}
        
        # Create one row per domain (not per year)
        row = {
            "domain": domain_val,
            "label": label,
            "sample_count": len(domain_group),
            "n_years": len(years_sorted),
            "years": ",".join(map(str, years_sorted)),
            **variability_features,
            **ce_geometry,
            **tfidf_geometry,
            **sbert_geometry,
        }
        if provider:
            row["provider"] = provider
        if level:
            row["level"] = level
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute trajectory features for macro dataset (grouped by domain+year)."
    )
    parser.add_argument(
        "--level",
        type=str,
        default=LEVEL,
        help=f"LLM level to process (default: {LEVEL}).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Computing Macro Trajectory Features (Grouped by Domain+Year)")
    print("=" * 80)
    
    all_results = []
    
    # Process human data
    for domain in args.domains:
        print(f"\nProcessing Human {domain}...")
        human_path = MACRO_DATA_ROOT / "human" / domain / "combined_with_embeddings.csv"
        df_human = process_domain_trajectory_features(
            human_path, domain=domain, label="human", provider="human", level="LV0"
        )
        if not df_human.empty:
            all_results.append(df_human)
            years_str = df_human['years'].iloc[0] if 'years' in df_human.columns else 'N/A'
            print(f"  ✓ Generated {len(df_human)} trajectory features (years: {years_str})")
    
    # Process LLM data
    for model in LLM_MODELS:
        for domain in args.domains:
            print(f"\nProcessing LLM {model} {args.level} {domain}...")
            llm_path = MACRO_DATA_ROOT / "LLM" / model / args.level / domain / "combined_with_embeddings.csv"
            df_llm = process_domain_trajectory_features(
                llm_path, domain=domain, label="llm", provider=model, level=args.level
            )
            if not df_llm.empty:
                all_results.append(df_llm)
                years_str = df_llm['years'].iloc[0] if 'years' in df_llm.columns else 'N/A'
                print(f"  ✓ Generated {len(df_llm)} trajectory features (years: {years_str})")
    
    if not all_results:
        print("\n⚠ No trajectory features generated!")
        return
    
    # Combine and save
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save per domain
    for domain in args.domains:
        domain_df = combined_df[combined_df["domain"] == domain]
        if not domain_df.empty:
            output_path = MACRO_DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
            domain_df.to_csv(output_path, index=False)
            print(f"\n✅ Saved {domain} trajectory features: {output_path} ({len(domain_df)} rows)")
    
    # Also save combined
    combined_output = MACRO_DATA_ROOT / "trajectory_features_combined_all.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"\n✅ Saved combined trajectory features: {combined_output} ({len(combined_df)} rows)")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total trajectory features: {len(combined_df)}")
    print(f"  Domains: {sorted(combined_df['domain'].unique())}")
    print(f"  Labels: {sorted(combined_df['label'].unique())}")
    if 'years' in combined_df.columns:
        all_years = set()
        for years_str in combined_df['years']:
            all_years.update(map(int, years_str.split(',')))
        print(f"  Years: {sorted(all_years)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
