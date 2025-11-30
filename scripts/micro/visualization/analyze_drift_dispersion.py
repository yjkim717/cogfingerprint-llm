#!/usr/bin/env python3
"""
Analyze dispersion of Human vs LLM LV3 in drift feature space.

This script quantifies:
1. Variance in PCA space
2. Cluster tightness (mean distance to centroid)
3. Spread metrics (standard deviation, IQR)

Usage:
    python scripts/micro/visualization/analyze_drift_dispersion.py \
        --space tfidf --domains academic blogs news
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
SPACES = ("tfidf", "sbert")


def load_trajectory_features(domain: str, space: str) -> pd.DataFrame:
    """Load trajectory features for a domain and space."""
    frames: List[pd.DataFrame] = []
    
    # Human
    human_path = DATA_ROOT / "human" / domain / f"{space}_trajectory_features.csv"
    if human_path.exists():
        df_human = pd.read_csv(human_path)
        frames.append(df_human)
    
    # LLM LV3 (all providers)
    for provider in PROVIDERS:
        llm_path = DATA_ROOT / "LLM" / provider / "LV3" / domain / f"{space}_trajectory_features.csv"
        if llm_path.exists():
            df_llm = pd.read_csv(llm_path)
            frames.append(df_llm)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def compute_dispersion_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute dispersion metrics for each class."""
    metrics = {}
    
    for label in np.unique(labels):
        mask = labels == label
        X_class = X[mask]
        
        if len(X_class) < 2:
            metrics[label] = {
                "n_samples": len(X_class),
                "variance_pc1": np.nan,
                "variance_pc2": np.nan,
                "total_variance": np.nan,
                "mean_dist_to_centroid": np.nan,
                "std_dist_to_centroid": np.nan,
                "max_dist_to_centroid": np.nan,
            }
            continue
        
        # Centroid
        centroid = X_class.mean(axis=0)
        
        # Distances to centroid
        dists_to_centroid = np.linalg.norm(X_class - centroid, axis=1)
        
        # Variance in each dimension
        var_pc1 = np.var(X_class[:, 0])
        var_pc2 = np.var(X_class[:, 1])
        total_var = np.var(X_class, axis=0).sum()
        
        metrics[label] = {
            "n_samples": len(X_class),
            "variance_pc1": var_pc1,
            "variance_pc2": var_pc2,
            "total_variance": total_var,
            "mean_dist_to_centroid": np.mean(dists_to_centroid),
            "std_dist_to_centroid": np.std(dists_to_centroid),
            "max_dist_to_centroid": np.max(dists_to_centroid),
        }
    
    return metrics


def analyze_domain(
    df: pd.DataFrame,
    domain: str,
    space: str,
) -> None:
    """Analyze dispersion for a single domain."""
    # Filter data
    human_df = df[df["label"] == "human"].copy()
    llm_df = df[(df["label"] == "llm") & (df["level"] == "LV3")].copy()
    
    if human_df.empty or llm_df.empty:
        print(f"⚠️  Skipping {domain}/{space}: No data")
        return
    
    # Prepare features
    feature_cols = ["mean_distance", "std_distance"]
    combined_df = pd.concat([human_df, llm_df], axis=0, ignore_index=True)
    combined_df = combined_df.dropna(subset=feature_cols)
    
    if combined_df.empty:
        return
    
    X = combined_df[feature_cols].values
    y = combined_df["label"].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Compute dispersion metrics
    metrics = compute_dispersion_metrics(X_pca, y)
    
    # Print results
    print(f"\n=== {domain.upper()} — {space.upper()} ===")
    print(f"\nHuman (n={metrics['human']['n_samples']}):")
    print(f"  Variance PC1: {metrics['human']['variance_pc1']:.4f}")
    print(f"  Variance PC2: {metrics['human']['variance_pc2']:.4f}")
    print(f"  Total variance: {metrics['human']['total_variance']:.4f}")
    print(f"  Mean dist to centroid: {metrics['human']['mean_dist_to_centroid']:.4f}")
    print(f"  Std dist to centroid: {metrics['human']['std_dist_to_centroid']:.4f}")
    print(f"  Max dist to centroid: {metrics['human']['max_dist_to_centroid']:.4f}")
    
    print(f"\nLLM LV3 (n={metrics['llm']['n_samples']}):")
    print(f"  Variance PC1: {metrics['llm']['variance_pc1']:.4f}")
    print(f"  Variance PC2: {metrics['llm']['variance_pc2']:.4f}")
    print(f"  Total variance: {metrics['llm']['total_variance']:.4f}")
    print(f"  Mean dist to centroid: {metrics['llm']['mean_dist_to_centroid']:.4f}")
    print(f"  Std dist to centroid: {metrics['llm']['std_dist_to_centroid']:.4f}")
    print(f"  Max dist to centroid: {metrics['llm']['max_dist_to_centroid']:.4f}")
    
    # Comparison
    print(f"\nComparison (Human / LLM):")
    print(f"  Total variance ratio: {metrics['human']['total_variance'] / metrics['llm']['total_variance']:.2f}x")
    print(f"  Mean dist ratio: {metrics['human']['mean_dist_to_centroid'] / metrics['llm']['mean_dist_to_centroid']:.2f}x")
    print(f"  Max dist ratio: {metrics['human']['max_dist_to_centroid'] / metrics['llm']['max_dist_to_centroid']:.2f}x")
    
    # Also check in original feature space
    human_mask = y == "human"
    llm_mask = y == "llm"
    
    print(f"\nOriginal feature space:")
    print(f"  Human mean_distance: mean={X[human_mask, 0].mean():.4f}, std={X[human_mask, 0].std():.4f}")
    print(f"  LLM mean_distance: mean={X[llm_mask, 0].mean():.4f}, std={X[llm_mask, 0].std():.4f}")
    print(f"  Human std_distance: mean={X[human_mask, 1].mean():.4f}, std={X[human_mask, 1].std():.4f}")
    print(f"  LLM std_distance: mean={X[llm_mask, 1].mean():.4f}, std={X[llm_mask, 1].std():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze dispersion of Human vs LLM LV3"
    )
    parser.add_argument(
        "--space",
        type=str,
        choices=SPACES,
        required=True,
        help="Embedding space: tfidf or sbert",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        choices=DOMAINS,
        default=DOMAINS,
        help="Domains to process (default: all)",
    )
    
    args = parser.parse_args()
    space = args.space
    domains = args.domains
    
    print(f"\n=== Analyzing {space.upper()} drift dispersion ===")
    print(f"Domains: {', '.join(domains)}\n")
    
    for domain in domains:
        df = load_trajectory_features(domain, space)
        if df.empty:
            print(f"⚠️  No data found for {domain}/{space}")
            continue
        analyze_domain(df, domain, space)
    
    print("\n" + "="*60)
    print("Interpretation:")
    print("  - If Human is more dispersed: variance ratio > 1, mean dist ratio > 1")
    print("  - If LLM is more dispersed: variance ratio < 1, mean dist ratio < 1")
    print("  - Expected: Human should be more dispersed (variance ratio > 1)")


if __name__ == "__main__":
    main()


