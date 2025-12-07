#!/usr/bin/env python3
"""
Visualize CE Geometry drift difference distribution in 3×4 panel format.

This script creates a single 3×4 panel plot for CE Geometry path_length feature:
- Rows: Academic / Blogs / News (domains)
- Columns: DS / G4B / G12B / LMK (models)
- Each cell: Drift difference for CE path_length feature (Human - LLM)

Uses only path_length feature as it shows stronger signal and better discriminative power.
Optimized specifically for CE Geometry's large value ranges.

Usage:
    python scripts/micro/visualization/visualize_ce_geometry_3x4.py \
        --level LV3 \
        --output plots/trajectory/drift_difference_distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"

# Color scheme
COLORS = {
    "histogram": "#2E86AB",  # Blue
    "zero_line": "#C73E1D",  # Red
    "mean_line": "#000000",  # Black
}

# Geometry metrics
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
)

# Metadata columns to exclude
METADATA_COLS = {
    "field",
    "author_id",
    "sample_count",
    "domain",
    "label",
    "provider",
    "level",
    "model",
}


def load_trajectory_features(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load trajectory features for all domains and models."""
    frames: List[pd.DataFrame] = []
    
    for domain in domains:
        # Human data
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)
        
        # LLM data
        for provider in models:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["domain"] = domain
                df["label"] = "llm"
                df["provider"] = provider
                df["level"] = level
                frames.append(df)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def get_ce_geometry_columns(df: pd.DataFrame) -> List[str]:
    """Get CE Geometry feature column names - only path_length."""
    return [
        col for col in df.columns
        if col.startswith("ce_") and col.endswith("_path_length")
    ]


def compute_differences(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Compute Human - LLM differences for matched author pairs."""
    differences = []
    
    # Get human authors
    human_df = df[df["label"] == "human"].copy()
    
    # For each human author, find matching LLM authors
    for _, human_row in human_df.iterrows():
        domain = human_row["domain"]
        field = human_row.get("field", None)
        author_id = human_row.get("author_id", None)
        
        # Find matching LLM authors (same domain, field, author_id)
        if field is not None and author_id is not None:
            llm_mask = (
                (df["label"] == "llm") &
                (df["domain"] == domain) &
                (df["field"] == field) &
                (df["author_id"] == author_id)
            )
        else:
            # Fallback: match by domain only
            llm_mask = (
                (df["label"] == "llm") &
                (df["domain"] == domain)
            )
        
        llm_df = df[llm_mask].copy()
        
        if llm_df.empty:
            continue
        
        # Compare with each LLM model separately
        for _, llm_row in llm_df.iterrows():
            for col in feature_cols:
                if col not in human_row.index or col not in llm_row.index:
                    continue
                
                human_val = human_row[col]
                llm_val = llm_row[col]
                
                # Skip if either value is missing
                if pd.isna(human_val) or pd.isna(llm_val):
                    continue
                
                diff = float(human_val) - float(llm_val)
                
                differences.append({
                    "domain": domain,
                    "field": field,
                    "author_id": author_id,
                    "llm_model": llm_row["provider"],
                    "feature": col,
                    "human_value": human_val,
                    "llm_value": llm_val,
                    "difference": diff,
                })
    
    return pd.DataFrame(differences)


def plot_ce_cell(differences: np.ndarray, ax) -> None:
    """Plot single cell for CE Geometry with optimized scaling for large ranges."""
    if len(differences) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Calculate statistics
    data_min = np.min(differences)
    data_max = np.max(differences)
    data_mean = np.mean(differences)
    data_std = np.std(differences)
    data_median = np.median(differences)
    data_range = data_max - data_min
    
    # Calculate percentiles for robust scaling
    q10 = np.percentile(differences, 10)
    q25 = np.percentile(differences, 25)
    q75 = np.percentile(differences, 75)
    q90 = np.percentile(differences, 90)
    iqr = q75 - q25
    
    # CE Geometry has very large ranges - use robust scaling with tighter focus
    if data_range > 1000:
        # Extremely large range: use IQR-based with 2*IQR whiskers (tighter)
        lower_bound = max(data_min, q25 - 2.0 * iqr)
        upper_bound = min(data_max, q75 + 2.0 * iqr)
        # Small padding for better resolution
        padding = max((upper_bound - lower_bound) * 0.02, data_std * 0.15)
        x_min = lower_bound - padding
        x_max = upper_bound + padding
    elif data_range > 500:
        # Very large range: use IQR-based with 1.5*IQR whiskers (tighter)
        lower_bound = max(data_min, q25 - 1.5 * iqr)
        upper_bound = min(data_max, q75 + 1.5 * iqr)
        padding = max((upper_bound - lower_bound) * 0.03, data_std * 0.2)
        x_min = lower_bound - padding
        x_max = upper_bound + padding
    elif data_range > 100:
        # Large range: use percentile-based (5th-95th for tighter focus)
        q5 = np.percentile(differences, 5)
        q95 = np.percentile(differences, 95)
        lower_bound = q5
        upper_bound = q95
        padding = max((upper_bound - lower_bound) * 0.04, data_std * 0.25)
        x_min = lower_bound - padding
        x_max = upper_bound + padding
    else:
        # Moderate range: use std-based
        padding = max(data_range * 0.06, data_std * 0.4)
        x_min = data_min - padding
        x_max = data_max + padding
    
    # Ensure zero is visible if data spans zero or is close to zero
    zero_span = (data_min <= 0 <= data_max)
    close_to_zero = (abs(data_median) < data_range * 0.15) or (abs(q25) < data_range * 0.1) or (abs(q75) < data_range * 0.1)
    
    if zero_span or close_to_zero:
        # Expand range to include zero with margin
        if x_min > 0:
            x_min = min(x_min, -abs(data_median) * 0.25, -abs(q25) * 0.2)
        if x_max < 0:
            x_max = max(x_max, abs(data_median) * 0.25, abs(q75) * 0.2)
    
    # Calculate optimal bins for large ranges - 2x more bins for finer resolution
    if data_range > 1000:
        n_bins = min(100, max(70, int(np.sqrt(len(differences)) * 3.0)))
    elif data_range > 500:
        n_bins = min(120, max(80, int(np.sqrt(len(differences)) * 3.2)))
    else:
        n_bins = min(140, max(100, int(np.sqrt(len(differences)) * 3.6)))
    
    # Plot histogram
    sns.histplot(
        differences,
        kde=True,
        color=COLORS["histogram"],
        alpha=0.5,
        ax=ax,
        stat="count",
        bins=n_bins,
        edgecolor="black",
        linewidth=0.5,
    )
    
    # Set x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Zero line
    ax.axvline(0, color=COLORS["zero_line"], linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)
    
    # Mean line
    ax.axvline(data_mean, color=COLORS["mean_line"], linestyle="--", linewidth=1.5, alpha=0.8, zorder=3)
    
    # Formatting
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)


def plot_ce_geometry_3x4_panel(
    diff_df: pd.DataFrame,
    domains: List[str],
    models: List[str],
    level: str,
    output_dir: Path,
) -> None:
    """Create 3×4 panel for CE Geometry with optimized scaling for large ranges."""
    n_rows = len(domains)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.reshape(n_rows, n_cols)
    
    # Plot each domain-model combination
    for row_idx, domain in enumerate(domains):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            # Filter to specific domain and model
            filtered_df = diff_df[
                (diff_df["domain"] == domain) & (diff_df["llm_model"] == model)
            ]
            differences = filtered_df["difference"].dropna().values
            
            plot_ce_cell(differences, ax)
            
            # Set row labels on left (only for first column)
            if col_idx == 0:
                ax.set_ylabel(domain.capitalize(), fontsize=10, fontweight="bold")
            
            # Set column labels on top (only for first row)
            if row_idx == 0:
                ax.set_title(model, fontsize=10, fontweight="bold", pad=5)
    
    # Main title
    plt.suptitle(
        "CE Geometry Path Length Drift Difference Distribution",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    # Add x-axis label to bottom row only
    for col_idx in range(n_cols):
        axes[-1, col_idx].set_xlabel("Human Drift - LLM Drift", fontsize=9, fontweight="bold")
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Output filename
    output_path = output_dir / "ce_geometry_3x4_panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CE Geometry drift difference distribution in 3×4 panel format"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to include (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM models to include (default: all)",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="LV3",
        choices=["LV3"],
        help="LLM level (only LV3 supported)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/trajectory/drift_difference_distribution",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CE GEOMETRY DRIFT DIFFERENCE VISUALIZATION (3×4 Panel)")
    print("=" * 80)
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Level: {args.level}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("Loading trajectory features...")
    df = load_trajectory_features(args.domains, args.models, args.level)
    
    if df.empty:
        print("ERROR: No data found!")
        return
    
    print(f"Loaded {len(df)} total samples")
    print(f"  Human: {len(df[df['label'] == 'human'])}")
    print(f"  LLM: {len(df[df['label'] == 'llm'])}")
    print()
    
    # Get CE Geometry path_length feature
    print("Processing CE Geometry path_length feature...")
    feature_cols = get_ce_geometry_columns(df)
    
    if not feature_cols:
        print("ERROR: No CE Geometry path_length feature found!")
        return
    
    print(f"Found {len(feature_cols)} feature(s): {', '.join(feature_cols)}")
    print()
    
    # Compute differences
    print("Computing differences (Human - LLM)...")
    diff_df = compute_differences(df, feature_cols)
    
    if diff_df.empty:
        print("ERROR: No differences computed!")
        return
    
    print(f"Computed {len(diff_df)} difference pairs")
    print()
    
    # Generate 3×4 panel
    print("Generating 3×4 panel visualization...")
    plot_ce_geometry_3x4_panel(
        diff_df,
        args.domains,
        args.models,
        args.level,
        output_dir,
    )
    
    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_dir / 'ce_geometry_3x4_panel.png'}")


if __name__ == "__main__":
    main()

