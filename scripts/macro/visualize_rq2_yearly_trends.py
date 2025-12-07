#!/usr/bin/env python3
"""
Visualize RQ2 yearly ML validation results: accuracy trends over years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("macro_results/rq2_yearly_ml_validation")
OUTPUT_DIR = Path("plots/rq2_yearly_ml_validation")


def load_results(feature_set: str) -> pd.DataFrame:
    """Load results JSON file."""
    json_file = RESULTS_DIR / f"rq2_yearly_validation_{feature_set}_all.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Results file not found: {json_file}")
    
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return pd.DataFrame(data["yearly_results"])


def plot_yearly_trends(ce_df: pd.DataFrame, all_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot accuracy trends over years for both feature sets."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    domains = ["academic", "blogs", "news"]
    
    for idx, domain in enumerate(domains):
        ax = axes[idx]
        
        ce_domain = ce_df[ce_df["domain"] == domain].sort_values("year")
        all_domain = all_df[all_df["domain"] == domain].sort_values("year")
        
        # Plot lines
        ax.plot(ce_domain["year"], ce_domain["test_accuracy"], 
                marker="o", label="CE Only (20 features)", linewidth=2, markersize=8)
        ax.plot(all_domain["year"], all_domain["test_accuracy"], 
                marker="s", label="All Features (414 features)", linewidth=2, markersize=8)
        
        # Add trend lines
        from scipy import stats
        ce_slope, ce_intercept, _, _, _ = stats.linregress(ce_domain["year"], ce_domain["test_accuracy"])
        all_slope, all_intercept, _, _, _ = stats.linregress(all_domain["year"], all_domain["test_accuracy"])
        
        ce_trend_line = ce_slope * ce_domain["year"] + ce_intercept
        all_trend_line = all_slope * all_domain["year"] + all_intercept
        
        ax.plot(ce_domain["year"], ce_trend_line, "--", alpha=0.5, color=ax.lines[0].get_color())
        ax.plot(all_domain["year"], all_trend_line, "--", alpha=0.5, color=ax.lines[1].get_color())
        
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(f"{domain.upper()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.85, 1.0])
        
        # Add trend annotation
        ce_trend_text = "↓" if ce_slope < 0 else "↑" if ce_slope > 0 else "→"
        all_trend_text = "↓" if all_slope < 0 else "↑" if all_slope > 0 else "→"
        ax.text(0.02, 0.98, f"CE: {ce_trend_text} All: {all_trend_text}", 
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "yearly_accuracy_trends.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot to {output_path}")
    plt.close()


def plot_yearly_trends_combined(ce_df: pd.DataFrame, all_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot accuracy trends over years for all domains in a single figure (414 features only) with line and bar charts."""
    sns.set_style("whitegrid")
    # Create figure with two subplots: line chart and bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
    domains = ["academic", "blogs", "news"]
    
    # Color scheme for domains
    domain_colors = {
        "academic": "#3498db",  # Blue
        "blogs": "#e74c3c",     # Red
        "news": "#2ecc71",       # Green
    }
    
    # Get all years
    all_years = sorted(all_df["year"].unique())
    
    # ===== Top subplot: Line chart =====
    for domain in domains:
        all_domain = all_df[all_df["domain"] == domain].sort_values("year")
        if len(all_domain) > 0:
            ax1.plot(all_domain["year"], all_domain["test_accuracy"], 
                    marker="s", linestyle="-", linewidth=4, markersize=14,
                    label=f"{domain.upper()}",
                    color=domain_colors[domain], alpha=0.9)
    
    ax1.set_xlabel("Year", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Test Accuracy", fontsize=18, fontweight="bold")
    ax1.set_title("RQ2 Yearly ML Validation: Accuracy Trends by Domain (414 Static Features) - Line Chart", 
                fontsize=20, fontweight="bold", pad=20)
    ax1.legend(fontsize=16, loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=1.5)
    ax1.set_ylim([0.85, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # ===== Bottom subplot: Bar chart grouped by domain =====
    x = np.arange(len(domains))
    width = 0.15  # Width of bars (smaller since we have more bars per group)
    
    # Prepare data: for each domain, get all years' accuracies
    domain_data = {}
    for domain in domains:
        all_domain = all_df[all_df["domain"] == domain].sort_values("year")
        if len(all_domain) > 0:
            values = []
            for year in all_years:
                year_data = all_domain[all_domain["year"] == year]
                if len(year_data) > 0:
                    values.append(year_data["test_accuracy"].values[0])
                else:
                    values.append(0)
            domain_data[domain] = values
    
    # Plot bars: each domain is a group, each year is a bar within the group
    for i, year in enumerate(all_years):
        year_int = int(year)
        offset = (i - len(all_years) / 2 + 0.5) * width
        
        for j, domain in enumerate(domains):
            if domain in domain_data and domain_data[domain][i] > 0:
                val = domain_data[domain][i]
                bars = ax2.bar(x[j] + offset, val, width,
                              label=f"{year_int}" if j == 0 else "",
                              color=domain_colors[domain], alpha=0.7 - i * 0.1,
                              edgecolor="black", linewidth=1.5)
                
                # Add value labels on bars
                ax2.text(x[j] + offset, val + 0.005, f"{val:.3f}",
                       ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax2.set_xlabel("Domain", fontsize=18, fontweight="bold")
    ax2.set_ylabel("Test Accuracy", fontsize=18, fontweight="bold")
    ax2.set_title("RQ2 Yearly ML Validation: Accuracy by Domain (414 Static Features) - Bar Chart", 
                fontsize=20, fontweight="bold", pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.upper() for d in domains], fontsize=14)
    ax2.legend(fontsize=14, loc="best", framealpha=0.9, title="Year", title_fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=1.5, axis="y")
    ax2.set_ylim([0.85, 1.0])
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    output_path = output_dir / "yearly_accuracy_trends_combined.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved combined plot to {output_path}")
    plt.close()


def plot_comparison_table(ce_df: pd.DataFrame, all_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a comparison table visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")
    
    # Prepare data
    data = []
    for domain in ["academic", "blogs", "news"]:
        ce_domain = ce_df[ce_df["domain"] == domain].sort_values("year")
        all_domain = all_df[all_df["domain"] == domain].sort_values("year")
        
        for _, row in ce_domain.iterrows():
            year = int(row["year"])
            all_row = all_domain[all_domain["year"] == year].iloc[0]
            
            data.append({
                "Domain": domain.upper(),
                "Year": year,
                "CE Only": f"{row['test_accuracy']:.4f}",
                "All Features": f"{all_row['test_accuracy']:.4f}",
                "Difference": f"{all_row['test_accuracy'] - row['test_accuracy']:+.4f}",
                "CE ROC-AUC": f"{row['test_roc_auc']:.4f}",
                "All ROC-AUC": f"{all_row['test_roc_auc']:.4f}",
            })
    
    df_table = pd.DataFrame(data)
    
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor("#4A90E2")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Style cells
    for i in range(1, len(df_table) + 1):
        for j in range(len(df_table.columns)):
            if j == 4:  # Difference column
                val = float(df_table.iloc[i-1, j])
                if val > 0:
                    table[(i, j)].set_facecolor("#90EE90")
                elif val < 0:
                    table[(i, j)].set_facecolor("#FFB6C1")
    
    plt.title("RQ2 Yearly ML Validation Results: CE Only vs All Features", 
              fontsize=14, fontweight="bold", pad=20)
    
    output_path = output_dir / "yearly_comparison_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved table to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RQ2 yearly ML validation results.")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    ce_df = load_results("ce_only")
    all_df = load_results("all_features")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_yearly_trends(ce_df, all_df, OUTPUT_DIR)
    plot_yearly_trends_combined(ce_df, all_df, OUTPUT_DIR)
    plot_comparison_table(ce_df, all_df, OUTPUT_DIR)
    
    print("\n✅ All visualizations created!")


if __name__ == "__main__":
    main()

