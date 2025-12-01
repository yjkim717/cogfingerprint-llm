#!/usr/bin/env python3
"""
Visualize RQ2 yearly ML validation results: accuracy trends over years.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    plot_comparison_table(ce_df, all_df, OUTPUT_DIR)
    
    print("\n✅ All visualizations created!")


if __name__ == "__main__":
    main()

