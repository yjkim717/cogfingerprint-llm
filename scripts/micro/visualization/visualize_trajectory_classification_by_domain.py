#!/usr/bin/env python3
"""
Visualize trajectory classification results by domain (RQ2).

Generates a single figure showing ML performance comparison across three domains:
- Academic
- Blogs  
- News

Usage:
    python scripts/micro/visualization/visualize_trajectory_classification_by_domain.py \
        --results-dir plots/trajectory/combined \
        --level LV3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 11

# Color scheme for domains
DOMAIN_COLORS = {
    "academic": "#3498db",  # Blue
    "blogs": "#e74c3c",     # Red
    "news": "#2ecc71",       # Green
}

# Feature set order and labels
FEATURE_SET_ORDER = [
    "variability",
    "ce_geometry",
    "tfidf_geometry",
    "sbert_geometry",
    "geometry_all",
    "unified",
]

FEATURE_SET_LABELS = {
    "variability": "CE-VAR",
    "ce_geometry": "CE-GEO",
    "tfidf_geometry": "TFIDF-GEO",
    "sbert_geometry": "SBERT-GEO",
    "geometry_all": "Geometry All",
    "unified": "Unified",
}


def load_domain_results(results_dir: Path, level: str) -> Dict[str, pd.DataFrame]:
    """Load classification results for each domain."""
    results = {}
    for domain in DOMAINS:
        csv_path = results_dir / f"classification_results_{domain}_{level}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["domain"] = domain.capitalize()
            results[domain] = df
        else:
            print(f"⚠ Warning: Results file not found for {domain}: {csv_path}")
    return results


def plot_performance_by_domain(results: Dict[str, pd.DataFrame], output_dir: Path, level: str) -> None:
    """Plot performance metrics comparison across domains in a single figure."""
    # Prepare data
    all_data = []
    for domain, df in results.items():
        for _, row in df.iterrows():
            all_data.append({
                "domain": domain.capitalize(),
                "feature_set": row["feature_set"],
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "roc_auc_mean": row["roc_auc_mean"],
                "roc_auc_std": row["roc_auc_std"],
                "f1_mean": row["f1_mean"],
                "f1_std": row["f1_std"],
            })
    
    plot_df = pd.DataFrame(all_data)
    
    # Filter to feature sets we want to show
    plot_df = plot_df[plot_df["feature_set"].isin(FEATURE_SET_ORDER)]
    plot_df["feature_set"] = pd.Categorical(plot_df["feature_set"], categories=FEATURE_SET_ORDER, ordered=True)
    plot_df = plot_df.sort_values("feature_set")
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy", axes[0]),
        ("roc_auc_mean", "roc_auc_std", "ROC AUC", axes[1]),
        ("f1_mean", "f1_std", "F1 Score", axes[2]),
    ]
    
    x = np.arange(len(FEATURE_SET_ORDER))
    width = 0.25  # Width of bars
    
    for idx, (mean_col, std_col, title, ax) in enumerate(metrics):
        # Plot bars for each domain
        for i, domain in enumerate(DOMAINS):
            domain_label = domain.capitalize()
            domain_data = plot_df[plot_df["domain"] == domain_label]
            
            # Align data with feature set order
            values = []
            errors = []
            for feat in FEATURE_SET_ORDER:
                feat_data = domain_data[domain_data["feature_set"] == feat]
                if len(feat_data) > 0:
                    values.append(feat_data[mean_col].values[0])
                    errors.append(feat_data[std_col].values[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=5,
                alpha=0.8,
                label=domain_label,
                color=DOMAIN_COLORS[domain],
                edgecolor="black",
                linewidth=1,
            )
            
            # Add value labels on bars
            for j, (val, err) in enumerate(zip(values, errors)):
                if val > 0:
                    ax.text(
                        j + offset,
                        val + err + 0.02,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
        
        # Customize axes
        ax.set_xlabel("Feature Set", fontweight="bold", fontsize=12)
        ax.set_ylabel(title, fontweight="bold", fontsize=12)
        ax.set_title(title, fontweight="bold", fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_SET_LABELS.get(f, f) for f in FEATURE_SET_ORDER], 
                          rotation=45, ha="right", fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
        
        # Highlight unified feature set
        unified_idx = FEATURE_SET_ORDER.index("unified")
        ax.axvline(x=unified_idx, color="red", linestyle="--", linewidth=2, alpha=0.5, zorder=0)
    
    plt.suptitle(
        f"ML Classification Performance by Domain (RQ2 - {level})",
        fontweight="bold",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    
    output_path = output_dir / f"performance_by_domain_{level.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_summary_table(results: Dict[str, pd.DataFrame], output_dir: Path, level: str) -> None:
    """Create a summary table showing unified model performance across domains."""
    summary_data = []
    for domain, df in results.items():
        unified_row = df[df["feature_set"] == "unified"]
        if len(unified_row) > 0:
            row = unified_row.iloc[0]
            summary_data.append({
                "Domain": domain.capitalize(),
                "Accuracy": f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
                "ROC AUC": f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}",
                "F1 Score": f"{row['f1_mean']:.3f} ± {row['f1_std']:.3f}",
                "N Samples": int(row["n_samples"]),
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Style rows
    for i in range(1, len(summary_df) + 1):
        domain = summary_df.iloc[i - 1]["Domain"].lower()
        color = DOMAIN_COLORS[domain]
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)
    
    plt.title(
        f"Unified Model Performance Summary by Domain ({level})",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    
    output_path = output_dir / f"summary_table_by_domain_{level.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize trajectory classification results by domain (RQ2)."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="plots/trajectory/combined",
        help="Directory containing domain-specific classification results CSV files.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="LV3",
        choices=["LV1", "LV2", "LV3"],
        help="LLM level to visualize (default: LV3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: plots/trajectory/rq2_by_domain).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    results_dir = PROJECT_ROOT / args.results_dir
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load results for each domain
    print(f"📊 Loading classification results from: {results_dir}")
    results = load_domain_results(results_dir, args.level)
    
    if not results:
        print("❌ No results found for any domain.")
        return
    
    print(f"✅ Loaded results for {len(results)} domain(s): {', '.join(results.keys())}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PLOTS_ROOT / "rq2_by_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Generating visualizations...")
    
    # Generate plots
    plot_performance_by_domain(results, output_dir, args.level)
    plot_summary_table(results, output_dir, args.level)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()


