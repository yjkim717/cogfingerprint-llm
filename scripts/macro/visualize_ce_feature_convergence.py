#!/usr/bin/env python3
"""
Visualize CE feature convergence analysis results, grouped by three layers:
1. Cognitive Layer (Big Five personality features)
2. Emotional Layer (NELA emotional features)
3. Stylistic Layer (NELA stylistic features)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Constants
DOMAINS = ["academic", "blogs", "news"]
RESULTS_DIR = Path("macro_results/rq2_ce_feature_convergence")
PLOTS_DIR = Path("plots/rq2_ce_feature_convergence")

# Feature categorization
COGNITIVE_FEATURES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

EMOTIONAL_FEATURES = [
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
]

STYLISTIC_FEATURES = [
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

ALL_FEATURES = COGNITIVE_FEATURES + EMOTIONAL_FEATURES + STYLISTIC_FEATURES

# Color scheme
COLORS = {
    "converging": "#2ecc71",  # Green
    "diverging": "#e74c3c",  # Red
    "stable": "#95a5a6",  # Gray
    "significant": "#f39c12",  # Orange for highlighting
}


def load_results(results_file: Path) -> pd.DataFrame:
    """Load convergence analysis results."""
    with results_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = []
    for item in data:
        feature = item["feature"]
        domain = item["domain"]
        trend_info = item["trend_analysis"]["mean_difference"]

        summary.append(
            {
                "feature": feature,
                "domain": domain,
                "trend": trend_info["trend"],
                "slope": trend_info["slope"],
                "correlation": trend_info["correlation"],
                "p_value": trend_info["p_value"],
                "significant": trend_info["significant"],
            }
        )

    return pd.DataFrame(summary)


def get_feature_layer(feature: str) -> str:
    """Get the layer category for a feature."""
    if feature in COGNITIVE_FEATURES:
        return "Cognitive"
    elif feature in EMOTIONAL_FEATURES:
        return "Emotional"
    elif feature in STYLISTIC_FEATURES:
        return "Stylistic"
    else:
        return "Unknown"


def create_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create a heatmap showing convergence trends by feature and domain."""
    # Create pivot table: feature x domain -> trend value
    # Values: 1 for converging, -1 for diverging, 0 for stable
    trend_map = {"converging": 1, "stable": 0, "diverging": -1}

    pivot_data = []
    for feature in ALL_FEATURES:
        row = {"feature": feature, "layer": get_feature_layer(feature)}
        for domain in DOMAINS:
            feat_domain = df[(df["feature"] == feature) & (df["domain"] == domain)]
            if len(feat_domain) > 0:
                trend = feat_domain.iloc[0]["trend"]
                significant = feat_domain.iloc[0]["significant"]
                # Use 1.5/-1.5 for significant trends
                value = trend_map[trend] * (1.5 if significant else 1.0)
                row[domain] = value
            else:
                row[domain] = 0
        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data)

    # Sort by layer, then by feature
    layer_order = {"Cognitive": 0, "Emotional": 1, "Stylistic": 2}
    pivot_df["layer_order"] = pivot_df["layer"].map(layer_order)
    pivot_df = pivot_df.sort_values(["layer_order", "feature"]).reset_index(drop=True)

    # Create the heatmap data
    heatmap_data = pivot_df[DOMAINS].values
    feature_labels = pivot_df["feature"].values
    layer_labels = pivot_df["layer"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Create custom colormap: diverging colormap centered at 0
    cmap = sns.diverging_palette(240, 10, as_cmap=True, center="light")
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=-1.5, vmax=1.5)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(DOMAINS)))
    ax.set_xticklabels([d.upper() for d in DOMAINS], fontsize=11)
    ax.set_yticks(np.arange(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=9)

    # Add layer separators
    current_layer = None
    for i, layer in enumerate(layer_labels):
        if layer != current_layer:
            if current_layer is not None:
                ax.axhline(i - 0.5, color="black", linewidth=1.5, linestyle="--")
            current_layer = layer

    # Add text annotations
    for i in range(len(feature_labels)):
        for j in range(len(DOMAINS)):
            value = heatmap_data[i, j]
            trend = "C" if value > 0 else "D" if value < 0 else "S"
            sig = "*" if abs(value) >= 1.5 else ""
            text = ax.text(
                j,
                i,
                f"{trend}{sig}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.5 else "black",
                fontsize=8,
                fontweight="bold" if sig else "normal",
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Trend (Converging → Diverging)", rotation=270, labelpad=20)
    cbar.set_ticks([-1.5, -1, 0, 1, 1.5])
    cbar.set_ticklabels(["Diverging*", "Diverging", "Stable", "Converging", "Converging*"])

    # Add layer labels on the right
    layer_positions = {}
    for i, layer in enumerate(layer_labels):
        if layer not in layer_positions:
            layer_positions[layer] = i
        layer_positions[layer] = (layer_positions[layer] + i) / 2

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(list(layer_positions.values()))
    ax2.set_yticklabels(list(layer_positions.keys()), fontsize=10, fontweight="bold")
    ax2.set_ylabel("Layer", fontsize=12, fontweight="bold")

    ax.set_xlabel("Domain", fontsize=12, fontweight="bold")
    ax.set_title(
        "CE Feature Convergence Trends by Layer\n(C=Converging, D=Diverging, S=Stable, *=p<0.05)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved heatmap to {output_path}")
    plt.close()


def create_layer_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar charts showing convergence summary by layer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    layers = ["Cognitive", "Emotional", "Stylistic"]
    layer_features = {
        "Cognitive": COGNITIVE_FEATURES,
        "Emotional": EMOTIONAL_FEATURES,
        "Stylistic": STYLISTIC_FEATURES,
    }

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        features = layer_features[layer]
        layer_df = df[df["feature"].isin(features)]

        # Count trends by domain
        domain_counts = {}
        for domain in DOMAINS:
            domain_df = layer_df[layer_df["domain"] == domain]
            counts = domain_df["trend"].value_counts().to_dict()
            domain_counts[domain] = {
                "converging": counts.get("converging", 0),
                "diverging": counts.get("diverging", 0),
                "stable": counts.get("stable", 0),
            }

        # Create grouped bar chart
        x = np.arange(len(DOMAINS))
        width = 0.25

        converging_counts = [domain_counts[d]["converging"] for d in DOMAINS]
        stable_counts = [domain_counts[d]["stable"] for d in DOMAINS]
        diverging_counts = [domain_counts[d]["diverging"] for d in DOMAINS]

        bars1 = ax.bar(
            x - width,
            converging_counts,
            width,
            label="Converging",
            color=COLORS["converging"],
            alpha=0.8,
        )
        bars2 = ax.bar(x, stable_counts, width, label="Stable", color=COLORS["stable"], alpha=0.8)
        bars3 = ax.bar(
            x + width,
            diverging_counts,
            width,
            label="Diverging",
            color=COLORS["diverging"],
            alpha=0.8,
        )

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_xlabel("Domain", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Features", fontsize=11, fontweight="bold")
        ax.set_title(f"{layer} Layer", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in DOMAINS])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, max(max(converging_counts), max(stable_counts), max(diverging_counts)) * 1.2)

    plt.suptitle(
        "CE Feature Convergence Summary by Layer",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved layer summary to {output_path}")
    plt.close()


def create_significant_features_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create a plot highlighting significant converging/diverging features."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    layers = ["Cognitive", "Emotional", "Stylistic"]
    layer_features = {
        "Cognitive": COGNITIVE_FEATURES,
        "Emotional": EMOTIONAL_FEATURES,
        "Stylistic": STYLISTIC_FEATURES,
    }

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        features = layer_features[layer]
        layer_df = df[df["feature"].isin(features)]

        # Get significant features
        sig_df = layer_df[layer_df["significant"]].copy()
        sig_df = sig_df.sort_values(["trend", "p_value"])

        if len(sig_df) == 0:
            ax.text(0.5, 0.5, "No significant features", ha="center", va="center", fontsize=12)
            ax.set_title(f"{layer} Layer - Significant Features", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        # Create grouped data
        y_pos = np.arange(len(sig_df))
        colors = [
            COLORS["converging"] if t == "converging" else COLORS["diverging"]
            for t in sig_df["trend"]
        ]

        # Plot bars
        abs_slopes = np.abs(sig_df["slope"].values)
        bars = ax.barh(
            y_pos,
            abs_slopes,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add feature labels with domain
        labels = []
        for _, row in sig_df.iterrows():
            label = f"{row['feature']} ({row['domain']})"
            labels.append(label)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)

        # Add p-value annotations
        for i, (_, row) in enumerate(sig_df.iterrows()):
            abs_slope = abs(row["slope"])
            ax.text(
                abs_slope * 1.05,
                i,
                f"p={row['p_value']:.4f}",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel("Absolute Slope (Convergence Rate)", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{layer} Layer - Significant Features (p<0.05)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=COLORS["converging"], label="Converging"),
            Patch(facecolor=COLORS["diverging"], label="Diverging"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved significant features plot to {output_path}")
    plt.close()


def create_trend_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create a comparison plot showing trends across layers."""
    fig, ax = plt.subplots(figsize=(12, 8))

    layers = ["Cognitive", "Emotional", "Stylistic"]
    layer_features = {
        "Cognitive": COGNITIVE_FEATURES,
        "Emotional": EMOTIONAL_FEATURES,
        "Stylistic": STYLISTIC_FEATURES,
    }

    # Calculate statistics for each layer
    layer_stats = []
    for layer in layers:
        features = layer_features[layer]
        layer_df = df[df["feature"].isin(features)]

        for domain in DOMAINS:
            domain_df = layer_df[layer_df["domain"] == domain]
            converging = len(domain_df[domain_df["trend"] == "converging"])
            diverging = len(domain_df[domain_df["trend"] == "diverging"])
            stable = len(domain_df[domain_df["trend"] == "stable"])
            total = len(domain_df)

            layer_stats.append(
                {
                    "layer": layer,
                    "domain": domain,
                    "converging_pct": converging / total * 100 if total > 0 else 0,
                    "diverging_pct": diverging / total * 100 if total > 0 else 0,
                    "stable_pct": stable / total * 100 if total > 0 else 0,
                }
            )

    stats_df = pd.DataFrame(layer_stats)

    # Create stacked bar chart
    x = np.arange(len(layers))
    width = 0.25

    for i, domain in enumerate(DOMAINS):
        domain_stats = stats_df[stats_df["domain"] == domain]
        converging_vals = [domain_stats[domain_stats["layer"] == l]["converging_pct"].values[0] for l in layers]
        stable_vals = [domain_stats[domain_stats["layer"] == l]["stable_pct"].values[0] for l in layers]
        diverging_vals = [domain_stats[domain_stats["layer"] == l]["diverging_pct"].values[0] for l in layers]

        bottom = np.zeros(len(layers))
        ax.bar(
            x + i * width,
            converging_vals,
            width,
            label=f"{domain.upper()} - Converging",
            color=COLORS["converging"],
            alpha=0.7,
            bottom=bottom,
        )
        bottom += converging_vals
        ax.bar(
            x + i * width,
            stable_vals,
            width,
            label=f"{domain.upper()} - Stable",
            color=COLORS["stable"],
            alpha=0.7,
            bottom=bottom,
        )
        bottom += stable_vals
        ax.bar(
            x + i * width,
            diverging_vals,
            width,
            label=f"{domain.upper()} - Diverging",
            color=COLORS["diverging"],
            alpha=0.7,
            bottom=bottom,
        )

    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Percentage of Features (%)", fontsize=12, fontweight="bold")
    ax.set_title("CE Feature Convergence Trends by Layer and Domain", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(layers)
    ax.legend(ncol=3, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved trend comparison to {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CE feature convergence analysis results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR / "ce_feature_convergence_all.json",
        help="Path to results JSON file.",
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading results from {args.results}...")
    df = load_results(args.results)
    print(f"Loaded {len(df)} feature-domain combinations")

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. Heatmap
    create_heatmap(df, PLOTS_DIR / "ce_feature_convergence_heatmap.png")

    # 2. Layer summary
    create_layer_summary(df, PLOTS_DIR / "ce_feature_convergence_by_layer.png")

    # 3. Significant features
    create_significant_features_plot(
        df, PLOTS_DIR / "ce_feature_convergence_significant.png"
    )

    # 4. Trend comparison
    create_trend_comparison(df, PLOTS_DIR / "ce_feature_convergence_trends.png")

    print(f"\n✅ All visualizations saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

