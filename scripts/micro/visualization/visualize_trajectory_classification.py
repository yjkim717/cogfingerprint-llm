#!/usr/bin/env python3
"""
Visualize trajectory classification results for unified feature set.

Generates:
1. Performance comparison bar charts (Accuracy, ROC-AUC, F1)
2. ROC curves for all feature sets
3. Confusion matrices
4. Feature importance (for unified model)

Usage:
    python scripts/micro/visualization/visualize_trajectory_classification.py \
        --results plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

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

# ---------------------------------------------------------------------------
# Feature groups (mirror run_trajectory_classification.py)
# ---------------------------------------------------------------------------
# CE-VAR: ONLY normalized variability metrics for CE features
#         - coefficient of variation (_cv)
#         - normalized RMSSD (_rmssd_norm)
#         - normalized MASD (_masd_norm)
VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")

GEOMETRY_PREFIXES = ("ce_", "tfidf_", "sbert_")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
)

# CE feature layers
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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def load_samples(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load trajectory features for all domains and models."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)
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


def geometry_columns(df: pd.DataFrame, spaces: List[str]) -> Dict[str, List[str]]:
    """Get geometry columns for specified spaces."""
    cols: Dict[str, List[str]] = {}
    for space in spaces:
        prefix = f"{space}_"
        space_cols = []
        for metric in GEOMETRY_METRICS:
            col = f"{prefix}{metric}"
            if col in df.columns:
                space_cols.append(col)
        cols[space] = space_cols
    cols["all"] = sum(cols.values(), [])
    return cols


def select_feature_sets(df: pd.DataFrame, geometry_spaces: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Select feature sets for classification.

    This mirrors the logic in scripts/micro/analysis/run_trajectory_classification.py:
      - variability : CE-VAR (CV, RMSSD_norm, MASD_norm only)
      - geometry_all: CE/TFIDF/SBERT geometry (5 metrics per space, no direction_consistency)
      - unified     : variability + all geometry metrics (no n_years columns)
    """
    # CE variability: ONLY CV / RMSSD_norm / MASD_norm columns
    variability_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]

    # Geometry
    geom_cols = geometry_columns(df, geometry_spaces + ["all"])

    # Unified = CE-VAR + all geometry metrics (no n_years, no direction_consistency)
    unified_cols = sorted(set(variability_cols + geom_cols["all"]))

    feature_sets: Dict[str, pd.DataFrame] = {
        "variability": df[variability_cols].fillna(0.0),
        "geometry_all": df[geom_cols["all"]].fillna(0.0),
        "unified": df[unified_cols].fillna(0.0),
    }
    for space in geometry_spaces:
        if geom_cols.get(space):
            feature_sets[f"{space}_geometry"] = df[geom_cols[space]].fillna(0.0)
    return feature_sets


def plot_performance_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance metrics comparison across feature sets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("roc_auc_mean", "roc_auc_std", "ROC-AUC"),
        ("f1_mean", "f1_std", "F1-Score"),
    ]

    for idx, (mean_col, std_col, title) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(results_df))
        bars = ax.bar(
            x,
            results_df[mean_col],
            yerr=results_df[std_col],
            capsize=5,
            alpha=0.8,
            color=sns.color_palette("husl", len(results_df)),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["feature_set"], rotation=45, ha="right")
        ax.set_ylabel(title)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)

        # Highlight unified
        for i, (bar, feature_set) in enumerate(zip(bars, results_df["feature_set"])):
            if feature_set == "unified":
                bar.set_edgecolor("red")
                bar.set_linewidth(3)

        # Add value labels
        for i, (mean_val, std_val) in enumerate(zip(results_df[mean_col], results_df[std_col])):
            ax.text(
                i,
                mean_val + std_val + 0.02,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = output_dir / "performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_roc_curves(
    df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot ROC curves for all feature sets."""
    y = (df["label"] == "human").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, X_df in feature_sets.items():
        X = X_df.to_numpy()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y[test_idx], proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc_score(y[test_idx], proba))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Highlight unified
        if name == "unified":
            ax.plot(
                mean_fpr,
                mean_tpr,
                label=f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
                linewidth=3,
                color="red",
            )
        else:
            ax.plot(
                mean_fpr,
                mean_tpr,
                label=f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
                alpha=0.7,
            )

    ax.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Human vs LLM Classification")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(
    df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot confusion matrices for all feature sets."""
    y = (df["label"] == "human").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_sets = len(feature_sets)
    cols = 3
    rows = (n_sets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (name, X_df) in enumerate(feature_sets.items()):
        ax = axes[idx]
        X = X_df.to_numpy()

        # Aggregate predictions across folds
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            all_y_true.extend(y[test_idx])
            all_y_pred.extend(y_pred)

        cm = confusion_matrix(all_y_true, all_y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar_kws={"shrink": 0.8},
            xticklabels=["LLM", "Human"],
            yticklabels=["LLM", "Human"],
        )
        ax.set_title(f"{name}\n(Acc: {accuracy_score(all_y_true, all_y_pred):.3f})")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    # Hide unused subplots
    for idx in range(len(feature_sets), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = output_dir / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def classify_feature_source(feature_name: str) -> Tuple[str, Optional[str]]:
    """
    Classify a feature into its source category and CE layer (if applicable).
    
    Returns:
        (source_category, ce_layer)
        source_category: 'CE-VAR', 'CE-GEO', 'TFIDF-GEO', 'SBERT-GEO', or 'Other'
        ce_layer: 'Cognitive', 'Emotional', 'Stylistic', or None
    """
    # CE-VAR: ends with variability suffixes
    if any(feature_name.endswith(suffix) for suffix in VAR_SUFFIXES):
        # Extract base feature name (remove suffix)
        base_name = feature_name
        for suffix in VAR_SUFFIXES:
            if feature_name.endswith(suffix):
                base_name = feature_name[: -len(suffix)]
                break
        
        # Determine CE layer
        ce_layer = None
        if base_name in COGNITIVE_FEATURES:
            ce_layer = "Cognitive"
        elif base_name in EMOTIONAL_FEATURES:
            ce_layer = "Emotional"
        elif base_name in STYLISTIC_FEATURES:
            ce_layer = "Stylistic"
        
        return ("CE-VAR", ce_layer)
    
    # CE-GEO: starts with "ce_" and ends with geometry metric
    if feature_name.startswith("ce_") and any(feature_name.endswith(f"_{m}") for m in GEOMETRY_METRICS):
        # Extract base feature name (remove "ce_" prefix and geometry metric suffix)
        base_name = feature_name[3:]  # Remove "ce_"
        for metric in GEOMETRY_METRICS:
            if base_name.endswith(f"_{metric}"):
                base_name = base_name[: -len(f"_{metric}")]
                break
        
        # Determine CE layer
        ce_layer = None
        if base_name in COGNITIVE_FEATURES:
            ce_layer = "Cognitive"
        elif base_name in EMOTIONAL_FEATURES:
            ce_layer = "Emotional"
        elif base_name in STYLISTIC_FEATURES:
            ce_layer = "Stylistic"
        
        return ("CE-GEO", ce_layer)
    
    # TFIDF-GEO
    if feature_name.startswith("tfidf_"):
        return ("TFIDF-GEO", None)
    
    # SBERT-GEO
    if feature_name.startswith("sbert_"):
        return ("SBERT-GEO", None)
    
    return ("Other", None)


def plot_feature_importance(
    df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot feature importance for unified model, grouped by source category."""
    if "unified" not in feature_sets:
        print("⚠ Unified feature set not found, skipping feature importance plot.")
        return

    y = (df["label"] == "human").astype(int).values
    X_df = feature_sets["unified"]
    X = X_df.to_numpy()

    # Train on full data to get feature importance
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Get feature importance
    importances = clf.feature_importances_
    feature_names = X_df.columns.tolist()

    # Classify all features
    feature_data = []
    for name, importance in zip(feature_names, importances):
        source, layer = classify_feature_source(name)
        feature_data.append({
            "name": name,
            "importance": importance,
            "source": source,
            "layer": layer,
        })
    
    # Group by source
    source_groups = {
        "CE-VAR": [],
        "CE-GEO": [],
        "TFIDF-GEO": [],
        "SBERT-GEO": [],
    }
    
    for feat in feature_data:
        source = feat["source"]
        if source in source_groups:
            source_groups[source].append(feat)
    
    # Sort each group by importance
    for source in source_groups:
        source_groups[source].sort(key=lambda x: x["importance"], reverse=True)
    
    # Color scheme
    source_colors = {
        "CE-VAR": "#3498db",      # Blue
        "CE-GEO": "#e74c3c",      # Red
        "TFIDF-GEO": "#2ecc71",   # Green
        "SBERT-GEO": "#f39c12",   # Orange
    }
    
    # Create subplots for each source category
    sources_to_plot = [s for s in ["CE-VAR", "CE-GEO", "TFIDF-GEO", "SBERT-GEO"] if len(source_groups[s]) > 0]
    n_sources = len(sources_to_plot)
    
    if n_sources == 0:
        print("⚠ No features found in any source category.")
        return
    
    # Create figure with subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    max_importance = max(feat["importance"] for feat in feature_data)
    
    for idx, source in enumerate(["CE-VAR", "CE-GEO", "TFIDF-GEO", "SBERT-GEO"]):
        ax = axes[idx]
        features = source_groups[source]
        
        if len(features) == 0:
            ax.text(0.5, 0.5, f"No {source} features", 
                   ha="center", va="center", transform=ax.transAxes,
                   fontsize=14, fontweight="bold")
            ax.set_title(f"{source}", fontweight="bold", fontsize=12)
            ax.axis("off")
            continue
        
        # Take top features for this source (or all if less than 15)
        top_n = min(15, len(features))
        top_features = features[:top_n]
        
        # Prepare data for plotting
        names = [f["name"] for f in top_features]
        imps = [f["importance"] for f in top_features]
        layers = [f["layer"] for f in top_features]
        
        # Create labels with layer info for CE features
        labels = []
        for name, layer in zip(names, layers):
            if layer:
                # Shorten name if too long
                if len(name) > 30:
                    name = name[:27] + "..."
                label = f"{name}\n({layer})"
            else:
                if len(name) > 30:
                    name = name[:27] + "..."
                label = name
            labels.append(label)
        
        # Plot horizontal bars
        y_pos = np.arange(len(labels))
        color = source_colors[source]
        
        bars = ax.barh(y_pos, imps, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance", fontweight="bold", fontsize=10)
        ax.set_title(f"{source} ({len(features)} features)", fontweight="bold", fontsize=12)
        ax.set_xlim(0, max_importance * 1.1)
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, imps)):
            ax.text(imp + max_importance * 0.01, i, f"{imp:.4f}",
                   va="center", fontsize=7)
    
    plt.suptitle(
        "Top Important Features by Source Category (Unified Model)",
        fontweight="bold",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Add summary statistics
    source_counts = {}
    layer_counts = {}
    for feat in feature_data:
        source = feat["source"]
        layer = feat["layer"]
        if source in source_groups:
            source_counts[source] = len(source_groups[source])
            if layer:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Print summary to console
    print("\n" + "="*60)
    print("Feature Source Distribution:")
    print("="*60)
    for source in ["CE-VAR", "CE-GEO", "TFIDF-GEO", "SBERT-GEO"]:
        count = source_counts.get(source, 0)
        if count > 0:
            print(f"  {source:12s}: {count:3d} features")
    
    if layer_counts:
        print("\nCE Features Layer Distribution:")
        print("-"*60)
        total_ce = sum(layer_counts.values())
        for layer in ["Cognitive", "Emotional", "Stylistic"]:
            count = layer_counts.get(layer, 0)
            if total_ce > 0:
                pct = count / total_ce * 100
                print(f"  {layer:12s}: {count:3d} ({pct:5.1f}% of CE features)")

    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trajectory classification results.")
    parser.add_argument(
        "--results",
        type=str,
        default="plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv",
        help="Path to classification results CSV",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to include (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="LLM models to include (default: all providers).",
    )
    parser.add_argument(
        "--level",
        default="LV3",
        choices=LEVELS,
        help="LLM level to include (default: LV3).",
    )
    parser.add_argument(
        "--geometry-spaces",
        nargs="+",
        choices=["ce", "tfidf", "sbert"],
        default=["ce", "tfidf", "sbert"],
        help="Which geometry spaces to include (default: all three).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: plots/trajectory/visualization)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load results
    results_path = PROJECT_ROOT / args.results
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return

    results_df = pd.read_csv(results_path)
    print(f"📊 Loaded results: {len(results_df)} feature sets")

    # Load data for visualization
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available for visualization.")
        return

    print(f"📈 Loaded {len(df)} samples for visualization")

    # Select feature sets
    feature_sets = select_feature_sets(df, args.geometry_spaces)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PLOTS_ROOT / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Generating visualizations...")

    # Generate plots
    plot_performance_comparison(results_df, output_dir)
    plot_roc_curves(df, feature_sets, output_dir)
    plot_confusion_matrices(df, feature_sets, output_dir)
    plot_feature_importance(df, feature_sets, output_dir)

    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

