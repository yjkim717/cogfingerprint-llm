#!/usr/bin/env python3
"""
Create a comprehensive visualization panel for Unified Trajectory Model results.

Combines:
1. Performance comparison (bar chart)
2. ROC curves
3. Confusion matrix
4. Feature importance (top features)

All in one figure.
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

VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")
GEOMETRY_PREFIXES = ("ce_", "tfidf_", "sbert_")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
    "direction_consistency",
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
plt.rcParams["font.size"] = 10


def load_samples(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load trajectory features."""
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
    """Select feature sets for classification."""
    # CE variability
    variability_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]

    # Geometry
    geom_cols = geometry_columns(df, geometry_spaces + ["all"])

    # Unified: all numeric features except metadata
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    unified_cols = [c for c in numeric_cols if c not in {"author_id", "sample_count"}]

    feature_sets: Dict[str, pd.DataFrame] = {
        "variability": df[variability_cols].fillna(0.0),
        "geometry_all": df[geom_cols["all"]].fillna(0.0),
        "unified": df[unified_cols].fillna(0.0),
    }
    for space in geometry_spaces:
        if geom_cols.get(space):
            feature_sets[f"{space}_geometry"] = df[geom_cols[space]].fillna(0.0)
    return feature_sets


def create_comprehensive_panel(
    df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame], results_df: pd.DataFrame, output_path: Path
) -> None:
    """Create a comprehensive visualization panel."""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Performance Comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_performance_comparison(ax1, results_df)

    # 2. ROC Curves (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_roc_curves_simple(ax2, df, feature_sets)

    # 3. Confusion Matrix for Unified (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_confusion_matrix_unified(ax3, df, feature_sets)

    # 4. Feature Importance (middle right, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, 1:])
    plot_feature_importance_unified(ax4, df, feature_sets)

    # 5. Feature Set Comparison Table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    plot_feature_set_table(ax5, results_df)

    # Add title
    fig.suptitle(
        "Unified Trajectory Model: CE-Var + CE-Geo + TF-IDF-Geo + SBERT-Geo\n"
        "Human vs LLM (LV3) Classification Results",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved comprehensive panel: {output_path}")
    plt.close()


def plot_performance_comparison(ax, results_df: pd.DataFrame) -> None:
    """Plot performance metrics comparison."""
    x = np.arange(len(results_df))
    width = 0.25

    metrics = ["accuracy_mean", "roc_auc_mean", "f1_mean"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1-Score"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        means = results_df[metric].values
        stds = results_df[f"{metric.replace('_mean', '_std')}"].values
        ax.bar(
            x + idx * width,
            means,
            width,
            yerr=stds,
            label=label,
            color=color,
            alpha=0.8,
            capsize=3,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["feature_set"], rotation=45, ha="right")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Performance Comparison Across Feature Sets", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left")
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)

    # Highlight unified
    unified_idx = results_df[results_df["feature_set"] == "unified"].index[0]
    for idx in range(len(metrics)):
        ax.bar(
            unified_idx + idx * width,
            results_df.loc[unified_idx, metrics[idx]],
            width,
            edgecolor="red",
            linewidth=2,
            fill=False,
        )


def plot_roc_curves_simple(ax, df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame]) -> None:
    """Plot ROC curves for key feature sets."""
    y = (df["label"] == "human").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    key_sets = ["variability", "geometry_all", "unified"]
    colors = {"variability": "#95a5a6", "geometry_all": "#f39c12", "unified": "#e74c3c"}
    linestyles = {"variability": "--", "geometry_all": "-.", "unified": "-"}

    for name in key_sets:
        if name not in feature_sets:
            continue
        X_df = feature_sets[name]
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

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{name}\n(AUC={mean_auc:.3f}±{std_auc:.3f})",
            linewidth=2.5 if name == "unified" else 2,
            color=colors[name],
            linestyle=linestyles[name],
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)


def plot_confusion_matrix_unified(ax, df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame]) -> None:
    """Plot confusion matrix for unified model."""
    if "unified" not in feature_sets:
        ax.text(0.5, 0.5, "Unified model not found", ha="center", va="center", transform=ax.transAxes)
        return

    y = (df["label"] == "human").astype(int).values
    X_df = feature_sets["unified"]
    X = X_df.to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    acc = accuracy_score(all_y_true, all_y_pred)

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
    ax.set_title(f"Confusion Matrix (Unified)\nAccuracy: {acc:.3f}", fontweight="bold", fontsize=12)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")


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


def plot_feature_importance_unified(ax, df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame]) -> None:
    """Plot feature importance for unified model with source and layer classification."""
    if "unified" not in feature_sets:
        ax.text(0.5, 0.5, "Unified model not found", ha="center", va="center", transform=ax.transAxes)
        return

    y = (df["label"] == "human").astype(int).values
    X_df = feature_sets["unified"]
    X = X_df.to_numpy()

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    feature_names = X_df.columns.tolist()

    indices = np.argsort(importances)[::-1]
    top_n = 20

    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Classify features
    feature_sources = []
    feature_layers = []
    colors = []
    
    # Color scheme
    source_colors = {
        "CE-VAR": "#3498db",      # Blue
        "CE-GEO": "#e74c3c",      # Red
        "TFIDF-GEO": "#2ecc71",   # Green
        "SBERT-GEO": "#f39c12",   # Orange
        "Other": "#95a5a6",        # Gray
    }
    
    for name in top_features:
        source, layer = classify_feature_source(name)
        feature_sources.append(source)
        feature_layers.append(layer)
        colors.append(source_colors.get(source, "#95a5a6"))
    
    # Create labels with source and layer info
    labels = []
    for i, name in enumerate(top_features):
        source = feature_sources[i]
        layer = feature_layers[i]
        if layer:
            label = f"{name}\n({source}, {layer})"
        else:
            label = f"{name}\n({source})"
        labels.append(label)
    
    # Plot
    bars = ax.barh(range(top_n), top_importances, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontweight="bold", fontsize=11)
    ax.set_title(f"Top {top_n} Most Important Features (Unified Model)\nSource & CE Layer Classification", 
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    
    # Add legend with source categories
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor="#3498db", label="CE-VAR"),
        Patch(facecolor="#e74c3c", label="CE-GEO"),
        Patch(facecolor="#2ecc71", label="TFIDF-GEO"),
        Patch(facecolor="#f39c12", label="SBERT-GEO"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)
    
    # Add summary statistics
    source_counts = {}
    layer_counts = {}
    for source, layer in zip(feature_sources, feature_layers):
        source_counts[source] = source_counts.get(source, 0) + 1
        if layer:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Print summary to console
    print("\n" + "="*60)
    print("Top Features Source Distribution:")
    print("="*60)
    for source in ["CE-VAR", "CE-GEO", "TFIDF-GEO", "SBERT-GEO"]:
        count = source_counts.get(source, 0)
        pct = count / top_n * 100
        print(f"  {source:12s}: {count:2d} ({pct:5.1f}%)")
    
    if layer_counts:
        print("\nCE Features Layer Distribution:")
        print("-"*60)
        for layer in ["Cognitive", "Emotional", "Stylistic"]:
            count = layer_counts.get(layer, 0)
            total_ce = sum(layer_counts.values())
            if total_ce > 0:
                pct = count / total_ce * 100
                print(f"  {layer:12s}: {count:2d} ({pct:5.1f}% of CE features)")


def plot_feature_set_table(ax, results_df: pd.DataFrame) -> None:
    """Plot feature set comparison as a table."""
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row["feature_set"],
            f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}",
            f"{row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}",
            f"{row['f1_mean']:.3f}±{row['f1_std']:.3f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Feature Set", "Accuracy", "ROC-AUC", "F1-Score"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Highlight unified row
    unified_idx = results_df[results_df["feature_set"] == "unified"].index[0] + 1
    for col in range(4):
        table[(unified_idx, col)].set_facecolor("#ffcccc")
        table[(unified_idx, col)].set_text_props(weight="bold")

    # Style header
    for col in range(4):
        table[(0, col)].set_facecolor("#34495e")
        table[(0, col)].set_text_props(weight="bold", color="white")

    ax.set_title("Feature Set Performance Summary", fontweight="bold", fontsize=12, pad=20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comprehensive visualization panel for unified trajectory model.")
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
        default=["DS", "G4B", "G12B"],
        help="LLM models to include (default: DS, G4B, G12B).",
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
        "--output",
        type=str,
        default=None,
        help="Output file path (default: plots/trajectory/visualization/unified_trajectory_panel.png)",
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

    # Load data
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available for visualization.")
        return

    print(f"📈 Loaded {len(df)} samples for visualization")

    # Select feature sets
    feature_sets = select_feature_sets(df, args.geometry_spaces)

    # Create output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PLOTS_ROOT / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "unified_trajectory_panel.png"

    print(f"\n🎨 Creating comprehensive visualization panel...")
    create_comprehensive_panel(df, feature_sets, results_df, output_path)

    print(f"\n✅ Visualization complete: {output_path}")


if __name__ == "__main__":
    main()

