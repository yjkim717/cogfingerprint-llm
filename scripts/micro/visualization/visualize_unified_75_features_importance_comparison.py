#!/usr/bin/env python3
"""
Feature Significance Visualization: Top feature importance comparison for unified 75 features (RQ1 LV3).

**Purpose**: Demonstrate which features are most important for distinguishing Human vs. LLM text.
This validates that our trajectory features (CE-VAR, Geometry) are indeed significant
and contribute meaningfully to classification.

Creates grouped visualizations comparing feature importances by feature type
(CE-VAR, CE-Geometry, TF-IDF-Geometry, SBERT-Geometry).

Usage:
    python scripts/micro/visualization/visualize_unified_75_features_importance_comparison.py \
        --results plots/trajectory/unified_75_features_ml/feature_importance_data.csv \
        --output plots/trajectory/unified_75_features_ml
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")

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
)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"

# CE feature layers (Cognitive, Emotional, Stylistic)
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
    "type_token_ratio",
]

COLORS = {
    # CE-VAR sub-layers (different shades of blue)
    'CE-VAR-Cognitive': '#1E5F8C',      # Darker blue
    'CE-VAR-Emotional': '#4A90C2',      # Medium blue
    'CE-VAR-Stylistic': '#87CEEB',      # Light blue
    # If CE-VAR cannot be classified
    'CE-VAR': '#2E86AB',                # Default blue
    # Geometry features
    'CE-Geometry': '#A23B72',           # Purple
    'TF-IDF-Geometry': '#F18F01',       # Orange
    'SBERT-Geometry': '#C73E1D',        # Red
}


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


def select_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select unified 75 features (CE-VAR + Geometry)."""
    # CE variability columns
    variability_cols = [
        col for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]
    
    # Geometry columns
    geometry_cols = []
    for prefix in GEOMETRY_PREFIXES:
        for metric in GEOMETRY_METRICS:
            col = f"{prefix}{metric}"
            if col in df.columns:
                geometry_cols.append(col)
    
    # Unified features
    unified_cols = sorted(set(variability_cols + geometry_cols))
    return df[unified_cols].fillna(0.0)


def classify_feature_type(feature: str) -> str:
    """Classify feature into category with CE-VAR sub-layers."""
    # Check if it's a CE-VAR feature
    if any(feature.endswith(suffix) for suffix in VAR_SUFFIXES):
        # Extract base feature name (remove suffix)
        base_name = feature
        for suffix in VAR_SUFFIXES:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        
        # Classify into sub-layers
        if base_name in COGNITIVE_FEATURES:
            return "CE-VAR-Cognitive"
        elif base_name in EMOTIONAL_FEATURES:
            return "CE-VAR-Emotional"
        elif base_name in STYLISTIC_FEATURES:
            return "CE-VAR-Stylistic"
        else:
            return "CE-VAR"  # Fallback for unclassified CE-VAR
    
    # Geometry features
    if feature.startswith("ce_"):
        return "CE-Geometry"
    if feature.startswith("tfidf_"):
        return "TF-IDF-Geometry"
    if feature.startswith("sbert_"):
        return "SBERT-Geometry"
    return "Other"


def compute_feature_importance(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Compute feature importance using Random Forest with CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_importances: List[np.ndarray] = []
    
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        feature_importances.append(clf.feature_importances_)
    
    # Average feature importance
    avg_importance = np.mean(feature_importances, axis=0)
    std_importance = np.std(feature_importances, axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': avg_importance,
        'importance_std': std_importance,
    })
    feature_importance_df['feature_type'] = feature_importance_df['feature'].apply(classify_feature_type)
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    return feature_importance_df


def plot_top_importance_comparison(fi_df: pd.DataFrame, output_dir: Path, top_n: int = 30):
    """Plot top N feature importances grouped by feature type."""
    # Get top N
    top_df = fi_df.head(top_n).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} Feature Importances: Unified 75 Features (RQ1 LV3)\nGrouped by Feature Type', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Overall top N (horizontal bar)
    ax1 = axes[0, 0]
    colors_list = [COLORS.get(ft, '#888888') for ft in top_df['feature_type']]
    bars = ax1.barh(range(len(top_df)), top_df['importance'], color=colors_list, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax1.errorbar(top_df['importance'], range(len(top_df)), 
                xerr=top_df['importance_std'], fmt='none', color='black', alpha=0.3)
    
    # Short feature names
    short_names = []
    for feat in top_df['feature']:
        if any(feat.endswith(s) for s in VAR_SUFFIXES):
            parts = feat.split('_')
            short = parts[0][:8] + '_' + parts[-1][:3] if len(parts) > 1 else parts[0][:10]
        else:
            short = feat.replace('ce_', 'ce_').replace('tfidf_', 'tf_').replace('sbert_', 'sb_')
        short_names.append(short)
    
    ax1.set_yticks(range(len(top_df)))
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Top 30 Features Overall', fontsize=12, fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 2. By feature type - boxplot (group CE-VAR sub-layers)
    ax2 = axes[0, 1]
    # Order: CE-VAR sub-layers first, then geometry
    type_order = ['CE-VAR-Cognitive', 'CE-VAR-Emotional', 'CE-VAR-Stylistic', 
                  'CE-VAR', 'CE-Geometry', 'TF-IDF-Geometry', 'SBERT-Geometry']
    type_data = [fi_df[fi_df['feature_type'] == ft]['importance'].values for ft in type_order]
    type_data = [d for d in type_data if len(d) > 0]  # Remove empty
    type_labels = [ft for ft in type_order if len(fi_df[fi_df['feature_type'] == ft]) > 0]
    
    bp = ax2.boxplot(type_data, labels=type_labels, patch_artist=True)
    for patch, ft in zip(bp['boxes'], type_labels):
        patch.set_facecolor(COLORS.get(ft, '#888888'))
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Feature Importance', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Distribution by Feature Type', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Top N per type (with CE-VAR sub-layers)
    ax3 = axes[1, 0]
    top_per_type = []
    type_colors = []
    type_labels_list = []
    # Use same order as panel B
    panel_b_type_order = ['CE-VAR-Cognitive', 'CE-VAR-Emotional', 'CE-VAR-Stylistic', 
                          'CE-VAR', 'CE-Geometry', 'TF-IDF-Geometry', 'SBERT-Geometry']
    for ft in panel_b_type_order:
        type_features = fi_df[fi_df['feature_type'] == ft].head(5)
        if len(type_features) > 0:
            top_per_type.extend(type_features['importance'].values)
            type_colors.extend([COLORS.get(ft, '#888888')] * len(type_features))
            # Shorten labels for readability
            short_ft = ft.replace('CE-VAR-', 'CE-')  # Shorten for display
            type_labels_list.extend([f"{short_ft}\n{feat[:18]}" for feat in type_features['feature']])
    
    if top_per_type:
        bars = ax3.barh(range(len(top_per_type)), top_per_type, color=type_colors, alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        ax3.set_yticks(range(len(top_per_type)))
        ax3.set_yticklabels(type_labels_list, fontsize=7)
        ax3.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
        ax3.set_title('(C) Top 5 per Feature Type', fontsize=12, fontweight='bold', pad=10)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 4. Summary statistics by type
    ax4 = axes[1, 1]
    type_stats = fi_df.groupby('feature_type').agg({
        'importance': ['mean', 'std', 'max', 'count']
    }).reset_index()
    type_stats.columns = ['feature_type', 'mean_importance', 'std_importance', 'max_importance', 'count']
    type_stats = type_stats.sort_values('mean_importance', ascending=False)
    
    bars = ax4.bar(range(len(type_stats)), type_stats['mean_importance'], 
                   color=[COLORS.get(ft, '#888888') for ft in type_stats['feature_type']],
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax4.errorbar(range(len(type_stats)), type_stats['mean_importance'],
                yerr=type_stats['std_importance'], fmt='none', color='black', linewidth=2, capsize=5)
    
    ax4.set_xticks(range(len(type_stats)))
    ax4.set_xticklabels(type_stats['feature_type'], rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Mean Feature Importance', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Mean Importance by Feature Type', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count annotations
    for i, (idx, row) in enumerate(type_stats.iterrows()):
        ax4.text(i, row['mean_importance'] + row['std_importance'] + 0.001,
                f'n={int(row["count"])}', ha='center', va='bottom', fontsize=9)
    
    # Add legend (group CE-VAR layers)
    from matplotlib.patches import Patch
    legend_order = [
        'CE-VAR-Cognitive', 'CE-VAR-Emotional', 'CE-VAR-Stylistic',
        'CE-Geometry', 'TF-IDF-Geometry', 'SBERT-Geometry'
    ]
    handles = [Patch(facecolor=COLORS.get(ft, '#888888'), label=ft.replace('CE-VAR-', 'CE-VAR-\n')) 
               for ft in legend_order if ft in COLORS or any(fi_df['feature_type'] == ft)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=6, fontsize=9, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    output_path = output_dir / 'feature_importance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize feature importance comparison for unified 75 features.")
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
        help="LLM models to include (default: all).",
    )
    parser.add_argument(
        "--level",
        default="LV3",
        help="LLM level (default: LV3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS_ROOT / "unified_75_features_ml",
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Path to pre-computed feature importance CSV (optional).",
    )
    args = parser.parse_args()
    
    # Load or compute feature importance
    if args.results and args.results.exists():
        print(f"Loading feature importance from: {args.results}")
        fi_df = pd.read_csv(args.results)
        if 'feature_type' not in fi_df.columns:
            fi_df['feature_type'] = fi_df['feature'].apply(classify_feature_type)
    else:
        print("Computing feature importance...")
        df = load_samples(args.domains, args.models, args.level)
        if df.empty:
            print("⚠ No data available for the requested configuration.")
            return
        
        X = select_unified_features(df)
        y = (df["label"] == "human").astype(int).values
        
        fi_df = compute_feature_importance(X, y)
        
        # Save feature importance data
        args.output.mkdir(parents=True, exist_ok=True)
        fi_path = args.output / "feature_importance_data.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"Saved feature importance data: {fi_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_top_importance_comparison(fi_df, args.output, top_n=30)
    
    # Print summary
    print("\n=== Feature Importance Summary ===")
    type_summary = fi_df.groupby('feature_type').agg({
        'importance': ['mean', 'max', 'count'],
        'feature': lambda x: ', '.join(x.head(3))
    })
    print(type_summary)
    
    print("\n=== Top 10 Features ===")
    print(fi_df.head(10)[['feature', 'importance', 'feature_type']].to_string(index=False))
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()

