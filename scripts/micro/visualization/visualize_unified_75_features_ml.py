#!/usr/bin/env python3
"""
Visualize ML classification results for unified 75 trajectory features (RQ1 LV3).

Generates:
1. Performance metrics (Accuracy, ROC-AUC, F1) for unified model
2. ROC curves
3. Confusion matrices
4. Feature importance analysis (top features from unified model)
5. Comparison with other feature sets

Usage:
    python scripts/micro/visualization/visualize_unified_75_features_ml.py \
        --results plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv \
        --output plots/trajectory/unified_75_features_ml
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
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"

COLORS = {
    'unified': '#6A994E',      # Green
    'variability': '#2E86AB',  # Blue
    'geometry_all': '#A23B72', # Purple
    'ce_geometry': '#2E86AB',
    'tfidf_geometry': '#F18F01',
    'sbert_geometry': '#C73E1D',
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


def get_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unified 75 features (CE-VAR + all geometry)."""
    # CE variability features
    variability_cols = [
        col for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]
    
    # All geometry features
    geometry_cols = []
    for prefix in GEOMETRY_PREFIXES:
        for metric in GEOMETRY_METRICS:
            col = f"{prefix}{metric}"
            if col in df.columns:
                geometry_cols.append(col)
    
    # Unified = variability + geometry (75 features total)
    unified_cols = sorted(set(variability_cols + geometry_cols))
    
    return df[unified_cols].fillna(0.0)


def evaluate_unified_model(df: pd.DataFrame, n_splits: int = 5) -> Dict:
    """Evaluate unified model with cross-validation."""
    X = get_unified_features(df)
    y = (df["label"] == "human").astype(int).values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accs: List[float] = []
    rocs: List[float] = []
    f1s: List[float] = []
    feature_importances: List[np.ndarray] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_proba_all: List[float] = []
    
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X.iloc[train_idx], y[train_idx])
        
        y_pred = clf.predict(X.iloc[test_idx])
        y_proba = clf.predict_proba(X.iloc[test_idx])[:, 1]
        
        accs.append(accuracy_score(y[test_idx], y_pred))
        rocs.append(roc_auc_score(y[test_idx], y_proba))
        f1s.append(f1_score(y[test_idx], y_pred))
        
        feature_importances.append(clf.feature_importances_)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
    
    # Average feature importance
    avg_importance = np.mean(feature_importances, axis=0)
    std_importance = np.std(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': avg_importance,
        'importance_std': std_importance
    }).sort_values('importance', ascending=False)
    
    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'roc_auc_mean': np.mean(rocs),
        'roc_auc_std': np.std(rocs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'n_features': len(X.columns),
        'n_samples': len(df),
        'feature_importance': feature_importance_df,
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all),
        'y_proba': np.array(y_proba_all),
    }


def plot_performance_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Plot performance comparison across feature sets."""
    # Filter to show unified vs others
    plot_df = results_df.copy()
    
    # Full version with all 3 metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['accuracy_mean', 'roc_auc_mean', 'f1_mean']
    metric_labels = ['Accuracy', 'ROC AUC', 'F1 Score']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        x = np.arange(len(plot_df))
        colors = [COLORS.get(fs, '#888888') for fs in plot_df['feature_set']]
        bars = ax.bar(x, plot_df[metric] * 100, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add error bars
        std_col = metric.replace('_mean', '_std')
        ax.errorbar(x, plot_df[metric] * 100, yerr=plot_df[std_col] * 100,
                   fmt='none', color='black', capsize=5, capthick=2)
        
        # Add value labels
        for i, (_, row) in enumerate(plot_df.iterrows()):
            y_pos = row[metric] * 100
            ax.text(i, y_pos + row[std_col] * 100 + 1, f'{y_pos:.1f}%', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(f'{label} (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Set', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['feature_set'], rotation=15, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('RQ1 LV3: ML Classification Performance (Unified 75 Features)', 
                fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    output_path = output_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Simplified version with only Accuracy and ROC AUC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics_simple = ['accuracy_mean', 'roc_auc_mean']
    metric_labels_simple = ['Accuracy', 'ROC AUC']
    
    for idx, (metric, label) in enumerate(zip(metrics_simple, metric_labels_simple)):
        ax = axes[idx]
        
        x = np.arange(len(plot_df))
        colors = [COLORS.get(fs, '#888888') for fs in plot_df['feature_set']]
        bars = ax.bar(x, plot_df[metric] * 100, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add error bars
        std_col = metric.replace('_mean', '_std')
        ax.errorbar(x, plot_df[metric] * 100, yerr=plot_df[std_col] * 100,
                   fmt='none', color='black', capsize=5, capthick=2)
        
        # Add value labels
        for i, (_, row) in enumerate(plot_df.iterrows()):
            y_pos = row[metric] * 100
            ax.text(i, y_pos + row[std_col] * 100 + 1, f'{y_pos:.1f}%', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(f'{label} (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Set', fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['feature_set'], rotation=15, ha='right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('RQ1 LV3: ML Classification Performance (Unified 75 Features)', 
                fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    output_path = output_dir / 'performance_comparison_acc_roc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_roc_curves(eval_results: Dict, output_dir: Path):
    """Plot ROC curves for unified model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_true = eval_results['y_true']
    y_proba = eval_results['y_proba']
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    ax.plot(fpr, tpr, color=COLORS['unified'], lw=3, 
           label=f'Unified 75 Features (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve: Unified 75 Features (RQ1 LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = output_dir / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(eval_results: Dict, output_dir: Path):
    """Plot confusion matrix for unified model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(eval_results['y_true'], eval_results['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['LLM', 'Human'], yticklabels=['LLM', 'Human'],
               cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Unified 75 Features (RQ1 LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_importance(eval_results: Dict, output_dir: Path, top_n: int = 15):
    """Plot top N feature importances."""
    fi_df = eval_results['feature_importance'].head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by feature type
    colors = []
    for feat in fi_df['feature']:
        if any(feat.endswith(s) for s in VAR_SUFFIXES):
            colors.append(COLORS['variability'])
        elif feat.startswith('ce_'):
            colors.append(COLORS['ce_geometry'])
        elif feat.startswith('tfidf_'):
            colors.append(COLORS['tfidf_geometry'])
        elif feat.startswith('sbert_'):
            colors.append(COLORS['sbert_geometry'])
        else:
            colors.append('#888888')
    
    bars = ax.barh(range(len(fi_df)), fi_df['importance'], color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Short feature names
    short_names = []
    for feat in fi_df['feature']:
        if any(feat.endswith(s) for s in VAR_SUFFIXES):
            parts = feat.split('_')
            short = parts[0][:8] + '_' + parts[-1][:3] if len(parts) > 1 else parts[0][:10]
        else:
            short = feat.replace('ce_', 'ce_').replace('tfidf_', 'tf_').replace('sbert_', 'sb_')
        short_names.append(short)
    
    ax.set_yticks(range(len(fi_df)))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances: Unified 75 Features (RQ1 LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLORS['variability'], label='CE-VAR'),
        Patch(facecolor=COLORS['ce_geometry'], label='CE-GEO'),
        Patch(facecolor=COLORS['tfidf_geometry'], label='TF-IDF-GEO'),
        Patch(facecolor=COLORS['sbert_geometry'], label='SBERT-GEO'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_panel(results_df: pd.DataFrame, eval_results: Dict, output_dir: Path):
    """Create a comprehensive summary panel."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Performance metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    unified_row = results_df[results_df['feature_set'] == 'unified'].iloc[0]
    metrics = ['accuracy_mean', 'roc_auc_mean', 'f1_mean']
    metric_labels = ['Accuracy', 'ROC AUC', 'F1']
    values = [unified_row[m] * 100 for m in metrics]
    stds = [unified_row[m.replace('_mean', '_std')] * 100 for m in metrics]
    
    x = np.arange(len(metrics))
    bars = ax1.bar(x, values, color=COLORS['unified'], alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax1.errorbar(x, values, yerr=stds, fmt='none', color='black', 
                capsize=5, capthick=2)
    
    for i, (v, s) in enumerate(zip(values, stds)):
        ax1.text(i, v + s + 1, f'{v:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Unified Model Performance', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Feature set comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_df = results_df[results_df['feature_set'].isin(['variability', 'geometry_all', 'unified'])]
    x = np.arange(len(plot_df))
    colors_comp = [COLORS.get(fs, '#888888') for fs in plot_df['feature_set']]
    bars = ax2.bar(x, plot_df['accuracy_mean'] * 100, color=colors_comp, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.errorbar(x, plot_df['accuracy_mean'] * 100, 
                yerr=plot_df['accuracy_std'] * 100,
                fmt='none', color='black', capsize=5, capthick=2)
    
    for i, (_, row) in enumerate(plot_df.iterrows()):
        y_pos = row['accuracy_mean'] * 100
        ax2.text(i, y_pos + row['accuracy_std'] * 100 + 1, f'{y_pos:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df['feature_set'], rotation=15, ha='right')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Feature Set Comparison', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. ROC curve (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    y_true = eval_results['y_true']
    y_proba = eval_results['y_proba']
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax3.plot(fpr, tpr, color=COLORS['unified'], lw=2, 
            label=f'AUC = {roc_auc:.3f}')
    ax3.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_title('(C) ROC Curve', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--')
    
    # 4. Confusion matrix (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_true, eval_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
               xticklabels=['LLM', 'Human'], yticklabels=['LLM', 'Human'],
               cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax4.set_ylabel('True', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    
    # 5. Top features (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    top15 = eval_results['feature_importance'].head(15)
    x = np.arange(len(top15))
    colors_fi = []
    for feat in top15['feature']:
        if any(feat.endswith(s) for s in VAR_SUFFIXES):
            colors_fi.append(COLORS['variability'])
        elif feat.startswith('ce_'):
            colors_fi.append(COLORS['ce_geometry'])
        elif feat.startswith('tfidf_'):
            colors_fi.append(COLORS['tfidf_geometry'])
        elif feat.startswith('sbert_'):
            colors_fi.append(COLORS['sbert_geometry'])
        else:
            colors_fi.append('#888888')
    
    bars = ax5.barh(x, top15['importance'], color=colors_fi, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    short_names = []
    for feat in top15['feature']:
        if any(feat.endswith(s) for s in VAR_SUFFIXES):
            parts = feat.split('_')
            short = parts[0][:10] + '_' + parts[-1][:3] if len(parts) > 1 else parts[0][:12]
        else:
            short = feat.replace('ce_', '').replace('tfidf_', 'tf_').replace('sbert_', 'sb_')
        short_names.append(short)
    
    ax5.set_yticks(x)
    ax5.set_yticklabels(short_names, fontsize=9)
    ax5.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax5.set_title('(E) Top 15 Feature Importances', fontsize=12, fontweight='bold', pad=10)
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.suptitle('RQ1 LV3: Unified 75 Features ML Classification Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'summary_panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ML classification results for unified 75 features"
    )
    parser.add_argument(
        '--results',
        type=str,
        default='plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv',
        help='Path to classification results CSV'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        default=['academic', 'blogs', 'news'],
        help='Domains to include'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['DS', 'G4B', 'G12B', 'LMK'],
        help='LLM models to include'
    )
    parser.add_argument(
        '--level',
        type=str,
        default='LV3',
        help='LLM level'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/trajectory/unified_75_features_ml',
        help='Output directory'
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Recompute unified model evaluation (slower but more detailed)'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VISUALIZING UNIFIED 75 FEATURES ML CLASSIFICATION")
    print("="*80)
    print(f"Results: {results_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load results
    results_df = pd.read_csv(results_path)
    print(f"Loaded results for {len(results_df)} feature sets")
    print()
    
    # Generate basic visualizations
    print("Generating visualizations...")
    plot_performance_comparison(results_df, output_dir)
    
    # If recompute is requested, run full evaluation for detailed analysis
    if args.recompute:
        print("\nRecomputing unified model evaluation (this may take a while)...")
        df = load_samples(args.domains, args.models, args.level)
        if df.empty:
            print("Warning: No data found, skipping detailed evaluation")
        else:
            eval_results = evaluate_unified_model(df)
            print(f"Unified model: {eval_results['accuracy_mean']*100:.2f}% accuracy")
            print(f"  ROC AUC: {eval_results['roc_auc_mean']*100:.2f}%")
            print(f"  F1 Score: {eval_results['f1_mean']*100:.2f}%")
            print(f"  Features: {eval_results['n_features']}")
            
            plot_roc_curves(eval_results, output_dir)
            plot_confusion_matrix(eval_results, output_dir)
            plot_feature_importance(eval_results, output_dir)
            plot_summary_panel(results_df, eval_results, output_dir)
            
            # Save feature importance data for further analysis
            fi_path = output_dir / "feature_importance_data.csv"
            eval_results['feature_importance'].to_csv(fi_path, index=False)
            print(f"Saved feature importance data: {fi_path}")
    else:
        print("\nSkipping detailed evaluation (use --recompute for ROC curves, confusion matrix, etc.)")
        print("Note: Basic performance comparison has been generated.")
    
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()


