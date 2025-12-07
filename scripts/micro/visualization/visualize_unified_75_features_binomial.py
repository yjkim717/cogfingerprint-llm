#!/usr/bin/env python3
"""
Visualize per-feature binomial test results for unified 75 trajectory features (RQ1 LV3).

Generates:
1. Win rate by feature group (4 groups summary)
2. Per-feature win rate bar chart (all 75 features, grouped by feature type)
3. Per-feature win rate heatmap (organized by feature group)
4. Top features summary

Usage:
    python scripts/micro/visualization/visualize_unified_75_features_binomial.py \
        --stats plots/binomial/unified_75_features/unified_75_features_binomial.csv \
        --output plots/binomial/unified_75_features
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

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"

# Color palette
COLORS = {
    'CE-VAR': '#2E86AB',      # Blue
    'CE-GEO': '#A23B72',      # Purple
    'TF-IDF-GEO': '#F18F01',  # Orange
    'SBERT-GEO': '#C73E1D',   # Red
    'baseline': '#CCCCCC'
}


def load_stats(stats_path: Path) -> pd.DataFrame:
    """Load per-feature binomial test statistics."""
    return pd.read_csv(stats_path)


def plot_win_rate_by_feature_group(stats_df: pd.DataFrame, output_dir: Path):
    """Plot average win rate by feature group (summary)."""
    # Aggregate by feature group
    group_stats = stats_df.groupby('feature_group').agg({
        'human_wins': 'sum',
        'n_comparisons': 'sum',
        'human_win_rate': lambda x: (stats_df.loc[x.index, 'human_wins'].sum() / 
                                     stats_df.loc[x.index, 'n_comparisons'].sum())
    }).reset_index()
    
    group_stats['human_win_rate'] = group_stats['human_wins'] / group_stats['n_comparisons']
    
    # Calculate p-values for groups
    from scipy.stats import binomtest
    group_stats['pvalue'] = group_stats.apply(
        lambda row: binomtest(int(row['human_wins']), int(row['n_comparisons']), 
                            p=0.5, alternative='greater').pvalue, axis=1
    )
    group_stats['significant'] = group_stats['pvalue'] < 0.05
    group_stats['significant_star'] = group_stats.apply(
        lambda row: '***' if row['pvalue'] < 0.001 else
                    '**' if row['pvalue'] < 0.01 else
                    '*' if row['pvalue'] < 0.05 else '', axis=1
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    group_stats['feature_group'] = pd.Categorical(group_stats['feature_group'], 
                                                   categories=feature_order, ordered=True)
    group_stats = group_stats.sort_values('feature_group')
    
    bars = ax.bar(
        group_stats['feature_group'],
        group_stats['human_win_rate'] * 100,
        color=[COLORS[fg] for fg in group_stats['feature_group']],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, 
              label='Baseline (50%)', zorder=0)
    
    # Add significance markers
    for i, (idx, row) in enumerate(group_stats.iterrows()):
        y_pos = row['human_win_rate'] * 100
        if row['significant']:
            ax.text(i, y_pos + 2, row['significant_star'], ha='center', va='bottom', 
                   fontsize=16, fontweight='bold')
        ax.text(i, y_pos / 2, f'{y_pos:.1f}%', ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
    
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_title('Unified 75 Features: Win Rate by Feature Group (RQ1 LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add sample size annotations
    for i, (idx, row) in enumerate(group_stats.iterrows()):
        ax.text(i, -3, f'n={row["n_comparisons"]:,}', ha='center', va='top', 
               fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_by_feature_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_per_feature_bar_chart(stats_df: pd.DataFrame, output_dir: Path):
    """Plot per-feature win rates, organized by feature group."""
    # Organize features by group
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    stats_df['feature_group'] = pd.Categorical(stats_df['feature_group'], 
                                               categories=feature_order, ordered=True)
    stats_df = stats_df.sort_values(['feature_group', 'human_win_rate'], 
                                   ascending=[True, False])
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, group in enumerate(feature_order):
        ax = axes[idx]
        group_data = stats_df[stats_df['feature_group'] == group].copy()
        
        if len(group_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(group, fontsize=12, fontweight='bold')
            continue
        
        x = np.arange(len(group_data))
        colors = [COLORS[group]] * len(group_data)
        
        bars = ax.bar(x, group_data['human_win_rate'] * 100, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', 
                  linewidth=2, zorder=0)
        
        # Add significance markers
        for i, (_, row) in enumerate(group_data.iterrows()):
            y_pos = row['human_win_rate'] * 100
            if row['significant_star']:
                ax.text(i, y_pos + 2, row['significant_star'], 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Feature names (shortened)
        feature_names = []
        for f in group_data['feature']:
            if group == 'CE-VAR':
                # e.g., Agreeableness_cv -> Agr_cv
                parts = f.split('_')
                base = parts[0][:4] if len(parts[0]) > 4 else parts[0]
                suffix = '_' + parts[-1] if len(parts) > 1 else ''
                feature_names.append(base + suffix)
            else:
                # Geometry: remove prefix
                feature_names.append(f.replace('ce_', '').replace('tfidf_', 'tf_')
                                   .replace('sbert_', 'sb_'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Human Win Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'{group} ({len(group_data)} features)', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, 90)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Unified 75 Features: Per-Feature Win Rate by Group (RQ1 LV3)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'per_feature_win_rate_by_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap(stats_df: pd.DataFrame, output_dir: Path):
    """Plot heatmap of win rates organized by feature group."""
    # Organize for heatmap
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    stats_df['feature_group'] = pd.Categorical(stats_df['feature_group'], 
                                               categories=feature_order, ordered=True)
    stats_df = stats_df.sort_values(['feature_group', 'human_win_rate'], 
                                   ascending=[True, False])
    
    # Create matrix for heatmap
    max_features = max(len(stats_df[stats_df['feature_group'] == g]) for g in feature_order)
    
    heatmap_data = []
    row_labels = []
    
    for group in feature_order:
        group_data = stats_df[stats_df['feature_group'] == group].copy()
        group_rates = group_data['human_win_rate'].values * 100
        
        # Pad to max_features
        padded = np.pad(group_rates, (0, max_features - len(group_rates)), 
                       constant_values=np.nan)
        heatmap_data.append(padded)
        
        # Short feature names
        feature_names = []
        for f in group_data['feature']:
            if group == 'CE-VAR':
                parts = f.split('_')
                base = parts[0][:6] if len(parts[0]) > 6 else parts[0]
                suffix = '_' + parts[-1][:3] if len(parts) > 1 else ''
                feature_names.append(base + suffix)
            else:
                feature_names.append(f.replace('ce_', '').replace('tfidf_', 'tf_')
                                   .replace('sbert_', 'sb_'))
        
        # Pad feature names
        padded_names = feature_names + [''] * (max_features - len(feature_names))
        row_labels.append(padded_names)
    
    heatmap_array = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(max(16, max_features * 0.3), 6))
    
    # Create heatmap
    im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Human Win Rate (%)', fontsize=11, fontweight='bold')
    
    # Set ticks and labels
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order, fontsize=11, fontweight='bold')
    ax.set_xticks(range(max_features))
    ax.set_xticklabels([])  # Too many features to label individually
    
    # Add feature names as text
    for i, group_labels in enumerate(row_labels):
        for j, label in enumerate(group_labels):
            if label and not np.isnan(heatmap_array[i, j]):
                ax.text(j, i, label, ha='center', va='center', 
                       fontsize=7, rotation=90)
    
    # Add baseline line
    baseline_idx = int(max_features * 0.5)  # Approximate 50% position
    ax.axvline(baseline_idx, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title('Unified 75 Features: Win Rate Heatmap (RQ1 LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_panel(stats_df: pd.DataFrame, output_dir: Path):
    """Create a summary panel with multiple views."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Group summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    group_stats = stats_df.groupby('feature_group').agg({
        'human_wins': 'sum',
        'n_comparisons': 'sum'
    }).reset_index()
    group_stats['human_win_rate'] = group_stats['human_wins'] / group_stats['n_comparisons']
    
    from scipy.stats import binomtest
    group_stats['pvalue'] = group_stats.apply(
        lambda row: binomtest(int(row['human_wins']), int(row['n_comparisons']), 
                            p=0.5, alternative='greater').pvalue, axis=1
    )
    group_stats['significant_star'] = group_stats.apply(
        lambda row: '***' if row['pvalue'] < 0.001 else
                    '**' if row['pvalue'] < 0.01 else
                    '*' if row['pvalue'] < 0.05 else '', axis=1
    )
    
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    group_stats['feature_group'] = pd.Categorical(group_stats['feature_group'], 
                                                  categories=feature_order, ordered=True)
    group_stats = group_stats.sort_values('feature_group')
    
    x = np.arange(len(feature_order))
    bars = ax1.bar(x, group_stats['human_win_rate'] * 100,
                   color=[COLORS[fg] for fg in feature_order], alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax1.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    
    for i, (_, row) in enumerate(group_stats.iterrows()):
        y_pos = row['human_win_rate'] * 100
        if row['significant_star']:
            ax1.text(i, y_pos + 2, row['significant_star'], ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        ax1.text(i, y_pos / 2, f'{y_pos:.1f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    ax1.set_ylabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Win Rate by Feature Group', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_order)
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Top 10 features (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    top10 = stats_df.nlargest(10, 'human_win_rate')
    x = np.arange(len(top10))
    colors_top = [COLORS[g] for g in top10['feature_group']]
    bars = ax2.barh(x, top10['human_win_rate'] * 100, color=colors_top, 
                   alpha=0.8, edgecolor='black', linewidth=1)
    ax2.axvline(x=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    
    # Short names
    short_names = []
    for f, g in zip(top10['feature'], top10['feature_group']):
        if g == 'CE-VAR':
            parts = f.split('_')
            short = parts[0][:8] + '_' + parts[-1][:3] if len(parts) > 1 else parts[0][:10]
        else:
            short = f.replace('ce_', '').replace('tfidf_', 'tf_').replace('sbert_', 'sb_')
        short_names.append(short)
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_xlabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Top 10 Features', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 90)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 3. Distribution by group (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    for group in feature_order:
        group_data = stats_df[stats_df['feature_group'] == group]
        ax3.hist(group_data['human_win_rate'] * 100, bins=20, alpha=0.6, 
                label=group, color=COLORS[group], edgecolor='black', linewidth=0.5)
    ax3.axvline(x=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    ax3.set_xlabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Distribution of Win Rates by Group', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Significance summary (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    sig_summary = stats_df.groupby(['feature_group', 'significant']).size().unstack(fill_value=0)
    sig_summary = sig_summary.reindex(feature_order)
    x = np.arange(len(feature_order))
    width = 0.35
    
    if True in sig_summary.columns:
        ax4.bar(x - width/2, sig_summary[True], width, label='Significant', 
               color='#4A90E2', alpha=0.8, edgecolor='black', linewidth=1)
    if False in sig_summary.columns:
        ax4.bar(x + width/2, sig_summary[False], width, label='Not Significant', 
               color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax4.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Feature Group', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Significance Summary', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(feature_order)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 5. Summary table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    summary_data = []
    for group in feature_order:
        group_data = stats_df[stats_df['feature_group'] == group]
        n_features = len(group_data)
        n_sig = group_data['significant'].sum()
        mean_rate = group_data['human_win_rate'].mean() * 100
        min_rate = group_data['human_win_rate'].min() * 100
        max_rate = group_data['human_win_rate'].max() * 100
        
        summary_data.append([
            group,
            f'{n_features}',
            f'{n_sig}',
            f'{mean_rate:.1f}%',
            f'{min_rate:.1f}%',
            f'{max_rate:.1f}%'
        ])
    
    table = ax5.table(cellText=summary_data,
                     colLabels=['Feature Group', 'N Features', 'N Significant', 
                               'Mean Win Rate', 'Min Win Rate', 'Max Win Rate'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i, group in enumerate(feature_order):
        for j in range(6):
            table[(i+1, j)].set_facecolor(COLORS[group] + '40')  # Add transparency
    
    ax5.set_title('(E) Summary Statistics', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('Unified 75 Features: Binomial Test Summary (RQ1 LV3)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'summary_panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-feature binomial test for unified 75 features"
    )
    parser.add_argument(
        '--stats',
        type=str,
        required=True,
        help='Path to per-feature statistics CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/binomial/unified_75_features',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    stats_path = Path(args.stats)
    output_dir = Path(args.output)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VISUALIZING UNIFIED 75 FEATURES BINOMIAL TEST")
    print("="*80)
    print(f"Stats: {stats_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    stats_df = load_stats(stats_path)
    print(f"Loaded {len(stats_df)} features")
    print(f"Feature groups: {', '.join(stats_df['feature_group'].unique())}")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_win_rate_by_feature_group(stats_df, output_dir)
    plot_per_feature_bar_chart(stats_df, output_dir)
    plot_heatmap(stats_df, output_dir)
    plot_summary_panel(stats_df, output_dir)
    
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()


