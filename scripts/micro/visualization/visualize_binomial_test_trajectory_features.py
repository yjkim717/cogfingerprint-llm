#!/usr/bin/env python3
"""
Visualize binomial test results for trajectory features.

Generates:
1. Human win rate bar chart by feature group
2. Human win rate by domain and feature group
3. Human win rate by model and feature group
4. Distribution comparison (Human vs LLM values)
5. Comparison with ML classification accuracy

Usage:
    python scripts/micro/visualization/visualize_binomial_test_trajectory_features.py \
        --results micro_results/binomial/binomial_test_trajectory_features_lv3.csv \
        --detailed micro_results/binomial/binomial_test_trajectory_features_lv3_detailed.csv \
        --output plots/binomial/trajectory_features_binomial_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

PLOTS_ROOT = PROJECT_ROOT / "plots" / "binomial"

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
    'OVERALL': '#6A994E',     # Green
    'Human': '#4A90E2',
    'LLM': '#E24A4A',
    'baseline': '#CCCCCC'
}


def load_data(results_path: Path, detailed_path: Optional[Path] = None) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load binomial test results."""
    results_df = pd.read_csv(results_path)
    
    detailed_df = None
    if detailed_path and detailed_path.exists():
        detailed_df = pd.read_csv(detailed_path)
    
    return results_df, detailed_df


def plot_win_rate_by_feature_group(results_df: pd.DataFrame, output_dir: Path):
    """Plot human win rate by feature group with significance markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out OVERALL for this plot
    plot_df = results_df[results_df['feature_group'] != 'OVERALL'].copy()
    
    # Order
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    plot_df['feature_group'] = pd.Categorical(plot_df['feature_group'], categories=feature_order, ordered=True)
    plot_df = plot_df.sort_values('feature_group')
    
    # Create bars
    bars = ax.bar(
        plot_df['feature_group'],
        plot_df['human_win_rate'] * 100,
        color=[COLORS[fg] for fg in plot_df['feature_group']],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add baseline (50%)
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, label='Baseline (50%)', zorder=0)
    
    # Add significance markers
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        x_pos = i
        y_pos = row['human_win_rate'] * 100
        if row['significant']:
            marker = row['significant_star']
            ax.text(x_pos, y_pos + 2, marker, ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        x_pos = i
        y_pos = row['human_win_rate'] * 100
        ax.text(x_pos, y_pos / 2, f'{y_pos:.1f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    # Customize
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_title('Human vs LLM: Win Rate by Feature Group (LV3)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add sample size annotations
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        x_pos = i
        ax.text(x_pos, -3, f'n={row["n_comparisons"]:,}', ha='center', va='top', 
                fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_by_feature_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_win_rate_by_domain(detailed_df: pd.DataFrame, output_dir: Path):
    """Plot human win rate by domain and feature group."""
    if detailed_df is None:
        return
    
    # Calculate win rate by domain and feature group
    domain_stats = detailed_df.groupby(['domain', 'feature_type']).agg({
        'human_wins': 'sum',
        'human_wins': lambda x: x.sum(),
    }).reset_index()
    
    domain_counts = detailed_df.groupby(['domain', 'feature_type']).size().reset_index(name='n_comparisons')
    domain_wins = detailed_df.groupby(['domain', 'feature_type'])['human_wins'].sum().reset_index(name='human_wins')
    domain_stats = domain_counts.merge(domain_wins, on=['domain', 'feature_type'])
    domain_stats['human_win_rate'] = domain_stats['human_wins'] / domain_stats['n_comparisons']
    
    # Filter feature groups
    feature_groups = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    domain_stats = domain_stats[domain_stats['feature_type'].isin(feature_groups)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create grouped bar chart
    x = np.arange(len(feature_groups))
    width = 0.25
    domains = sorted(domain_stats['domain'].unique())
    
    for i, domain in enumerate(domains):
        domain_data = domain_stats[domain_stats['domain'] == domain]
        values = []
        for fg in feature_groups:
            fg_data = domain_data[domain_data['feature_type'] == fg]
            if len(fg_data) > 0:
                values.append(fg_data['human_win_rate'].iloc[0] * 100)
            else:
                values.append(0)
        
        offset = (i - len(domains) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=domain.capitalize(), alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + offset, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, label='Baseline (50%)', zorder=0)
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_title('Human Win Rate by Domain and Feature Group (LV3)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_groups)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_by_domain.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_win_rate_by_model(detailed_df: pd.DataFrame, output_dir: Path):
    """Plot human win rate by model and feature group."""
    if detailed_df is None:
        return
    
    # Calculate win rate by model and feature group
    model_counts = detailed_df.groupby(['model', 'feature_type']).size().reset_index(name='n_comparisons')
    model_wins = detailed_df.groupby(['model', 'feature_type'])['human_wins'].sum().reset_index(name='human_wins')
    model_stats = model_counts.merge(model_wins, on=['model', 'feature_type'])
    model_stats['human_win_rate'] = model_stats['human_wins'] / model_stats['n_comparisons']
    
    # Filter feature groups
    feature_groups = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    model_stats = model_stats[model_stats['feature_type'].isin(feature_groups)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create grouped bar chart
    x = np.arange(len(feature_groups))
    width = 0.2
    models = sorted(model_stats['model'].unique())
    
    for i, model in enumerate(models):
        model_data = model_stats[model_stats['model'] == model]
        values = []
        for fg in feature_groups:
            fg_data = model_data[model_data['feature_type'] == fg]
            if len(fg_data) > 0:
                values.append(fg_data['human_win_rate'].iloc[0] * 100)
            else:
                values.append(0)
        
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + offset, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, label='Baseline (50%)', zorder=0)
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_title('Human Win Rate by LLM Model and Feature Group (LV3)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_groups)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_by_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_with_ml(results_df: pd.DataFrame, output_dir: Path):
    """Compare binomial test win rate with ML classification accuracy."""
    # ML results from RQ1 (from documentation)
    ml_results = {
        'CE-VAR': {'accuracy': 91.70, 'roc_auc': 96.06},
        'CE-GEO': {'accuracy': 78.11, 'roc_auc': 57.12},
        'TF-IDF-GEO': {'accuracy': 80.49, 'roc_auc': 73.94},
        'SBERT-GEO': {'accuracy': 79.37, 'roc_auc': 60.98},
    }
    
    # Filter results
    plot_df = results_df[results_df['feature_group'] != 'OVERALL'].copy()
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    plot_df['feature_group'] = pd.Categorical(plot_df['feature_group'], categories=feature_order, ordered=True)
    plot_df = plot_df.sort_values('feature_group')
    
    # Prepare data
    feature_groups = plot_df['feature_group'].tolist()
    binomial_rates = plot_df['human_win_rate'].values * 100
    ml_accuracies = [ml_results[fg]['accuracy'] for fg in feature_groups]
    
    x = np.arange(len(feature_groups))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Binomial test bars
    bars1 = ax1.bar(x - width/2, binomial_rates, width, label='Binomial Test: Human Win Rate', 
                    color=COLORS['CE-VAR'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # ML accuracy bars
    bars2 = ax1.bar(x + width/2, ml_accuracies, width, label='ML Classification: Accuracy', 
                    color=COLORS['SBERT-GEO'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (br, ma) in enumerate(zip(binomial_rates, ml_accuracies)):
        ax1.text(i - width/2, br + 1, f'{br:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, ma + 1, f'{ma:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Rate / Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax1.set_title('Binomial Test Win Rate vs ML Classification Accuracy (LV3)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_groups)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_with_ml.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_value_distribution_comparison(detailed_df: pd.DataFrame, output_dir: Path, n_samples: int = 5000):
    """Plot distribution comparison of Human vs LLM values for each feature group."""
    if detailed_df is None:
        return
    
    feature_groups = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, fg in enumerate(feature_groups):
        ax = axes[idx]
        fg_data = detailed_df[detailed_df['feature_type'] == fg].copy()
        
        if len(fg_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(fg, fontsize=12, fontweight='bold')
            continue
        
        # Sample if too large
        if len(fg_data) > n_samples:
            fg_data = fg_data.sample(n=n_samples, random_state=42)
        
        # Plot distributions
        ax.hist(fg_data['human_value'], bins=50, alpha=0.6, label='Human', 
                color=COLORS['Human'], edgecolor='black', linewidth=0.5)
        ax.hist(fg_data['llm_value'], bins=50, alpha=0.6, label='LLM', 
                color=COLORS['LLM'], edgecolor='black', linewidth=0.5)
        
        # Add mean lines
        human_mean = fg_data['human_value'].mean()
        llm_mean = fg_data['llm_value'].mean()
        ax.axvline(human_mean, color=COLORS['Human'], linestyle='--', linewidth=2, label=f'Human mean: {human_mean:.3f}')
        ax.axvline(llm_mean, color=COLORS['LLM'], linestyle='--', linewidth=2, label=f'LLM mean: {llm_mean:.3f}')
        
        ax.set_xlabel('Feature Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{fg}\n(Human Win Rate: {fg_data["human_wins"].mean()*100:.1f}%)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Distribution Comparison: Human vs LLM Feature Values (LV3)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / 'value_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_panel(results_df: pd.DataFrame, detailed_df: Optional[pd.DataFrame], output_dir: Path):
    """Create a summary panel with all key results."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Win rate by feature group (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_df = results_df[results_df['feature_group'] != 'OVERALL'].copy()
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    plot_df['feature_group'] = pd.Categorical(plot_df['feature_group'], categories=feature_order, ordered=True)
    plot_df = plot_df.sort_values('feature_group')
    
    bars = ax1.bar(plot_df['feature_group'], plot_df['human_win_rate'] * 100,
                   color=[COLORS[fg] for fg in plot_df['feature_group']], alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax1.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, label='Baseline (50%)', zorder=0)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        y_pos = row['human_win_rate'] * 100
        if row['significant']:
            ax1.text(i, y_pos + 2, row['significant_star'], ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
        ax1.text(i, y_pos / 2, f'{y_pos:.1f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    ax1.set_ylabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Win Rate by Feature Group', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    
    # 2. Comparison with ML (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ml_results = {
        'CE-VAR': {'accuracy': 91.70, 'roc_auc': 96.06},
        'CE-GEO': {'accuracy': 78.11, 'roc_auc': 57.12},
        'TF-IDF-GEO': {'accuracy': 80.49, 'roc_auc': 73.94},
        'SBERT-GEO': {'accuracy': 79.37, 'roc_auc': 60.98},
    }
    feature_groups = plot_df['feature_group'].tolist()
    binomial_rates = plot_df['human_win_rate'].values * 100
    ml_accuracies = [ml_results[fg]['accuracy'] for fg in feature_groups]
    x = np.arange(len(feature_groups))
    width = 0.35
    ax2.bar(x - width/2, binomial_rates, width, label='Binomial: Win Rate', 
            color=COLORS['CE-VAR'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.bar(x + width/2, ml_accuracies, width, label='ML: Accuracy', 
            color=COLORS['SBERT-GEO'], alpha=0.8, edgecolor='black', linewidth=1.5)
    for i, (br, ma) in enumerate(zip(binomial_rates, ml_accuracies)):
        ax2.text(i - width/2, br + 1, f'{br:.1f}%', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, ma + 1, f'{ma:.1f}%', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Rate / Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Binomial Test vs ML Classification', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_groups, rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9)
    
    # 3. Win rate by domain (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if detailed_df is not None:
        domain_counts = detailed_df.groupby(['domain', 'feature_type']).size().reset_index(name='n_comparisons')
        domain_wins = detailed_df.groupby(['domain', 'feature_type'])['human_wins'].sum().reset_index(name='human_wins')
        domain_stats = domain_counts.merge(domain_wins, on=['domain', 'feature_type'])
        domain_stats['human_win_rate'] = domain_stats['human_wins'] / domain_stats['n_comparisons']
        domain_stats = domain_stats[domain_stats['feature_type'].isin(feature_order)]
        
        x = np.arange(len(feature_order))
        width = 0.25
        domains = sorted(domain_stats['domain'].unique())
        for i, domain in enumerate(domains):
            domain_data = domain_stats[domain_stats['domain'] == domain]
            values = [domain_data[domain_data['feature_type'] == fg]['human_win_rate'].iloc[0] * 100 
                     if len(domain_data[domain_data['feature_type'] == fg]) > 0 else 0 
                     for fg in feature_order]
            offset = (i - len(domains) / 2 + 0.5) * width
            ax3.bar(x + offset, values, width, label=domain.capitalize(), alpha=0.8, 
                   edgecolor='black', linewidth=1)
    ax3.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    ax3.set_ylabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Feature Group', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Win Rate by Domain', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_order, rotation=15, ha='right')
    ax3.set_ylim(0, 80)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', fontsize=9)
    
    # 4. Win rate by model (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    if detailed_df is not None:
        model_counts = detailed_df.groupby(['model', 'feature_type']).size().reset_index(name='n_comparisons')
        model_wins = detailed_df.groupby(['model', 'feature_type'])['human_wins'].sum().reset_index(name='human_wins')
        model_stats = model_counts.merge(model_wins, on=['model', 'feature_type'])
        model_stats['human_win_rate'] = model_stats['human_wins'] / model_stats['n_comparisons']
        model_stats = model_stats[model_stats['feature_type'].isin(feature_order)]
        
        models = sorted(model_stats['model'].unique())
        width = 0.2
        for i, model in enumerate(models):
            model_data = model_stats[model_stats['model'] == model]
            values = [model_data[model_data['feature_type'] == fg]['human_win_rate'].iloc[0] * 100 
                     if len(model_data[model_data['feature_type'] == fg]) > 0 else 0 
                     for fg in feature_order]
            offset = (i - len(models) / 2 + 0.5) * width
            ax4.bar(x + offset, values, width, label=model, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    ax4.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    ax4.set_ylabel('Human Win Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Feature Group', fontsize=11, fontweight='bold')
    ax4.set_title('(D) Win Rate by LLM Model', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(feature_order, rotation=15, ha='right')
    ax4.set_ylim(0, 80)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right', fontsize=9)
    
    # 5. Summary statistics table (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in plot_df.iterrows():
        summary_data.append([
            row['feature_group'],
            f"{row['n_comparisons']:,}",
            f"{row['human_wins']:,}",
            f"{row['human_win_rate']*100:.1f}%",
            f"{row['pvalue']:.2e}" if row['pvalue'] > 0 else "< 1e-300",
            row['significant_star']
        ])
    
    # Add overall
    overall = results_df[results_df['feature_group'] == 'OVERALL'].iloc[0]
    summary_data.append([
        'OVERALL',
        f"{overall['n_comparisons']:,}",
        f"{overall['human_wins']:,}",
        f"{overall['human_win_rate']*100:.1f}%",
        f"{overall['pvalue']:.2e}" if overall['pvalue'] > 0 else "< 1e-300",
        overall['significant_star']
    ])
    
    table = ax5.table(cellText=summary_data,
                     colLabels=['Feature Group', 'N Comparisons', 'Human Wins', 'Win Rate', 'p-value', 'Significance'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style significant rows
    for i in range(len(summary_data)):
        if summary_data[i][-1] != '':
            for j in range(6):
                table[(i+1, j)].set_facecolor('#E8F4F8')
    
    ax5.set_title('(E) Summary Statistics', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('Binomial Test: Human vs LLM Trajectory Features (LV3, Excluding direction_consistency)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'summary_panel.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize binomial test results for trajectory features"
    )
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to binomial test results CSV'
    )
    parser.add_argument(
        '--detailed',
        type=str,
        default=None,
        help='Path to detailed comparisons CSV (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/binomial/trajectory_features_binomial_test',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    detailed_path = Path(args.detailed) if args.detailed else None
    output_dir = Path(args.output)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("VISUALIZING BINOMIAL TEST RESULTS")
    print("="*80)
    print(f"Results: {results_path}")
    if detailed_path:
        print(f"Detailed: {detailed_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    results_df, detailed_df = load_data(results_path, detailed_path)
    print(f"Loaded {len(results_df)} result groups")
    if detailed_df is not None:
        print(f"Loaded {len(detailed_df):,} detailed comparisons")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_win_rate_by_feature_group(results_df, output_dir)
    
    if detailed_df is not None:
        plot_win_rate_by_domain(detailed_df, output_dir)
        plot_win_rate_by_model(detailed_df, output_dir)
        plot_value_distribution_comparison(detailed_df, output_dir)
    
    plot_comparison_with_ml(results_df, output_dir)
    plot_summary_panel(results_df, detailed_df, output_dir)
    
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()

