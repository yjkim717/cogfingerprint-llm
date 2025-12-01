#!/usr/bin/env python3
"""
Visualize robustness test: Compare binomial test results across LV1, LV2, LV3.

This script compares the binomial test results across different prompting levels
to assess robustness of the findings.

Usage:
    python scripts/micro/visualization/visualize_binomial_test_robustness.py \
        --results-lv1 micro_results/binomial/binomial_test_trajectory_features_lv1.csv \
        --results-lv2 micro_results/binomial/binomial_test_trajectory_features_lv2.csv \
        --results-lv3 micro_results/binomial/binomial_test_trajectory_features_lv3.csv \
        --output plots/binomial/robustness_test
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
    'LV1': '#2E86AB',      # Blue
    'LV2': '#A23B72',      # Purple
    'LV3': '#F18F01',      # Orange
    'CE-VAR': '#2E86AB',
    'CE-GEO': '#A23B72',
    'TF-IDF-GEO': '#F18F01',
    'SBERT-GEO': '#C73E1D',
    'baseline': '#CCCCCC'
}


def load_results(results_paths: Dict[str, Path]) -> pd.DataFrame:
    """Load results from all levels and combine."""
    all_results = []
    
    for level, path in results_paths.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {level}")
            continue
        
        df = pd.read_csv(path)
        df['level'] = level
        all_results.append(df)
    
    if not all_results:
        raise ValueError("No results files found!")
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined


def plot_win_rate_comparison(combined_df: pd.DataFrame, output_dir: Path):
    """Plot win rate comparison across levels by feature group."""
    # Filter out OVERALL for main plot
    plot_df = combined_df[combined_df['feature_group'] != 'OVERALL'].copy()
    
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    level_order = ['LV1', 'LV2', 'LV3']
    
    plot_df['feature_group'] = pd.Categorical(plot_df['feature_group'], 
                                              categories=feature_order, ordered=True)
    plot_df['level'] = pd.Categorical(plot_df['level'], 
                                      categories=level_order, ordered=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_order))
    width = 0.25
    
    for i, level in enumerate(level_order):
        level_data = plot_df[plot_df['level'] == level]
        values = []
        for fg in feature_order:
            fg_data = level_data[level_data['feature_group'] == fg]
            if len(fg_data) > 0:
                values.append(fg_data['human_win_rate'].iloc[0] * 100)
            else:
                values.append(0)
        
        offset = (i - len(level_order) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=level, 
                     color=COLORS[level], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + offset, v + 1, f'{v:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # Add baseline
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, 
              label='Baseline (50%)', zorder=0)
    
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Test: Human Win Rate Across Prompting Levels (LV1, LV2, LV3)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_order)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'win_rate_comparison_by_level.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall_comparison(combined_df: pd.DataFrame, output_dir: Path):
    """Plot overall win rate comparison across levels."""
    overall_df = combined_df[combined_df['feature_group'] == 'OVERALL'].copy()
    level_order = ['LV1', 'LV2', 'LV3']
    overall_df['level'] = pd.Categorical(overall_df['level'], 
                                       categories=level_order, ordered=True)
    overall_df = overall_df.sort_values('level')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(level_order))
    values = overall_df['human_win_rate'].values * 100
    colors = [COLORS[level] for level in level_order]
    
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(i, val / 2, f'{val:.2f}%', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        ax.text(i, val + 0.5, f'n={overall_df.iloc[i]["n_comparisons"]:,}', 
               ha='center', va='bottom', fontsize=9, style='italic', color='gray')
    
    # Add baseline
    ax.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, 
              label='Baseline (50%)', zorder=0)
    
    # Calculate range
    min_val = values.min()
    max_val = values.max()
    range_val = max_val - min_val
    
    ax.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prompting Level', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Human Win Rate Across Levels\n(Range: {range_val:.2f}%, Min: {min_val:.2f}%, Max: {max_val:.2f}%)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(level_order)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    output_path = output_dir / 'overall_win_rate_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_variability_analysis(combined_df: pd.DataFrame, output_dir: Path):
    """Plot variability (standard deviation) of win rates across levels."""
    # Filter out OVERALL
    plot_df = combined_df[combined_df['feature_group'] != 'OVERALL'].copy()
    
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO']
    
    # Calculate statistics for each feature group
    stats = []
    for fg in feature_order:
        fg_data = plot_df[plot_df['feature_group'] == fg]
        if len(fg_data) > 0:
            win_rates = fg_data['human_win_rate'].values * 100
            mean_rate = win_rates.mean()
            std_rate = win_rates.std()
            min_rate = win_rates.min()
            max_rate = win_rates.max()
            range_rate = max_rate - min_rate
            
            stats.append({
                'feature_group': fg,
                'mean': mean_rate,
                'std': std_rate,
                'min': min_rate,
                'max': max_rate,
                'range': range_rate
            })
    
    stats_df = pd.DataFrame(stats)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean with error bars (std)
    x = np.arange(len(feature_order))
    ax1.bar(x, stats_df['mean'], yerr=stats_df['std'], 
           color=[COLORS[fg] for fg in feature_order], alpha=0.8, 
           edgecolor='black', linewidth=1.5, capsize=5, error_kw={'linewidth': 2})
    
    ax1.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    ax1.set_ylabel('Mean Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Win Rate with Std Dev Across Levels', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_order)
    ax1.set_ylim(0, 80)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(stats_df['mean'], stats_df['std'])):
        ax1.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Range (max - min)
    ax2.bar(x, stats_df['range'], 
           color=[COLORS[fg] for fg in feature_order], alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Range (Max - Min) (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Group', fontsize=12, fontweight='bold')
    ax2.set_title('Variability: Range of Win Rates Across Levels', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_order)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, range_val in enumerate(stats_df['range']):
        ax2.text(i, range_val + 0.1, f'{range_val:.2f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Robustness Analysis: Variability Across Prompting Levels', 
                fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    output_path = output_dir / 'variability_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(combined_df: pd.DataFrame, output_dir: Path):
    """Create a summary table comparing all levels."""
    feature_order = ['CE-VAR', 'CE-GEO', 'TF-IDF-GEO', 'SBERT-GEO', 'OVERALL']
    level_order = ['LV1', 'LV2', 'LV3']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for fg in feature_order:
        row = [fg]
        fg_data = combined_df[combined_df['feature_group'] == fg]
        
        for level in level_order:
            level_data = fg_data[fg_data['level'] == level]
            if len(level_data) > 0:
                win_rate = level_data['human_win_rate'].iloc[0] * 100
                n_comparisons = level_data['n_comparisons'].iloc[0]
                significant = level_data['significant'].iloc[0]
                sig_star = level_data['significant_star'].iloc[0] if significant else ''
                row.append(f'{win_rate:.2f}%{sig_star}\n(n={n_comparisons:,})')
            else:
                row.append('N/A')
        
        # Calculate range across levels
        if len(fg_data) > 0:
            win_rates = fg_data['human_win_rate'].values * 100
            range_val = win_rates.max() - win_rates.min()
            row.append(f'{range_val:.2f}%')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    col_labels = ['Feature Group'] + level_order + ['Range']
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style significant cells
    for i, row in enumerate(table_data):
        for j in range(1, len(level_order) + 1):
            if '***' in row[j] or '**' in row[j] or '*' in row[j]:
                table[(i+1, j)].set_facecolor('#E8F4F8')
    
    ax.set_title('Robustness Test: Summary Comparison Across Prompting Levels', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize robustness test: Compare binomial test results across levels"
    )
    parser.add_argument(
        '--results-lv1',
        type=str,
        required=True,
        help='Path to LV1 binomial test results CSV'
    )
    parser.add_argument(
        '--results-lv2',
        type=str,
        required=True,
        help='Path to LV2 binomial test results CSV'
    )
    parser.add_argument(
        '--results-lv3',
        type=str,
        required=True,
        help='Path to LV3 binomial test results CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/binomial/robustness_test',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    results_paths = {
        'LV1': Path(args.results_lv1),
        'LV2': Path(args.results_lv2),
        'LV3': Path(args.results_lv3)
    }
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ROBUSTNESS TEST VISUALIZATION")
    print("="*80)
    for level, path in results_paths.items():
        print(f"{level}: {path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    combined_df = load_results(results_paths)
    print(f"Loaded results from {combined_df['level'].nunique()} levels")
    print(f"Total result groups: {len(combined_df)}")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_win_rate_comparison(combined_df, output_dir)
    plot_overall_comparison(combined_df, output_dir)
    plot_variability_analysis(combined_df, output_dir)
    plot_summary_table(combined_df, output_dir)
    
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-"*80)
    overall_df = combined_df[combined_df['feature_group'] == 'OVERALL']
    for level in ['LV1', 'LV2', 'LV3']:
        level_data = overall_df[overall_df['level'] == level]
        if len(level_data) > 0:
            win_rate = level_data['human_win_rate'].iloc[0] * 100
            print(f"{level}: {win_rate:.2f}%")
    
    # Calculate range
    if len(overall_df) > 0:
        win_rates = overall_df['human_win_rate'].values * 100
        print(f"\nRange: {win_rates.max() - win_rates.min():.2f}%")
        print(f"Mean: {win_rates.mean():.2f}%")
        print(f"Std Dev: {win_rates.std():.2f}%")


if __name__ == '__main__':
    main()

