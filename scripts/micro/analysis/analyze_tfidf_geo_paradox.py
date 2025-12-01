#!/usr/bin/env python3
"""
Analyze the TF-IDF-GEO paradox:
- Binomial test: 35.4% human win rate (LLM values are higher on average)
- ML classification: 80.49% accuracy (ML can distinguish well)

This script analyzes why ML can distinguish even when LLM values are higher on average.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 11

COLORS = {
    'Human': '#4A90E2',
    'LLM': '#E24A4A'
}


def load_trajectory_data(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load trajectory features for Human and LLM."""
    DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
    frames = []
    
    for domain in domains:
        # Human data
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            frames.append(df_h)
        
        # LLM data
        for model in models:
            llm_path = DATA_ROOT / "LLM" / model / level / domain / "trajectory_features_combined.csv"
            if llm_path.exists():
                df_l = pd.read_csv(llm_path)
                df_l["domain"] = domain
                df_l["label"] = "llm"
                df_l["model"] = model
                frames.append(df_l)
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def extract_tfidf_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract TF-IDF geometry features."""
    tfidf_features = [
        'tfidf_mean_distance',
        'tfidf_std_distance',
        'tfidf_net_displacement',
        'tfidf_path_length',
        'tfidf_tortuosity'
    ]
    
    # Filter to only TF-IDF features
    feature_cols = [col for col in tfidf_features if col in df.columns]
    metadata_cols = ['domain', 'label', 'model', 'author_id', 'field']
    
    result_df = df[metadata_cols + feature_cols].copy()
    
    # Melt to long format for easier analysis
    melted = result_df.melt(
        id_vars=metadata_cols,
        value_vars=feature_cols,
        var_name='feature',
        value_name='value'
    )
    
    return melted


def calculate_statistics(human_values: np.ndarray, llm_values: np.ndarray) -> Dict:
    """Calculate statistical measures for comparison."""
    stats_dict = {
        'human_mean': np.mean(human_values),
        'llm_mean': np.mean(llm_values),
        'human_std': np.std(human_values),
        'llm_std': np.std(llm_values),
        'human_median': np.median(human_values),
        'llm_median': np.median(llm_values),
        'mean_diff': np.mean(llm_values) - np.mean(human_values),
        'mean_diff_pct': ((np.mean(llm_values) - np.mean(human_values)) / np.mean(human_values)) * 100 if np.mean(human_values) > 0 else 0,
    }
    
    # Coefficient of variation
    stats_dict['human_cv'] = stats_dict['human_std'] / stats_dict['human_mean'] if stats_dict['human_mean'] > 0 else 0
    stats_dict['llm_cv'] = stats_dict['llm_std'] / stats_dict['llm_mean'] if stats_dict['llm_mean'] > 0 else 0
    
    # Skewness and kurtosis
    stats_dict['human_skew'] = stats.skew(human_values)
    stats_dict['llm_skew'] = stats.skew(llm_values)
    stats_dict['human_kurtosis'] = stats.kurtosis(human_values)
    stats_dict['llm_kurtosis'] = stats.kurtosis(llm_values)
    
    # Kolmogorov-Smirnov test (distribution difference)
    ks_stat, ks_pvalue = stats.ks_2samp(human_values, llm_values)
    stats_dict['ks_statistic'] = ks_stat
    stats_dict['ks_pvalue'] = ks_pvalue
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(human_values, llm_values, alternative='two-sided')
    stats_dict['mannwhitney_u'] = u_stat
    stats_dict['mannwhitney_pvalue'] = u_pvalue
    
    return stats_dict


def plot_distribution_comparison(melted_df: pd.DataFrame, output_dir: Path):
    """Plot distribution comparison for each TF-IDF geometry feature."""
    features = melted_df['feature'].unique()
    n_features = len(features)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_stats = []
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        feature_data = melted_df[melted_df['feature'] == feature].copy()
        
        human_data = feature_data[feature_data['label'] == 'human']['value'].dropna()
        llm_data = feature_data[feature_data['label'] == 'llm']['value'].dropna()
        
        if len(human_data) == 0 or len(llm_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate statistics
        stats_dict = calculate_statistics(human_data.values, llm_data.values)
        stats_dict['feature'] = feature
        all_stats.append(stats_dict)
        
        # Plot histograms
        ax.hist(human_data, bins=50, alpha=0.6, label='Human', 
               color=COLORS['Human'], edgecolor='black', linewidth=0.5, density=True)
        ax.hist(llm_data, bins=50, alpha=0.6, label='LLM', 
               color=COLORS['LLM'], edgecolor='black', linewidth=0.5, density=True)
        
        # Add mean lines
        human_mean = stats_dict['human_mean']
        llm_mean = stats_dict['llm_mean']
        ax.axvline(human_mean, color=COLORS['Human'], linestyle='--', linewidth=2, 
                  label=f'Human mean: {human_mean:.4f}')
        ax.axvline(llm_mean, color=COLORS['LLM'], linestyle='--', linewidth=2, 
                  label=f'LLM mean: {llm_mean:.4f}')
        
        # Add statistics text
        stats_text = (
            f"Human: μ={human_mean:.4f}, σ={stats_dict['human_std']:.4f}\n"
            f"LLM: μ={llm_mean:.4f}, σ={stats_dict['llm_std']:.4f}\n"
            f"KS test: D={stats_dict['ks_statistic']:.3f}, p={stats_dict['ks_pvalue']:.2e}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Feature Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(feature.replace('tfidf_', '').replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplot
    if n_features < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('TF-IDF Geometry Features: Distribution Comparison (Human vs LLM)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'tfidf_geo_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return pd.DataFrame(all_stats)


def plot_boxplot_comparison(melted_df: pd.DataFrame, output_dir: Path):
    """Plot boxplot comparison showing quartiles and outliers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create boxplot
    sns.boxplot(data=melted_df, x='feature', y='value', hue='label', 
               palette=[COLORS['Human'], COLORS['LLM']], ax=ax)
    
    ax.set_xlabel('TF-IDF Geometry Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
    ax.set_title('TF-IDF Geometry Features: Boxplot Comparison (Human vs LLM)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels([f.get_text().replace('tfidf_', '').replace('_', ' ').title() 
                        for f in ax.get_xticklabels()], rotation=15, ha='right')
    ax.legend(title='Label', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'tfidf_geo_boxplot_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_violin_comparison(melted_df: pd.DataFrame, output_dir: Path):
    """Plot violin plot showing full distribution shape."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.violinplot(data=melted_df, x='feature', y='value', hue='label', 
                  palette=[COLORS['Human'], COLORS['LLM']], ax=ax, inner='box')
    
    ax.set_xlabel('TF-IDF Geometry Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
    ax.set_title('TF-IDF Geometry Features: Violin Plot (Distribution Shape)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels([f.get_text().replace('tfidf_', '').replace('_', ' ').title() 
                        for f in ax.get_xticklabels()], rotation=15, ha='right')
    ax.legend(title='Label', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'tfidf_geo_violin_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_report(stats_df: pd.DataFrame, output_dir: Path):
    """Create a summary report explaining the paradox."""
    report_path = output_dir / 'tfidf_geo_paradox_analysis.md'
    
    with open(report_path, 'w') as f:
        f.write("# TF-IDF-GEO Paradox Analysis\n\n")
        f.write("## The Paradox\n\n")
        f.write("- **Binomial Test**: 35.4% human win rate (LLM values are higher on average)\n")
        f.write("- **ML Classification**: 80.49% accuracy (ML can distinguish well)\n\n")
        f.write("## Why This Happens\n\n")
        f.write("### Key Insight\n\n")
        f.write("**Binomial test compares means, ML learns distribution differences.**\n\n")
        f.write("Even if LLM values are higher on average, ML can still distinguish if:\n")
        f.write("1. **Variance differences**: Human and LLM have different spreads\n")
        f.write("2. **Distribution shape**: Different skewness, kurtosis, or overall shape\n")
        f.write("3. **Feature interactions**: ML can learn complex patterns across multiple features\n")
        f.write("4. **Non-linear boundaries**: ML can find non-linear decision boundaries\n\n")
        f.write("## Statistical Summary\n\n")
        f.write("| Feature | Human Mean | LLM Mean | Mean Diff | Human Std | LLM Std | CV Ratio | KS p-value |\n")
        f.write("|---------|------------|----------|-----------|-----------|---------|----------|------------|\n")
        
        for _, row in stats_df.iterrows():
            cv_ratio = row['llm_cv'] / row['human_cv'] if row['human_cv'] > 0 else 0
            f.write(f"| {row['feature']} | {row['human_mean']:.4f} | {row['llm_mean']:.4f} | "
                   f"{row['mean_diff']:.4f} | {row['human_std']:.4f} | {row['llm_std']:.4f} | "
                   f"{cv_ratio:.2f} | {row['ks_pvalue']:.2e} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write("1. **Mean Difference**: LLM values are higher (negative mean_diff or positive when LLM > Human)\n")
        f.write("2. **Variance**: If CV ratios differ significantly, distributions have different spreads\n")
        f.write("3. **KS Test**: Low p-value indicates significantly different distributions\n")
        f.write("4. **ML Advantage**: ML can learn these distribution differences, not just mean differences\n\n")
        f.write("## Conclusion\n\n")
        f.write("The paradox is resolved: **ML classification accuracy does not depend solely on mean differences.**\n")
        f.write("Even when LLM values are higher on average, ML can distinguish based on:\n")
        f.write("- Distribution shape differences\n")
        f.write("- Variance differences\n")
        f.write("- Multi-feature patterns\n")
        f.write("- Non-linear boundaries\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TF-IDF-GEO paradox: Why ML can distinguish when LLM values are higher"
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        default=['academic', 'blogs', 'news'],
        help='Domains to analyze'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['DS', 'G4B', 'G12B', 'LMK'],
        help='LLM models to include'
    )
    parser.add_argument(
        '--level',
        default='LV3',
        help='LLM level to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/analysis/tfidf_geo_paradox',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TF-IDF-GEO PARADOX ANALYSIS")
    print("="*80)
    print(f"Domains: {args.domains}")
    print(f"Models: {args.models}")
    print(f"Level: {args.level}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("Loading trajectory data...")
    df = load_trajectory_data(args.domains, args.models, args.level)
    if df.empty:
        print("Error: No data found!")
        return
    
    print(f"Loaded {len(df)} samples")
    print(f"  Human: {len(df[df['label'] == 'human'])}")
    print(f"  LLM: {len(df[df['label'] == 'llm'])}")
    print()
    
    # Extract TF-IDF features
    print("Extracting TF-IDF geometry features...")
    melted_df = extract_tfidf_geo_features(df)
    print(f"Extracted {len(melted_df)} feature-value pairs")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    stats_df = plot_distribution_comparison(melted_df, output_dir)
    plot_boxplot_comparison(melted_df, output_dir)
    plot_violin_comparison(melted_df, output_dir)
    
    # Save statistics
    stats_path = output_dir / 'tfidf_geo_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    # Create summary report
    create_summary_report(stats_df, output_dir)
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

