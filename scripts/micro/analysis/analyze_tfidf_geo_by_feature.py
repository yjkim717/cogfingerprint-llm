#!/usr/bin/env python3
"""
Analyze TF-IDF-GEO win rate breakdown by individual features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

COLORS = {
    'Human > LLM': '#4A90E2',
    'LLM > Human': '#E24A4A',
    'baseline': '#CCCCCC'
}


def analyze_tfidf_features(detailed_csv: str, output_dir: Path):
    """Analyze TF-IDF-GEO features individually."""
    df = pd.read_csv(detailed_csv)
    
    # Filter TF-IDF-GEO features
    tfidf_df = df[df['feature_type'] == 'TF-IDF-GEO'].copy()
    
    # Calculate win rate for each feature
    feature_stats = []
    for feature in tfidf_df['feature'].unique():
        feature_data = tfidf_df[tfidf_df['feature'] == feature]
        n = len(feature_data)
        wins = feature_data['human_wins'].sum()
        win_rate = wins / n * 100
        
        feature_stats.append({
            'feature': feature.replace('tfidf_', ''),
            'n_comparisons': n,
            'human_wins': wins,
            'human_win_rate': win_rate,
            'llm_win_rate': 100 - win_rate
        })
    
    feature_stats_df = pd.DataFrame(feature_stats).sort_values('human_win_rate')
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart with win rates
    colors = [COLORS['Human > LLM'] if rate > 50 else COLORS['LLM > Human'] 
              for rate in feature_stats_df['human_win_rate']]
    
    bars = ax1.bar(range(len(feature_stats_df)), feature_stats_df['human_win_rate'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, 
               label='Baseline (50%)', zorder=0)
    
    # Add value labels
    for i, (idx, row) in enumerate(feature_stats_df.iterrows()):
        y_pos = row['human_win_rate']
        ax1.text(i, y_pos + 1, f'{y_pos:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        ax1.text(i, -3, f'n={row["n_comparisons"]:,}', ha='center', va='top', 
                fontsize=9, style='italic', color='gray')
    
    ax1.set_xticks(range(len(feature_stats_df)))
    ax1.set_xticklabels(feature_stats_df['feature'], rotation=15, ha='right')
    ax1.set_ylabel('Human Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('TF-IDF Geometry Feature', fontsize=12, fontweight='bold')
    ax1.set_title('TF-IDF-GEO: Human Win Rate by Feature', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 60)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Plot 2: Stacked bar chart showing Human vs LLM wins
    x = range(len(feature_stats_df))
    width = 0.6
    
    ax2.bar(x, feature_stats_df['human_win_rate'], width, 
           label='Human Wins', color=COLORS['Human > LLM'], alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    ax2.bar(x, feature_stats_df['llm_win_rate'], width, 
           bottom=feature_stats_df['human_win_rate'],
           label='LLM Wins', color=COLORS['LLM > Human'], alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    
    ax2.axhline(y=50, color=COLORS['baseline'], linestyle='--', linewidth=2, zorder=0)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(feature_stats_df.iterrows()):
        human_rate = row['human_win_rate']
        llm_rate = row['llm_win_rate']
        if human_rate > 10:
            ax2.text(i, human_rate / 2, f'{human_rate:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
        if llm_rate > 10:
            ax2.text(i, human_rate + llm_rate / 2, f'{llm_rate:.1f}%', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='white')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_stats_df['feature'], rotation=15, ha='right')
    ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('TF-IDF Geometry Feature', fontsize=12, fontweight='bold')
    ax2.set_title('TF-IDF-GEO: Human vs LLM Win Distribution', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('TF-IDF-GEO Win Rate Breakdown: Which Features Cause Low Human Win Rate?', 
                fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    output_path = output_dir / 'tfidf_geo_feature_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save statistics
    stats_path = output_dir / 'tfidf_geo_feature_statistics.csv'
    feature_stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TF-IDF-GEO FEATURE BREAKDOWN")
    print("="*80)
    print(f"\nOverall TF-IDF-GEO Win Rate: {tfidf_df['human_wins'].sum() / len(tfidf_df) * 100:.2f}%")
    print(f"  ({tfidf_df['human_wins'].sum():,}/{len(tfidf_df):,} comparisons)\n")
    print("Individual Features (sorted by win rate):")
    print("-"*80)
    for _, row in feature_stats_df.iterrows():
        status = "Human > LLM" if row['human_win_rate'] > 50 else "LLM > Human"
        print(f"{row['feature']:25s} | {row['human_win_rate']:6.2f}% | {status}")
    print("\n" + "="*80)
    print("KEY FINDING:")
    print(f"- Features with Human < 50%: {len(feature_stats_df[feature_stats_df['human_win_rate'] < 50])}")
    print(f"- Features with Human > 50%: {len(feature_stats_df[feature_stats_df['human_win_rate'] > 50])}")
    print(f"- Lowest win rate: {feature_stats_df['human_win_rate'].min():.2f}% ({feature_stats_df.loc[feature_stats_df['human_win_rate'].idxmin(), 'feature']})")
    print(f"- Highest win rate: {feature_stats_df['human_win_rate'].max():.2f}% ({feature_stats_df.loc[feature_stats_df['human_win_rate'].idxmax(), 'feature']})")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze TF-IDF-GEO features individually")
    parser.add_argument('--detailed', type=str, 
                      default='micro_results/binomial/binomial_test_trajectory_features_lv3_detailed.csv',
                      help='Path to detailed comparisons CSV')
    parser.add_argument('--output', type=str, 
                      default='plots/analysis/tfidf_geo_breakdown',
                      help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_tfidf_features(args.detailed, output_dir)

