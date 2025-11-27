#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on Time-Series Metrics
==============================================================

Compare Human vs LLM for every author × feature × metric × LLM model.
Uses time-series statistics (VAR, CV, RMSSD, MASD) computed from raw features over years.
Each comparison yields a binary outcome (Human_metric > LLM_metric).
Binomial tests evaluate whether Human wins occur significantly more than the 50% random baseline.

Flow:
  1. Raw Features over Years (18 linguistic & emotional)
     ↓
  2. Compute Time-Series Metrics (per author × feature):
     • VAR   – overall temporal fluctuation
     • CV    – normalized variability
     • RMSSD – short-term volatility (year-to-year jump)
     • MASD  – absolute drift magnitude
     ↓
  3. Human vs LLM Comparison (per author × feature × metric × LLM model)
     ↓
  4. Outcome: 1 (Human wins) or 0 (LLM wins)
     ↓
  5. Binomial Test: H₀: p = 0.5, H₁: p > 0.5
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Time-series metrics (normalized, length-insensitive)
# CV is the normalized version of VAR (CV = VAR / Mean²), but we include VAR for validation
# RMSSD_norm and MASD_norm are also normalized versions
# Note: VAR is included to validate that VAR and CV are equivalent (they should show similar results)
METRICS = ['variance', 'cv', 'rmssd_norm', 'masd_norm']
METRIC_DISPLAY = {
    'variance': 'VAR',
    'cv': 'CV',
    'rmssd_norm': 'RMSSD (norm)',
    'masd_norm': 'MASD (norm)',
    # Legacy metrics (kept for reference but not used in main analysis)
    'rmssd': 'RMSSD',
    'masd': 'MASD'
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM comparison on time-series metrics"
    )
    parser.add_argument(
        '--human-data',
        type=str,
        required=True,
        help='Path to human author_timeseries_stats_merged.csv file'
    )
    parser.add_argument(
        '--llm-data-dir',
        type=str,
        required=True,
        help='Directory containing LLM author_timeseries_stats_merged.csv files'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['DS', 'G4B', 'G12B', 'LMK'],
        help='LLM models to include (default: DS G4B G12B LMK)'
    )
    parser.add_argument(
        '--level',
        type=str,
        default='LV1',
        help='LLM level to use (default: LV1)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=['academic', 'news', 'blogs'],
        help='Domains to analyze (default: academic news blogs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='micro_results/binomial/binomial_test_results.csv',
        help='Output CSV file path (level suffix will be appended automatically if missing)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    return parser.parse_args()


def load_human_data(human_path: Path, domain: str) -> pd.DataFrame:
    """Load Human time-series statistics data."""
    domain_file = human_path / domain / 'author_timeseries_stats_merged.csv'
    if not domain_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(domain_file)
    df['domain'] = domain
    return df


def load_llm_data(llm_data_dir: Path, model: str, level: str, domain: str) -> pd.DataFrame:
    """Load LLM time-series statistics data for a specific model."""
    model_file = llm_data_dir / model / level / domain / 'author_timeseries_stats_merged.csv'
    if not model_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(model_file)
    df['domain'] = domain
    df['model'] = model
    df['level'] = level
    return df


def get_feature_metric_columns(df: pd.DataFrame, metric: str) -> List[str]:
    """Get feature columns for a specific metric (e.g., columns ending with '_cv')."""
    metric_suffix = f'_{metric}'
    feature_cols = [col for col in df.columns if col.endswith(metric_suffix)]
    return feature_cols


def extract_feature_name(feature_metric_col: str, metric: str) -> str:
    """Extract base feature name from feature_metric column."""
    # e.g., 'Openness_cv' -> 'Openness'
    metric_suffix = f'_{metric}'
    if feature_metric_col.endswith(metric_suffix):
        return feature_metric_col[:-len(metric_suffix)]
    return feature_metric_col


def compare_human_vs_llm_metric(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature: str,
    metric: str,
    model: str
) -> pd.DataFrame:
    """
    Compare Human vs LLM for each author × feature × metric combination.
    Returns a DataFrame with comparison results.
    """
    results = []
    
    feature_metric_col = f'{feature}_{metric}'
    
    # Check if columns exist
    if feature_metric_col not in human_df.columns:
        return pd.DataFrame()
    if feature_metric_col not in llm_df.columns:
        return pd.DataFrame()
    
    # Get unique author-field combinations from Human data
    human_groups = human_df.groupby(['domain', 'field', 'author_id'])
    
    for (domain, field, author_id), human_group in human_groups:
        # Get Human value for this author-field-feature-metric
        human_value = human_group[feature_metric_col].dropna()
        
        if len(human_value) == 0:
            continue
        
        human_val = human_value.iloc[0]  # Should be unique per author-field
        
        # Get LLM value for the same author-field-feature-metric-model
        llm_group = llm_df[
            (llm_df['domain'] == domain) &
            (llm_df['field'] == field) &
            (llm_df['author_id'] == author_id) &
            (llm_df['model'] == model)
        ]
        
        if len(llm_group) == 0:
            continue
        
        llm_value = llm_group[feature_metric_col].dropna()
        
        if len(llm_value) == 0:
            continue
        
        llm_val = llm_value.iloc[0]  # Should be unique per author-field-model
        
        # Binary outcome: Human > LLM
        human_wins = 1 if human_val > llm_val else 0
        
        results.append({
            'domain': domain,
            'field': field,
            'author_id': author_id,
            'feature': feature,
            'metric': metric,
            'model': model,
            'human_value': human_val,
            'llm_value': llm_val,
            'human_wins': human_wins
        })
    
    return pd.DataFrame(results)


def perform_binomial_test(comparisons_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform binomial test for each metric (and overall).
    H0: p = 0.5 (Human wins 50% of the time by chance)
    H1: p > 0.5 (Human wins significantly more than 50%)
    """
    results = []
    
    # Test for each metric
    for metric in METRICS:
        metric_comparisons = comparisons_df[comparisons_df['metric'] == metric]
        
        if len(metric_comparisons) == 0:
            continue
        
        n = len(metric_comparisons)
        k = metric_comparisons['human_wins'].sum()
        
        # Proportion of Human wins
        p_observed = k / n
        
        # Binomial test (one-sided: p > 0.5)
        test_result = binomtest(k, n, p=0.5, alternative='greater')
        
        # Significance
        is_significant = test_result.pvalue < alpha
        
        results.append({
            'metric': METRIC_DISPLAY.get(metric, metric.upper()),
            'n_comparisons': n,
            'human_wins': k,
            'human_win_rate': p_observed,
            'pvalue': test_result.pvalue,
            'significant': is_significant,
            'significant_star': '***' if test_result.pvalue < 0.001 else
                               '**' if test_result.pvalue < 0.01 else
                               '*' if test_result.pvalue < 0.05 else ''
        })
    
    # Overall test (all metrics combined)
    n_total = len(comparisons_df)
    k_total = comparisons_df['human_wins'].sum()
    
    if n_total > 0:
        p_observed_total = k_total / n_total
        test_result_total = binomtest(k_total, n_total, p=0.5, alternative='greater')
        is_significant_total = test_result_total.pvalue < alpha
        
        results.append({
            'metric': 'OVERALL',
            'n_comparisons': n_total,
            'human_wins': k_total,
            'human_win_rate': p_observed_total,
            'pvalue': test_result_total.pvalue,
            'significant': is_significant_total,
            'significant_star': '***' if test_result_total.pvalue < 0.001 else
                               '**' if test_result_total.pvalue < 0.01 else
                               '*' if test_result_total.pvalue < 0.05 else ''
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    human_path = Path(args.human_data)
    llm_data_dir = Path(args.llm_data_dir)
    
    if not human_path.exists():
        raise FileNotFoundError(f"Human data directory not found: {human_path}")
    if not llm_data_dir.exists():
        raise FileNotFoundError(f"LLM data directory not found: {llm_data_dir}")
    
    print("="*80)
    print("BINOMIAL TEST: Human vs LLM Time-Series Metrics")
    print("="*80)
    print(f"Domains: {args.domains}")
    print(f"Models: {args.models}")
    print(f"Level: {args.level}")
    print(f"Metrics: {', '.join([METRIC_DISPLAY[m] for m in METRICS])}")
    
    # Load and combine data from all domains
    all_comparisons = []
    
    for domain in args.domains:
        print(f"\nProcessing domain: {domain}")
        
        # Load Human data
        human_df = load_human_data(human_path, domain)
        if len(human_df) == 0:
            print(f"  Warning: No Human data found for {domain}")
            continue
        print(f"  Human samples: {len(human_df)}")
        
        # Load LLM data for each model
        for model in args.models:
            llm_df = load_llm_data(llm_data_dir, model, args.level, domain)
            if len(llm_df) == 0:
                print(f"  Warning: No LLM data found for {model}/{args.level}/{domain}")
                continue
            
            print(f"  {model} samples: {len(llm_df)}")
            
            # For each metric
            for metric in METRICS:
                # Get feature columns for this metric
                feature_metric_cols = get_feature_metric_columns(human_df, metric)
                
                for feature_metric_col in feature_metric_cols:
                    feature = extract_feature_name(feature_metric_col, metric)
                    
                    # Compare Human vs LLM
                    comparisons = compare_human_vs_llm_metric(
                        human_df, llm_df, feature, metric, model
                    )
                    
                    if len(comparisons) > 0:
                        all_comparisons.append(comparisons)
    
    if len(all_comparisons) == 0:
        print("\nNo comparisons found!")
        return
    
    comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    print(f"\nTotal comparisons: {len(comparisons_df):,}")
    print(f"  - Unique authors: {comparisons_df.groupby(['domain', 'field', 'author_id']).ngroups}")
    print(f"  - Unique features: {comparisons_df['feature'].nunique()}")
    print(f"  - Unique models: {comparisons_df['model'].nunique()}")
    print(f"  - Metrics: {', '.join(comparisons_df['metric'].unique())}")
    
    # Perform binomial tests
    print("\nPerforming binomial tests...")
    test_results = perform_binomial_test(comparisons_df, args.alpha)
    
    # Sort by metric order
    metric_order = {METRIC_DISPLAY.get(m, m.upper()): i for i, m in enumerate(METRICS)}
    metric_order['OVERALL'] = len(METRICS)
    test_results['sort_order'] = test_results['metric'].map(metric_order)
    test_results = test_results.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Save results with level-specific suffix
    output_path = Path(args.output)
    level_suffix = f"_{args.level.lower()}"
    if not output_path.stem.lower().endswith(level_suffix):
        output_path = output_path.with_name(f"{output_path.stem}{level_suffix}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparisons
    comparisons_output = output_path.parent / f"{output_path.stem}_detailed.csv"
    comparisons_df.to_csv(comparisons_output, index=False)
    print(f"\nDetailed comparisons saved to: {comparisons_output}")
    
    # Save test results
    test_results.to_csv(output_path, index=False)
    print(f"Test results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BINOMIAL TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Significance level (alpha): {args.alpha}")
    print(f"\nH₀: Human and LLM have equal variability (p = 0.5)")
    print(f"H₁: Human variability is greater (p > 0.5)")
    print("\nResults by Metric:")
    print("-"*80)
    
    for _, row in test_results.iterrows():
        sig_marker = row['significant_star'] if row['significant'] else ''
        print(f"{row['metric']:10s} | Win Rate: {row['human_win_rate']:.1%} | "
              f"Wins: {row['human_wins']:6,}/{row['n_comparisons']:6,} | "
              f"p-value: {row['pvalue']:.6f} {sig_marker}")
    
    # Overall summary
    overall = test_results[test_results['metric'] == 'OVERALL'].iloc[0]
    print("\n" + "="*80)
    print("OVERALL RESULT:")
    print(f"Human wins: {overall['human_win_rate']:.1%} ({overall['human_wins']:,}/{overall['n_comparisons']:,})")
    print(f"p-value: {overall['pvalue']:.6f} {overall['significant_star']}")
    if overall['significant']:
        print("→ Human evolution is systematically > LLM")
    print("="*80)


if __name__ == '__main__':
    main()

