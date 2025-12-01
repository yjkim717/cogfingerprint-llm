#!/usr/bin/env python3
"""
Binomial Test: Human vs LLM Comparison on TF-IDF and SBERT Geometry Features
from representation_pipeline data.

This script uses data from dataset/process/representation_pipeline/:
- *_tfidf_euclid.csv: TF-IDF geometry features
- *_embedding_euclid.csv: SBERT geometry features
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from pathlib import Path
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

# Geometry features to test
GEOMETRY_FEATURES = [
    'mean_yearly_distance',  # mean_distance
    'std_yearly_distance',    # std_distance
    'net_displacement',
    'path_length',
    'tortuosity'
]

# Exclude direction_consistency (not in this dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binomial test for Human vs LLM comparison on representation pipeline features"
    )
    parser.add_argument(
        '--pipeline-dir',
        type=str,
        default='dataset/process/representation_pipeline',
        help='Path to representation_pipeline directory'
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
        default='LV3',
        help='LLM level to use (default: LV3)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=['academic', 'blogs', 'news'],
        help='Domains to analyze (default: academic blogs news)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='micro_results/binomial/binomial_test_representation_pipeline_lv3.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    return parser.parse_args()


def parse_author_id(author_str: str) -> Dict:
    """Parse author ID to extract domain, field, author_id, model, level."""
    # Format: MODEL_LEVEL_Domain_FIELD_XX or human_Domain_FIELD_XX
    parts = author_str.split('_')
    
    if parts[0] in ['DS', 'G4B', 'G12B', 'LMK']:
        # LLM format: MODEL_LEVEL_Domain_FIELD_XX
        # e.g., DS_LV3_Academic_BIOLOGY_01
        level_str = parts[1]  # LV3
        level_int = int(level_str.replace('LV', '')) if 'LV' in level_str else int(level_str)
        return {
            'model': parts[0],
            'level': level_int,
            'domain': parts[2].lower(),
            'field': parts[3],
            'author_id': int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
            'is_llm': True
        }
    elif parts[0] == 'human':
        # Human format: human_Domain_FIELD_XX
        # e.g., human_Academic_BIOLOGY_01
        return {
            'model': None,
            'level': None,
            'domain': parts[1].lower(),
            'field': parts[2],
            'author_id': int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
            'is_llm': False
        }
    else:
        # Fallback: assume human format without 'human' prefix
        return {
            'model': None,
            'level': None,
            'domain': parts[0].lower(),
            'field': parts[1],
            'author_id': int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None,
            'is_llm': False
        }


def load_pipeline_data(pipeline_dir: Path, domain: str, feature_type: str) -> pd.DataFrame:
    """Load data from representation_pipeline directory.
    
    feature_type: 'tfidf' or 'embedding' (for SBERT)
    """
    if feature_type == 'tfidf':
        filename = f"{domain}_tfidf_euclid.csv"
    elif feature_type == 'embedding':
        filename = f"{domain}_embedding_euclid.csv"
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
    
    file_path = pipeline_dir / filename
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Reset index to avoid duplicate index issues
    df = df.reset_index(drop=True)
    
    # Parse author information
    parsed_info = df['author'].apply(parse_author_id)
    parsed_df = pd.DataFrame(list(parsed_info))
    
    # Reset index for parsed_df as well
    parsed_df = parsed_df.reset_index(drop=True)
    
    # Combine with original data - ensure no duplicate indices
    # First check if there are any duplicate indices
    if df.index.duplicated().any():
        df = df.reset_index(drop=True)
    if parsed_df.index.duplicated().any():
        parsed_df = parsed_df.reset_index(drop=True)
    
    # Align indices before concat
    df_index = df.index
    parsed_index = parsed_df.index
    if not df_index.equals(parsed_index):
        # Reset both to ensure alignment
        df = df.reset_index(drop=True)
        parsed_df = parsed_df.reset_index(drop=True)
    
    # Combine with original data - use assign to avoid column conflicts
    for col in parsed_df.columns:
        df[col] = parsed_df[col].values
    
    # Ensure no duplicate indices
    df = df.reset_index(drop=True)
    
    df['domain'] = domain
    df['feature_type'] = 'TF-IDF-GEO' if feature_type == 'tfidf' else 'SBERT-GEO'
    
    return df


def compare_human_vs_llm_feature(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    feature: str,
    model: str
) -> pd.DataFrame:
    """
    Compare Human vs LLM for each author × feature combination.
    Returns a DataFrame with comparison results.
    """
    results = []
    
    # Check if feature exists
    if feature not in human_df.columns or feature not in llm_df.columns:
        return pd.DataFrame()
    
    # Get unique author-field combinations from Human data
    human_groups = human_df.groupby(['domain', 'field', 'author_id'])
    
    for (domain, field, author_id), human_group in human_groups:
        # Get Human value for this author-field-feature
        human_value = human_group[feature].dropna()
        
        if len(human_value) == 0:
            continue
        
        human_val = human_value.iloc[0]  # Should be unique per author-field
        
        # Get LLM value for the same author-field-feature-model
        llm_group = llm_df[
            (llm_df['domain'] == domain) &
            (llm_df['field'] == field) &
            (llm_df['author_id'] == author_id) &
            (llm_df['model'] == model)
        ]
        
        if len(llm_group) == 0:
            continue
        
        llm_value = llm_group[feature].dropna()
        
        if len(llm_value) == 0:
            continue
        
        llm_val = llm_value.iloc[0]  # Should be unique per author-field-model
        
        # Skip if either value is NaN
        if pd.isna(human_val) or pd.isna(llm_val):
            continue
        
        # Binary outcome: Human > LLM
        human_wins = 1 if human_val > llm_val else 0
        
        feature_type = human_group['feature_type'].iloc[0]
        
        results.append({
            'domain': domain,
            'field': field,
            'author_id': author_id,
            'feature': feature,
            'feature_type': feature_type,
            'model': model,
            'human_value': human_val,
            'llm_value': llm_val,
            'human_wins': human_wins
        })
    
    return pd.DataFrame(results)


def perform_binomial_test(comparisons_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform binomial test for each feature group.
    For TF-IDF-GEO, use two-sided test (since LLM values may be higher).
    For SBERT-GEO, use one-sided test (greater).
    """
    results = []
    
    # Test for each feature group
    feature_groups = ['TF-IDF-GEO', 'SBERT-GEO']
    
    for group in feature_groups:
        group_comparisons = comparisons_df[comparisons_df['feature_type'] == group]
        
        if len(group_comparisons) == 0:
            continue
        
        n = len(group_comparisons)
        k = group_comparisons['human_wins'].sum()
        
        # Proportion of Human wins
        p_observed = k / n
        
        # For TF-IDF-GEO, use two-sided test
        # For SBERT-GEO, use one-sided (greater)
        if group == 'TF-IDF-GEO':
            test_result = binomtest(k, n, p=0.5, alternative='two-sided')
            direction = 'two-sided'
            # Also test both directions for interpretation
            test_greater = binomtest(k, n, p=0.5, alternative='greater')
            test_less = binomtest(k, n, p=0.5, alternative='less')
            
            if p_observed > 0.5:
                direction_note = 'Human > LLM' if test_greater.pvalue < alpha else 'not significant'
            elif p_observed < 0.5:
                direction_note = 'LLM > Human' if test_less.pvalue < alpha else 'not significant'
            else:
                direction_note = 'equal'
        else:
            # One-sided test: p > 0.5 (Human wins more)
            test_result = binomtest(k, n, p=0.5, alternative='greater')
            direction = 'greater'
            direction_note = 'Human > LLM' if test_result.pvalue < alpha else 'not significant'
        
        # Significance
        is_significant = test_result.pvalue < alpha
        
        results.append({
            'feature_group': group,
            'n_comparisons': n,
            'human_wins': k,
            'human_win_rate': p_observed,
            'pvalue': test_result.pvalue,
            'test_direction': direction,
            'direction_note': direction_note,
            'significant': is_significant,
            'significant_star': '***' if test_result.pvalue < 0.001 else
                              '**' if test_result.pvalue < 0.01 else
                              '*' if test_result.pvalue < 0.05 else ''
        })
    
    # Overall test (all groups combined)
    n_total = len(comparisons_df)
    k_total = comparisons_df['human_wins'].sum()
    
    if n_total > 0:
        p_observed_total = k_total / n_total
        test_result_total = binomtest(k_total, n_total, p=0.5, alternative='two-sided')
        is_significant_total = test_result_total.pvalue < alpha
        
        results.append({
            'feature_group': 'OVERALL',
            'n_comparisons': n_total,
            'human_wins': k_total,
            'human_win_rate': p_observed_total,
            'pvalue': test_result_total.pvalue,
            'test_direction': 'two-sided',
            'direction_note': None,
            'significant': is_significant_total,
            'significant_star': '***' if test_result_total.pvalue < 0.001 else
                              '**' if test_result_total.pvalue < 0.01 else
                              '*' if test_result_total.pvalue < 0.05 else ''
        })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    pipeline_dir = Path(args.pipeline_dir)
    
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"Pipeline directory not found: {pipeline_dir}")
    
    print("="*80)
    print("BINOMIAL TEST: Human vs LLM (Representation Pipeline Data)")
    print("="*80)
    print(f"Pipeline Dir: {pipeline_dir}")
    print(f"Domains: {args.domains}")
    print(f"Models: {args.models}")
    print(f"Level: {args.level}")
    print(f"Feature Groups: TF-IDF-GEO, SBERT-GEO")
    print()
    
    # Load and combine data from all domains
    all_comparisons = []
    
    # Convert level to int for comparison (LV3 -> 3)
    level_int = int(args.level.replace('LV', ''))
    
    for domain in args.domains:
        print(f"Processing domain: {domain}")
        
        # Load TF-IDF data
        tfidf_df = load_pipeline_data(pipeline_dir, domain, 'tfidf')
        if len(tfidf_df) == 0:
            print(f"  Warning: No TF-IDF data found for {domain}")
        else:
            total_samples = len(tfidf_df)
            human_samples = len(tfidf_df[tfidf_df['label'] == 'human'])
            llm_lv3_samples = len(tfidf_df[(tfidf_df['label'].values == 'llm') & (tfidf_df['level'].values == level_int)])
            print(f"  TF-IDF: {total_samples} total (Human: {human_samples}, LLM LV{level_int}: {llm_lv3_samples})")
        
        # Load SBERT (embedding) data
        sbert_df = load_pipeline_data(pipeline_dir, domain, 'embedding')
        if len(sbert_df) == 0:
            print(f"  Warning: No SBERT data found for {domain}")
        else:
            total_samples = len(sbert_df)
            human_samples = len(sbert_df[sbert_df['label'] == 'human'])
            llm_lv3_samples = len(sbert_df[(sbert_df['label'].values == 'llm') & (sbert_df['level'].values == level_int)])
            print(f"  SBERT: {total_samples} total (Human: {human_samples}, LLM LV{level_int}: {llm_lv3_samples})")
        
        # Separate Human and LLM data
        
        if len(tfidf_df) > 0:
            tfidf_df = tfidf_df.reset_index(drop=True).copy()
            # Use .values to avoid index alignment issues
            human_mask = (tfidf_df['label'].values == 'human')
            llm_mask = (tfidf_df['label'].values == 'llm') & (tfidf_df['level'].values == level_int)
            human_tfidf = tfidf_df.loc[human_mask].copy().reset_index(drop=True)
            llm_tfidf = tfidf_df.loc[llm_mask].copy().reset_index(drop=True)
        else:
            human_tfidf = pd.DataFrame()
            llm_tfidf = pd.DataFrame()
        
        if len(sbert_df) > 0:
            sbert_df = sbert_df.reset_index(drop=True).copy()
            # Use .values to avoid index alignment issues
            human_mask = (sbert_df['label'].values == 'human')
            llm_mask = (sbert_df['label'].values == 'llm') & (sbert_df['level'].values == level_int)
            human_sbert = sbert_df.loc[human_mask].copy().reset_index(drop=True)
            llm_sbert = sbert_df.loc[llm_mask].copy().reset_index(drop=True)
        else:
            human_sbert = pd.DataFrame()
            llm_sbert = pd.DataFrame()
        
        # Compare all features for each model
        for model in args.models:
            # TF-IDF comparisons
            if len(human_tfidf) > 0 and len(llm_tfidf) > 0:
                for feature in GEOMETRY_FEATURES:
                    comparisons = compare_human_vs_llm_feature(
                        human_tfidf, llm_tfidf, feature, model
                    )
                    if len(comparisons) > 0:
                        all_comparisons.append(comparisons)
            
            # SBERT comparisons
            if len(human_sbert) > 0 and len(llm_sbert) > 0:
                for feature in GEOMETRY_FEATURES:
                    comparisons = compare_human_vs_llm_feature(
                        human_sbert, llm_sbert, feature, model
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
    print(f"  - Feature groups: {', '.join(comparisons_df['feature_type'].unique())}")
    
    # Perform binomial tests
    print("\nPerforming binomial tests...")
    test_results = perform_binomial_test(comparisons_df, args.alpha)
    
    # Sort by feature group order
    group_order = {'TF-IDF-GEO': 0, 'SBERT-GEO': 1, 'OVERALL': 2}
    test_results['sort_order'] = test_results['feature_group'].map(group_order)
    test_results = test_results.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Save results
    output_path = Path(args.output)
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
    print(f"\nH₀: Human and LLM have equal values (p = 0.5)")
    print("\nResults by Feature Group:")
    print("-"*80)
    
    for _, row in test_results.iterrows():
        sig_marker = row['significant_star'] if row['significant'] else ''
        direction_info = ""
        if row.get('direction_note'):
            direction_info = f" [{row['direction_note']}]"
        print(f"{row['feature_group']:15s} | Win Rate: {row['human_win_rate']:.1%} | "
              f"Wins: {row['human_wins']:6,}/{row['n_comparisons']:6,} | "
              f"p-value: {row['pvalue']:.6f} {sig_marker}{direction_info}")
    
    # Overall summary
    overall = test_results[test_results['feature_group'] == 'OVERALL'].iloc[0]
    print("\n" + "="*80)
    print("OVERALL RESULT:")
    print(f"Human wins: {overall['human_win_rate']:.1%} ({overall['human_wins']:,}/{overall['n_comparisons']:,})")
    print(f"p-value: {overall['pvalue']:.6f} {overall['significant_star']}")
    if overall['significant']:
        if overall['human_win_rate'] > 0.5:
            print("→ Human trajectory features are systematically > LLM")
        else:
            print("→ LLM trajectory features are systematically > Human")
    print("="*80)


if __name__ == '__main__':
    main()
