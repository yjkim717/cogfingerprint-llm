#!/usr/bin/env python3
"""
Remove outliers from combined_merged.csv files.
For each model-feature combination, detect outliers using IQR method and set them to NaN.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

def remove_outliers_iqr(series: pd.Series, iqr_factor: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method. Returns boolean mask (True = keep, False = outlier)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        # No variability, keep all values
        return pd.Series([True] * len(series), index=series.index)
    
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    # Return mask: True for non-outliers, False for outliers
    return (series >= lower_bound) & (series <= upper_bound)


def remove_outliers_from_combined_merged(
    input_file: Path,
    output_file: Path,
    iqr_factor: float = 1.5,
    feature_columns: List[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Remove outliers from combined_merged.csv file.
    
    Args:
        input_file: Path to input combined_merged.csv
        output_file: Path to output file
        iqr_factor: IQR factor for outlier detection (default: 1.5)
        feature_columns: List of feature columns to process (None = auto-detect)
    
    Returns:
        Tuple of (processed DataFrame, statistics dictionary)
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Identify metadata columns
    metadata_cols = ['filename', 'path', 'label', 'domain', 'field', 'author_id', 'model', 'level']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    
    # Auto-detect feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in df.columns 
                          if col not in metadata_cols 
                          and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"  Processing {len(feature_columns)} feature columns: {feature_columns[:5]}...")
    
    # Check if 'model' column exists
    if 'model' not in df.columns:
        print("  Warning: 'model' column not found. Treating all data as single model.")
        df['model'] = 'ALL'
    else:
        df['model'] = df['model'].fillna('ALL')

    if 'domain' in df.columns:
        df['_outlier_group'] = df['model'] + "_" + df['domain'].fillna("UNKNOWN")
    else:
        df['_outlier_group'] = df['model']
    
    # Statistics
    stats = {
        'total_rows': len(df),
        'total_values': 0,
        'total_outliers': 0,
        'by_model_feature': {}
    }
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Process each model-feature combination
    groups = df['_outlier_group'].unique()
    print(f"\nProcessing {len(groups)} model-domain group(s): {list(groups)}")
    
    for group_key in groups:
        group_data = df[df['_outlier_group'] == group_key]
        print(f"\n  Group: {group_key} ({len(group_data)} rows)")
        
        for feature_col in feature_columns:
            if feature_col not in df.columns:
                continue
            
            # Get all non-null values for this model-feature combination
            feature_series = group_data[feature_col].dropna()
            
            if len(feature_series) == 0:
                continue
            
            # Detect outliers using IQR
            mask = remove_outliers_iqr(feature_series, iqr_factor=iqr_factor)
            outliers_mask = ~mask  # True for outliers
            
            # Count outliers
            n_outliers = outliers_mask.sum()
            n_total = len(feature_series)
            
            if n_outliers > 0:
                # Set outliers to NaN in the processed DataFrame
                model_indices = group_data.index
                feature_indices = feature_series.index
                outlier_indices = feature_indices[outliers_mask]
                
                df_processed.loc[outlier_indices, feature_col] = np.nan
                
                stats['by_model_feature'][f"{group_key}_{feature_col}"] = {
                    'total': n_total,
                    'outliers': n_outliers,
                    'percentage': (n_outliers / n_total * 100) if n_total > 0 else 0
                }
                
                stats['total_values'] += n_total
                stats['total_outliers'] += n_outliers
            
            # Always count total values processed
            stats['total_values'] += n_total if 'total_values' in stats else n_total
    
    # Save processed DataFrame
    print(f"\nSaving processed data to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if '_outlier_group' in df_processed.columns:
        df_processed = df_processed.drop(columns=['_outlier_group'])
    df_processed.to_csv(output_file, index=False)
    print(f"  Saved {len(df_processed)} rows to {output_file}")
    
    return df_processed, stats


def main():
    parser = argparse.ArgumentParser(
        description="Remove outliers from combined_merged.csv files using IQR method per model-feature combination."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input combined_merged.csv file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output file (default: input file with '_outliers_removed' suffix)"
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR factor for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="List of feature columns to process (default: auto-detect all numeric columns)"
    )
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if args.output is None:
        input_stem = args.input.stem
        input_suffix = args.input.suffix
        output_dir = args.input.parent
        args.output = output_dir / f"{input_stem}_outliers_removed{input_suffix}"
    
    print("="*80)
    print("Remove Outliers from Combined Merged CSV")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"IQR Factor: {args.iqr_factor}")
    print("="*80)
    
    # Process file
    df_processed, stats = remove_outliers_from_combined_merged(
        input_file=args.input,
        output_file=args.output,
        iqr_factor=args.iqr_factor,
        feature_columns=args.feature_columns
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Total values processed: {stats['total_values']}")
    print(f"Total outliers removed: {stats['total_outliers']}")
    if stats['total_values'] > 0:
        print(f"Outlier percentage: {stats['total_outliers'] / stats['total_values'] * 100:.2f}%")
    
    print(f"\nOutliers by model-feature combination:")
    print("-" * 80)
    for key, value in sorted(stats['by_model_feature'].items()):
        if value['outliers'] > 0:
            print(f"  {key}: {value['outliers']}/{value['total']} ({value['percentage']:.2f}%)")
    
    print("\n" + "="*80)
    print("✅ Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()


