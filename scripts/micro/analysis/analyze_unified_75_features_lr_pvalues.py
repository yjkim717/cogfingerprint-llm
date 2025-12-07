#!/usr/bin/env python3
"""
Statistical Significance Validation: Logistic Regression p-values for unified 75 features (RQ1 LV3).

**Purpose**: Verify that the 75 features are statistically significant in distinguishing
Human vs. LLM text. This is a core validation for RQ1.

Uses statsmodels to fit logistic regression and compute p-values for each feature,
providing statistical significance testing to prove that our features have genuine
discriminative power (not just by chance).

Usage:
    python scripts/micro/analysis/analyze_unified_75_features_lr_pvalues.py \
        --output plots/trajectory/unified_75_features_ml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels is not available. Install with: pip install statsmodels")

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


def compute_lr_pvalues(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Fit a statsmodels logistic regression (with intercept) and return coefficient stats.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not available. Install statsmodels to compute LR p-values.")
    
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_with_const)
    
    try:
        result = model.fit(disp=False, maxiter=1000, method='bfgs')
    except Exception as e:
        print(f"⚠️ Warning: Optimization failed, trying alternative method: {e}")
        try:
            result = model.fit(disp=False, maxiter=2000, method='lbfgs')
        except Exception as e2:
            print(f"⚠️ Warning: Second optimization attempt failed: {e2}")
            raise
    
    summary_df = pd.DataFrame({
        "feature": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values,
    })
    summary_df["odds_ratio"] = np.exp(summary_df["coef"])
    
    # Add significance markers
    summary_df["significant"] = summary_df["p_value"] < 0.05
    summary_df["significance"] = summary_df["p_value"].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Add feature type
    summary_df["feature_type"] = summary_df["feature"].apply(classify_feature_type)
    
    return summary_df


def classify_feature_type(feature: str) -> str:
    """Classify feature into category."""
    if feature == "const":
        return "intercept"
    if any(feature.endswith(suffix) for suffix in VAR_SUFFIXES):
        return "CE-VAR"
    if feature.startswith("ce_"):
        return "CE-Geometry"
    if feature.startswith("tfidf_"):
        return "TF-IDF-Geometry"
    if feature.startswith("sbert_"):
        return "SBERT-Geometry"
    return "other"


def compute_cv_lr_pvalues(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Compute LR p-values with cross-validation and aggregate results."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not available.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    all_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else pd.Series(y[train_idx])
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        try:
            result_df = compute_lr_pvalues(X_train_scaled, y_train)
            result_df["fold"] = fold
            all_results.append(result_df)
            print(f"Fold {fold + 1}/{n_splits} completed")
        except Exception as e:
            print(f"⚠️ Warning: Fold {fold + 1} failed: {e}")
            continue
    
    if not all_results:
        raise ValueError("All CV folds failed")
    
    # Aggregate results across folds
    combined = pd.concat(all_results, ignore_index=True)
    
    # Group by feature and compute statistics
    agg_results = combined.groupby('feature').agg({
        'coef': ['mean', 'std'],
        'std_err': 'mean',
        'z_value': ['mean', 'std'],
        'p_value': ['mean', 'min', 'max'],
        'odds_ratio': ['mean', 'std'],
        'feature_type': 'first',
    }).reset_index()
    
    # Flatten column names
    agg_results.columns = [
        'feature',
        'coef_mean', 'coef_std',
        'std_err_mean',
        'z_value_mean', 'z_value_std',
        'p_value_mean', 'p_value_min', 'p_value_max',
        'odds_ratio_mean', 'odds_ratio_std',
        'feature_type',
    ]
    
    # Add significance based on mean p-value
    agg_results["significant"] = agg_results["p_value_mean"] < 0.05
    agg_results["significance"] = agg_results["p_value_mean"].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Sort by p-value
    agg_results = agg_results.sort_values('p_value_mean')
    
    return agg_results, combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic Regression p-values for unified 75 features.")
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
        help="Output directory for results.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation (aggregate p-values across folds).",
    )
    args = parser.parse_args()
    
    if not STATSMODELS_AVAILABLE:
        print("❌ statsmodels is required for this script.")
        print("Install with: pip install statsmodels")
        return
    
    # Load data
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available for the requested configuration.")
        return
    
    # Select unified features
    X = select_unified_features(df)
    y = pd.Series((df["label"] == "human").astype(int).values)
    
    print(f"\n=== Logistic Regression p-value Analysis for Unified 75 Features ===")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Level: {args.level}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(X.columns)}")
    print(f"Use CV: {args.cv}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    if args.cv:
        # Cross-validated p-values
        print("\nComputing p-values with 5-fold cross-validation...")
        agg_results, fold_results = compute_cv_lr_pvalues(X, y, n_splits=5)
        
        # Save aggregated results
        args.output.mkdir(parents=True, exist_ok=True)
        agg_path = args.output / "lr_pvalues_cv_aggregated.csv"
        agg_results.to_csv(agg_path, index=False)
        print(f"\nSaved aggregated CV results: {agg_path}")
        
        # Save fold-level results
        fold_path = args.output / "lr_pvalues_cv_folds.csv"
        fold_results.to_csv(fold_path, index=False)
        print(f"Saved fold-level results: {fold_path}")
        
        # Summary
        n_sig = agg_results["significant"].sum()
        print(f"\n=== Summary ===")
        print(f"Significant features (p < 0.05): {n_sig}/{len(agg_results)} ({100*n_sig/len(agg_results):.1f}%)")
        print(f"\nTop 10 most significant features:")
        print(agg_results.head(10)[['feature', 'coef_mean', 'p_value_mean', 'significance', 'feature_type']].to_string(index=False))
        
    else:
        # Single model on full data
        print("\nComputing p-values on full dataset...")
        result_df = compute_lr_pvalues(X_scaled, y)
        
        # Save results
        args.output.mkdir(parents=True, exist_ok=True)
        result_path = args.output / "lr_pvalues_full.csv"
        result_df.to_csv(result_path, index=False)
        print(f"\nSaved results: {result_path}")
        
        # Summary
        n_sig = result_df[result_df["feature"] != "const"]["significant"].sum()
        print(f"\n=== Summary ===")
        print(f"Significant features (p < 0.05): {n_sig}/{len(result_df)-1} ({100*n_sig/(len(result_df)-1):.1f}%)")
        print(f"\nTop 10 most significant features:")
        top10 = result_df[result_df["feature"] != "const"].head(10)
        print(top10[['feature', 'coef', 'p_value', 'significance', 'feature_type']].to_string(index=False))
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()

