#!/usr/bin/env python3
"""
Analyze feature importance from ML classification models.

Extracts feature importance from the best performing models (Random Forest, 
Gradient Boosting, Logistic Regression) for each time series statistic.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Import functions from ml_classify_author_by_timeseries.py
from ml_classify_author_by_timeseries import (
    prepare_classification_data,
    train_and_evaluate_models,
    DEFAULT_MODELS,
    DEFAULT_LEVEL,
    STAT_NAMES,
)

OUTPUT_DIR = Path("results")


def extract_and_save_feature_importance(
    domains: List[str],
    models: List[str],
    level: str,
    stats_to_analyze: List[str],
    output_dir: Path = OUTPUT_DIR,
    random_state: int = 42,
):
    """
    Extract feature importance from best models and save to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    print(f"Domains: {', '.join([d.upper() for d in domains])}")
    print(f"LLM Models: {', '.join(models)}")
    print(f"LLM Level: {level}")
    print(f"Statistics: {', '.join([s.upper() for s in stats_to_analyze])}")
    print("="*70)
    
    all_feature_importance = {}
    
    for stat_name in stats_to_analyze:
        print(f"\n{'#'*70}")
        print(f"ANALYZING FEATURE IMPORTANCE: {stat_name.upper()}")
        print(f"{'#'*70}")
        
        # Prepare data
        data, features = prepare_classification_data(domains, models, level, stat_name)
        X = data[features]
        y = data["label"]
        
        # Remove rows with missing values
        missing_mask = X.isna().any(axis=1)
        if missing_mask.sum() > 0:
            X = X[~missing_mask]
            y = y[~missing_mask]
            data = data[~missing_mask]
        
        # Train and evaluate models
        train_result = train_and_evaluate_models(
            X, y, features, data, stat_name, random_state=random_state
        )
        results = train_result[0]  # First element is results dict
        
        # Extract feature importance from best models
        # Use Random Forest (usually best) and Logistic Regression (for interpretability)
        feature_importance_data = []
        
        # Random Forest (has feature_importances_)
        if "Random Forest" in results:
            rf_result = results["Random Forest"]
            if rf_result.get("feature_importance"):
                for feat, imp in rf_result["feature_importance"].items():
                    # Remove stat suffix for display
                    feat_base = feat.replace(f"_{stat_name}", "")
                    feature_importance_data.append({
                        "statistic": stat_name.upper(),
                        "model": "Random Forest",
                        "feature": feat_base,
                        "importance": imp,
                        "rank": None,  # Will fill later
                    })
        
        # Gradient Boosting (has feature_importances_)
        if "Gradient Boosting" in results:
            gb_result = results["Gradient Boosting"]
            if gb_result.get("feature_importance"):
                for feat, imp in gb_result["feature_importance"].items():
                    feat_base = feat.replace(f"_{stat_name}", "")
                    feature_importance_data.append({
                        "statistic": stat_name.upper(),
                        "model": "Gradient Boosting",
                        "feature": feat_base,
                        "importance": imp,
                        "rank": None,
                    })
        
        # Logistic Regression (has coef_)
        if "Logistic Regression" in results:
            lr_result = results["Logistic Regression"]
            if lr_result.get("feature_importance"):
                for feat, imp in lr_result["feature_importance"].items():
                    feat_base = feat.replace(f"_{stat_name}", "")
                    feature_importance_data.append({
                        "statistic": stat_name.upper(),
                        "model": "Logistic Regression",
                        "feature": feat_base,
                        "importance": imp,
                        "rank": None,
                    })
        
        if feature_importance_data:
            df_importance = pd.DataFrame(feature_importance_data)
            
            # Add rank within each statistic-model combination
            for stat in df_importance["statistic"].unique():
                for model in df_importance["model"].unique():
                    mask = (df_importance["statistic"] == stat) & (df_importance["model"] == model)
                    df_importance.loc[mask, "rank"] = (
                        df_importance.loc[mask, "importance"]
                        .rank(ascending=False, method="min")
                        .astype(int)
                    )
            
            # Save per-statistic feature importance
            output_path = output_dir / f"feature_importance_{stat_name}.csv"
            df_importance.to_csv(output_path, index=False)
            print(f"\nSaved: {output_path}")
            
            # Print top features for best model (Random Forest)
            if "Random Forest" in df_importance["model"].values:
                rf_features = df_importance[
                    (df_importance["statistic"] == stat_name.upper()) &
                    (df_importance["model"] == "Random Forest")
                ].sort_values("importance", ascending=False)
                
                print(f"\nTop 10 Features for {stat_name.upper()} (Random Forest):")
                for idx, row in rf_features.head(10).iterrows():
                    print(f"  {row['rank']:2d}. {row['feature']:25s}: {row['importance']:.6f}")
            
            all_feature_importance[stat_name] = df_importance
    
    # Create combined summary
    if all_feature_importance:
        all_combined = pd.concat(all_feature_importance.values(), ignore_index=True)
        
        # Create summary: average importance across models for each statistic
        summary_data = []
        for stat_name in stats_to_analyze:
            stat_data = all_combined[all_combined["statistic"] == stat_name.upper()]
            
            for feature in stat_data["feature"].unique():
                feature_data = stat_data[stat_data["feature"] == feature]
                avg_importance = feature_data["importance"].mean()
                max_importance = feature_data["importance"].max()
                min_rank = feature_data["rank"].min()
                avg_rank = feature_data["rank"].mean()
                
                summary_data.append({
                    "statistic": stat_name.upper(),
                    "feature": feature,
                    "avg_importance": avg_importance,
                    "max_importance": max_importance,
                    "min_rank": min_rank,
                    "avg_rank": avg_rank,
                    "appears_in_models": len(feature_data),
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by avg_importance within each statistic
        summary_df = summary_df.sort_values(["statistic", "avg_importance"], ascending=[True, False])
        
        # Save summary
        summary_path = output_dir / "feature_importance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")
        
        # Print cross-statistic summary
        print(f"\n{'='*70}")
        print("CROSS-STATISTIC FEATURE IMPORTANCE SUMMARY")
        print("="*70)
        
        for stat_name in stats_to_analyze:
            stat_summary = summary_df[summary_df["statistic"] == stat_name.upper()].head(10)
            print(f"\nTop 10 Features for {stat_name.upper()}:")
            for idx, row in stat_summary.iterrows():
                print(f"  {row['avg_rank']:4.1f}. {row['feature']:25s}: "
                      f"avg={row['avg_importance']:.6f}, max={row['max_importance']:.6f}")
        
        # Find most consistently important features across all statistics
        print(f"\n{'='*70}")
        print("MOST CONSISTENTLY IMPORTANT FEATURES (across all statistics)")
        print("="*70)
        
        # Calculate average rank across statistics for each feature
        feature_consistency = []
        for feature in summary_df["feature"].unique():
            feature_data = summary_df[summary_df["feature"] == feature]
            avg_rank_across_stats = feature_data["avg_rank"].mean()
            avg_importance_across_stats = feature_data["avg_importance"].mean()
            num_stats = len(feature_data)
            
            feature_consistency.append({
                "feature": feature,
                "avg_rank": avg_rank_across_stats,
                "avg_importance": avg_importance_across_stats,
                "num_statistics": num_stats,
            })
        
        consistency_df = pd.DataFrame(feature_consistency).sort_values("avg_rank")
        
        print("\nTop 15 Most Important Features (averaged across all statistics):")
        for idx, row in consistency_df.head(15).iterrows():
            print(f"  {row['avg_rank']:5.1f}. {row['feature']:25s}: "
                  f"avg_importance={row['avg_importance']:.6f}")
        
        # Save consistency summary
        consistency_path = output_dir / "feature_importance_consistency.csv"
        consistency_df.to_csv(consistency_path, index=False)
        print(f"\nSaved consistency summary: {consistency_path}")
        
        return all_combined, summary_df, consistency_df
    
    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance from ML classification models."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["academic", "news", "blogs"],
        help="Domains to analyze (default: academic news blogs)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"LLM models to include (default: {' '.join(DEFAULT_MODELS)})."
    )
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        help=f"LLM level (default: {DEFAULT_LEVEL})."
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        choices=STAT_NAMES,
        default=STAT_NAMES,
        help=f"Statistics to analyze (default: {' '.join(STAT_NAMES)})."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results (default: results)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)."
    )
    
    args = parser.parse_args()
    
    extract_and_save_feature_importance(
        domains=args.domains,
        models=args.models,
        level=args.level,
        stats_to_analyze=args.stats,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

