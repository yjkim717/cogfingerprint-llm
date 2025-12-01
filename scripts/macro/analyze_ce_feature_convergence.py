#!/usr/bin/env python3
"""
Analyze which CE features are converging (human-LLM difference decreasing over years)
and which remain stable.

For each CE feature, calculate:
1. Mean difference between human and LLM each year
2. Effect size (Cohen's d) each year
3. Trend analysis (is the difference decreasing?)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Constants
DOMAINS = ["academic", "blogs", "news"]
LLM_MODELS = ["DS", "G4B", "G12B", "LMK"]
LEVEL = "LV3"

CE_FEATURES = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neu",
    "vader_neg",
    "word_diversity",
    "flesch_reading_ease",
    "gunning_fog",
    "average_word_length",
    "num_words",
    "avg_sentence_length",
    "verb_ratio",
    "function_word_ratio",
    "content_word_ratio",
]

HUMAN_BASE = Path("macro_dataset/process/human")
LLM_BASE = Path("macro_dataset/process/LLM")
RESULTS_DIR = Path("macro_results/rq2_ce_feature_convergence")


def parse_year_from_filename(filename: str) -> Optional[int]:
    """Extract 4-digit year from filename."""
    tokens = filename.replace(".txt", "").split("_")
    for token in tokens:
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def load_data(domain: str) -> pd.DataFrame:
    """Load all data for a domain."""
    frames = []
    
    # Load human
    human_path = HUMAN_BASE / domain / "combined_with_embeddings.csv"
    if human_path.exists():
        df_human = pd.read_csv(human_path)
        df_human["year"] = df_human["filename"].apply(parse_year_from_filename)
        df_human["label"] = "human"
        frames.append(df_human)
    
    # Load LLM
    for model in LLM_MODELS:
        llm_path = LLM_BASE / model / LEVEL / domain / "combined_with_embeddings.csv"
        if llm_path.exists():
            df_llm = pd.read_csv(llm_path)
            df_llm["year"] = df_llm["filename"].apply(parse_year_from_filename)
            df_llm["label"] = "llm"
            frames.append(df_llm)
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["year"])
    return df


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def analyze_feature_convergence(df: pd.DataFrame, feature: str, domain: str) -> dict:
    """Analyze convergence trend for a single feature."""
    if feature not in df.columns:
        return None
    
    years = sorted(df["year"].unique())
    yearly_stats = []
    
    for year in years:
        year_df = df[df["year"] == year].copy()
        
        human_vals = year_df[year_df["label"] == "human"][feature].dropna().values
        llm_vals = year_df[year_df["label"] == "llm"][feature].dropna().values
        
        if len(human_vals) == 0 or len(llm_vals) == 0:
            continue
        
        # Calculate statistics
        human_mean = np.mean(human_vals)
        llm_mean = np.mean(llm_vals)
        mean_diff = human_mean - llm_mean
        
        # Effect size
        d = cohens_d(human_vals, llm_vals)
        
        # Statistical test
        try:
            t_stat, p_value = stats.ttest_ind(human_vals, llm_vals)
        except:
            t_stat, p_value = np.nan, np.nan
        
        yearly_stats.append({
            "year": int(year),
            "human_mean": float(human_mean),
            "llm_mean": float(llm_mean),
            "mean_difference": float(mean_diff),
            "cohens_d": float(d),
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "human_n": int(len(human_vals)),
            "llm_n": int(len(llm_vals)),
        })
    
    if len(yearly_stats) < 2:
        return None
    
    # Analyze trend
    stats_df = pd.DataFrame(yearly_stats)
    years_arr = stats_df["year"].values
    diff_arr = np.abs(stats_df["mean_difference"].values)  # Use absolute difference
    cohens_d_arr = np.abs(stats_df["cohens_d"].values)  # Use absolute effect size
    
    # Linear regression on absolute difference (convergence = decreasing difference)
    if len(years_arr) >= 2 and np.std(diff_arr) > 0:
        slope_diff, intercept_diff, r_diff, p_diff, _ = stats.linregress(years_arr, diff_arr)
        trend_diff = "converging" if slope_diff < 0 else "diverging" if slope_diff > 0 else "stable"
    else:
        slope_diff, r_diff, p_diff = 0, 0, 1
        trend_diff = "stable"
    
    # Linear regression on effect size
    if len(years_arr) >= 2 and np.std(cohens_d_arr) > 0:
        slope_d, intercept_d, r_d, p_d, _ = stats.linregress(years_arr, cohens_d_arr)
        trend_d = "converging" if slope_d < 0 else "diverging" if slope_d > 0 else "stable"
    else:
        slope_d, r_d, p_d = 0, 0, 1
        trend_d = "stable"
    
    return {
        "feature": feature,
        "domain": domain,
        "yearly_stats": yearly_stats,
        "trend_analysis": {
            "mean_difference": {
                "trend": trend_diff,
                "slope": float(slope_diff),
                "correlation": float(r_diff),
                "p_value": float(p_diff),
                "significant": bool(p_diff < 0.05),
            },
            "cohens_d": {
                "trend": trend_d,
                "slope": float(slope_d),
                "correlation": float(r_d),
                "p_value": float(p_d),
                "significant": bool(p_d < 0.05),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze which CE features are converging over years."
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS + ["all"],
        default="all",
        help="Domain to analyze (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: auto-generated).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    domains_to_process = DOMAINS if args.domain == "all" else [args.domain]
    all_results = []

    for domain in domains_to_process:
        print(f"\n{'='*70}")
        print(f"Analyzing {domain.upper()}")
        print(f"{'='*70}")
        
        df = load_data(domain)
        if df.empty:
            print(f"⚠️  No data loaded for {domain}")
            continue
        
        print(f"Loaded {len(df)} samples")
        print(f"Years: {sorted(df['year'].unique())}")
        
        for feature in CE_FEATURES:
            result = analyze_feature_convergence(df, feature, domain)
            if result:
                all_results.append(result)
                
                trend_info = result["trend_analysis"]["mean_difference"]
                trend_d = result["trend_analysis"]["cohens_d"]
                
                sig_marker = "***" if trend_info["significant"] else "**" if trend_info["p_value"] < 0.1 else ""
                print(f"{feature:25s} | {trend_info['trend']:12s} | "
                      f"slope={trend_info['slope']:8.6f} | "
                      f"r={trend_info['correlation']:6.3f} | "
                      f"p={trend_info['p_value']:.4f} {sig_marker}")

    # Save results
    if all_results:
        output_path = args.output or (
            RESULTS_DIR / f"ce_feature_convergence_{args.domain}.json"
        )
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
        # Create summary table
        summary_data = []
        for result in all_results:
            trend_info = result["trend_analysis"]["mean_difference"]
            summary_data.append({
                "feature": result["feature"],
                "domain": result["domain"],
                "trend": trend_info["trend"],
                "slope": trend_info["slope"],
                "correlation": trend_info["correlation"],
                "p_value": trend_info["p_value"],
                "significant": trend_info["significant"],
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = output_path.with_suffix(".csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"✅ Summary CSV saved to {csv_path}")
        
        # Print convergence summary
        print(f"\n{'='*70}")
        print("CONVERGENCE SUMMARY")
        print(f"{'='*70}")
        
        for domain in domains_to_process:
            domain_df = summary_df[summary_df["domain"] == domain]
            converging = domain_df[domain_df["trend"] == "converging"]
            stable = domain_df[domain_df["trend"] == "stable"]
            diverging = domain_df[domain_df["trend"] == "diverging"]
            
            print(f"\n{domain.upper()}:")
            print(f"  Converging: {len(converging)} features")
            if len(converging) > 0:
                sig_converging = converging[converging["significant"]]
                print(f"    - Significant: {len(sig_converging)}")
                if len(sig_converging) > 0:
                    print(f"      {', '.join(sig_converging['feature'].tolist())}")
            
            print(f"  Stable: {len(stable)} features")
            print(f"  Diverging: {len(diverging)} features")
            if len(diverging) > 0:
                sig_diverging = diverging[diverging["significant"]]
                if len(sig_diverging) > 0:
                    print(f"    - Significant: {len(sig_diverging)}")
                    print(f"      {', '.join(sig_diverging['feature'].tolist())}")


if __name__ == "__main__":
    main()

