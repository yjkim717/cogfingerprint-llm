#!/usr/bin/env python3
"""
Detailed verification of trajectory hypotheses by comparing Human vs each LLM model separately.

Focuses on:
- CE Geometry features
- TF-IDF Geometry features  
- SBERT Geometry features
- Variability features (CV/RMSSD/MASD)

For each LLM model (DS, G4B, G12B, LMK) at LV3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
    "direction_consistency",
)


def load_human_data(domains: List[str]) -> pd.DataFrame:
    """Load human trajectory features."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            frames.append(df_h)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_llm_data(domains: List[str], provider: str, level: str) -> pd.DataFrame:
    """Load LLM trajectory features for a specific provider and level."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["domain"] = domain
            df["label"] = "llm"
            df["provider"] = provider
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def calculate_geometry_stats(human_df: pd.DataFrame, llm_df: pd.DataFrame, space: str) -> Dict[str, Dict]:
    """Calculate geometry statistics for a specific space (ce, tfidf, sbert)."""
    prefix = f"{space}_"
    stats = {}
    
    for metric in GEOMETRY_METRICS:
        col = f"{prefix}{metric}"
        if col not in human_df.columns or col not in llm_df.columns:
            continue
        
        human_vals = human_df[col].dropna()
        llm_vals = llm_df[col].dropna()
        
        if human_vals.empty or llm_vals.empty:
            continue
        
        # Statistical test
        try:
            u_stat, p_value = mannwhitneyu(human_vals, llm_vals, alternative="two-sided")
        except:
            p_value = np.nan
        
        stats[metric] = {
            "human_mean": float(human_vals.mean()),
            "human_std": float(human_vals.std()),
            "human_median": float(human_vals.median()),
            "llm_mean": float(llm_vals.mean()),
            "llm_std": float(llm_vals.std()),
            "llm_median": float(llm_vals.median()),
            "ratio": float(human_vals.mean() / llm_vals.mean()) if llm_vals.mean() > 0 else np.inf,
            "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
        }
    
    return stats


def calculate_variability_stats(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate variability statistics (CV/RMSSD/MASD)."""
    variability_cols = [
        col for col in human_df.columns
        if any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]
    
    stats = {}
    for col in variability_cols:
        if col not in llm_df.columns:
            continue
        
        human_vals = human_df[col].dropna()
        llm_vals = llm_df[col].dropna()
        
        if human_vals.empty or llm_vals.empty:
            continue
        
        # Statistical test
        try:
            u_stat, p_value = mannwhitneyu(human_vals, llm_vals, alternative="two-sided")
        except:
            p_value = np.nan
        
        stats[col] = {
            "human_mean": float(human_vals.mean()),
            "llm_mean": float(llm_vals.mean()),
            "ratio": float(human_vals.mean() / llm_vals.mean()) if llm_vals.mean() > 0 else np.inf,
            "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            "significant": p_value < 0.05 if not np.isnan(p_value) else False,
        }
    
    return stats


def print_hypothesis_table(human_df: pd.DataFrame, llm_dfs: Dict[str, pd.DataFrame], spaces: List[str]):
    """Print detailed hypothesis verification table."""
    print("\n" + "="*100)
    print("TRAJECTORY HYPOTHESIS VERIFICATION - DETAILED ANALYSIS")
    print("="*100)
    print(f"\nHuman authors: {len(human_df)}")
    for provider, df in llm_dfs.items():
        print(f"LLM {provider} LV3 authors: {len(df)}")
    
    # 1. Variability (CV/RMSSD/MASD)
    print("\n" + "="*100)
    print("1. VARIABILITY METRICS (CV / RMSSD / MASD)")
    print("="*100)
    print("Hypothesis: High in Humans, Very Low in LLM → evolution vs stasis")
    print("-"*100)
    print(f"{'Provider':<10} {'Metric':<35} {'Human Mean':<12} {'LLM Mean':<12} {'Ratio':<10} {'p-value':<10} {'Sig':<5}")
    print("-"*100)
    
    # Show average variability across all features
    for provider, llm_df in llm_dfs.items():
        var_stats = calculate_variability_stats(human_df, llm_df)
        if var_stats:
            avg_ratio = np.mean([s["ratio"] for s in var_stats.values() if s["ratio"] < 100])
            sig_count = sum(1 for s in var_stats.values() if s["significant"])
            print(f"{provider:<10} {'Average (all features)':<35} {'-':<12} {'-':<12} {avg_ratio:>8.2f}x {'-':<10} {sig_count}/{len(var_stats)}")
    
    # 2. Geometry features for each space
    for space in spaces:
        print("\n" + "="*100)
        print(f"2. {space.upper()} GEOMETRY FEATURES")
        print("="*100)
        print("-"*100)
        print(f"{'Provider':<10} {'Metric':<25} {'Human Mean':<12} {'LLM Mean':<12} {'Ratio':<10} {'p-value':<10} {'Sig':<5}")
        print("-"*100)
        
        for provider, llm_df in llm_dfs.items():
            geom_stats = calculate_geometry_stats(human_df, llm_df, space)
            for metric, stats in geom_stats.items():
                sig_mark = "***" if stats["significant"] else ""
                print(
                    f"{provider:<10} "
                    f"{metric:<25} "
                    f"{stats['human_mean']:>10.4f}   "
                    f"{stats['llm_mean']:>10.4f}   "
                    f"{stats['ratio']:>8.2f}x   "
                    f"{stats['p_value']:>8.4f}   "
                    f"{sig_mark:<5}"
                )
    
    # Summary by hypothesis
    print("\n" + "="*100)
    print("HYPOTHESIS SUMMARY")
    print("="*100)
    
    # Hypothesis 1: Variability
    print("\n1. CV/RMSSD/MASD: High in Humans, Very Low in LLM")
    for provider, llm_df in llm_dfs.items():
        var_stats = calculate_variability_stats(human_df, llm_df)
        if var_stats:
            avg_ratio = np.mean([s["ratio"] for s in var_stats.values() if s["ratio"] < 100])
            sig_count = sum(1 for s in var_stats.values() if s["significant"])
            status = "✅" if avg_ratio > 2.0 else "⚠️" if avg_ratio > 1.2 else "❌"
            print(f"   {status} {provider}: Human is {avg_ratio:.2f}x higher ({sig_count}/{len(var_stats)} features significant)")
    
    # Hypothesis 2-5: Geometry
    hypotheses = [
        ("Path Length", "path_length", "Long in Humans, Short/Zero in LLM"),
        ("Net Displacement", "net_displacement", "Large in Humans, Small in LLM"),
        ("Tortuosity", "tortuosity", "Large in Humans, ~1 in LLM"),
        ("Direction Consistency", "direction_consistency", "Low in Humans, High in LLM"),
    ]
    
    for hyp_name, metric_key, hyp_desc in hypotheses:
        print(f"\n{hyp_name}: {hyp_desc}")
        for space in spaces:
            print(f"   {space.upper()} space:")
            for provider, llm_df in llm_dfs.items():
                geom_stats = calculate_geometry_stats(human_df, llm_df, space)
                if metric_key in geom_stats:
                    stats = geom_stats[metric_key]
                    ratio = stats["ratio"]
                    sig = "***" if stats["significant"] else ""
                    
                    if metric_key == "tortuosity":
                        status = "✅" if stats["human_mean"] > 1.5 and abs(stats["llm_mean"] - 1.0) < 0.3 else "⚠️"
                        print(f"      {status} {provider}: Human={stats['human_mean']:.2f}, LLM={stats['llm_mean']:.2f} {sig}")
                    elif metric_key == "direction_consistency":
                        status = "✅" if stats["human_mean"] < stats["llm_mean"] else "⚠️"
                        print(f"      {status} {provider}: Human={stats['human_mean']:.3f}, LLM={stats['llm_mean']:.3f} {sig}")
                    else:
                        status = "✅" if ratio > 1.5 and stats["significant"] else "⚠️" if ratio > 1.2 else "❌"
                        print(f"      {status} {provider}: {ratio:.2f}x {sig}")
    
    print("\n" + "="*100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed trajectory hypothesis verification.")
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
        help="LLM models to include (default: all providers).",
    )
    parser.add_argument(
        "--level",
        default="LV3",
        choices=LEVELS,
        help="LLM level to include (default: LV3).",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=["ce", "tfidf", "sbert"],
        default=["ce", "tfidf", "sbert"],
        help="Geometry spaces to analyze (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading data for domains: {', '.join(args.domains)}")
    print(f"LLM level: {args.level}")
    print(f"Geometry spaces: {', '.join(args.spaces)}")
    
    # Load human data
    human_df = load_human_data(args.domains)
    if human_df.empty:
        print("⚠ No human data available.")
        return
    
    # Load LLM data for each provider
    llm_dfs = {}
    for provider in args.models:
        llm_df = load_llm_data(args.domains, provider, args.level)
        if not llm_df.empty:
            llm_dfs[provider] = llm_df
    
    if not llm_dfs:
        print("⚠ No LLM data available.")
        return
    
    # Print detailed analysis
    print_hypothesis_table(human_df, llm_dfs, args.spaces)


if __name__ == "__main__":
    main()

