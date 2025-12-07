#!/usr/bin/env python3
"""
Verify trajectory hypotheses by comparing Human vs LLM statistics.

Checks:
1. CV/RMSSD/MASD: High in Humans, Very Low in LLM
2. Path length: Long in Humans, Short/Zero in LLM
3. Net displacement: Large in Humans, Small in LLM
4. Tortuosity: Large in Humans, ~1 in LLM
5. Direction consistency: Low in Humans, High in LLM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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


def load_samples(domains: List[str], models: List[str], level: str) -> pd.DataFrame:
    """Load trajectory features for all domains and models."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            frames.append(df_h)
        for provider in models:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["domain"] = domain
                df["label"] = "llm"
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def calculate_variability_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate mean variability statistics for Human vs LLM."""
    variability_cols = [
        col for col in df.columns
        if any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]
    
    stats = {}
    for col in variability_cols:
        human_vals = df.loc[df["label"] == "human", col].dropna()
        llm_vals = df.loc[df["label"] == "llm", col].dropna()
        
        if not human_vals.empty and not llm_vals.empty:
            stats[col] = {
                "human_mean": float(human_vals.mean()),
                "human_std": float(human_vals.std()),
                "llm_mean": float(llm_vals.mean()),
                "llm_std": float(llm_vals.std()),
                "ratio": float(human_vals.mean() / llm_vals.mean()) if llm_vals.mean() > 0 else np.inf,
            }
    
    return stats


def calculate_geometry_stats(df: pd.DataFrame, space: str = "all") -> Dict[str, Dict[str, float]]:
    """Calculate mean geometry statistics for Human vs LLM."""
    if space == "all":
        prefixes = ["ce_", "tfidf_", "sbert_"]
    else:
        prefixes = [f"{space}_"]
    
    stats = {}
    for prefix in prefixes:
        for metric in GEOMETRY_METRICS:
            col = f"{prefix}{metric}"
            if col not in df.columns:
                continue
            
            human_vals = df.loc[df["label"] == "human", col].dropna()
            llm_vals = df.loc[df["label"] == "llm", col].dropna()
            
            if not human_vals.empty and not llm_vals.empty:
                stats[col] = {
                    "human_mean": float(human_vals.mean()),
                    "human_std": float(human_vals.std()),
                    "llm_mean": float(llm_vals.mean()),
                    "llm_std": float(llm_vals.std()),
                    "ratio": float(human_vals.mean() / llm_vals.mean()) if llm_vals.mean() > 0 else np.inf,
                }
    
    return stats


def print_summary_table(variability_stats: Dict, geometry_stats: Dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("TRAJECTORY HYPOTHESIS VERIFICATION")
    print("="*80)
    
    print("\n📊 VARIABILITY METRICS (CV / RMSSD / MASD)")
    print("-" * 80)
    print(f"{'Metric':<40} {'Human Mean':<15} {'LLM Mean':<15} {'Ratio':<10}")
    print("-" * 80)
    
    # Group by feature type
    feature_types = {}
    for col, stats in variability_stats.items():
        for suffix in VAR_SUFFIXES:
            if col.endswith(suffix):
                feature_name = col.replace(suffix, "")
                if feature_name not in feature_types:
                    feature_types[feature_name] = {}
                feature_types[feature_name][suffix] = stats
                break
    
    for feature_name in sorted(feature_types.keys())[:5]:  # Show first 5 features
        for suffix in VAR_SUFFIXES:
            if suffix in feature_types[feature_name]:
                stats = feature_types[feature_name][suffix]
                metric_name = f"{feature_name}{suffix}"
                print(
                    f"{metric_name:<40} "
                    f"{stats['human_mean']:>12.4f}   "
                    f"{stats['llm_mean']:>12.4f}   "
                    f"{stats['ratio']:>8.2f}x"
                )
    
    print("\n📐 GEOMETRY METRICS")
    print("-" * 80)
    print(f"{'Metric':<40} {'Human Mean':<15} {'LLM Mean':<15} {'Ratio':<10}")
    print("-" * 80)
    
    # Key metrics to highlight
    key_metrics = {
        "path_length": "Path Length",
        "net_displacement": "Net Displacement",
        "tortuosity": "Tortuosity",
        "direction_consistency": "Direction Consistency",
    }
    
    for metric_key, metric_label in key_metrics.items():
        for prefix in ["ce_", "tfidf_", "sbert_"]:
            col = f"{prefix}{metric_key}"
            if col in geometry_stats:
                stats = geometry_stats[col]
                print(
                    f"{col:<40} "
                    f"{stats['human_mean']:>12.4f}   "
                    f"{stats['llm_mean']:>12.4f}   "
                    f"{stats['ratio']:>8.2f}x"
                )
    
    print("\n" + "="*80)
    print("HYPOTHESIS VERIFICATION")
    print("="*80)
    
    # Verify each hypothesis
    print("\n1. CV/RMSSD/MASD: High in Humans, Very Low in LLM")
    avg_variability_ratio = np.mean([s["ratio"] for s in variability_stats.values() if s["ratio"] < 100])
    if avg_variability_ratio > 2.0:
        print(f"   ✅ CONFIRMED: Human variability is {avg_variability_ratio:.2f}x higher than LLM")
    else:
        print(f"   ⚠️  PARTIAL: Human variability is {avg_variability_ratio:.2f}x higher (expected >2x)")
    
    print("\n2. Path Length: Long in Humans, Short/Zero in LLM")
    path_length_cols = [col for col in geometry_stats.keys() if "path_length" in col]
    if path_length_cols:
        avg_path_ratio = np.mean([geometry_stats[col]["ratio"] for col in path_length_cols])
        if avg_path_ratio > 1.5:
            print(f"   ✅ CONFIRMED: Human path length is {avg_path_ratio:.2f}x longer than LLM")
        else:
            print(f"   ⚠️  PARTIAL: Human path length is {avg_path_ratio:.2f}x longer (expected >1.5x)")
    
    print("\n3. Net Displacement: Large in Humans, Small in LLM")
    net_disp_cols = [col for col in geometry_stats.keys() if "net_displacement" in col]
    if net_disp_cols:
        avg_disp_ratio = np.mean([geometry_stats[col]["ratio"] for col in net_disp_cols])
        if avg_disp_ratio > 1.5:
            print(f"   ✅ CONFIRMED: Human net displacement is {avg_disp_ratio:.2f}x larger than LLM")
        else:
            print(f"   ⚠️  PARTIAL: Human net displacement is {avg_disp_ratio:.2f}x larger (expected >1.5x)")
    
    print("\n4. Tortuosity: Large in Humans, ~1 in LLM")
    tortuosity_cols = [col for col in geometry_stats.keys() if "tortuosity" in col]
    if tortuosity_cols:
        human_tort = np.mean([geometry_stats[col]["human_mean"] for col in tortuosity_cols])
        llm_tort = np.mean([geometry_stats[col]["llm_mean"] for col in tortuosity_cols])
        if human_tort > 1.5 and abs(llm_tort - 1.0) < 0.3:
            print(f"   ✅ CONFIRMED: Human tortuosity={human_tort:.2f}, LLM tortuosity={llm_tort:.2f} (~1)")
        else:
            print(f"   ⚠️  PARTIAL: Human tortuosity={human_tort:.2f}, LLM tortuosity={llm_tort:.2f}")
    
    print("\n5. Direction Consistency: Low in Humans, High in LLM")
    dir_cons_cols = [col for col in geometry_stats.keys() if "direction_consistency" in col]
    if dir_cons_cols:
        human_dir = np.mean([geometry_stats[col]["human_mean"] for col in dir_cons_cols])
        llm_dir = np.mean([geometry_stats[col]["llm_mean"] for col in dir_cons_cols])
        if human_dir < llm_dir:
            print(f"   ✅ CONFIRMED: Human direction consistency={human_dir:.3f} < LLM={llm_dir:.3f}")
        else:
            print(f"   ⚠️  PARTIAL: Human direction consistency={human_dir:.3f}, LLM={llm_dir:.3f}")
    
    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify trajectory hypotheses.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading data for domains: {', '.join(args.domains)}")
    print(f"LLM level: {args.level}")
    
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available.")
        return
    
    print(f"Loaded {len(df)} samples ({len(df[df['label']=='human'])} human, {len(df[df['label']=='llm'])} LLM)")
    
    # Calculate statistics
    variability_stats = calculate_variability_stats(df)
    geometry_stats = calculate_geometry_stats(df, space="all")
    
    # Print summary
    print_summary_table(variability_stats, geometry_stats)


if __name__ == "__main__":
    main()


