#!/usr/bin/env python3
"""
Statistical and visual analysis for unified trajectory features.

Generates:
1. Mann-Whitney U tests (Human vs LLM) for selected metrics.
2. Boxplots per metric.
3. PCA scatter plots over the full ~72-dim trajectory vector space.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

DEFAULT_METRICS = [
    "ce_net_displacement",
    "ce_path_length",
    "ce_tortuosity",
    "tfidf_net_displacement",
    "tfidf_path_length",
    "tfidf_tortuosity",
    "sbert_net_displacement",
    "sbert_path_length",
    "sbert_tortuosity",
]


def load_split(domain: str, provider: str, level: str, label: str) -> pd.DataFrame:
    path = (
        DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if label == "human"
        else DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["label"] = label
    df["domain"] = domain
    df["provider"] = provider
    df["level"] = level
    df["model_tag"] = provider if label == "llm" else "human"
    return df


def collect_domain(domain: str, llm_level: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = [load_split(domain, "human", "LV0", "human")]
    for provider in PROVIDERS:
        frames.append(load_split(domain, provider, llm_level, "llm"))
    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def ensure_output(domain: str) -> Path:
    out_dir = PLOTS_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_stats(df: pd.DataFrame, metrics: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        human_vals = df.loc[df["label"] == "human", metric].dropna()
        llm_vals = df.loc[df["label"] == "llm", metric].dropna()
        if human_vals.empty or llm_vals.empty:
            continue
        stat, p_value = mannwhitneyu(human_vals, llm_vals, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "human_mean": human_vals.mean(),
                "llm_mean": llm_vals.mean(),
                "u_statistic": stat,
                "p_value": p_value,
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "metric_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Stats → {stats_path}")
    return stats_df


def plot_metrics(df: pd.DataFrame, metrics: List[str], out_dir: Path, domain: str) -> None:
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="label", y=metric, hue="label", palette="Set2")
        sns.stripplot(
            data=df,
            x="label",
            y=metric,
            color="black",
            alpha=0.2,
            jitter=True,
            dodge=True,
        )
        plt.title(f"{domain.upper()} – {metric}")
        plt.tight_layout()
        out_path = out_dir / f"{metric}_boxplot.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Plot → {out_path}")


def plot_pca(df: pd.DataFrame, out_dir: Path, domain: str) -> None:
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in {"author_id", "sample_count"}]
    X = df[feature_cols].fillna(0.0).to_numpy()
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=y,
        style=df.get("provider", y),
        palette="Set2",
        alpha=0.7,
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title(f"{domain.upper()} – Trajectory PCA")
    plt.tight_layout()
    out_path = out_dir / "trajectory_pca.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  PCA → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize / analyze trajectory feature tables.")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to process (default: all).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to visualize / test.",
    )
    parser.add_argument(
        "--llm-level",
        default="LV3",
        choices=LEVELS,
        help="LLM level to include (default: LV3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for domain in args.domains:
        print(f"\n=== Analyzing trajectory features for {domain} ===")
        df = collect_domain(domain, args.llm_level)
        if df.empty:
            print("  ⚠ No data.")
            continue
        out_dir = ensure_output(domain)
        stats_df = run_stats(df, args.metrics, out_dir)
        plot_metrics(df, args.metrics, out_dir, domain)
        plot_pca(df, out_dir, domain)


if __name__ == "__main__":
    main()

