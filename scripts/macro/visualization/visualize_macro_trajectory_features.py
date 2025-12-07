#!/usr/bin/env python3
"""
Visualize macro trajectory features comparison: Human vs each LLM model by domain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

VAR_SUFFIXES = ("_cv", "_rmssd_norm", "_masd_norm")
GEOMETRY_METRICS = (
    "mean_distance",
    "std_distance",
    "net_displacement",
    "path_length",
    "tortuosity",
)

PROVIDERS = ["human", "DS", "G4B", "G12B", "LMK"]
COLORS = {
    "human": "#2E8B57",
    "DS": "#1f77b4",
    "G4B": "#ff7f0e",
    "G12B": "#d62728",
    "LMK": "#9467bd",
}

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11


def load_features(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def feature_groups(df: pd.DataFrame) -> dict:
    cols = df.columns.tolist()
    var_cols = [c for c in cols if any(c.endswith(s) for s in VAR_SUFFIXES)]
    ce_geom = [c for c in cols if c.startswith("ce_") and any(c.endswith(f"_{m}") for m in GEOMETRY_METRICS)]
    tfidf_geom = [c for c in cols if c.startswith("tfidf_") and any(c.endswith(f"_{m}") for m in GEOMETRY_METRICS)]
    sbert_geom = [c for c in cols if c.startswith("sbert_") and any(c.endswith(f"_{m}") for m in GEOMETRY_METRICS)]
    return {
        "variability": var_cols,
        "ce_geom": ce_geom,
        "tfidf_geom": tfidf_geom,
        "sbert_geom": sbert_geom,
        "all": var_cols + ce_geom + tfidf_geom + sbert_geom,
    }


def get_row(df: pd.DataFrame, domain: str, provider: str) -> pd.Series | None:
    sub = df[(df["domain"] == domain) & (df["provider"] == provider)]
    return None if sub.empty else sub.iloc[0]


def mean_feature(row: pd.Series | None, feats: List[str]) -> float:
    if row is None or not feats:
        return np.nan
    return float(row[feats].mean())


def plot_bars(df: pd.DataFrame, out_dir: Path) -> None:
    groups = feature_groups(df)
    domains = sorted(df["domain"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    titles = [
        ("variability", "CE Variability"),
        ("ce_geom", "CE Geometry"),
        ("tfidf_geom", "TFIDF Geometry"),
        ("sbert_geom", "SBERT Geometry"),
    ]

    for ax, (key, title) in zip(axes, titles):
        feats = groups[key]
        x = np.arange(len(domains))
        width = 0.8 / len(PROVIDERS)
        for i, provider in enumerate(PROVIDERS):
            means = [mean_feature(get_row(df, dom, provider), feats) for dom in domains]
            bars = ax.bar(x - 0.4 + i * width, means, width,
                          label=provider.upper(), color=COLORS.get(provider, "gray"),
                          edgecolor="black", alpha=0.85)
            for bar, val in zip(bars, means):
                if np.isnan(val):
                    continue
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in domains])
        ax.set_ylabel("Mean Value")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Macro Trajectory Features by Domain & Model", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "trajectory_features_by_model.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved: {out}")


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    groups = feature_groups(df)
    domains = sorted(df["domain"].unique())
    base_feats = groups["all"]
    human_rows = {d: get_row(df, d, "human") for d in domains}

    diff_cols, diff_rows = [], []
    for prov in PROVIDERS:
        if prov == "human":
            continue
        for dom in domains:
            row = get_row(df, dom, prov)
            base = human_rows.get(dom)
            if row is None or base is None:
                continue
            diff_rows.append(row[base_feats].values - base[base_feats].values)
            diff_cols.append(f"{dom.upper()}-{prov}")
    if not diff_rows:
        return

    matrix = np.array(diff_rows, dtype=float).T
    importances = np.abs(matrix).mean(axis=1)
    top = min(30, len(base_feats))
    idx = np.argsort(importances)[-top:][::-1]
    matrix = matrix[idx]
    top_feats = [base_feats[i] for i in idx]

    plt.figure(figsize=(len(diff_cols) * 1.2, top * 0.4 + 4))
    sns.heatmap(matrix, cmap="RdBu_r", center=0,
                xticklabels=diff_cols,
                yticklabels=[f.replace("_", " ").title() for f in top_feats])
    plt.title("Feature Difference (LLM - Human)", fontweight="bold")
    plt.tight_layout()
    out = out_dir / "trajectory_features_heatmap_by_model.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved: {out}")


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    groups = feature_groups(df)
    domains = sorted(df["domain"].unique())
    rows = []
    for dom in domains:
        human_row = get_row(df, dom, "human")
        human_mean = mean_feature(human_row, groups["all"])
        for prov in PROVIDERS:
            row = get_row(df, dom, prov)
            if row is None:
                continue
            mean_val = mean_feature(row, groups["all"])
            delta = mean_val - human_mean if not np.isnan(human_mean) else np.nan
            rows.append({
                "Domain": dom.upper(),
                "Provider": prov.upper(),
                "Mean": f"{mean_val:.4f}",
                "Δ vs Human": f"{delta:+.4f}" if not np.isnan(delta) else "N/A",
            })
    table_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.35)))
    ax.axis("off")
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                   cellLoc="center", loc="center", colColours=["#4a90e2"] * len(table_df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.4)
    for (i, j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_text_props(color="white", weight="bold")
        else:
            prov = table_df.iloc[i-1]["Provider"].lower()
            cell.set_facecolor(COLORS.get(prov, "#dddddd"))
            cell.set_alpha(0.25)
    ax.set_title("Summary: Mean Trajectory Features", fontsize=14, fontweight="bold")
    out = out_dir / "trajectory_features_summary_table.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_domain_model_grid(df: pd.DataFrame, out_dir: Path) -> None:
    """
    3×4 grid:
      - rows = domains (Academic / Blogs / News)
      - cols = feature groups (CE-VAR, CE-GEO, TFIDF-GEO, SBERT-GEO)
      - bars in each subplot = Human + 4 LLM models
    """
    groups = feature_groups(df)
    domains = sorted(df["domain"].unique())

    # 3 rows (domains) × 4 cols (feature groups)
    feature_labels = ["CE-VAR", "CE-GEO", "TFIDF-GEO", "SBERT-GEO"]
    feature_keys = ["variability", "ce_geom", "tfidf_geom", "sbert_geom"]

    fig, axes = plt.subplots(
        nrows=len(domains),
        ncols=len(feature_keys),
        figsize=(len(feature_keys) * 4.5, len(domains) * 3.0),
        sharey=False,
    )

    # Ensure 2D array
    if len(domains) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, domain in enumerate(domains):
        for j, (feat_key, feat_label) in enumerate(zip(feature_keys, feature_labels)):
            ax = axes[i][j]
            feats = groups[feat_key]

            # X-axis: providers (Human + 4 LLM models)
            x = np.arange(len(PROVIDERS))
            values = []
            for prov in PROVIDERS:
                row = get_row(df, domain, prov if prov != "human" else "human")
                values.append(mean_feature(row, feats))

            bars = ax.bar(
                x,
                values,
                color=[COLORS.get(p, "gray") for p in PROVIDERS],
                alpha=0.9,
                edgecolor="black",
                linewidth=1.0,
            )

            # X tick labels: provider names
            ax.set_xticks(x)
            ax.set_xticklabels([p.upper() for p in PROVIDERS], rotation=45, ha="right", fontsize=9)

            # Annotate bars
            for bar, val in zip(bars, values):
                if np.isnan(val):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.001,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            # Row labels on the left
            if j == 0:
                ax.set_ylabel(domain.upper(), fontweight="bold")

            # Column titles on the top row
            if i == 0:
                ax.set_title(feat_label, fontweight="bold", fontsize=12)

            ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(
        "Trajectory Features Comparison: Human vs LLM Models by Domain",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "trajectory_features_domain_model_grid.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize macro trajectory features per model")
    parser.add_argument("--input", type=str, default="macro_dataset/process/trajectory_features_combined_all.csv")
    parser.add_argument("--output", type=str, default="plots/macro/trajectory_features_by_model")
    args = parser.parse_args()

    csv = PROJECT_ROOT / args.input
    if not csv.exists():
        print(f"❌ Input file not found: {csv}")
        return
    out_dir = PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(csv)
    print("Loaded rows:", len(df))

    plot_bars(df, out_dir)
    plot_heatmap(df, out_dir)
    plot_summary(df, out_dir)
    plot_domain_model_grid(df, out_dir)
    print(f"Done. Plots at {out_dir}")


if __name__ == "__main__":
    main()
