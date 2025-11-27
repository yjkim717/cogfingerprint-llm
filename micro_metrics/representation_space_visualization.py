#!/usr/bin/env python3
"""
representation_difference_visualization.py
------------------------------------------------------
Generates Human vs Provider difference-distribution plots
for each domain × provider × method.

- Domains: academic, blogs, news
- Providers: DS, G12B, G4B, LMK
- Methods: embedding, tfidf

Creates TWO versions:
  1) All levels (provider LV1–LV5)
  2) LV3 only (provider level == 3)

Each plot:
  - Human – LLM Difference Distribution
  - Red vertical line at 0 (no difference)
  - Black vertical line showing mean difference
------------------------------------------------------
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------
# Configurations
# ------------------------------------------------------
DOMAINS = ["academic", "blogs", "news"]
PROVIDERS = ["DS", "G12B", "G4B", "LMK"]
METHODS = ["embedding", "tfidf"]

INPUT_ROOT = "dataset/process/representation_pipeline"
OUTPUT_ROOT = "dataset/process/representation_visualization_diff"


# ------------------------------------------------------
# Create diff-distribution plot
# ------------------------------------------------------
def plot_difference_distribution(df, provider, domain, method, lv3=False):
    # Human data
    human_df = df[df["label"] == "human"]

    # Provider data (all or lv3 only)
    if lv3:
        llm_df = df[(df["label"] == "llm") &
                    (df["provider"] == provider) &
                    (df["level"] == 3)]
    else:
        llm_df = df[(df["label"] == "llm") &
                    (df["provider"] == provider)]

    if human_df.empty or llm_df.empty:
        print(f"⚠️ Skipping: No data for {domain}/{provider}/{method} (lv3={lv3})")
        return

    # Compute all pairwise differences (Human – Provider)
    diffs = []
    for h in human_df["mean_distance"]:
        for l in llm_df["mean_distance"]:
            diffs.append(h - l)

    diffs = pd.Series(diffs)

    plt.figure(figsize=(10, 6))
    sns.histplot(diffs, kde=True, bins=30, color="skyblue", alpha=0.8)

    # 0 reference line
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="No Difference (0)")

    # Mean difference line
    mean_val = diffs.mean()
    plt.axvline(mean_val, color="black", linestyle="--",
                linewidth=2, label=f"Mean Diff = {mean_val:.3f}")

    lv_tag = "LV3 Only" if lv3 else "All Levels"

    plt.title(f"{domain.capitalize()} — {method.upper()} — Human vs {provider} ({lv_tag})")
    plt.xlabel("Human Drift – LLM Drift")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    # Save path
    if lv3:
        out_dir = os.path.join(OUTPUT_ROOT, "lv3", domain)
        fname = f"{domain}_{provider}_{method}_lv3_diff.png"
    else:
        out_dir = os.path.join(OUTPUT_ROOT, domain)
        fname = f"{domain}_{provider}_{method}_diff.png"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out_path}")


# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
def load_distance_csv(domain, method):
    path = os.path.join(INPUT_ROOT, f"{domain}_{method}_euclid.csv")
    if not os.path.exists(path):
        print(f"⚠ Missing file: {path}")
        return None
    return pd.read_csv(path)


# ------------------------------------------------------
# Main runner
# ------------------------------------------------------
def main():
    for domain in DOMAINS:
        for method in METHODS:
            df = load_distance_csv(domain, method)
            if df is None:
                continue

            # Needed columns check
            needed = {"label", "provider", "level", "mean_distance"}
            if not needed.issubset(df.columns):
                print(f"⚠ Missing required columns in {domain}/{method}")
                continue

            for provider in PROVIDERS:
                # 1) All Levels
                plot_difference_distribution(
                    df, provider, domain, method, lv3=False
                )

                # 2) LV3 Only
                plot_difference_distribution(
                    df, provider, domain, method, lv3=True
                )

    print("\n🎉 All difference-distribution plots generated successfully!")


if __name__ == "__main__":
    main()
