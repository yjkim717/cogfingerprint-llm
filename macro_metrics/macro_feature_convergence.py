#!/usr/bin/env python3
"""
macro_feature_convergence.py
------------------------------------------------------
Human vs Each LLM Model (LV3 Only for LLM)
Feature-level convergence + normalized layer plots + trend summary

Outputs:
    macro_dataset/process_output/Macro/
        - <domain>_<MODEL>_lv3_feature_convergence.csv
        - <domain>_<MODEL>_lv3_feature_trend.csv
        - layer_plots/<domain>/<MODEL>_lv3/*.png
        - layer_summary/<domain>/<MODEL>_lv3_summary.csv
------------------------------------------------------
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "macro_dataset"
INPUT_DIR = os.path.join(BASE_DIR, "process")
OUTPUT_DIR = os.path.join(BASE_DIR, "process_output", "Macro")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAINS = ["academic", "blogs", "news"]
LLM_MODELS = ["G4B", "G12B", "DS", "LMK"]

# Layer grouping
COGNITIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
EMOTIONAL = ["polarity", "subjectivity", "vader_compound", "vader_pos", "vader_neu", "vader_neg"]
STYLISTIC = [
    "word_diversity", "flesch_reading_ease", "gunning_fog",
    "average_word_length", "num_words", "avg_sentence_length",
    "verb_ratio", "function_word_ratio", "content_word_ratio"
]

def extract_year(fname):
    m = re.search(r"_(\d{4})_", str(fname))
    return int(m.group(1)) if m else None


# ======================================
# 🔥 Layer Plot (Normalized ABS)
# ======================================
def plot_layer_abs(layer_name, feature_list, feature_diff, trend_df, outdir):
    plt.figure(figsize=(10, 6))

    years = feature_diff["year"].values

    colors = {
        "converging": "#1f77b4",   # blue
        "diverging":  "#d62728",   # red
        "insufficient": "#7f7f7f"  # gray
    }

    for feature in feature_list:
        if feature not in feature_diff.columns:
            continue

        series = feature_diff[feature].dropna()
        if len(series) == 0:
            continue

        abs_vals = np.abs(series.values)

        # Normalize ABS (0–1)
        fmin, fmax = abs_vals.min(), abs_vals.max()
        if fmax - fmin == 0:
            norm_vals = abs_vals * 0
        else:
            norm_vals = (abs_vals - fmin) / (fmax - fmin)

        # Trend color
        row = trend_df[trend_df["feature"] == feature]
        trend_type = row["trend"].values[0] if not row.empty else "insufficient"
        color = colors.get(trend_type, "black")

        plt.plot(
            years[:len(norm_vals)],
            norm_vals,
            marker="o",
            linewidth=2,
            label=f"{feature} ({trend_type})",
            color=color
        )

    plt.title(f"{layer_name} Layer — Normalized ABS")
    plt.xlabel("Year")
    plt.ylabel("Normalized |Human – LLM|")
    plt.grid(alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(outdir, f"{layer_name}_layer_abs_norm.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================
# 🔥 Layer Summary (Converging ratio etc.)
# ======================================
def save_layer_summary(domain, model, trend_df, outdir):
    summary_rows = []

    def calc(layer_name, features):
        df = trend_df[trend_df["feature"].isin(features)]
        total = len(df)
        conv = (df["trend"] == "converging").sum()
        div = (df["trend"] == "diverging").sum()
        ins = (df["trend"] == "insufficient").sum()
        return [layer_name, total, conv, div, ins, round(conv/total, 3) if total else 0]

    summary_rows.append(calc("Cognitive", COGNITIVE))
    summary_rows.append(calc("Emotional", EMOTIONAL))
    summary_rows.append(calc("Stylistic", STYLISTIC))

    summary_df = pd.DataFrame(summary_rows,
                              columns=["Layer", "Total", "Converging", "Diverging", "Insufficient", "Converging_Ratio"])

    out_csv = os.path.join(outdir, f"{domain}_{model}_lv3_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"      📄 Saved layer summary → {out_csv}")


# ======================================
# MAIN ANALYSIS
# ======================================
def analyze_domain(domain_name):
    print(f"\n=====================================================")
    print(f"📌 Processing domain: {domain_name}")
    print(f"=====================================================")

    csv_path = os.path.join(INPUT_DIR, f"combined_{domain_name}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str).str.lower()

    if "year" not in df.columns:
        df["year"] = df["filename"].apply(extract_year)

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= 2020]

    # select numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["year", "author_id"]]

    df[numeric_cols] = df[numeric_cols].replace("", np.nan)

    # human rows
    human_df = df[df["label"] == "human"]

    # ---------------------
    # LLM models lv3
    # ---------------------
    for model in LLM_MODELS:
        print(f"\n  🔍 Model LV3: {model}")

        llm_df = df[(df["label"] == "llm") &
                    (df["model"] == model) &
                    (df["level"] == "LV3")]

        if llm_df.empty:
            print(f"     ⚠ No {model} LV3 data.")
            continue

        # yearly means
        human_yearly = human_df.groupby("year")[numeric_cols].mean()
        llm_yearly = llm_df.groupby("year")[numeric_cols].mean()

        common_years = human_yearly.index.intersection(llm_yearly.index)
        if len(common_years) == 0:
            print("     ⚠ No overlapping years.")
            continue

        human_yearly = human_yearly.loc[common_years]
        llm_yearly = llm_yearly.loc[common_years]

        feature_diff = human_yearly.subtract(llm_yearly, axis=1)
        feature_diff.insert(0, "year", common_years)

        # save raw diff csv
        conv_path = os.path.join(OUTPUT_DIR, f"{domain_name}_{model}_lv3_feature_convergence.csv")
        feature_diff.to_csv(conv_path, index=False)

        # ------------- trend calc -------------
        trend_rows = []
        for feature in numeric_cols:
            s = feature_diff[feature].dropna()
            if len(s) <= 1:
                trend_rows.append([feature, np.nan, "insufficient"])
                continue

            abs_vals = np.abs(s.values)
            years = feature_diff.loc[s.index, "year"].values

            slope = np.polyfit(years, abs_vals, 1)[0]

            if slope < 0:
                t = "converging"
            elif slope > 0:
                t = "diverging"
            else:
                t = "flat"

            trend_rows.append([feature, slope, t])

        trend_df = pd.DataFrame(trend_rows,
                                columns=["feature", "slope_abs", "trend"])

        trend_path = os.path.join(OUTPUT_DIR, f"{domain_name}_{model}_lv3_feature_trend.csv")
        trend_df.to_csv(trend_path, index=False)

        # ------------- layer plotting -------------
        LAYER_DIR = os.path.join(OUTPUT_DIR, "layer_plots", domain_name, f"{model}_lv3")
        SUMMARY_DIR = os.path.join(OUTPUT_DIR, "layer_summary", domain_name)
        os.makedirs(LAYER_DIR, exist_ok=True)
        os.makedirs(SUMMARY_DIR, exist_ok=True)

        plot_layer_abs("Cognitive", COGNITIVE, feature_diff, trend_df, LAYER_DIR)
        plot_layer_abs("Emotional", EMOTIONAL, feature_diff, trend_df, LAYER_DIR)
        plot_layer_abs("Stylistic", STYLISTIC, feature_diff, trend_df, LAYER_DIR)

        # ------------- summary save -------------
        save_layer_summary(domain_name, model, trend_df, SUMMARY_DIR)

        print(f"      📈 Saved layer plots → {LAYER_DIR}")


if __name__ == "__main__":
    for d in DOMAINS:
        analyze_domain(d)

    print("\n🌟 Macro LV3 feature convergence — with Layer Plot & Summary complete!")
