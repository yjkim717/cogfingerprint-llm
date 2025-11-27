# baseline_plot_grouped.py
# ------------------------------------------------------
# Visualize static baseline prototype by feature groups
# (Cognitive / Emotional / Stylistic)
# ------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

DATA_PATH = "Datasets/Processed/combined_features.csv"
SAVE_PATH = "Datasets/Processed/baseline_prototype_grouped.png"

assert os.path.exists(DATA_PATH), "❌ combined_features.csv not found."
df = pd.read_csv(DATA_PATH)

# === Compute static baseline ===
metrics_of_interest = [
    "Openness", "Conscientiousness",        # Cognitive
    "polarity", "subjectivity",             # Emotional
    "word_diversity", "avg_sentence_length" # Stylistic
]
baseline = (
    df.groupby("label")[metrics_of_interest]
    .mean(numeric_only=True)
    .reset_index()
)

# === Split into groups ===
groups = {
    "Cognitive": ["Openness", "Conscientiousness"],
    "Emotional": ["polarity", "subjectivity"],
    "Stylistic": ["word_diversity", "avg_sentence_length"]
}

sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

palette = {"human": "#2a9d8f", "llm": "#e76f51"}

for ax, (group_name, feats) in zip(axes, groups.items()):
    melted = baseline.melt(id_vars="label", value_vars=feats,
                           var_name="metric", value_name="mean_value")
    melted["metric"] = melted["metric"].str.replace("_", " ").str.title()
    sns.barplot(data=melted, x="metric", y="mean_value",
                hue="label", palette=palette, ax=ax)
    ax.set_title(f"{group_name} Features")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Value")
    ax.tick_params(axis='x', rotation=30)
    ax.legend().set_title("Source")

plt.suptitle("🧩 Static Baseline Prototype — Cognitive · Emotional · Stylistic", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(SAVE_PATH, dpi=300)
plt.show()

print(f"✅ Saved grouped baseline visualization to {SAVE_PATH}")
