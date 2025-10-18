# trajectory_plot.ipynb
# ------------------------------------------------------
# Visualize cognitive–emotional–stylistic trajectories
# and compare with static baseline means
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load combined feature data ===
DATA_PATH = "Datasets/Processed/combined_features.csv"
assert os.path.exists(DATA_PATH), "❌ combined_features.csv not found."
df = pd.read_csv(DATA_PATH)

# === Extract year from path ===
df["year"] = df["path"].str.extract(r"(\d{4})")
df["year"] = df["year"].astype(str)

# === Compute yearly averages ===
metrics_of_interest = [
    "openness", "conscientiousness",        # Cognitive
    "polarity", "subjectivity",             # Emotional
    "word_diversity", "avg_sentence_length" # Stylistic
]
df_yearly = df.groupby(["label", "year"])[metrics_of_interest].mean(numeric_only=True).reset_index()

# === Compute static baseline (overall mean per label) ===
baseline = (
    df.groupby("label")[metrics_of_interest]
    .mean(numeric_only=True)
    .reset_index()
    .rename(columns={m: f"baseline_{m}" for m in metrics_of_interest})
)
print("✅ Baseline means:")
print(baseline)

# === Plot setup ===
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(15, 10))

# === 1. Cognitive Trajectories ===
plt.subplot(3, 1, 1)
for label, color in zip(["human", "llm"], ["#2a9d8f", "#e76f51"]):
    subset = df_yearly[df_yearly["label"] == label]
    base = baseline[baseline["label"] == label]
    # Trajectories
    plt.plot(subset["year"], subset["openness"], marker="o", label=f"{label} – Openness", color=color)
    plt.plot(subset["year"], subset["conscientiousness"], marker="s", linestyle="--", label=f"{label} – Conscientiousness", color=color, alpha=0.7)
    # Baseline (static mean)
    plt.axhline(y=float(base[f"baseline_openness"]), color=color, linestyle=":", alpha=0.6)
plt.title("🧠 Cognitive Trajectories vs Static Baseline")
plt.ylabel("Mean Score")
plt.legend()

# === 2. Emotional Trajectories ===
plt.subplot(3, 1, 2)
for label, color in zip(["human", "llm"], ["#2a9d8f", "#e76f51"]):
    subset = df_yearly[df_yearly["label"] == label]
    base = baseline[baseline["label"] == label]
    plt.plot(subset["year"], subset["polarity"], marker="o", label=f"{label} – Polarity", color=color)
    plt.plot(subset["year"], subset["subjectivity"], marker="s", linestyle="--", label=f"{label} – Subjectivity", color=color, alpha=0.7)
    plt.axhline(y=float(base[f"baseline_subjectivity"]), color=color, linestyle=":", alpha=0.6)
plt.title("❤️ Emotional Trajectories vs Static Baseline")
plt.ylabel("Mean Score")
plt.legend()

# === 3. Stylistic Trajectories ===
plt.subplot(3, 1, 3)
for label, color in zip(["human", "llm"], ["#2a9d8f", "#e76f51"]):
    subset = df_yearly[df_yearly["label"] == label]
    base = baseline[baseline["label"] == label]
    plt.plot(subset["year"], subset["word_diversity"], marker="o", label=f"{label} – Word Diversity", color=color)
    plt.plot(subset["year"], subset["avg_sentence_length"], marker="s", linestyle="--", label=f"{label} – Sentence Length", color=color, alpha=0.7)
    plt.axhline(y=float(base[f"baseline_word_diversity"]), color=color, linestyle=":", alpha=0.6)
plt.title("✍️ Stylistic Trajectories vs Static Baseline")
plt.xlabel("Year")
plt.ylabel("Mean Value")
plt.legend()

plt.tight_layout()
plt.savefig("Datasets/Processed/trajectory_vs_baseline.png", dpi=300)
plt.show()

print("✅ Saved visualization to Datasets/Processed/trajectory_vs_baseline.png")
