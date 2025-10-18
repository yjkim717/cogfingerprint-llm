# analyze_metrics.py
# ------------------------------------------------------
# Run full metrics extraction pipeline (all .txt files)
# Supports partial reruns (e.g., only NELA or only merge)
# ------------------------------------------------------

import os
import pandas as pd
from utils.metrics_big5 import extract_big5_features
from utils.metrics_nela import extract_nela_features

# === Execution Options ===
# Set True for the stages you want to run
RUN_BIG5 = True      # Run Big Five extraction
RUN_NELA = True      # Run NELA extraction
RUN_COMBINE = True   # Merge and save combined CSV

# === Path Configuration ===
DATA_ROOT = "Datasets"
HUMAN_DIR = os.path.join(DATA_ROOT, "Human")
LLM_DIR = os.path.join(DATA_ROOT, "LLM")
OUTPUT_DIR = os.path.join(DATA_ROOT, "Processed")

BIG5_HUMAN_PATH = os.path.join(OUTPUT_DIR, "big5_human.csv")
BIG5_LLM_PATH = os.path.join(OUTPUT_DIR, "big5_llm.csv")
NELA_HUMAN_PATH = os.path.join(OUTPUT_DIR, "nela_human.csv")
NELA_LLM_PATH = os.path.join(OUTPUT_DIR, "nela_llm.csv")
COMBINED_PATH = os.path.join(OUTPUT_DIR, "combined_features.csv")

# === Step 1: Big Five (Optional) ===
if RUN_BIG5:
    print("Extracting Big Five features from all .txt files...")
    df_big5_human = extract_big5_features(HUMAN_DIR, "human", BIG5_HUMAN_PATH)
    df_big5_llm = extract_big5_features(LLM_DIR, "llm", BIG5_LLM_PATH)
    df_big5_all = pd.concat([df_big5_human, df_big5_llm], ignore_index=True)
else:
    print("Skipping Big Five extraction — loading from saved files.")
    if os.path.exists(BIG5_HUMAN_PATH) and os.path.exists(BIG5_LLM_PATH):
        df_big5_human = pd.read_csv(BIG5_HUMAN_PATH)
        df_big5_llm = pd.read_csv(BIG5_LLM_PATH)
        df_big5_all = pd.concat([df_big5_human, df_big5_llm], ignore_index=True)
    else:
        raise FileNotFoundError("Big Five feature files not found — run with RUN_BIG5=True first.")

# === Step 2: NELA ===
if RUN_NELA:
    print("Extracting NELA emotional and stylistic features from all .txt files...")
    df_nela_human = extract_nela_features(HUMAN_DIR, "human", NELA_HUMAN_PATH)
    df_nela_llm = extract_nela_features(LLM_DIR, "llm", NELA_LLM_PATH)
    df_nela_all = pd.concat([df_nela_human, df_nela_llm], ignore_index=True)
else:
    print("Skipping NELA extraction — loading from saved files.")
    if os.path.exists(NELA_HUMAN_PATH) and os.path.exists(NELA_LLM_PATH):
        df_nela_human = pd.read_csv(NELA_HUMAN_PATH)
        df_nela_llm = pd.read_csv(NELA_LLM_PATH)
        df_nela_all = pd.concat([df_nela_human, df_nela_llm], ignore_index=True)
    else:
        raise FileNotFoundError("NELA feature files not found — run with RUN_NELA=True first.")

# === Step 3: Combine ===
if RUN_COMBINE:
    print("Combining Big Five and NELA results...")
    df_combined = pd.merge(df_big5_all, df_nela_all, on=["filename", "path", "label"], how="inner")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_combined.to_csv(COMBINED_PATH, index=False)
    print(f"Combined features saved to {COMBINED_PATH}")
    print("\n=== Sample preview ===")
    print(df_combined.head(3))
else:
    print("Skipping combination step.")
