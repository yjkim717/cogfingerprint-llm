#!/usr/bin/env python3
"""
Micro Dataset Static ML Validation by Level: Compare Human vs LLM across LV1-LV3.

This script performs ML classification on Micro dataset using 414 static features:
- 20 CE features
- 10 TF-IDF features  
- 384 SBERT features

Data:
- Human: All ~6000 samples across all domains
- LLM: 4 models (DS, G4B, G12B, LMK) × 3 levels (LV1, LV2, LV3) = mirror samples
- Comparison: Separate analysis for each level

Features:
- All 414 static features (CE + TF-IDF + SBERT)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
DOMAINS = ["academic", "blogs", "news"]
LLM_MODELS = ["DS", "G4B", "G12B", "LMK"]
LEVELS = ["LV1", "LV2", "LV3"]

# 20 CE features
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

HUMAN_BASE = Path("dataset/process/human")
LLM_BASE = Path("dataset/process/LLM")
RESULTS_DIR = Path("micro_results/micro_static_ml_by_level")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_metadata_from_filename(filename: str) -> dict:
    """Parse metadata from micro dataset filename."""
    # Human: <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>.txt
    # LLM: <Domain>_<Field>_<AuthorID>_<Year>_<ItemIndex>_<Model>_<Level>.txt
    parts = filename.replace(".txt", "").split("_")
    metadata = {
        "domain": None,
        "field": None,
        "author_id": None,
        "year": None,
        "item_index": None,
        "model": None,
        "level": None,
    }
    
    if len(parts) >= 5:
        metadata["domain"] = parts[0].lower()
        metadata["field"] = parts[1]
        metadata["author_id"] = parts[2]
        metadata["year"] = parts[3] if parts[3].isdigit() and len(parts[3]) == 4 else None
        metadata["item_index"] = parts[4] if parts[4].isdigit() else None
        
        # Check for LLM suffix: _Model_Level
        if len(parts) >= 7:
            metadata["model"] = parts[5]
            metadata["level"] = parts[6]
    
    return metadata


def load_micro_features(
    path: Path, 
    source: str, 
    model_hint: Optional[str] = None,
    level_hint: Optional[str] = None
) -> pd.DataFrame:
    """Load micro dataset features from combined file."""
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df["source"] = source
    df["label"] = source
    
    # Parse metadata from filename
    if "filename" in df.columns:
        metadata_list = df["filename"].apply(parse_metadata_from_filename)
        for key in ["domain", "field", "author_id", "year", "item_index", "model", "level"]:
            df[key] = [m.get(key) for m in metadata_list]
        
        # Fill model/level for human samples
        if source == "human":
            df["model"] = "HUMAN"
            df["level"] = "HUMAN"
        elif model_hint:
            df["model"] = df["model"].fillna(model_hint)
        if level_hint:
            df["level"] = df["level"].fillna(level_hint)
    
    df["binary_target"] = np.where(df["label"].str.lower() == "llm", 1, 0)
    
    return df


def build_dataset_by_level(level: str, verbose: bool = True) -> pd.DataFrame:
    """Build dataset for a specific level."""
    frames: list[pd.DataFrame] = []
    
    # Load human data (all domains)
    # Human uses: combined_with_embeddings.csv (merged from combined_merged.csv + tfidf + sbert)
    for domain in DOMAINS:
        human_path = HUMAN_BASE / domain / "combined_with_embeddings.csv"
        
        if human_path.exists():
            if verbose:
                print(f"[LOAD] Human {domain}: {human_path}")
            df_human = load_micro_features(human_path, source="human")
            frames.append(df_human)
        else:
            if verbose:
                print(f"[WARN] Missing human file: {human_path}")
                print(f"       Please run: python scripts/micro/merge_features_micro.py --human-only")
    
    # Load LLM data for specified level
    # LLM uses: combined_with_embeddings.csv (merged from combined_merged_outliers_removed.csv + tfidf + sbert)
    for model in LLM_MODELS:
        for domain in DOMAINS:
            llm_path = LLM_BASE / model / level / domain / "combined_with_embeddings.csv"
            
            if llm_path.exists():
                if verbose:
                    print(f"[LOAD] {model} {level} {domain}: {llm_path}")
                df_llm = load_micro_features(llm_path, source="llm", model_hint=model, level_hint=level)
                frames.append(df_llm)
            else:
                if verbose:
                    print(f"[WARN] Missing LLM file: {llm_path}")
                    print(f"       Please run: python scripts/micro/merge_features_micro.py")
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    
    if verbose:
        print(f"\n[INFO] Total samples: {len(df)}")
        print(f"[INFO] Label distribution:")
        print(df["label"].value_counts())
        print(f"[INFO] Domain distribution:")
        print(df["domain"].value_counts())
        print(f"[INFO] Model distribution:")
        print(df["model"].value_counts())
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all 414 feature columns (CE + TF-IDF + SBERT)."""
    ce_cols = [c for c in CE_FEATURES if c in df.columns]
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    sbert_cols = [c for c in df.columns if c.startswith("sbert_")]
    
    all_features = ce_cols + tfidf_cols + sbert_cols
    
    # Remove any duplicates
    return list(dict.fromkeys(all_features))


def make_pipeline() -> Pipeline:
    """Create ML pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )


def run_ml_validation(df: pd.DataFrame, feature_cols: list[str], level: str, domain: Optional[str] = None) -> dict:
    """Run ML validation."""
    if domain:
        df = df[df["domain"] == domain].copy()
    
    if len(df) == 0:
        return {}
    
    # Check class balance
    class_counts = df["binary_target"].value_counts()
    if len(class_counts) < 2:
        print(f"⚠️  Only one class present, skipping")
        return {}
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df["binary_target"].values
    
    # Split: 70% train, 30% test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = make_pipeline()
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    
    # Train on full training set
    pipeline.fit(X_train, y_train)
    
    # Test set predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Counts
    human_count = (df["label"].str.lower() == "human").sum()
    llm_count = (df["label"].str.lower() == "llm").sum()
    
    result = {
        "level": level,
        "domain": domain or "all",
        "human_samples": int(human_count),
        "llm_samples": int(llm_count),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": len(feature_cols),
        "test_accuracy": float(test_accuracy),
        "test_roc_auc": float(test_roc_auc),
        "test_f1": float(test_f1),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Micro dataset static ML validation by level"
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS + ["all"],
        default="all",
        help="Domain to analyze (default: all)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=LEVELS,
        help="Levels to analyze (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    domains_to_process = DOMAINS if args.domain == "all" else [args.domain]
    
    all_results = []
    
    for level in args.levels:
        print(f"\n{'='*80}")
        print(f"Processing Level: {level}")
        print(f"{'='*80}\n")
        
        # Build dataset for this level
        df = build_dataset_by_level(level, verbose=True)
        
        if len(df) == 0:
            print(f"⚠️  No data found for {level}, skipping")
            continue
        
        # Get feature columns
        feature_cols = get_feature_columns(df)
        print(f"\n[INFO] Feature columns: {len(feature_cols)}")
        print(f"  - CE: {len([c for c in CE_FEATURES if c in df.columns])}")
        print(f"  - TF-IDF: {len([c for c in df.columns if c.startswith('tfidf_')])}")
        print(f"  - SBERT: {len([c for c in df.columns if c.startswith('sbert_')])}")
        
        if len(feature_cols) < 20:
            print(f"⚠️  Too few features ({len(feature_cols)}), skipping")
            continue
        
        # Run validation for each domain
        for domain in domains_to_process:
            print(f"\n--- {domain.upper()} ---")
            result = run_ml_validation(df, feature_cols, level, domain=domain)
            if result:
                all_results.append(result)
                print(f"Accuracy: {result['test_accuracy']:.4f}")
                print(f"ROC AUC: {result['test_roc_auc']:.4f}")
                print(f"F1 Score: {result['test_f1']:.4f}")
        
        # Also run for all domains combined
        if args.domain == "all":
            print(f"\n--- ALL DOMAINS ---")
            result = run_ml_validation(df, feature_cols, level, domain=None)
            if result:
                all_results.append(result)
                print(f"Accuracy: {result['test_accuracy']:.4f}")
                print(f"ROC AUC: {result['test_roc_auc']:.4f}")
                print(f"F1 Score: {result['test_f1']:.4f}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = args.output / "micro_static_ml_by_level.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        # Also save as JSON
        output_json = args.output / "micro_static_ml_by_level.json"
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Results saved to {output_json}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))
    else:
        print("\n⚠️  No results to save")


if __name__ == "__main__":
    main()

