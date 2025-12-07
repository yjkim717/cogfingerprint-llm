#!/usr/bin/env python3
"""
RQ2 Yearly ML Validation: Analyze if classification accuracy decreases over years.

This script performs yearly ML validation to test if human and LLM texts converge over time.
For each year, we train a classifier on that year's data and test on the same year.
If convergence occurs, accuracy should decrease over years.

Data:
- Human: 1000 samples per year per domain
- LLM: 4000 samples per year per domain (4 models × 1000 each)
- Total: 5000 samples per year per domain

Features:
- Option 1: 20 CE features only
- Option 2: All features (20 CE + 10 TF-IDF + 384 SBERT)
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
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
DOMAINS = ["academic", "blogs", "news"]
LLM_MODELS = ["DS", "G4B", "G12B", "LMK"]
LEVEL = "LV3"

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

HUMAN_BASE = Path("macro_dataset/process/human")
LLM_BASE = Path("macro_dataset/process/LLM")
RESULTS_DIR = Path("macro_results/rq2_yearly_ml_validation")


def parse_year_from_filename(filename: str) -> Optional[int]:
    """Extract 4-digit year from filename."""
    tokens = filename.replace(".txt", "").split("_")
    for token in tokens:
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def load_combined_with_embeddings(path: Path, source: str, model_hint: Optional[str] = None) -> pd.DataFrame:
    """Load combined_with_embeddings.csv file."""
    df = pd.read_csv(path)
    df["source"] = source
    if "model" not in df.columns or df["model"].isna().all():
        df["model"] = model_hint or ("HUMAN" if source == "human" else "UNKNOWN")
    else:
        df["model"] = df["model"].fillna(model_hint or "HUMAN")
    
    # Parse year from filename
    df["year"] = df["filename"].apply(parse_year_from_filename)
    
    # Ensure label is set
    if "label" not in df.columns:
        df["label"] = source
    df["label"] = df["label"].fillna(source).str.lower()
    df["binary_target"] = np.where(df["label"] == "llm", 1, 0)
    
    return df


def build_dataset(use_all_features: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build dataset from combined_with_embeddings.csv files."""
    frames: list[pd.DataFrame] = []

    # Load human data
    for domain in DOMAINS:
        human_path = HUMAN_BASE / domain / "combined_with_embeddings.csv"
        if human_path.exists():
            if verbose:
                print(f"[LOAD] Human {domain}: {human_path}")
            frames.append(load_combined_with_embeddings(human_path, source="human", model_hint="HUMAN"))
        else:
            print(f"[WARN] Missing human file: {human_path}")

    # Load LLM data
    for model in LLM_MODELS:
        for domain in DOMAINS:
            llm_path = LLM_BASE / model / LEVEL / domain / "combined_with_embeddings.csv"
            if llm_path.exists():
                if verbose:
                    print(f"[LOAD] {model} {domain}: {llm_path}")
                frames.append(load_combined_with_embeddings(llm_path, source="llm", model_hint=model))
            else:
                print(f"[WARN] Missing LLM file: {llm_path}")

    if not frames:
        raise SystemExit("No feature tables loaded. Abort.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["year"])
    
    if verbose:
        print(f"\n[INFO] Total samples: {len(df)}")
        print(f"[INFO] Year distribution:")
        print(df.groupby(["year", "label"]).size().unstack(fill_value=0))
    
    return df


def get_feature_columns(df: pd.DataFrame, use_all_features: bool) -> list[str]:
    """Get feature columns based on feature set choice."""
    if use_all_features:
        # All features: CE + TF-IDF + SBERT
        ce_cols = [c for c in CE_FEATURES if c in df.columns]
        tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
        sbert_cols = [c for c in df.columns if c.startswith("sbert_")]
        return ce_cols + tfidf_cols + sbert_cols
    else:
        # Only CE features
        return [c for c in CE_FEATURES if c in df.columns]


def make_pipeline() -> Pipeline:
    """Create ML pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )


def run_yearly_validation(df: pd.DataFrame, feature_cols: list[str], domain: Optional[str] = None) -> pd.DataFrame:
    """Run ML validation for each year independently."""
    if domain:
        df = df[df["domain"] == domain].copy()
    
    years = sorted(df["year"].unique())
    results = []
    
    print(f"\n{'='*70}")
    print(f"Yearly ML Validation{' - ' + domain.upper() if domain else ' - All Domains'}")
    print(f"{'='*70}")
    print(f"Feature set: {len(feature_cols)} features")
    print(f"Years: {years}")
    print()
    
    for year in years:
        year_df = df[df["year"] == year].copy()
        
        if len(year_df) == 0:
            continue
        
        # Check class balance
        class_counts = year_df["binary_target"].value_counts()
        if len(class_counts) < 2:
            print(f"⚠️  Year {year}: Only one class present, skipping")
            continue
        
        # Split: 70% train, 30% test (stratified)
        from sklearn.model_selection import train_test_split
        
        X = year_df[feature_cols].values
        y = year_df["binary_target"].values
        
        # Handle missing values
        X = pd.DataFrame(X, columns=feature_cols).fillna(0).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train and evaluate
        pipeline = make_pipeline()
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = np.nan
        
        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        result = {
            "year": int(year),
            "domain": domain or "all",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "human_samples": int((year_df["binary_target"] == 0).sum()),
            "llm_samples": int((year_df["binary_target"] == 1).sum()),
            "test_accuracy": float(accuracy),
            "test_roc_auc": float(roc_auc),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
        }
        
        results.append(result)
        
        print(f"Year {year}: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}, "
              f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, "
              f"Train={len(X_train)}, Test={len(X_test)}")
    
    return pd.DataFrame(results)


def analyze_trends(results_df: pd.DataFrame) -> dict:
    """Analyze if accuracy decreases over years."""
    if len(results_df) < 2:
        return {"trend": "insufficient_data", "slope": None, "correlation": None}
    
    years = results_df["year"].values
    accuracies = results_df["test_accuracy"].values
    
    # Linear regression
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, accuracies)
    
    trend = "decreasing" if slope < 0 else "increasing" if slope > 0 else "stable"
    
    return {
        "trend": trend,
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "yearly_accuracies": {int(y): float(a) for y, a in zip(years, accuracies)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ2 Yearly ML Validation: Test if classification accuracy decreases over years."
    )
    parser.add_argument(
        "--use-all-features",
        action="store_true",
        help="Use all features (CE + TF-IDF + SBERT). Default: CE features only.",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS + ["all"],
        default="all",
        help="Domain to analyze (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path (default: auto-generated).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build dataset
    df = build_dataset(use_all_features=args.use_all_features, verbose=True)
    
    # Get feature columns
    feature_cols = get_feature_columns(df, args.use_all_features)
    print(f"\n[INFO] Using {len(feature_cols)} features")
    if args.use_all_features:
        ce_count = len([c for c in feature_cols if c in CE_FEATURES])
        tfidf_count = len([c for c in feature_cols if c.startswith("tfidf_")])
        sbert_count = len([c for c in feature_cols if c.startswith("sbert_")])
        print(f"  - CE: {ce_count}, TF-IDF: {tfidf_count}, SBERT: {sbert_count}")

    # Run yearly validation
    if args.domain == "all":
        domains_to_process = DOMAINS
    else:
        domains_to_process = [args.domain]

    all_results = []
    all_trends = {}

    for domain in domains_to_process:
        results_df = run_yearly_validation(df, feature_cols, domain=domain)
        if len(results_df) > 0:
            all_results.append(results_df)
            trends = analyze_trends(results_df)
            all_trends[domain] = trends
            
            print(f"\n{domain.upper()} Trend Analysis:")
            print(f"  Trend: {trends['trend']}")
            print(f"  Slope: {trends['slope']:.6f}")
            print(f"  Correlation: {trends['correlation']:.4f}")
            print(f"  p-value: {trends['p_value']:.4f}")

    # Combine results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Overall trend (all domains combined)
        overall_trends = analyze_trends(combined_results.groupby("year")["test_accuracy"].mean().reset_index())
        
        # Save results
        output_path = args.output or (
            RESULTS_DIR / f"rq2_yearly_validation_{'all_features' if args.use_all_features else 'ce_only'}_{args.domain}.json"
        )
        
        summary = {
            "settings": {
                "feature_set": "all_features" if args.use_all_features else "ce_only",
                "domain": args.domain,
                "level": LEVEL,
                "feature_count": len(feature_cols),
            },
            "yearly_results": combined_results.to_dict(orient="records"),
            "trend_analysis": {
                "overall": overall_trends,
                "by_domain": all_trends,
            },
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        
        # Save CSV for easy analysis
        csv_path = output_path.with_suffix(".csv")
        combined_results.to_csv(csv_path, index=False)
        print(f"✅ CSV saved to {csv_path}")
    else:
        print("⚠️  No results generated")


if __name__ == "__main__":
    main()


