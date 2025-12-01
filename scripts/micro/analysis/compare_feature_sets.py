#!/usr/bin/env python3
"""
Compare classification performance using different feature combinations:
1. CE-Var only
2. CE-Var + CE-Geo
3. CE-Var + CE-Geo + TF-IDF-Geo
4. CE-Var + CE-Geo + SBERT-Geo
5. Unified (all features)

This will help determine if TF-IDF/SBERT-Geo are necessary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "trajectory"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B", "LMK")
LEVELS = ("LV1", "LV2", "LV3")

METADATA_COLS = {
    "field",
    "author_id",
    "sample_count",
    "domain",
    "label",
    "provider",
    "level",
    "model",
}

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
    """Load trajectory features."""
    frames: List[pd.DataFrame] = []
    for domain in domains:
        human_path = DATA_ROOT / "human" / domain / "trajectory_features_combined.csv"
        if human_path.exists():
            df_h = pd.read_csv(human_path)
            df_h["domain"] = domain
            df_h["label"] = "human"
            df_h["provider"] = "human"
            df_h["level"] = "LV0"
            frames.append(df_h)
        for provider in models:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["domain"] = domain
                df["label"] = "llm"
                df["provider"] = provider
                df["level"] = level
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def select_feature_sets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Select different feature combinations."""
    # CE Variability
    variability_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and any(col.endswith(suffix) for suffix in VAR_SUFFIXES)
    ]

    # CE Geometry
    ce_geom_cols = [
        col for col in df.columns
        if col.startswith("ce_") and any(col.endswith(metric) for metric in GEOMETRY_METRICS)
    ]

    # TF-IDF Geometry
    tfidf_geom_cols = [
        col for col in df.columns
        if col.startswith("tfidf_") and any(col.endswith(metric) for metric in GEOMETRY_METRICS)
    ]

    # SBERT Geometry
    sbert_geom_cols = [
        col for col in df.columns
        if col.startswith("sbert_") and any(col.endswith(metric) for metric in GEOMETRY_METRICS)
    ]

    feature_sets: Dict[str, pd.DataFrame] = {
        "1_CE-Var_only": df[variability_cols].fillna(0.0),
        "2_CE-Var+CE-Geo": df[variability_cols + ce_geom_cols].fillna(0.0),
        "3_CE-Var+CE-Geo+TFIDF-Geo": df[variability_cols + ce_geom_cols + tfidf_geom_cols].fillna(0.0),
        "4_CE-Var+CE-Geo+SBERT-Geo": df[variability_cols + ce_geom_cols + sbert_geom_cols].fillna(0.0),
        "5_Unified": df[[c for c in df.select_dtypes(include=[np.number]).columns 
                         if c not in {"author_id", "sample_count"}]].fillna(0.0),
    }

    return feature_sets


def evaluate(df: pd.DataFrame, feature_sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate classification performance for each feature set."""
    y = (df["label"] == "human").astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: List[Dict] = []

    for name, X_df in feature_sets.items():
        X = X_df.to_numpy()
        accs: List[float] = []
        rocs: List[float] = []
        f1s: List[float] = []

        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            accs.append(accuracy_score(y[test_idx], y_pred))
            rocs.append(roc_auc_score(y[test_idx], proba))
            f1s.append(f1_score(y[test_idx], y_pred))

        rows.append(
            {
                "feature_set": name,
                "n_features": X_df.shape[1],
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "roc_auc_mean": np.mean(rocs),
                "roc_auc_std": np.std(rocs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
                "n_samples": len(df),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare different feature set combinations.")
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
        default=["DS", "G4B", "G12B"],  # Exclude LMK by default
        help="LLM models to include (default: DS, G4B, G12B).",
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
    print(f"LLM models: {', '.join(args.models)}")
    print(f"LLM level: {args.level}")

    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available.")
        return

    print(f"Loaded {len(df)} samples ({len(df[df['label']=='human'])} human, {len(df[df['label']=='llm'])} LLM)")

    feature_sets = select_feature_sets(df)
    print(f"\nFeature sets:")
    for name, X_df in feature_sets.items():
        print(f"  {name}: {X_df.shape[1]} features")

    print("\nEvaluating classification performance...")
    results = evaluate(df, feature_sets)

    # Print results
    print("\n" + "="*80)
    print("CLASSIFICATION PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Feature Set':<35} {'# Features':<12} {'Accuracy':<15} {'ROC-AUC':<15} {'F1-Score':<15}")
    print("-"*80)

    for _, row in results.iterrows():
        print(
            f"{row['feature_set']:<35} "
            f"{int(row['n_features']):<12} "
            f"{row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}   "
            f"{row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}   "
            f"{row['f1_mean']:.3f}±{row['f1_std']:.3f}"
        )

    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)

    baseline = results[results['feature_set'] == '1_CE-Var_only'].iloc[0]
    ce_geo = results[results['feature_set'] == '2_CE-Var+CE-Geo'].iloc[0]
    unified = results[results['feature_set'] == '5_Unified'].iloc[0]

    print(f"\n1. CE-Var only:")
    print(f"   Accuracy: {baseline['accuracy_mean']:.3f}, ROC-AUC: {baseline['roc_auc_mean']:.3f}")

    print(f"\n2. Adding CE-Geo (+{int(ce_geo['n_features'] - baseline['n_features'])} features):")
    acc_improve = ce_geo['accuracy_mean'] - baseline['accuracy_mean']
    auc_improve = ce_geo['roc_auc_mean'] - baseline['roc_auc_mean']
    print(f"   Accuracy: {ce_geo['accuracy_mean']:.3f} ({acc_improve:+.3f})")
    print(f"   ROC-AUC: {ce_geo['roc_auc_mean']:.3f} ({auc_improve:+.3f})")

    print(f"\n3. Adding TF-IDF/SBERT-Geo (Unified, +{int(unified['n_features'] - ce_geo['n_features'])} features):")
    acc_improve = unified['accuracy_mean'] - ce_geo['accuracy_mean']
    auc_improve = unified['roc_auc_mean'] - ce_geo['roc_auc_mean']
    print(f"   Accuracy: {unified['accuracy_mean']:.3f} ({acc_improve:+.3f})")
    print(f"   ROC-AUC: {unified['roc_auc_mean']:.3f} ({auc_improve:+.3f})")

    # Save results
    output_dir = PLOTS_ROOT / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feature_set_comparison_{'_'.join(args.domains)}_{args.level}.csv"
    results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

