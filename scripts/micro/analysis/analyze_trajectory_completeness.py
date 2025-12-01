#!/usr/bin/env python3
"""
Analyze whether CE-Var + CE-Geo alone can constitute a complete trajectory,
or if multiple spaces (TF-IDF/SBERT) are necessary.

Key questions:
1. Do CE-Var and CE-Geo capture the same information?
2. Do different spaces (CE, TF-IDF, SBERT) provide complementary information?
3. Is a multi-space trajectory more complete than single-space?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "dataset" / "process"
DOMAINS = ("academic", "blogs", "news")
PROVIDERS = ("DS", "G4B", "G12B")
LEVELS = ("LV1", "LV2", "LV3")

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
            frames.append(df_h)
        for provider in models:
            csv_path = DATA_ROOT / "LLM" / provider / level / domain / "trajectory_features_combined.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["domain"] = domain
                df["label"] = "llm"
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_feature_correlations(df: pd.DataFrame) -> None:
    """Analyze correlations between CE-Var, CE-Geo, TF-IDF-Geo, SBERT-Geo."""
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Extract feature groups
    ce_var_cols = [c for c in df.columns if any(c.endswith(s) for s in VAR_SUFFIXES)]
    ce_geo_cols = [c for c in df.columns if c.startswith("ce_") and any(c.endswith(m) for m in GEOMETRY_METRICS)]
    tfidf_geo_cols = [c for c in df.columns if c.startswith("tfidf_") and any(c.endswith(m) for m in GEOMETRY_METRICS)]
    sbert_geo_cols = [c for c in df.columns if c.startswith("sbert_") and any(c.endswith(m) for m in GEOMETRY_METRICS)]
    
    print(f"\nFeature counts:")
    print(f"  CE-Var: {len(ce_var_cols)}")
    print(f"  CE-Geo: {len(ce_geo_cols)}")
    print(f"  TF-IDF-Geo: {len(tfidf_geo_cols)}")
    print(f"  SBERT-Geo: {len(sbert_geo_cols)}")
    
    # Calculate average correlations between groups
    print("\n" + "-"*80)
    print("Average correlations between feature groups:")
    print("-"*80)
    
    def avg_correlation(group1_cols, group2_cols, name1, name2):
        correlations = []
        for col1 in group1_cols[:10]:  # Sample to avoid too many
            for col2 in group2_cols:
                try:
                    corr, p = pearsonr(df[col1].fillna(0), df[col2].fillna(0))
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    pass
        if correlations:
            avg_corr = np.mean(correlations)
            print(f"  {name1} vs {name2}: {avg_corr:.3f} (n={len(correlations)})")
    
    avg_correlation(ce_var_cols, ce_geo_cols, "CE-Var", "CE-Geo")
    avg_correlation(ce_var_cols, tfidf_geo_cols, "CE-Var", "TF-IDF-Geo")
    avg_correlation(ce_var_cols, sbert_geo_cols, "CE-Var", "SBERT-Geo")
    avg_correlation(ce_geo_cols, tfidf_geo_cols, "CE-Geo", "TF-IDF-Geo")
    avg_correlation(ce_geo_cols, sbert_geo_cols, "CE-Geo", "SBERT-Geo")
    avg_correlation(tfidf_geo_cols, sbert_geo_cols, "TF-IDF-Geo", "SBERT-Geo")


def analyze_space_separation(df: pd.DataFrame) -> None:
    """Analyze how well different spaces separate Human vs LLM."""
    print("\n" + "="*80)
    print("SPACE SEPARATION ANALYSIS")
    print("="*80)
    print("How well does each space separate Human vs LLM?")
    print("-"*80)
    
    spaces = {
        "CE-Var": [c for c in df.columns if any(c.endswith(s) for s in VAR_SUFFIXES)],
        "CE-Geo": [c for c in df.columns if c.startswith("ce_") and any(c.endswith(m) for m in GEOMETRY_METRICS)],
        "CE-Var+Geo": [c for c in df.columns if (any(c.endswith(s) for s in VAR_SUFFIXES) or 
                                                  (c.startswith("ce_") and any(c.endswith(m) for m in GEOMETRY_METRICS)))],
        "TF-IDF-Geo": [c for c in df.columns if c.startswith("tfidf_") and any(c.endswith(m) for m in GEOMETRY_METRICS)],
        "SBERT-Geo": [c for c in df.columns if c.startswith("sbert_") and any(c.endswith(m) for m in GEOMETRY_METRICS)],
    }
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    
    y = (df["label"] == "human").astype(int).values
    
    print(f"{'Space':<20} {'# Features':<12} {'Accuracy':<15} {'ROC-AUC':<15}")
    print("-"*80)
    
    for space_name, cols in spaces.items():
        if not cols:
            continue
        X = df[cols].fillna(0.0).to_numpy()
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, rocs = [], []
        
        for train_idx, test_idx in skf.split(X, y):
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            accs.append(accuracy_score(y[test_idx], y_pred))
            rocs.append(roc_auc_score(y[test_idx], proba))
        
        print(f"{space_name:<20} {len(cols):<12} {np.mean(accs):.3f}±{np.std(accs):.3f}   {np.mean(rocs):.3f}±{np.std(rocs):.3f}")


def analyze_pca_variance(df: pd.DataFrame) -> None:
    """Analyze variance explained by PCA in different feature combinations."""
    print("\n" + "="*80)
    print("PCA VARIANCE ANALYSIS")
    print("="*80)
    print("How much variance is explained by first 2 PCs in different spaces?")
    print("-"*80)
    
    feature_combos = {
        "CE-Var only": [c for c in df.columns if any(c.endswith(s) for s in VAR_SUFFIXES)],
        "CE-Var + CE-Geo": [c for c in df.columns if (any(c.endswith(s) for s in VAR_SUFFIXES) or 
                                                       (c.startswith("ce_") and any(c.endswith(m) for m in GEOMETRY_METRICS)))],
        "CE-Var + CE-Geo + TF-IDF-Geo": [c for c in df.columns if (any(c.endswith(s) for s in VAR_SUFFIXES) or 
                                                                    (c.startswith("ce_") and any(c.endswith(m) for m in GEOMETRY_METRICS)) or
                                                                    (c.startswith("tfidf_") and any(c.endswith(m) for m in GEOMETRY_METRICS)))],
        "All spaces": [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in {"author_id", "sample_count"}],
    }
    
    print(f"{'Feature Combination':<40} {'PC1':<10} {'PC2':<10} {'PC1+PC2':<10}")
    print("-"*80)
    
    for combo_name, cols in feature_combos.items():
        if not cols:
            continue
        X = df[cols].fillna(0.0).to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X_scaled)
        
        print(f"{combo_name:<40} {pca.explained_variance_ratio_[0]:.3f}     {pca.explained_variance_ratio_[1]:.3f}     {pca.explained_variance_ratio_.sum():.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trajectory completeness.")
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
        default=list(PROVIDERS),
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
    
    print("Loading data...")
    df = load_samples(args.domains, args.models, args.level)
    if df.empty:
        print("⚠ No data available.")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # Run analyses
    analyze_feature_correlations(df)
    analyze_space_separation(df)
    analyze_pca_variance(df)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
    A complete trajectory description likely requires:
    1. Multiple representation spaces (CE, TF-IDF, SBERT) because:
       - Each space captures different aspects of text (handcrafted vs lexical vs semantic)
       - Low correlation between spaces suggests complementary information
       - Multi-space provides more robust trajectory characterization
    
    2. Both variability and geometry because:
       - Variability describes "how much" change occurs
       - Geometry describes "what shape" the path takes
       - Together they describe the full trajectory structure
    
    3. CE-Var + CE-Geo alone may be incomplete because:
       - CE space is only one perspective (handcrafted features)
       - Different spaces may reveal different trajectory patterns
       - Multi-space trajectory is more robust to space-specific biases
    """)


if __name__ == "__main__":
    main()

