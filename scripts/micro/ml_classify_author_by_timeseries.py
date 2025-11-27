#!/usr/bin/env python3
"""
Machine Learning Classification: Identify Human vs LLM Authors based on Time Series Features.

Research Question: Can we classify whether an author is Human or LLM based on their 
time series statistics (CV, RMSSD, MASD)?

Approach:
1. Extract CV, RMSSD, MASD features from author_variance.csv files
2. Label: Human = 1, LLM = 0
3. Train classification models for each statistic (CV, RMSSD, MASD) and combined
4. Evaluate model performance (accuracy, AUC, etc.)
5. Compare with variance-based model
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    PerfectSeparationError = Exception
    STATSMODELS_AVAILABLE = False

# Try to import XGBoost if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

DATA_ROOT = Path("dataset")
PROCESS_ROOT = DATA_ROOT / "process"
DEFAULT_MODELS = ["DS", "G4B", "G12B", "LMK"]
DEFAULT_LEVEL = "LV1"
# Primary statistics (normalized, length-insensitive)
STAT_NAMES = ["cv", "rmssd_norm", "masd_norm"]
# Original statistics (length-sensitive) - kept for reference but not used in main analysis
STAT_NAMES_ORIGINAL = ["variance", "rmssd", "masd"]

# All feature columns for merged version (base names without stat suffix)
# Merged version: 5 Big5 + 15 NELA (6 Emotional + 9 Stylistic)
ALL_FEATURE_COLUMNS_MERGED = [
    # Big5 (5 features)
    "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
    # NELA Emotional (6 features)
    "polarity", "subjectivity", "vader_compound", "vader_pos", "vader_neu", "vader_neg",
    # NELA Stylistic (9 features)
    "word_diversity", "flesch_reading_ease", "gunning_fog", "average_word_length",
    "num_words", "avg_sentence_length", "verb_ratio", "function_word_ratio", "content_word_ratio"
]

# Old enhanced version features (for backward compatibility)
ALL_FEATURE_COLUMNS_ENHANCED = [
    "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
    "polarity", "subjectivity", "vader_compound", "vader_pos", "vader_neu", "vader_neg",
    "word_diversity", "flesch_reading_ease", "average_word_length", "sentence_density",
    "punctuation_density", "smog_index", "verb_ratio"
]

# Default to merged version
ALL_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS_MERGED


def load_timeseries_data(domain: str, source: str, model: str = None, level: str = None, 
                         use_merged: bool = True, normalize_approach: str = None,
                         outliers_removed: bool = False, outlier_strategy: str = None,
                         outlier_method: str = None) -> pd.DataFrame:
    """
    Load time series statistics data (variance, CV, RMSSD, MASD).
    
    Args:
        domain: Domain name (academic, news, blogs)
        source: "human" or "llm"
        model: Model name (required for LLM)
        level: Level name (required for LLM)
        use_merged: If True, use merged version (default: True)
        normalize_approach: None, "A" (normalize values first), or "B" (normalize statistics)
        outliers_removed: If True, load outliers-removed version
        outlier_strategy: Strategy used for outlier removal (feature, model_feature, etc.)
        outlier_method: Method used for outlier removal (iqr, mad, etc.)
    
    Returns:
        DataFrame with time series statistics
    """
    if outliers_removed:
        # When outliers are removed, use merged file which contains normalized versions
        # (generated from combined_merged_outliers_removed.csv using generate_timeseries_stats_from_outliers_removed.py)
        # This file contains both original and normalized statistics (cv, rmssd_norm, masd_norm)
        filename = "author_timeseries_stats_merged.csv" if use_merged else "author_timeseries_stats.csv"
    elif normalize_approach is None:
        filename = "author_timeseries_stats_merged.csv" if use_merged else "author_timeseries_stats.csv"
    elif normalize_approach == "A":
        filename = "author_timeseries_stats_normalized_A.csv"
    elif normalize_approach == "B":
        filename = "author_timeseries_stats_normalized_B.csv"
    else:
        raise ValueError(f"Unknown normalize_approach: {normalize_approach}")
    
    if source == "human":
        path = PROCESS_ROOT / "human" / domain / filename
    elif source == "llm":
        if not model or not level:
            raise ValueError("model and level required for LLM")
        path = PROCESS_ROOT / "LLM" / model.upper() / level.upper() / domain / filename
    else:
        raise ValueError(f"Unknown source: {source}")
    
    if not path.exists():
        raise FileNotFoundError(f"Time series file not found: {path}")
    
    df = pd.read_csv(path)
    return df


def extract_feature_columns(df: pd.DataFrame, stat_name: str) -> List[str]:
    """Extract feature columns for a specific statistic (e.g., '_cv', '_rmssd', '_masd', '_variance')."""
    metadata_cols = {"field", "author_id", "sample_count", "model", "level", "domain"}
    
    # Find columns ending with the statistic name
    stat_cols = [col for col in df.columns 
                if col.endswith(f"_{stat_name}") 
                and col not in metadata_cols]
    
    # Extract base feature names (remove stat suffix)
    base_features = [col.replace(f"_{stat_name}", "") for col in stat_cols]
    
    # Return only columns that match known features
    feature_cols = [col for col in stat_cols 
                   if col.replace(f"_{stat_name}", "") in ALL_FEATURE_COLUMNS]
    
    return feature_cols


def prepare_classification_data(
    domains: List[str],
    models: List[str],
    level: str,
    stat_name: str,
    use_merged: bool = True,
    normalize_approach: str = None,
    outliers_removed: bool = False,
    outlier_strategy: str = None,
    outlier_method: str = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for classification using a specific statistic (CV, RMSSD, MASD, or variance).
    
    Args:
        domains: List of domain names (e.g., ["academic", "news", "blogs"])
        models: List of LLM models
        level: LLM level
        stat_name: Statistic name ('cv', 'rmssd', 'masd', or 'variance')
        use_merged: If True, use merged version features (default: True)
        normalize_approach: None, "A" (normalize values first), or "B" (normalize statistics)
        outliers_removed: If True, use outliers-removed version
        outlier_strategy: Strategy used for outlier removal
        outlier_method: Method used for outlier removal
    
    Returns:
        DataFrame with columns: [domain, field, author_id, feature1_stat, ..., featureN_stat, label]
        List of feature column names
    """
    global ALL_FEATURE_COLUMNS
    # Set feature columns based on version
    ALL_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS_MERGED if use_merged else ALL_FEATURE_COLUMNS_ENHANCED
    
    all_human_data = []
    all_llm_data = []
    all_features = set()
    
    # Collect data from all domains
    for domain in domains:
        print(f"\n{'='*70}")
        version_tag = "MERGED" if use_merged else "ENHANCED"
        outlier_tag = f" (Outliers removed: {outlier_strategy}_{outlier_method})" if outliers_removed else ""
        # Update outlier tag to reflect that only LLM data uses outliers removal
        if outliers_removed:
            outlier_tag = f" (LLM: Outliers removed: {outlier_strategy}_{outlier_method}; Human: Original)"
        else:
            outlier_tag = ""
        print(f"Processing domain: {domain.upper()} (Statistic: {stat_name.upper()}, Version: {version_tag}{outlier_tag})")
        print(f"{'='*70}")
        
        # Load human data
        # NOTE: Human data does NOT need outliers removal (already verified/checked)
        # Only LLM data uses outliers-removed version
        # 
        # Rationale for outlier removal in LLM data:
        # - Some models (especially LMK) reach max token limit, causing generation failures
        # - This produces either irrelevant content or content with incorrect length
        # - These failures significantly change some authors' feature values
        # - Outliers represent technical errors (data quality issues), not statistical anomalies
        # - Removal strategy: per-model, per-feature (a sample may be outlier for one feature
        #   but valid for another feature)
        # 
        # Test results show: Human Original achieves 92.99% avg accuracy vs 90.61% when both removed
        print(f"Loading human time series data...")
        try:
            human_df = load_timeseries_data(
                domain, "human", 
                use_merged=use_merged, 
                normalize_approach=normalize_approach,
                outliers_removed=False,  # Human data: always use original (no outliers removal)
                outlier_strategy=None,
                outlier_method=None
            )
            features = extract_feature_columns(human_df, stat_name)
            
            if not features:
                print(f"Warning: No {stat_name} features found for domain {domain}, skipping...")
                continue
            
            all_features.update(features)
            
            # Prepare human data
            human_data = human_df[["field", "author_id"] + features].copy()
            human_data["label"] = 1  # Human = 1
            human_data["source"] = "human"
            human_data["domain"] = domain
            
            all_human_data.append(human_data)
            print(f"  Human authors: {len(human_data)} with {len(features)} features")
            
            # Prepare LLM data (combine all models)
            for model in models:
                print(f"Loading LLM time series data for model: {model}")
                try:
                    llm_df = load_timeseries_data(
                        domain, "llm", 
                        model=model, 
                        level=level, 
                        use_merged=use_merged, 
                        normalize_approach=normalize_approach,
                        outliers_removed=outliers_removed,
                        outlier_strategy=outlier_strategy,
                        outlier_method=outlier_method
                    )
                    
                    # Ensure same features
                    available_features = extract_feature_columns(llm_df, stat_name)
                    common_features = [f for f in features if f in available_features]
                    
                    if len(common_features) < len(features):
                        print(f"  Warning: Model {model} missing some features. Using {len(common_features)}/{len(features)} features.")
                    
                    llm_model_data = llm_df[["field", "author_id"] + common_features].copy()
                    llm_model_data["label"] = 0  # LLM = 0
                    llm_model_data["source"] = f"llm_{model}"
                    llm_model_data["domain"] = domain
                    
                    all_llm_data.append(llm_model_data)
                    print(f"    {model} authors: {len(llm_model_data)}")
                except FileNotFoundError as e:
                    print(f"  Warning: {e}, skipping model {model} for domain {domain}")
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping domain {domain}")
            continue
    
    if not all_human_data:
        raise ValueError("No human data loaded from any domain")
    
    if not all_llm_data:
        raise ValueError("No LLM data loaded from any domain")
    
    # Find common features across all domains
    common_features = sorted(list(all_features))
    
    # Combine all human data
    human_combined = pd.concat(all_human_data, ignore_index=True)
    
    # Combine all LLM data
    llm_combined = pd.concat(all_llm_data, ignore_index=True)
    
    # Ensure all dataframes have the same feature columns (fill missing with NaN)
    for df in [human_combined, llm_combined]:
        for feat in common_features:
            if feat not in df.columns:
                df[feat] = np.nan
    
    # Combine human and LLM data
    all_data = pd.concat([
        human_combined[["domain", "field", "author_id"] + common_features + ["label", "source"]],
        llm_combined[["domain", "field", "author_id"] + common_features + ["label", "source"]]
    ], ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"COMBINED DATA SUMMARY ({stat_name.upper()})")
    print(f"{'='*70}")
    print(f"Total samples: {len(all_data)}")
    print(f"  Human (label=1): {(all_data['label'] == 1).sum()}")
    print(f"  LLM (label=0): {(all_data['label'] == 0).sum()}")
    print(f"\nBy domain:")
    for domain in domains:
        domain_data = all_data[all_data["domain"] == domain]
        if len(domain_data) > 0:
            print(f"  {domain}: {len(domain_data)} samples "
                  f"(Human: {(domain_data['label'] == 1).sum()}, "
                  f"LLM: {(domain_data['label'] == 0).sum()})")
    print(f"\nFeatures used: {len(common_features)}")
    
    return all_data, common_features


def train_and_evaluate_models_unified_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    features: List[str],
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    stat_name: str,
    random_state: int = 42
) -> Dict:
    """
    Train multiple classification models and evaluate performance.
    Note: Uses pre-split train/test data (same split across all statistics).
    """
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"  Human: {(y_train == 1).sum()} train, {(y_test == 1).sum()} test")
    print(f"  LLM: {(y_train == 0).sum()} train, {(y_test == 0).sum()} test")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": (LogisticRegression(random_state=random_state, max_iter=1000), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1), False),
        "SVM": (SVC(probability=True, random_state=random_state, kernel="rbf"), True),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=random_state), False),
        "Naive Bayes": (GaussianNB(), True),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=random_state, max_depth=10), False),
        "Neural Network (MLP)": (MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=random_state), True),
        "AdaBoost": (AdaBoostClassifier(n_estimators=50, random_state=random_state), False),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric="logloss"), False)
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE ({stat_name.upper()})")
    print("="*70)
    
    for model_name, (model, use_scaled) in models.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        
        # Use scaled or original data
        if use_scaled:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Train
        model.fit(X_tr, y_train)
        
        # Predict
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 1:
            TN = FP = FN = TP = 0
            if y_pred[0] == 0:
                TN = len(y_pred)
            else:
                TP = len(y_pred)
        elif cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
        
        # ROC AUC
        auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) > 1:
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                pass
        
        # Cross-validation
        cv_scores = None
        try:
            if len(np.unique(y_train)) == 2 and min(pd.Series(y_train).value_counts()) >= 5:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
            else:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
        except:
            pass
        
        # Feature importance
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(features, model.feature_importances_))
        elif hasattr(model, "coef_"):
            try:
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                # Normalize coef_ to percentages for consistency
                coef_abs = np.abs(coef)
                coef_total = coef_abs.sum()
                if coef_total > 0:
                    coef_normalized = coef_abs / coef_total  # Normalize to [0, 1] range
                else:
                    coef_normalized = coef_abs
                feature_importance = dict(zip(features, coef_normalized))
            except (AttributeError, IndexError, ValueError) as e:
                # coef_ extraction failed, will use permutation importance as fallback
                pass
        
        # If feature importance is still None, use permutation importance as fallback
        if feature_importance is None:
            try:
                # Use permutation importance for models without direct feature importance
                perm_importance = permutation_importance(
                    model, X_te, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
                )
                # Normalize to percentages
                perm_scores = perm_importance.importances_mean
                perm_total = perm_scores.sum()
                if perm_total > 0:
                    perm_normalized = perm_scores / perm_total
                else:
                    perm_normalized = perm_scores
                feature_importance = dict(zip(features, perm_normalized))
            except Exception as e:
                # Permutation importance failed, feature_importance remains None
                pass
        
        # Store results
        results[model_name] = {
            "accuracy": accuracy,
            "auc": auc,
            "cv_mean": cv_scores.mean() if cv_scores is not None else None,
            "cv_std": cv_scores.std() if cv_scores is not None else None,
            "confusion_matrix": {"TN": TN, "FP": FP, "FN": FN, "TP": TP},
            "feature_importance": feature_importance
        }
        
        # Print results
        print(f"  Accuracy: {accuracy:.4f}")
        if auc is not None:
            print(f"  ROC AUC: {auc:.4f}")
        if cv_scores is not None:
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
        
        # Print top features if available
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 5 Features:")
            for feat, imp in sorted_features[:5]:
                # Remove stat suffix for display
                feat_display = feat.replace(f"_{stat_name}", "")
                print(f"    {feat_display}: {imp:.4f}")
    
    return results


def compute_lr_pvalues_df(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Fit a statsmodels logistic regression (with intercept) and return coefficient stats.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is not available. Install statsmodels to compute LR p-values.")
    
    X_with_const = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_with_const)
    result = model.fit(disp=False, maxiter=1000)
    
    summary_df = pd.DataFrame({
        "feature": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values,
    })
    summary_df["odds_ratio"] = np.exp(summary_df["coef"])
    return summary_df


def compare_statistics(
    all_results: Dict[str, Dict],
    output_dir: Path = Path("results")
):
    """Compare performance across different statistics (variance, CV, RMSSD, MASD) and concatenated model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison DataFrame
    comparison_data = []
    for stat_name, results in all_results.items():
        stat_display = "CONCATENATED (All Stats)" if stat_name == "concatenated" else stat_name.upper()
        for model_name, result in results.items():
            comparison_data.append({
                "Statistic": stat_display,
                "Model": model_name,
                "Accuracy": result["accuracy"],
                "ROC_AUC": result["auc"] if result["auc"] is not None else np.nan,
                "CV_Mean": result["cv_mean"] if result["cv_mean"] is not None else np.nan,
                "CV_Std": result["cv_std"] if result["cv_std"] is not None else np.nan,
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_path = output_dir / "timeseries_statistics_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    acc_pivot = comparison_df.pivot(index="Model", columns="Statistic", values="Accuracy")
    acc_pivot.plot(kind="bar", ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title("Accuracy Comparison by Statistic", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy", fontsize=12)
    axes[0, 0].set_xlabel("Model", fontsize=12)
    axes[0, 0].legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].tick_params(axis="x", rotation=45)
    
    # 2. ROC AUC comparison
    auc_pivot = comparison_df.pivot(index="Model", columns="Statistic", values="ROC_AUC")
    auc_pivot.plot(kind="bar", ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title("ROC AUC Comparison by Statistic", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylabel("ROC AUC", fontsize=12)
    axes[0, 1].set_xlabel("Model", fontsize=12)
    axes[0, 1].legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].tick_params(axis="x", rotation=45)
    
    # 3. Average accuracy by statistic
    avg_acc = comparison_df.groupby("Statistic")["Accuracy"].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(avg_acc)), avg_acc.values, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    axes[1, 0].set_xticks(range(len(avg_acc)))
    axes[1, 0].set_xticklabels(avg_acc.index)
    axes[1, 0].set_title("Average Accuracy by Statistic", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Average Accuracy", fontsize=12)
    axes[1, 0].set_xlabel("Statistic", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for i, (stat, acc) in enumerate(avg_acc.items()):
        axes[1, 0].text(i, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    # 4. Average ROC AUC by statistic
    avg_auc = comparison_df.groupby("Statistic")["ROC_AUC"].mean().sort_values(ascending=False)
    axes[1, 1].bar(range(len(avg_auc)), avg_auc.values, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    axes[1, 1].set_xticks(range(len(avg_auc)))
    axes[1, 1].set_xticklabels(avg_auc.index)
    axes[1, 1].set_title("Average ROC AUC by Statistic", fontsize=14, fontweight="bold")
    axes[1, 1].set_ylabel("Average ROC AUC", fontsize=12)
    axes[1, 1].set_xlabel("Statistic", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for i, (stat, auc) in enumerate(avg_auc.items()):
        axes[1, 1].text(i, auc + 0.01, f"{auc:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    comparison_plot_path = output_dir / "timeseries_statistics_comparison.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison plot saved to: {comparison_plot_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("STATISTICS COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Statistic':<25} {'Avg Accuracy':<15} {'Avg ROC AUC':<15}")
    print("-" * 70)
    # Use actual statistics from the comparison DataFrame
    unique_stats = sorted(comparison_df["Statistic"].unique())
    for stat in unique_stats:
        stat_data = comparison_df[comparison_df["Statistic"] == stat]
        if len(stat_data) > 0:
            avg_acc = stat_data["Accuracy"].mean()
            avg_auc = stat_data["ROC_AUC"].mean()
            print(f"{stat:<25} {avg_acc:<15.4f} {avg_auc:<15.4f}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Classify Human vs LLM authors based on time series statistics (CV, RMSSD, MASD)."
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        help="Domain name(s) (e.g., academic, news, blogs). Can specify multiple domains."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Alternative way to specify multiple domains (same as --domain)."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=f"LLM models to include (default: {' '.join(DEFAULT_MODELS)})."
    )
    parser.add_argument(
        "--level",
        default=DEFAULT_LEVEL,
        help=f"LLM level (default: {DEFAULT_LEVEL})."
    )
    parser.add_argument(
        "--stat",
        choices=STAT_NAMES,
        help="Specific statistic to use (cv, rmssd, masd, variance). If not specified, analyze all statistics."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("micro_results/ml"),
        help="Output directory for results (default: micro_results/ml)."
    )
    parser.add_argument(
        "--normalize-approach",
        choices=["A", "B"],
        help="Normalization approach: A (normalize values first), B (normalize statistics). If not specified, use original (no normalization)."
    )
    parser.add_argument(
        "--outliers-removed",
        action="store_true",
        help="Use outliers-removed version of time series statistics."
    )
    parser.add_argument(
        "--outlier-strategy",
        choices=["global", "feature", "model_feature", "domain_feature"],
        help="Outlier removal strategy (required if --outliers-removed is set)."
    )
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "zscore", "mad", "percentile"],
        help="Outlier removal method (required if --outliers-removed is set)."
    )
    parser.add_argument(
        "--skip-lr-pvalues",
        action="store_true",
        help="Disable statsmodels logistic regression with p-value reporting."
    )
    
    args = parser.parse_args()
    
    # Validate outliers arguments
    if args.outliers_removed:
        if not args.outlier_strategy or not args.outlier_method:
            parser.error("--outlier-strategy and --outlier-method are required when --outliers-removed is set.")
    
    with_lr_pvalues = not args.skip_lr_pvalues
    if with_lr_pvalues and not STATSMODELS_AVAILABLE:
        print("⚠️ statsmodels is not installed. Skipping logistic regression p-value computation.")
        with_lr_pvalues = False
    
    # Get domains list
    domains = []
    if args.domains:
        domains.extend(args.domains)
    if args.domain:
        domains.extend(args.domain)
    
    if not domains:
        raise ValueError("Please specify at least one domain using --domain or --domains")
    
    domains = [d.lower() for d in domains]
    models = [m.upper() for m in args.models]
    level = args.level.upper()
    level_tag = level.lower()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which statistics to analyze (default: normalized versions)
    stats_to_analyze = [args.stat] if args.stat else ["cv", "rmssd_norm", "masd_norm"]
    
    # Determine normalization approach
    normalize_approach = args.normalize_approach
    
    # Determine which version to use
    use_merged = True  # Default to merged version
    version_tag = "MERGED"
    
    if normalize_approach:
        if normalize_approach == "A":
            norm_tag = " (Normalized Approach A: normalize values first, then compute statistics)"
        elif normalize_approach == "B":
            norm_tag = " (Normalized Approach B: normalize statistics, normalized_variance = CV^2)"
    else:
        norm_tag = " (Original - No normalization)"
    
    print("="*70)
    print(f"AUTHOR CLASSIFICATION: Human vs LLM (Time Series Statistics)")
    print(f"Version: {version_tag}{norm_tag}")
    print(f"Domain(s): {', '.join([d.upper() for d in domains])}")
    print(f"LLM Models: {', '.join(models)}")
    print(f"LLM Level: {level}")
    print(f"Statistics to analyze: {', '.join([s.upper() for s in stats_to_analyze])}")
    print("="*70)
    
    # Store results for all statistics
    all_results = {}
    
    # ==========================================
    # STEP 1: Find common samples across all statistics
    # ==========================================
    print(f"\n{'='*70}")
    print("STEP 1: Finding common samples across all statistics")
    print(f"{'='*70}")
    
    # Load data for all statistics to find common samples
    all_stat_data = {}
    all_stat_features = {}
    
    for stat_name in stats_to_analyze:
        print(f"\nLoading data for {stat_name.upper()}...")
        data, features = prepare_classification_data(
            domains, models, level, stat_name, 
            use_merged=use_merged, 
            normalize_approach=normalize_approach,
            outliers_removed=args.outliers_removed,
            outlier_strategy=args.outlier_strategy,
            outlier_method=args.outlier_method
        )
        
        # Create identifier for each sample
        # Note: author_id is field-specific (academic_CS_author_id=1 != academic_MEDICINE_author_id=1, they are different people)
        # sample_id = domain + field + author_id + source
        #   - Human: "academic_PHYSICS_17_human"
        #   - LLM: "academic_PHYSICS_17_llm_DS", "academic_PHYSICS_17_llm_G4B", etc.
        # This ensures each sample_id is unique (unique author-source combination) and has consistent label
        data['sample_id'] = (
            data['domain'].astype(str) + '_' + 
            data['field'].astype(str) + '_' + 
            data['author_id'].astype(str) + '_' + 
            data['source'].astype(str)
        )
        
        # Check for missing values in features
        X_temp = data[features]
        missing_mask = X_temp.isna().any(axis=1)
        
        if missing_mask.sum() > 0:
            print(f"  Removing {missing_mask.sum()} rows with missing values in {stat_name}")
            data_clean = data[~missing_mask].copy()
        else:
            data_clean = data.copy()
        
        # Store samples with complete data
        complete_samples = set(data_clean['sample_id'].values)
        all_stat_data[stat_name] = {
            'data': data_clean,
            'features': features,
            'complete_samples': complete_samples
        }
        all_stat_features[stat_name] = features
        
        print(f"  Complete samples for {stat_name}: {len(complete_samples)}")
    
    # Find intersection of all complete samples
    common_samples = set.intersection(*[info['complete_samples'] for info in all_stat_data.values()])
    
    print(f"\n{'='*70}")
    print(f"Common samples (present in all statistics): {len(common_samples)}")
    print(f"{'='*70}")
    
    if len(common_samples) == 0:
        raise ValueError("No common samples found across all statistics! Cannot ensure fair comparison.")
    
    # ==========================================
    # STEP 2: Create unified train/test split on common samples (by AUTHOR, not by sample)
    # ==========================================
    print(f"\n{'='*70}")
    print("STEP 2: Creating unified train/test split (by SAMPLE_ID)")
    print(f"{'='*70}")
    
    # Use the first statistic's data to create split
    first_stat = stats_to_analyze[0]
    first_data = all_stat_data[first_stat]['data']
    
    # Filter to common samples only
    common_mask = first_data['sample_id'].isin(common_samples)
    common_data = first_data[common_mask].copy()
    
    # Note: sample_id = domain + field + author_id + source (e.g., "academic_PHYSICS_17_human" or "academic_PHYSICS_17_llm_DS")
    # author_id is field-specific (academic_CS_author_id=1 != academic_MEDICINE_author_id=1, they are different people)
    # source distinguishes Human from different LLM models, ensuring each sample_id has consistent label
    # So we can directly split by sample_id without worrying about data leakage
    
    # Get labels for each sample_id (should be consistent since each sample_id is a unique author-source combination)
    sample_labels = {}
    for sample_id in common_samples:
        sample_data = common_data[common_data['sample_id'] == sample_id]
        # Label should be consistent for all rows with the same sample_id
        # (each sample_id corresponds to either a Human author or an LLM model simulating an author)
        labels = sample_data['label'].unique()
        if len(labels) > 1:
            raise ValueError(f"Inconsistent labels for sample_id={sample_id}: {labels}")
        sample_labels[sample_id] = labels[0]
    
    # Convert to arrays for train_test_split
    sample_ids_list = list(common_samples)
    sample_labels_list = [sample_labels[sid] for sid in sample_ids_list]
    
    # Perform train_test_split on SAMPLE_IDs directly
    train_sample_ids, test_sample_ids = train_test_split(
        sample_ids_list,
        test_size=0.2,
        random_state=args.random_state,
        stratify=sample_labels_list
    )
    
    train_sample_ids = set(train_sample_ids)
    test_sample_ids = set(test_sample_ids)
    
    print(f"Train samples: {len(train_sample_ids)} samples")
    print(f"Test samples: {len(test_sample_ids)} samples")
    print(f"  Human: {(common_data[common_data['sample_id'].isin(train_sample_ids)]['label'] == 1).sum()} train, "
          f"{(common_data[common_data['sample_id'].isin(test_sample_ids)]['label'] == 1).sum()} test")
    print(f"  LLM: {(common_data[common_data['sample_id'].isin(train_sample_ids)]['label'] == 0).sum()} train, "
          f"{(common_data[common_data['sample_id'].isin(test_sample_ids)]['label'] == 0).sum()} test")
    
    # ==========================================
    # STEP 3: Train and evaluate models for each statistic using SAME split
    # ==========================================
    
    # Analyze each statistic using the same train/test split
    for stat_name in stats_to_analyze:
        print(f"\n{'#'*70}")
        print(f"ANALYZING STATISTIC: {stat_name.upper()} (using unified train/test split)")
        print(f"{'#'*70}")
        
        # Get data for this statistic
        stat_info = all_stat_data[stat_name]
        data_full = stat_info['data'].copy()
        features = stat_info['features']
        
        # Filter to common samples only
        common_mask = data_full['sample_id'].isin(common_samples)
        data_common = data_full[common_mask].copy()
        
        # Split based on sample_id (NOT index, to ensure same samples across stats)
        train_mask = data_common['sample_id'].isin(train_sample_ids)
        test_mask = data_common['sample_id'].isin(test_sample_ids)
        
        data_train = data_common[train_mask].copy()
        data_test = data_common[test_mask].copy()
        
        # Prepare features and labels
        X_train = data_train[features]
        X_test = data_test[features]
        y_train = data_train['label']
        y_test = data_test['label']
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        data = pd.concat([data_train, data_test])
        
        print(f"\nFinal dataset: {len(X)} samples, {len(features)} features")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Train and evaluate models (but skip train_test_split since we already have it)
        results = train_and_evaluate_models_unified_split(
            X_train, X_test, y_train, y_test, features, 
            data_train, data_test, stat_name, 
            random_state=args.random_state
        )
        
        all_results[stat_name] = results
        
        # Domain-specific metrics calculation is skipped for now
        # (would require storing trained models or re-training, which is computationally expensive)
        
        # Save results for this statistic
        results_data = {
            "Model": list(results.keys()),
            "Accuracy": [r["accuracy"] for r in results.values()],
            "ROC_AUC": [r["auc"] if r["auc"] is not None else np.nan for r in results.values()],
            "CV_Mean": [r["cv_mean"] if r["cv_mean"] is not None else np.nan for r in results.values()],
            "CV_Std": [r["cv_std"] if r["cv_std"] is not None else np.nan for r in results.values()],
            "TN": [r["confusion_matrix"]["TN"] for r in results.values()],
            "FP": [r["confusion_matrix"]["FP"] for r in results.values()],
            "FN": [r["confusion_matrix"]["FN"] for r in results.values()],
            "TP": [r["confusion_matrix"]["TP"] for r in results.values()],
        }
        results_df = pd.DataFrame(results_data)
        results_path = output_dir / f"ml_classification_{stat_name}.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        # Summary for this statistic
        print(f"\n{'='*70}")
        print(f"SUMMARY ({stat_name.upper()})")
        print("="*70)
        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
        best_model_name = best_model[0]
        best_model_result = best_model[1]
        
        print(f"Best Model: {best_model_name} (Accuracy: {best_model_result['accuracy']:.4f})")
        if best_model_result["auc"] is not None:
            print(f"ROC AUC: {best_model_result['auc']:.4f}")
        if best_model_result.get("cv_mean") is not None:
            print(f"CV Accuracy: {best_model_result['cv_mean']:.4f} (+/- {best_model_result['cv_std'] * 2:.4f})")
        
        # Display feature importance for best model
        feature_importance = best_model_result.get("feature_importance")
        if feature_importance:
            print(f"\n{'─'*70}")
            print(f"TOP 10 FEATURES ({best_model_name})")
            print(f"{'─'*70}")
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Normalize to percentages if needed
            total_importance = sum(feature_importance.values())
            if total_importance > 0 and total_importance <= 1.5:  # Likely already normalized
                # Already percentages, use as is
                sorted_features_pct = [(feat, imp * 100) for feat, imp in sorted_features]
            else:
                # Normalize to percentages
                sorted_features_pct = [(feat, (imp / total_importance) * 100) for feat, imp in sorted_features]
            
            print(f"{'Rank':<6} {'Feature':<35} {'Importance %':<15}")
            print(f"{'─'*70}")
            for rank, (feat, imp_pct) in enumerate(sorted_features_pct[:10], 1):
                # Remove stat suffix for display
                feat_display = feat.replace(f"_{stat_name}", "")
                print(f"{rank:<6} {feat_display:<35} {imp_pct:>6.2f}%")
            
            # Save feature importance to CSV
            importance_data = []
            for rank, (feat, imp_pct) in enumerate(sorted_features_pct, 1):
                feat_base = feat.replace(f"_{stat_name}", "")
                # Find original importance value
                orig_imp = next(imp for f, imp in sorted_features if f == feat)
                importance_data.append({
                    "rank": rank,
                    "feature": feat_base,
                    "full_name": feat,
                    "importance": orig_imp,
                    "importance_pct": imp_pct
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_path = output_dir / f"feature_importance_{stat_name}_{best_model_name.lower().replace(' ', '_')}.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"\nFeature importance saved to: {importance_path}")
        else:
            print(f"\n⚠️ Feature importance not available for {best_model_name}")
        
        # Statsmodels logistic regression for p-values
        if with_lr_pvalues:
            lr_dir = output_dir / "lr_pvalues"
            lr_dir.mkdir(parents=True, exist_ok=True)
            try:
                scaler_lr = StandardScaler()
                X_all_scaled = pd.DataFrame(
                    scaler_lr.fit_transform(X.reset_index(drop=True)),
                    columns=features
                )
                y_all = y.reset_index(drop=True)
                lr_df = compute_lr_pvalues_df(X_all_scaled, y_all)
                lr_df.insert(0, "statistic", stat_name.upper())
                lr_path = lr_dir / f"lr_pvalues_{stat_name}_{level_tag}.csv"
                lr_df.to_csv(lr_path, index=False)
                print(f"Logistic regression p-values saved to: {lr_path}")
            except PerfectSeparationError as exc:
                print(f"[LR-PVALUE][WARN] Perfect separation detected for {stat_name}: {exc}")
            except Exception as exc:
                print(f"[LR-PVALUE][WARN] Failed to compute logistic regression p-values for {stat_name}: {exc}")
    
    # ==========================================
    # STEP 4: Feature Concatenation - Unified Model with All Statistics
    # ==========================================
    
    if len(stats_to_analyze) > 1:
        print(f"\n{'#'*70}")
        print("STEP 4: FEATURE CONCATENATION - UNIFIED MODEL")
        print("Concatenating all statistics' features together")
        print(f"{'#'*70}")
        
        # Merge all statistics' features for each sample
        # Start with the first statistic's data as base
        first_stat = stats_to_analyze[0]
        base_data = all_stat_data[first_stat]['data'].copy()
        base_features = all_stat_data[first_stat]['features']
        
        # Filter to common samples
        common_mask = base_data['sample_id'].isin(common_samples)
        concatenated_data = base_data[common_mask].copy()
        
        # Keep metadata columns
        metadata_cols = ['sample_id', 'domain', 'field', 'author_id', 'label', 'source']
        concatenated_data = concatenated_data[metadata_cols + base_features].copy()
        
        # Merge features from other statistics
        for stat_name in stats_to_analyze[1:]:
            stat_info = all_stat_data[stat_name]
            stat_data = stat_info['data'].copy()
            stat_features = stat_info['features']
            
            # Filter to common samples
            stat_common_mask = stat_data['sample_id'].isin(common_samples)
            stat_data_common = stat_data[stat_common_mask].copy()
            
            # Merge on sample_id (should be 1-to-1 match)
            concatenated_data = concatenated_data.merge(
                stat_data_common[['sample_id'] + stat_features],
                on='sample_id',
                how='inner',
                suffixes=('', '')
            )
            
            print(f"  Added {len(stat_features)} features from {stat_name.upper()}")
        
        # Get all feature columns (all statistics concatenated)
        all_feature_cols = []
        for stat_name in stats_to_analyze:
            all_feature_cols.extend(all_stat_data[stat_name]['features'])
        
        print(f"\nTotal concatenated features: {len(all_feature_cols)}")
        print(f"  - {len(base_features)} from {first_stat.upper()}")
        for stat_name in stats_to_analyze[1:]:
            print(f"  - {len(all_stat_data[stat_name]['features'])} from {stat_name.upper()}")
        
        # Split based on sample_id (same split as before)
        train_mask = concatenated_data['sample_id'].isin(train_sample_ids)
        test_mask = concatenated_data['sample_id'].isin(test_sample_ids)
        
        data_train_concatenated = concatenated_data[train_mask].copy()
        data_test_concatenated = concatenated_data[test_mask].copy()
        
        # Prepare features and labels
        X_train_concatenated = data_train_concatenated[all_feature_cols]
        X_test_concatenated = data_test_concatenated[all_feature_cols]
        y_train_concatenated = data_train_concatenated['label']
        y_test_concatenated = data_test_concatenated['label']
        
        print(f"\nFinal concatenated dataset: {len(concatenated_data)} samples, {len(all_feature_cols)} features")
        print(f"  Train: {len(X_train_concatenated)} samples")
        print(f"  Test: {len(X_test_concatenated)} samples")
        
        # Train and evaluate unified model
        unified_results = train_and_evaluate_models_unified_split(
            X_train_concatenated, X_test_concatenated, 
            y_train_concatenated, y_test_concatenated, 
            all_feature_cols, 
            data_train_concatenated, data_test_concatenated, 
            stat_name="concatenated",  # Special name for concatenated model
            random_state=args.random_state
        )
        
        all_results["concatenated"] = unified_results
        
        # Save results for concatenated model
        results_data = {
            "Model": list(unified_results.keys()),
            "Accuracy": [r["accuracy"] for r in unified_results.values()],
            "ROC_AUC": [r["auc"] if r["auc"] is not None else np.nan for r in unified_results.values()],
            "CV_Mean": [r["cv_mean"] if r["cv_mean"] is not None else np.nan for r in unified_results.values()],
            "CV_Std": [r["cv_std"] if r["cv_std"] is not None else np.nan for r in unified_results.values()],
            "TN": [r["confusion_matrix"]["TN"] for r in unified_results.values()],
            "FP": [r["confusion_matrix"]["FP"] for r in unified_results.values()],
            "FN": [r["confusion_matrix"]["FN"] for r in unified_results.values()],
            "TP": [r["confusion_matrix"]["TP"] for r in unified_results.values()],
        }
        results_df = pd.DataFrame(results_data)
        results_path = output_dir / "ml_classification_concatenated.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        # Summary for concatenated model
        print(f"\n{'='*70}")
        print(f"SUMMARY (CONCATENATED MODEL - All Statistics Combined)")
        print("="*70)
        best_model = max(unified_results.items(), key=lambda x: x[1]["accuracy"])
        best_model_name = best_model[0]
        best_model_result = best_model[1]
        
        print(f"Best Model: {best_model_name} (Accuracy: {best_model_result['accuracy']:.4f})")
        if best_model_result["auc"] is not None:
            print(f"ROC AUC: {best_model_result['auc']:.4f}")
        if best_model_result.get("cv_mean") is not None:
            print(f"CV Accuracy: {best_model_result['cv_mean']:.4f} (+/- {best_model_result['cv_std'] * 2:.4f})")
        
        # Display feature importance for best model
        feature_importance = best_model_result.get("feature_importance")
        if feature_importance:
            print(f"\n{'─'*70}")
            print(f"TOP 15 FEATURES ({best_model_name}) - All Statistics Combined")
            print(f"{'─'*70}")
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Normalize to percentages if needed
            total_importance = sum(feature_importance.values())
            if total_importance > 0 and total_importance <= 1.5:  # Likely already normalized
                sorted_features_pct = [(feat, imp * 100) for feat, imp in sorted_features]
            else:
                sorted_features_pct = [(feat, (imp / total_importance) * 100) for feat, imp in sorted_features]
            
            print(f"{'Rank':<6} {'Feature':<40} {'Statistic':<15} {'Importance %':<15}")
            print(f"{'─'*70}")
            for rank, (feat, imp_pct) in enumerate(sorted_features_pct[:15], 1):
                # Extract base feature name and statistic
                for stat in stats_to_analyze:
                    if feat.endswith(f"_{stat}"):
                        feat_base = feat.replace(f"_{stat}", "")
                        stat_display = stat.upper()
                        break
                else:
                    feat_base = feat
                    stat_display = "UNKNOWN"
                
                print(f"{rank:<6} {feat_base:<40} {stat_display:<15} {imp_pct:>6.2f}%")
            
            # Save feature importance to CSV
            importance_data = []
            for rank, (feat, imp_pct) in enumerate(sorted_features_pct, 1):
                # Extract base feature name and statistic
                for stat in stats_to_analyze:
                    if feat.endswith(f"_{stat}"):
                        feat_base = feat.replace(f"_{stat}", "")
                        stat_display = stat
                        break
                else:
                    feat_base = feat
                    stat_display = "unknown"
                
                orig_imp = next(imp for f, imp in sorted_features if f == feat)
                importance_data.append({
                    "rank": rank,
                    "feature": feat_base,
                    "statistic": stat_display,
                    "full_name": feat,
                    "importance": orig_imp,
                    "importance_pct": imp_pct
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_path = output_dir / f"feature_importance_concatenated_{best_model_name.lower().replace(' ', '_')}.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"\nFeature importance saved to: {importance_path}")
        else:
            print(f"\n⚠️ Feature importance not available for {best_model_name}")
        
        if with_lr_pvalues:
            lr_dir = output_dir / "lr_pvalues"
            lr_dir.mkdir(parents=True, exist_ok=True)
            try:
                scaler_lr_concat = StandardScaler()
                X_concat_scaled = pd.DataFrame(
                    scaler_lr_concat.fit_transform(concatenated_data[all_feature_cols].reset_index(drop=True)),
                    columns=all_feature_cols
                )
                y_concat_all = concatenated_data['label'].reset_index(drop=True)
                lr_df_concat = compute_lr_pvalues_df(X_concat_scaled, y_concat_all)
                lr_df_concat.insert(0, "statistic", "CONCATENATED")
                lr_path_concat = lr_dir / f"lr_pvalues_concatenated_{level_tag}.csv"
                lr_df_concat.to_csv(lr_path_concat, index=False)
                print(f"Logistic regression p-values saved to: {lr_path_concat}")
            except PerfectSeparationError as exc:
                print(f"[LR-PVALUE][WARN] Perfect separation detected for concatenated model: {exc}")
            except Exception as exc:
                print(f"[LR-PVALUE][WARN] Failed to compute logistic regression p-values for concatenated model: {exc}")
    
    # Compare all statistics (including concatenated model)
    if len(stats_to_analyze) > 1:
        print(f"\n{'#'*70}")
        print("COMPARING ALL STATISTICS (Including Concatenated Model)")
        print(f"{'#'*70}")
        comparison_df = compare_statistics(all_results, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nMethods Implemented:")
    print("  1. Same Training/Test Split: ✓ (All statistics use same split)")
    print("  2. Feature Concatenation in Unified Model: ✓ (All features combined)")
    
    return all_results


if __name__ == "__main__":
    main()

