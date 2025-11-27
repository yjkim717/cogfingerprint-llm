#!/usr/bin/env python3
"""
Analyze feature importance from ML models after outliers removal.
Extract and compare feature importance rankings across all statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Import from ml_classify script
from ml_classify_author_by_timeseries import (
    load_timeseries_data,
    prepare_classification_data,
    extract_feature_columns,
    ALL_FEATURE_COLUMNS_MERGED,
)

STATS = ["cv", "rmssd", "masd", "variance"]
FEATURE_LAYERS = {
    "Cognitive": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"],
    "Emotional": ["polarity", "subjectivity", "vader_compound", "vader_pos", "vader_neu", "vader_neg"],
    "Stylistic": ["word_diversity", "flesch_reading_ease", "gunning_fog", "average_word_length",
                  "num_words", "avg_sentence_length", "verb_ratio", "function_word_ratio", "content_word_ratio"]
}


def get_feature_layer(feature_name: str) -> str:
    """Get the layer (Cognitive, Emotional, Stylistic) for a feature."""
    # Remove stat suffix (e.g., "_cv", "_variance")
    base_name = feature_name.split("_")[0]
    for i, layer_name in enumerate(["Cognitive", "Emotional", "Stylistic"]):
        if any(base_name.startswith(f.split("_")[0]) or f.split("_")[0] in base_name 
               for f in FEATURE_LAYERS[layer_name]):
            return layer_name
    # Try direct match
    for layer_name, features in FEATURE_LAYERS.items():
        for feat in features:
            if feat in feature_name:
                return layer_name
    return "Unknown"


def analyze_feature_importance(
    stat_name: str,
    domains: List[str],
    models: List[str],
    level: str,
    random_state: int = 42
) -> Dict:
    """Analyze feature importance for a specific statistic."""
    print(f"\n{'='*70}")
    print(f"Analyzing Feature Importance: {stat_name.upper()}")
    print(f"{'='*70}")
    
    # Prepare data
    data, features = prepare_classification_data(
        domains, models, level, stat_name,
        use_merged=True,
        outliers_removed=True,
        outlier_strategy="model_feature",
        outlier_method="iqr"
    )
    
    X = data[features]
    y = data["label"]
    
    # Remove missing values
    missing_mask = X.isna().any(axis=1)
    if missing_mask.sum() > 0:
        X = X[~missing_mask]
        y = y[~missing_mask]
        data = data[~missing_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Analyze multiple models
    models_to_analyze = {
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=random_state), False),
        "Logistic Regression": (LogisticRegression(random_state=random_state, max_iter=1000), True),
        "SVM": (SVC(probability=True, random_state=random_state, kernel="rbf"), True),
        "Neural Network": (MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=random_state), True),
    }
    
    for model_name, (model, use_scaled) in models_to_analyze.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        
        # Use scaled or original data
        if use_scaled:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Train
        model.fit(X_tr, y_train)
        
        # Evaluate
        y_pred = model.predict(X_te)
        y_pred_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Extract feature importance
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(features, model.feature_importances_))
        elif hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_importance = dict(zip(features, np.abs(coef)))
        
        if feature_importance:
            # Normalize to percentages
            total = sum(feature_importance.values())
            feature_importance_pct = {k: (v / total * 100) if total > 0 else 0 
                                     for k, v in feature_importance.items()}
            
            # Sort by importance
            sorted_features = sorted(feature_importance_pct.items(), key=lambda x: x[1], reverse=True)
            
            # Add layer information
            feature_with_layer = []
            for feat, imp in sorted_features:
                base_name = feat.replace(f"_{stat_name}", "")
                layer = get_feature_layer(base_name)
                feature_with_layer.append({
                    "feature": base_name,
                    "full_name": feat,
                    "importance": imp,
                    "layer": layer
                })
            
            results[model_name] = {
                "accuracy": accuracy,
                "auc": auc,
                "features": feature_with_layer
            }
            
            # Print top 10
            print(f"  Accuracy: {accuracy:.4f}")
            if auc:
                print(f"  ROC AUC: {auc:.4f}")
            print(f"  Top 10 Features:")
            for i, item in enumerate(feature_with_layer[:10], 1):
                print(f"    {i:2d}. {item['feature']:<25} [{item['layer']:<12}] {item['importance']:>6.2f}%")
    
    return results


def create_summary_report(all_results: Dict[str, Dict], output_dir: Path):
    """Create a comprehensive summary report of feature importance."""
    
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE SUMMARY REPORT")
    print(f"{'='*70}")
    
    # Find best model for each statistic
    best_models = {}
    for stat_name, models_dict in all_results.items():
        best_model = None
        best_accuracy = 0
        for model_name, result in models_dict.items():
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_model = model_name
        best_models[stat_name] = best_model
        print(f"\n{stat_name.upper()}: Best Model = {best_model} (Accuracy: {best_accuracy:.4f})")
    
    # Aggregate feature importance across all statistics
    print(f"\n{'='*70}")
    print("AGGREGATED FEATURE IMPORTANCE (Top 20)")
    print(f"{'='*70}")
    
    feature_aggregate = {}
    for stat_name, models_dict in all_results.items():
        best_model = best_models[stat_name]
        if best_model in models_dict:
            for feat_info in models_dict[best_model]["features"]:
                feat_name = feat_info["feature"]
                if feat_name not in feature_aggregate:
                    feature_aggregate[feat_name] = {
                        "total_importance": 0,
                        "count": 0,
                        "layer": feat_info["layer"]
                    }
                feature_aggregate[feat_name]["total_importance"] += feat_info["importance"]
                feature_aggregate[feat_name]["count"] += 1
    
    # Calculate average importance
    for feat_name in feature_aggregate:
        feature_aggregate[feat_name]["avg_importance"] = (
            feature_aggregate[feat_name]["total_importance"] / 
            feature_aggregate[feat_name]["count"]
        )
    
    # Sort by average importance
    sorted_aggregate = sorted(feature_aggregate.items(), 
                             key=lambda x: x[1]["avg_importance"], 
                             reverse=True)
    
    print(f"\n{'Rank':<6} {'Feature':<30} {'Layer':<15} {'Avg Importance':<15}")
    print("-" * 70)
    for rank, (feat_name, info) in enumerate(sorted_aggregate[:20], 1):
        print(f"{rank:<6} {feat_name:<30} {info['layer']:<15} {info['avg_importance']:>6.2f}%")
    
    # Layer-wise analysis
    print(f"\n{'='*70}")
    print("LAYER-WISE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    
    layer_totals = {"Cognitive": 0, "Emotional": 0, "Stylistic": 0, "Unknown": 0}
    layer_counts = {"Cognitive": 0, "Emotional": 0, "Stylistic": 0, "Unknown": 0}
    
    for feat_name, info in sorted_aggregate:
        layer = info["layer"]
        layer_totals[layer] += info["avg_importance"]
        layer_counts[layer] += 1
    
    print(f"\n{'Layer':<15} {'Total Importance':<20} {'Avg per Feature':<20} {'Feature Count':<15}")
    print("-" * 70)
    for layer in ["Cognitive", "Emotional", "Stylistic"]:
        total = layer_totals[layer]
        count = layer_counts[layer]
        avg_per_feat = total / count if count > 0 else 0
        print(f"{layer:<15} {total:>6.2f}%{'':<13} {avg_per_feat:>6.2f}%{'':<13} {count:<15}")
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-statistic results
    for stat_name, models_dict in all_results.items():
        best_model = best_models[stat_name]
        if best_model in models_dict:
            df = pd.DataFrame(models_dict[best_model]["features"])
            output_path = output_dir / f"feature_importance_{stat_name}_{best_model.lower().replace(' ', '_')}.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved: {output_path}")
    
    # Save aggregated results
    aggregate_df = pd.DataFrame([
        {
            "rank": rank,
            "feature": feat_name,
            "layer": info["layer"],
            "avg_importance": info["avg_importance"],
            "total_importance": info["total_importance"],
            "appeared_in_stats": info["count"]
        }
        for rank, (feat_name, info) in enumerate(sorted_aggregate, 1)
    ])
    aggregate_path = output_dir / "feature_importance_aggregated.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"Saved: {aggregate_path}")
    
    # Save summary JSON
    summary = {
        "best_models": best_models,
        "layer_totals": layer_totals,
        "top_features": [
            {
                "feature": feat_name,
                "layer": info["layer"],
                "avg_importance": info["avg_importance"]
            }
            for feat_name, info in sorted_aggregate[:20]
        ]
    }
    summary_path = output_dir / "feature_importance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance after outliers removal."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["academic", "news", "blogs"],
        help="Domains to analyze (default: academic news blogs)."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DS", "G4B", "G12B", "LMK"],
        help="LLM models to include (default: DS G4B G12B LMK)."
    )
    parser.add_argument(
        "--level",
        default="LV1",
        help="LLM level (default: LV1)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/feature_importance"),
        help="Output directory for results (default: results/feature_importance)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)."
    )
    
    args = parser.parse_args()
    
    # Analyze all statistics
    all_results = {}
    for stat_name in STATS:
        try:
            results = analyze_feature_importance(
                stat_name,
                args.domains,
                args.models,
                args.level,
                args.random_state
            )
            all_results[stat_name] = results
        except Exception as e:
            print(f"Error analyzing {stat_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary report
    create_summary_report(all_results, args.output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



