#!/usr/bin/env python3
"""
Analyze why Logistic Regression has low accuracy (63%) but high AUC (0.98).

This script investigates:
1. Prediction probability distribution
2. Optimal threshold
3. Confusion matrix at different thresholds
4. Class imbalance effects
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

DATA_ROOT = Path("dataset")
PROCESS_ROOT = DATA_ROOT / "process"

def load_and_prepare_data(domain="academic"):
    """Load variance data and prepare for classification."""
    from ml_classification_variance import (
        load_variance_data,
        get_feature_columns,
        prepare_classification_data,
    )
    
    DEFAULT_MODELS = ["DS", "G4B", "G12B"]
    DEFAULT_LEVEL = "LV1"
    
    # Load data
    human_df = load_variance_data(domain, "human")
    llm_data_by_model = {}
    for model in DEFAULT_MODELS:
        llm_df = load_variance_data(domain, "llm", model=model, level=DEFAULT_LEVEL)
        llm_data_by_model[model.upper()] = llm_df
    
    # Get features
    human_features = set(get_feature_columns(human_df))
    llm_features = set()
    for llm_df in llm_data_by_model.values():
        llm_features.update(get_feature_columns(llm_df))
    features = sorted(human_features & llm_features)
    
    # Prepare data
    data_df = prepare_classification_data(human_df, llm_data_by_model, features, domain)
    
    # Prepare features
    feature_cols = ["human_var"] + [f"llm_var_{m}" for m in llm_data_by_model.keys()]
    X = data_df[feature_cols].copy()
    y = data_df["label"]
    
    # Remove missing values
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y, data_df

def analyze_threshold_effect(X, y, X_train, X_test, y_train, y_test, model):
    """Analyze how different thresholds affect accuracy."""
    print("\n" + "="*80)
    print("阈值分析 (Threshold Analysis)")
    print("="*80)
    
    # Get probabilities
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.3, 0.71, 0.05)
    results = []
    
    print("\n不同阈值下的准确率：")
    print(f"{'阈值':<8} {'准确率':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    
    for threshold in thresholds:
        y_pred = (y_proba_test >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            if y_pred[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        marker = " <-- 默认" if threshold == 0.5 else ""
        print(f"{threshold:.2f}     {acc:.4f}     {tp:<6} {tn:<6} {fp:<6} {fn:<6} {precision:.4f}     {recall:.4f}{marker}")
        
        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall
        })
    
    # Find optimal threshold (maximizing accuracy)
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\n最优阈值（最大化准确率）: {best['threshold']:.2f}")
    print(f"  准确率: {best['accuracy']:.4f}")
    print(f"  在默认阈值0.5下: {results[len(results)//2]['accuracy']:.4f}")
    print(f"  提升: {(best['accuracy'] - results[len(results)//2]['accuracy']):.4f}")
    
    return results, best

def analyze_probability_distribution(y_proba, y_true):
    """Analyze the distribution of prediction probabilities."""
    print("\n" + "="*80)
    print("预测概率分布分析")
    print("="*80)
    
    df = pd.DataFrame({
        "probability": y_proba,
        "true_label": y_true
    })
    
    print(f"\n概率分布统计：")
    print(f"  均值: {y_proba.mean():.4f}")
    print(f"  中位数: {np.median(y_proba):.4f}")
    print(f"  标准差: {y_proba.std():.4f}")
    print(f"  最小值: {y_proba.min():.4f}")
    print(f"  最大值: {y_proba.max():.4f}")
    
    # Distribution by class
    print(f"\n按真实类别分组的概率分布：")
    print(f"  正类 (Human > LLM):")
    pos_proba = y_proba[y_true == 1]
    print(f"    均值: {pos_proba.mean():.4f}")
    print(f"    中位数: {np.median(pos_proba):.4f}")
    print(f"    标准差: {pos_proba.std():.4f}")
    
    print(f"  负类 (Human <= LLM):")
    neg_proba = y_proba[y_true == 0]
    print(f"    均值: {neg_proba.mean():.4f}")
    print(f"    中位数: {np.median(neg_proba):.4f}")
    print(f"    标准差: {neg_proba.std():.4f}")
    
    # Samples near threshold
    near_threshold = np.abs(y_proba - 0.5) < 0.1
    print(f"\n接近阈值0.5的样本（±0.1范围内）: {near_threshold.sum()} / {len(y_proba)} ({near_threshold.sum()/len(y_proba)*100:.1f}%)")
    
    if near_threshold.sum() > 0:
        near_proba = y_proba[near_threshold]
        near_labels = y_true[near_threshold]
        print(f"  其中正类: {(near_labels == 1).sum()} ({((near_labels == 1).sum()/near_threshold.sum()*100):.1f}%)")
        print(f"  其中负类: {(near_labels == 0).sum()} ({((near_labels == 0).sum()/near_threshold.sum()*100):.1f}%)")

def main():
    print("="*80)
    print("逻辑回归性能分析：为什么准确率63%但AUC 0.98？")
    print("="*80)
    
    # Load data
    print("\n加载数据...")
    X, y, data_df = load_and_prepare_data("academic")
    
    print(f"总样本数: {len(X)}")
    print(f"类别分布: {y.value_counts().to_dict()}")
    print(f"正类比例: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n模型性能：")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  ROC AUC: {auc:.4f}")
    
    # Confusion matrix at 0.5 threshold
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    print(f"\n混淆矩阵（阈值=0.5）：")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    
    # Analyze probability distribution
    analyze_probability_distribution(y_proba, y_test)
    
    # Analyze threshold effect
    threshold_results, best_threshold = analyze_threshold_effect(
        X, y, X_train_scaled, X_test_scaled, y_train, y_test, model
    )
    
    # Summary
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"\n1. 类别不平衡：")
    print(f"   - 正类占 {y.mean():.2%}，负类占 {1-y.mean():.2%}")
    print(f"   - 这可能导致在0.5阈值下准确率不是最优")
    
    print(f"\n2. 模型区分能力：")
    print(f"   - ROC AUC = {auc:.4f} 说明模型能很好地区分两类")
    print(f"   - 正类的平均概率应该高于负类")
    
    print(f"\n3. 阈值选择：")
    print(f"   - 默认阈值0.5可能不是最优")
    print(f"   - 最优阈值约为 {best_threshold['threshold']:.2f}")
    print(f"   - 在最优阈值下，准确率可提升到 {best_threshold['accuracy']:.4f}")
    
    print(f"\n4. 为什么AUC高但准确率低？")
    print(f"   - AUC衡量的是模型的排序能力（能否将正样本排在负样本前面）")
    print(f"   - 即使很多样本的概率在0.5附近，只要正样本概率>负样本概率，AUC就会高")
    print(f"   - 但准确率需要明确的分类决策（阈值），在0.5阈值下可能不是最优")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()




