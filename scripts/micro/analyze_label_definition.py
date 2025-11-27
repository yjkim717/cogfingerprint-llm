#!/usr/bin/env python3
"""
分析为什么用平均值定义标签，以及是否有更好的方式

当前方法：
  label = 1 if human_var > average(llm_var_DS, llm_var_G4B, llm_var_G12B)

其他可能的方法：
  1. 与每个模型分别比较，然后多数投票
  2. 与每个模型分别比较，然后取"至少一个"或"全部"
  3. 加权平均（根据模型重要性）
  4. 最大值/最小值
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ml_classification_variance import (
    load_variance_data,
    get_feature_columns,
)

DATA_ROOT = Path("dataset")
PROCESS_ROOT = DATA_ROOT / "process"
DEFAULT_MODELS = ["DS", "G4B", "G12B"]

def compare_label_definitions(domain="academic"):
    """比较不同的标签定义方法"""
    print("="*80)
    print("标签定义方法对比分析")
    print("="*80)
    
    # Load data
    human_df = load_variance_data(domain, "human")
    llm_data_by_model = {}
    for model in DEFAULT_MODELS:
        llm_df = load_variance_data(domain, "llm", model=model, level="LV1")
        llm_data_by_model[model.upper()] = llm_df
    
    # Get features
    human_features = set(get_feature_columns(human_df))
    llm_features = set()
    for llm_df in llm_data_by_model.values():
        llm_features.update(get_feature_columns(llm_df))
    features = sorted(human_features & llm_features)
    
    # Prepare sample data for one feature
    feature = features[0]
    merge_cols = ["field", "author_id"]
    
    # Merge all models
    merged = human_df.merge(
        llm_data_by_model["DS"],
        on=merge_cols,
        suffixes=("_human", "_llm_DS"),
        how="inner"
    )
    merged = merged.merge(
        llm_data_by_model["G4B"],
        on=merge_cols,
        suffixes=("", "_llm_G4B"),
        how="inner"
    )
    merged = merged.merge(
        llm_data_by_model["G12B"],
        on=merge_cols,
        suffixes=("", "_llm_G12B"),
        how="inner"
    )
    
    human_col = f"{feature}_human"
    llm_ds_col = f"{feature}_llm_DS"
    llm_g4b_col = f"{feature}_llm_G4B"
    llm_g12b_col = f"{feature}_llm_G12B"
    
    if not all(col in merged.columns for col in [human_col, llm_ds_col, llm_g4b_col, llm_g12b_col]):
        print("Error: Missing columns")
        return
    
    # Extract data
    data = merged[[human_col, llm_ds_col, llm_g4b_col, llm_g12b_col]].dropna()
    
    if data.empty:
        print("No data")
        return
    
    human_var = data[human_col].values
    llm_ds_var = data[llm_ds_col].values
    llm_g4b_var = data[llm_g4b_col].values
    llm_g12b_var = data[llm_g12b_col].values
    
    n = len(data)
    
    print(f"\n样本数: {n}")
    print(f"特征: {feature}")
    
    # Method 1: Average (当前方法)
    avg_llm = (llm_ds_var + llm_g4b_var + llm_g12b_var) / 3
    label_avg = (human_var > avg_llm).astype(int)
    prop_avg = label_avg.mean()
    
    # Method 2: Majority vote (与每个模型比较，多数投票)
    votes = (
        (human_var > llm_ds_var).astype(int) +
        (human_var > llm_g4b_var).astype(int) +
        (human_var > llm_g12b_var).astype(int)
    )
    label_majority = (votes >= 2).astype(int)  # 至少2个模型
    prop_majority = label_majority.mean()
    
    # Method 3: All models (Human > 所有模型)
    label_all = (
        (human_var > llm_ds_var) &
        (human_var > llm_g4b_var) &
        (human_var > llm_g12b_var)
    ).astype(int)
    prop_all = label_all.mean()
    
    # Method 4: At least one (Human > 至少一个模型)
    label_any = (
        (human_var > llm_ds_var) |
        (human_var > llm_g4b_var) |
        (human_var > llm_g12b_var)
    ).astype(int)
    prop_any = label_any.mean()
    
    # Method 5: Maximum (Human > 最大LLM variance)
    max_llm = np.maximum.reduce([llm_ds_var, llm_g4b_var, llm_g12b_var])
    label_max = (human_var > max_llm).astype(int)
    prop_max = label_max.mean()
    
    # Method 6: Minimum (Human > 最小LLM variance)
    min_llm = np.minimum.reduce([llm_ds_var, llm_g4b_var, llm_g12b_var])
    label_min = (human_var > min_llm).astype(int)
    prop_min = label_min.mean()
    
    # Compare results
    print("\n" + "="*80)
    print("不同标签定义方法的结果对比")
    print("="*80)
    
    results = [
        ("Average (当前方法)", prop_avg, label_avg),
        ("Majority Vote (多数投票)", prop_majority, label_majority),
        ("All Models (全部模型)", prop_all, label_all),
        ("At Least One (至少一个)", prop_any, label_any),
        ("Maximum (最大值)", prop_max, label_max),
        ("Minimum (最小值)", prop_min, label_min),
    ]
    
    print(f"\n{'方法':<25} {'Human>LLM比例':<15} {'说明'}")
    print("-" * 80)
    for method, prop, labels in results:
        print(f"{method:<25} {prop*100:>6.2f}%        ", end="")
        if method == "Average (当前方法)":
            print("(当前使用)")
        elif method == "Majority Vote (多数投票)":
            print("(需要至少2/3模型同意)")
        elif method == "All Models (全部模型)":
            print("(最严格)")
        elif method == "At Least One (至少一个)":
            print("(最宽松)")
        elif method == "Maximum (最大值)":
            print("(与最强LLM比较)")
        elif method == "Minimum (最小值)":
            print("(与最弱LLM比较)")
    
    # Agreement analysis
    print("\n" + "="*80)
    print("方法之间的一致性分析")
    print("="*80)
    
    # Compare Average vs Majority
    agreement_avg_maj = (label_avg == label_majority).mean()
    print(f"\nAverage vs Majority Vote: {agreement_avg_maj*100:.2f}% 一致")
    
    # Compare Average vs All
    agreement_avg_all = (label_avg == label_all).mean()
    print(f"Average vs All Models: {agreement_avg_all*100:.2f}% 一致")
    
    # Compare Average vs Any
    agreement_avg_any = (label_avg == label_any).mean()
    print(f"Average vs At Least One: {agreement_avg_any*100:.2f}% 一致")
    
    # Disagreement cases
    disagree_avg_maj = np.where(label_avg != label_majority)[0]
    if len(disagree_avg_maj) > 0:
        print(f"\n不一致的样本数: {len(disagree_avg_maj)} ({len(disagree_avg_maj)/n*100:.2f}%)")
        print("示例不一致样本:")
        for idx in disagree_avg_maj[:5]:
            print(f"  样本{idx}: Human={human_var[idx]:.6f}, "
                  f"DS={llm_ds_var[idx]:.6f}, G4B={llm_g4b_var[idx]:.6f}, "
                  f"G12B={llm_g12b_var[idx]:.6f}, "
                  f"Avg={avg_llm[idx]:.6f}")
            print(f"    Average方法: {label_avg[idx]}, Majority方法: {label_majority[idx]}")

def explain_why_average():
    """解释为什么用平均值"""
    print("\n" + "="*80)
    print("为什么用平均值？")
    print("="*80)
    
    print("""
优点：

  1. 稳健性 (Robustness)
     - 不偏向某个特定模型
     - 减少单个模型的异常值影响
     - 更代表"一般LLM"的表现

  2. 理论合理性
     - 平均值代表"典型LLM"的variance
     - 如果Human > 典型LLM，说明Human确实变化更大
     - 符合研究问题的表述（Human vs LLM，而不是vs某个特定模型）

  3. 与二项检验一致
     - 二项检验也是比较Human vs 所有LLM模型
     - 用平均值与二项检验的逻辑一致
     - 结果可以相互印证

  4. 简单直观
     - 容易理解和解释
     - 不需要复杂的投票机制
     - 计算简单

  5. 统计意义
     - 平均值是中心趋势的度量
     - 如果Human > 平均值，说明Human确实高于"平均水平"
     - 这是统计上合理的比较

缺点：

  1. 可能掩盖模型差异
     - 如果某个模型特别高或特别低，平均值可能不准确
     - 但这也可能是优点（减少异常值影响）

  2. 不考虑模型重要性
     - 所有模型权重相同
     - 如果某些模型更重要，可能需要加权平均

  3. 可能不够严格
     - 如果Human只比平均值高一点点，可能不够显著
     - 但这也取决于研究目标
""")

def compare_with_alternatives():
    """对比其他方法"""
    print("\n" + "="*80)
    print("其他方法的优缺点")
    print("="*80)
    
    print("""
方法1: Majority Vote (多数投票)
  优点：
    - 需要至少2/3模型同意，更严格
    - 减少单个模型的影响
    - 符合"民主"原则
  
  缺点：
    - 如果Human只比2个模型高，但低于1个，会被标记为0
    - 可能过于严格
    - 与二项检验的逻辑不完全一致

方法2: All Models (全部模型)
  优点：
    - 最严格的标准
    - 确保Human > 所有LLM
  
  缺点：
    - 可能过于严格
    - 如果某个模型特别高，可能无法满足
    - 可能丢失很多正样本

方法3: At Least One (至少一个)
  优点：
    - 最宽松的标准
    - 只要Human > 任何一个模型就标记为1
  
  缺点：
    - 可能过于宽松
    - 如果Human只比最弱的模型高，可能不够有意义
    - 可能包含很多噪声

方法4: Maximum (最大值)
  优点：
    - 与最强的LLM比较
    - 如果Human > 最强LLM，说明确实很强
  
  缺点：
    - 可能过于严格
    - 只考虑一个模型（最强的）
    - 可能丢失信息

方法5: Minimum (最小值)
  优点：
    - 与最弱的LLM比较
    - 容易满足
  
  缺点：
    - 可能过于宽松
    - 只考虑一个模型（最弱的）
    - 可能不够有意义
""")

def provide_recommendation():
    """提供建议"""
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    
    print("""
当前方法（平均值）是合理的，因为：

  1. 与研究问题一致
     - 研究问题是"Human vs LLM"，不是"Human vs 某个特定模型"
     - 平均值代表"典型LLM"，这是合理的比较对象

  2. 稳健性
     - 不依赖单个模型
     - 减少异常值影响
     - 结果更可靠

  3. 与统计检验一致
     - 二项检验也是比较Human vs 所有LLM
     - 用平均值与二项检验逻辑一致

  4. 简单可解释
     - 容易理解和解释
     - 不需要复杂的机制

可能的改进：

  1. 如果发现模型差异很大，可以考虑：
     - 加权平均（根据模型重要性）
     - Majority vote（更严格）
  
  2. 作为稳健性检验，可以：
     - 用多种方法定义标签
     - 比较结果是否一致
     - 如果一致，增强结论可信度

  3. 在论文中可以说明：
     "We define the label as Human > average(LLM) to represent 
     a comparison against typical LLM performance, rather than 
     any specific model. Robustness checks using majority vote 
     and other methods showed consistent results."
""")

def main():
    compare_label_definitions("academic")
    explain_why_average()
    compare_with_alternatives()
    provide_recommendation()

if __name__ == "__main__":
    main()




