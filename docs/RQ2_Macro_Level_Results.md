# RQ2 Macro-level 结果详细报告
## Field Convergence & Resilience Analysis

## 概述

本文档详细记录了RQ2（Macro-level: Field Convergence & Resilience）的研究结果。RQ2旨在回答：在群体层面，人类写作风格与LLM生成文本是否存在收敛趋势？以及哪些静态特征能够持续区分人类和LLM？

**研究问题**: 
- RQ2: 在领域/群体层面，人类写作风格与LLM生成文本是否存在收敛？哪些特征具有最强的区分能力？

**实验类型**:
1. **Yearly Static ML Validation**: 按年份进行Human vs. LLM分类，检测分离度是否随时间下降
2. **Macro Static Classification**: 使用静态特征进行整体分类二分类
3. **Feature Convergence Analysis**: 特征级别的收敛分析

---

## 1. Yearly Static ML Validation (按年份分类验证)

### 1.1 实验设计

**目标**: 检测如果人类和LLM在群体层面收敛，那么按年份训练的分类器应该发现分离度随时间下降。

**数据构成**:
- **Human样本**: `macro_dataset/process/human/<domain>/combined.csv`
  - 每个领域每年1,000个文档
- **LLM样本**: `macro_dataset/process/LLM/<model>/LV3/<domain>/combined_outliers_removed.csv`
  - 每个领域每年每个模型1,000个文档
  - 4个模型（DS, G4B, G12B, LMK）总计4,000个LLM文档
- **每年总计**: 约5,000个训练样本（1k human + 4k LLM）
- **Prompting Level**: LV3（固定）

**特征集**: 20个静态特征
- Big Five (5个): Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- Emotional (6个): polarity, subjectivity, vader_compound, vader_pos, vader_neu, vader_neg
- Stylistic (9个): word_diversity, flesch_reading_ease, gunning_fog, average_word_length, num_words, avg_sentence_length, verb_ratio, function_word_ratio, content_word_ratio

**模型配置**:
- **Pipeline**: `StandardScaler → LogisticRegression(class_weight='balanced', max_iter=2000)`
- **数据分割**: 每个年份/领域独立70/30分层分割（`random_state=42`）
- **选择Logistic Regression的原因**:
  1. 系数可解释，直接对应RQ2特征分析
  2. statsmodels提供p值用于显著性检验

### 1.2 主要结果（按领域）

#### 1.2.1 合并领域（All Domains）

- **准确率**: 保持在0.83-0.85之间
- **ROC AUC**: 保持在0.89-0.91之间
- **结论**: 仍然高度可分离，无明显收敛趋势

#### 1.2.2 News领域

| 年份 | 准确率 | ROC AUC | 趋势 |
|------|--------|---------|------|
| 2020 | 0.965 | 0.995 | 基准 |
| 2021 | ~0.965 | ~0.99 | 稳定 |
| 2022 | ~0.965 | ~0.99 | 稳定 |
| 2023 | ~0.965 | ~0.99 | 稳定 |
| 2024 | 0.965 | 0.986 | 轻微下降 |

**关键发现**:
- 准确率保持在96.5%，几乎完美分离
- ROC AUC从0.995轻微降至0.986，但仍接近完美
- **结论**: News领域没有明显的收敛趋势，分离度保持极高

#### 1.2.3 Blogs领域

| 年份 | 准确率 | ROC AUC | 趋势 |
|------|--------|---------|------|
| 2020-2024 | 0.881-0.895 | 0.95-0.96 | 基本稳定 |

**关键发现**:
- 准确率在88.1%-89.5%之间波动（±0.02）
- ROC AUC稳定在0.95-0.96
- **结论**: 没有可见的收敛趋势，分离度保持稳定

#### 1.2.4 Academic领域

| 年份 | 准确率 | ROC AUC | 趋势 |
|------|--------|---------|------|
| 2020 | 0.932 | 0.976 | 基准 |
| 2021 | ~0.92 | ~0.97 | 轻微下降 |
| 2022 | ~0.91 | ~0.96 | 继续下降 |
| 2023 | ~0.90 | ~0.95 | 继续下降 |
| 2024 | 0.891 | 0.949 | 最低点 |

**关键发现**:
- 准确率从93.2%降至89.1%（下降4.1%）
- ROC AUC从0.976降至0.949（下降2.7%）
- **结论**: Academic领域存在轻微收敛趋势，但准确率仍保持在89%以上，说明核心分离能力仍然存在

### 1.3 特征重要性分析（Logistic Regression系数）

#### 1.3.1 News领域 - Top 5特征（按|系数|）

| 年份 | Top 5特征（系数） |
|------|------------------|
| 2020 | `flesch_reading_ease (3.486)`, `avg_sentence_length (2.785)`, `gunning_fog (-2.330)`, `vader_compound (-1.967)`, `Agreeableness (1.635)` |
| 2021 | `gunning_fog (-3.306)`, `avg_sentence_length (3.072)`, `flesch_reading_ease (2.330)`, `num_words (1.646)`, `vader_compound (-1.474)` |
| 2022 | `avg_sentence_length (2.752)`, `gunning_fog (-2.702)`, `flesch_reading_ease (2.011)`, `vader_compound (-1.959)`, `average_word_length (-1.756)` |
| 2023 | `avg_sentence_length (3.742)`, `flesch_reading_ease (2.477)`, `gunning_fog (-2.421)`, `vader_compound (-2.220)`, `Agreeableness (1.461)` |
| 2024 | `flesch_reading_ease (3.518)`, `avg_sentence_length (2.061)`, `gunning_fog (-1.813)`, `vader_compound (-1.682)`, `num_words (1.112)` |

**模式**:
- **Stylistic特征占主导**: `flesch_reading_ease`, `avg_sentence_length`, `gunning_fog`始终在前5
- **Emotional特征稳定**: `vader_compound`每年都出现
- **Cognitive特征**: `Agreeableness`在部分年份出现

#### 1.3.2 Blogs领域 - Top 5特征

| 年份 | Top 5特征（系数） |
|------|------------------|
| 2020 | `flesch_reading_ease (2.873)`, `avg_sentence_length (1.791)`, `Openness (1.552)`, `vader_neu (1.400)`, `Agreeableness (1.173)` |
| 2021 | `flesch_reading_ease (2.295)`, `Openness (1.849)`, `Agreeableness (1.551)`, `vader_neu (1.188)`, `avg_sentence_length (1.163)` |
| 2022 | `flesch_reading_ease (2.532)`, `Openness (1.783)`, `Agreeableness (1.472)`, `Extraversion (1.404)`, `Neuroticism (1.323)` |
| 2023 | `flesch_reading_ease (2.687)`, `Openness (1.890)`, `Agreeableness (1.698)`, `avg_sentence_length (1.477)`, `word_diversity (-1.209)` |
| 2024 | `Openness (1.874)`, `flesch_reading_ease (1.797)`, `Agreeableness (1.690)`, `vader_neu (1.237)`, `Neuroticism (1.128)` |

**模式**:
- **Cognitive特征更强**: `Openness`, `Agreeableness`始终在前列
- **Stylistic特征**: `flesch_reading_ease`稳定重要
- **Emotional特征**: `vader_neu`在部分年份重要

#### 1.3.3 Academic领域 - Top 5特征

| 年份 | Top 5特征（系数） |
|------|------------------|
| 2020 | `Agreeableness (1.546)`, `gunning_fog (-1.397)`, `flesch_reading_ease (1.364)`, `avg_sentence_length (1.038)`, `num_words (-0.775)` |
| 2021 | `flesch_reading_ease (1.658)`, `Agreeableness (1.417)`, `avg_sentence_length (1.079)`, `gunning_fog (-1.002)`, `Neuroticism (0.842)` |
| 2022 | `Agreeableness (1.597)`, `gunning_fog (-1.354)`, `avg_sentence_length (1.083)`, `flesch_reading_ease (1.078)`, `vader_neg (1.045)` |
| 2023 | `flesch_reading_ease (1.494)`, `Agreeableness (1.268)`, `gunning_fog (-1.037)`, `vader_neg (0.999)`, `num_words (-0.866)` |
| 2024 | `Agreeableness (1.952)`, `flesch_reading_ease (1.646)`, `vader_neg (1.225)`, `avg_sentence_length (0.990)`, `gunning_fog (-0.957)` |

**模式**:
- **Cognitive特征主导**: `Agreeableness`始终是最重要的特征之一
- **Stylistic特征稳定**: `flesch_reading_ease`, `avg_sentence_length`, `gunning_fog`持续重要
- **Emotional特征**: `vader_neg`在2022-2024年变得重要

### 1.4 统计显著性分析（p值）

#### 1.4.1 显著特征数量（p ≤ 0.01）

| 领域 | 2020 | 2021 | 2022 | 2023 | 2024 | 趋势 |
|------|------|------|------|------|------|------|
| News | 11 | 9 | 12 | 9 | 11 | 稳定（9-12个） |
| Blogs | 14 | 10 | 13 | 14 | 13 | 稳定（10-14个） |
| Academic | 9 | 9 | 6 | 9 | 10 | 2022年短暂下降后恢复 |

**关键发现**:
- **News**: 每年都有两位数显著特征，主要由长度/可读性指标主导
- **Blogs**: 每年至少10个显著特征，stylistic差异最难消除
- **Academic**: 2022年短暂降至6个，但2024年恢复至10个，说明核心维度仍然显著

#### 1.4.2 Top 5显著特征（按p值）

**News领域**:
- 2020: `avg_sentence_length (p=5.7e-35)`, `flesch_reading_ease (p=2.4e-10)`, `num_words (p=3.5e-10)`, `gunning_fog (p=4.6e-7)`, `vader_compound (p=5.0e-6)`
- 2024: `avg_sentence_length (p=2.1e-24)`, `num_words (p=5.0e-13)`, `flesch_reading_ease (p=7.3e-10)`, `vader_compound (p=1.3e-7)`, `gunning_fog (p=9.1e-5)`

**Blogs领域**:
- 2020: `flesch_reading_ease (p=2.9e-36)`, `avg_sentence_length (p=3.3e-18)`, `Openness (p=7.9e-17)`, `vader_compound (p=7.8e-11)`, `Extraversion (p=1.2e-10)`
- 2024: `flesch_reading_ease (p=2.0e-16)`, `Openness (p=2.1e-15)`, `word_diversity (p=4.5e-14)`, `vader_compound (p=5.3e-12)`, `Extraversion (p=5.6e-11)`

**Academic领域**:
- 2020: `avg_sentence_length (p=6.5e-23)`, `subjectivity (p=1.8e-14)`, `gunning_fog (p=1.1e-12)`, `flesch_reading_ease (p=7.4e-8)`, `vader_neg (p=1.1e-7)`
- 2024: `flesch_reading_ease (p=8.7e-16)`, `avg_sentence_length (p=1.1e-12)`, `Agreeableness (p=1.7e-11)`, `vader_neg (p=5.8e-10)`, `subjectivity (p=3.4e-9)`

**结论**: 即使在2024年，核心特征（如`flesch_reading_ease`, `avg_sentence_length`）的p值仍然极低（<1e-10），说明这些维度持续显著区分人类和LLM。

---

## 2. Macro Static Classification (整体静态分类)

### 2.1 实验设计

**数据规模**:
- **总样本数**: 74,995
  - Human: 15,000 (每个领域5,000)
  - LLM: 59,995 (4个模型 × 3个领域 × ~5,000)
- **数据分割**:
  - 训练集: 2020-2023年（59,996个样本）
  - 测试集: 2024年（14,999个样本）
- **交叉验证**: 5折分层交叉验证（仅训练集）

**特征**: 20个静态特征（与Yearly Validation相同）

**模型配置**:
- **Pipeline**: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`
- **任务**: 二分类（Human vs. LLM）

### 2.2 二分类结果（Human vs. LLM）- **RQ2核心结果**

**这是RQ2的主要任务**：使用20个静态特征区分Human和LLM文本。

| 指标 | 数值 |
|------|------|
| **测试准确率** | **88.84%** |
| **CV平均准确率** | **89.75% ± 0.24%** |
| **Human Precision** | 0.793 |
| **Human Recall** | 0.598 |
| **Human F1** | 0.682 |
| **LLM Precision** | 0.905 |
| **LLM Recall** | 0.961 |
| **LLM F1** | 0.932 |

**混淆矩阵** (行=真实标签, 列=预测标签):
```
                Human    LLM
Human           1,793   1,207
LLM               467  11,532
```

**关键发现**:
1. **整体性能**: **88.84%准确率**，接近micro time-series的~90%基准
   - 说明静态特征在群体层面可以有效区分Human和LLM
   - 即使使用最简单的静态特征（无时间序列信息），仍能达到接近90%的准确率
   
2. **类别不平衡影响**: Human recall较低（0.598），说明许多人类样本被误分类为LLM
   - 1,207个Human样本被误分类为LLM
   - 可能原因：部分人类文本的静态特征与LLM相似
   
3. **LLM识别**: LLM recall很高（0.961），说明LLM样本更容易被识别
   - 11,532个LLM样本被正确识别
   - 只有467个LLM样本被误分类为Human
   
4. **错误模式**: 主要错误是Human→LLM（1,207个），而非LLM→Human（467个）
   - 说明模型更倾向于将不确定的样本分类为LLM（由于类别不平衡）

**RQ2核心结论**: 
- ✅ **静态特征可以有效区分Human和LLM**（88.84%准确率）
- ✅ **即使在LV3最强prompting下，人类和LLM的静态特征指纹仍然可分离**
- ✅ **Stylistic特征（可读性、长度等）是最可靠的区分指标**

### 2.3 与RQ1对比

| 指标 | RQ1 (Micro Trajectory) | RQ2 (Macro Static) | 差异 |
|------|------------------------|-------------------|------|
| **任务** | Human vs. LLM (二分类) | Human vs. LLM (二分类) | 相同任务 |
| **准确率** | 94.17% (Unified Model) | **88.84% (Binary)** | -5.33% |
| **特征类型** | Variability + Geometry | Static only | 轨迹特征更强 |
| **数据规模** | 2,060轨迹 | 74,995文档 | Macro更大 |
| **结论** | 轨迹特征提供更强的区分能力 | 静态特征仍有88%准确率 | 两者互补 |

**关键对比**:
- **RQ1**: 使用轨迹特征（variability + geometry），达到94.17%准确率
- **RQ2**: 仅使用静态特征（无时间序列信息），达到88.84%准确率
- **差异**: 5.33%，说明轨迹特征提供了额外的区分能力
- **但**: 静态特征单独使用也能达到接近90%的准确率，说明静态特征本身就很有效

---

## 3. 主要结论

### 3.1 收敛趋势分析

1. **News领域**: 无收敛趋势
   - 准确率保持在96.5%
   - ROC AUC保持在0.986-0.995
   - 结论：News领域人类和LLM保持高度分离

2. **Blogs领域**: 无收敛趋势
   - 准确率稳定在88.1%-89.5%
   - ROC AUC稳定在0.95-0.96
   - 结论：Blogs领域分离度保持稳定

3. **Academic领域**: 轻微收敛
   - 准确率从93.2%降至89.1%（-4.1%）
   - ROC AUC从0.976降至0.949（-2.7%）
   - 结论：存在轻微收敛，但核心分离能力仍然存在（>89%）

### 3.2 特征重要性

**Stylistic特征（最重要）**:
- `flesch_reading_ease`: 在所有领域和年份都保持高重要性
- `avg_sentence_length`: 持续显著（p < 1e-10）
- `gunning_fog`: News和Academic领域的关键特征
- `num_words`: 长度相关特征稳定重要

**Cognitive特征**:
- `Agreeableness`: Academic领域最重要的特征
- `Openness`: Blogs领域的关键特征
- Big Five特征在部分领域/年份保持重要性

**Emotional特征**:
- `vader_compound`: News领域稳定重要
- `vader_neg`: Academic领域在2022-2024年变得重要
- `subjectivity`: Academic领域持续重要

### 3.3 研究意义

1. **方法学贡献**:
   - 首次系统性地按年份追踪Human vs. LLM分离度的变化
   - 证明了静态特征在群体层面的有效性（88.84%准确率）
   - 揭示了不同领域的收敛模式差异

2. **实践意义**:
   - 即使在LV3（最强prompting）下，静态特征仍能有效区分
   - Stylistic特征（可读性、长度）是最可靠的区分指标
   - 可用于大规模内容真实性检测

3. **理论意义**:
   - 证明了人类写作风格在群体层面的稳定性
   - 揭示了LLM生成文本的系统性差异（即使在最强prompting下）
   - 为理解人类写作的"指纹"提供了实证基础

---

## 4. 结果文件位置

### 4.1 Yearly Validation结果
- **分类结果**: `macro_results/yearly_static_ml_validation_<domain>.csv`
- **特征重要性**: `macro_results/yearly_feature_importance_<domain>.csv`
- **p值分析**: `macro_results/yearly_static_ml_lr_pvalues_<domain>.csv`
- **文档**: `macro_results/yearly_static_ml_validation.md`

### 4.2 Macro Static Classification结果
- **汇总结果**: `results/macro_static_classification/macro_static_classification_summary.json`
- **文档**: `docs/macro_static_classification_experiment.md`

### 4.3 Feature Convergence分析
- **收敛分析**: `macro_metrics/macro_feature_convergence.py`
- **输出**: `macro_dataset/process_output/Macro/<domain>_<MODEL>_lv3_feature_convergence.csv`

---

## 5. 附录

### 5.1 完整数据集统计

| 标签 | 模型 | 领域 | 样本数 |
|------|------|------|--------|
| human | HUMAN | academic | 5,000 |
| human | HUMAN | blogs | 5,000 |
| human | HUMAN | news | 5,000 |
| llm | DS | academic | 4,999 |
| llm | DS | blogs | 5,000 |
| llm | DS | news | 4,997 |
| llm | G12B | academic | 5,000 |
| llm | G12B | blogs | 5,000 |
| llm | G12B | news | 5,000 |
| llm | G4B | academic | 5,000 |
| llm | G4B | blogs | 5,000 |
| llm | G4B | news | 5,000 |
| llm | LMK | academic | 5,000 |
| llm | LMK | blogs | 4,999 |
| llm | LMK | news | 5,000 |
| **总计** | | | **74,995** |

### 5.2 20个静态特征列表

1. **Cognitive (5个)**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
2. **Emotional (6个)**: polarity, subjectivity, vader_compound, vader_pos, vader_neu, vader_neg
3. **Stylistic (9个)**: word_diversity, flesch_reading_ease, gunning_fog, average_word_length, num_words, avg_sentence_length, verb_ratio, function_word_ratio, content_word_ratio

### 5.3 模型参数配置

```python
# Logistic Regression
LogisticRegression(
    class_weight='balanced',  # 处理类别不平衡
    max_iter=2000,           # 最大迭代次数
    random_state=42          # 随机种子
)

# Data Preprocessing
SimpleImputer(strategy='median')  # 中位数填充缺失值
StandardScaler()                  # 标准化特征
```

---

**文档生成日期**: 2025年1月  
**最后更新**: 基于最新实验结果  
**维护者**: Research Team
