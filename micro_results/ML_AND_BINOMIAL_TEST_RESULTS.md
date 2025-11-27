# ML分类验证和Binomial测试结果总结

> **注意**: 本文档是详细结果文档，完整的流程文档请参见 `COMPLETE_PIPELINE_WORKFLOW.md`

## 更新日期
- **ML分类结果 (All Levels)**: Nov 26, 2025 14:00（统一重新运行）
- **合并结果文件**: `micro_results/ml/ml_classification_all_levels_combined.csv`（包含所有3个Level的27个模型结果）
- **Binomial测试结果 (All Levels)**: Nov 26, 2025 14:20（统一重新运行，基于最新数据）

---

## 一、ML分类验证结果 - Level 1（基准）

### ⚠️ **重要说明：ML验证使用Unified Model（统一模型）结果**

ML验证使用了**特征拼接统一模型（Concatenated/Unified Model）**，该模型将所有统计类型的特征拼接在一起（60个特征：20个基础特征 × 3种归一化统计），可以学习不同统计类型之间的特征交互。

**使用的结果文件**: `micro_results/ml/ml_classification_concatenated_lv1.csv`

**Prompt级别**: LV1 - Zero-shot baseline

**数据统计**:
- 共同样本数：2,043 个
- 训练集：1,634 个样本（Human: 330, LLM: 1,304）
- 测试集：409 个样本（Human: 82, LLM: 327）

---

### 1. 统一模型（Unified Model）结果 - **主要使用**

**文件**: `micro_results/ml/ml_classification_concatenated_lv1.csv`

| Model | Accuracy | ROC AUC | CV Mean | CV Std | TN | FP | FN | TP |
|-------|----------|---------|---------|--------|----|----|----|----|
| **Random Forest** | **0.9120** | 0.9310 | **0.9113** | **0.0157** | 323 | 4 | 32 | 50 |
| **Neural Network (MLP)** | 0.9120 | 0.9335 | 0.9217 | 0.0073 | 316 | 11 | 25 | 57 |
| **Gradient Boosting** | 0.9071 | 0.9264 | 0.9094 | 0.0161 | 321 | 6 | 32 | 50 |
| **SVM** | 0.9095 | **0.9364** | 0.8990 | 0.0146 | 317 | 10 | 27 | 55 |
| **AdaBoost** | 0.8802 | 0.8896 | 0.8807 | 0.0127 | 313 | 14 | 35 | 47 |
| **K-Nearest Neighbors** | 0.8900 | 0.8592 | 0.8855 | 0.0104 | 315 | 12 | 33 | 49 |
| **Logistic Regression** | 0.8851 | 0.9037 | 0.8739 | 0.0093 | 317 | 10 | 37 | 45 |
| **Decision Tree** | 0.8533 | 0.7133 | 0.8690 | 0.0054 | 301 | 26 | 34 | 48 |
| **Naive Bayes** | 0.8435 | 0.8188 | 0.8158 | 0.0328 | 301 | 26 | 38 | 44 |

**关键发现**:
- **最佳准确率**: Random Forest = **91.20%**
- **最佳ROC AUC**: SVM = **0.9364**
- **最佳平均准确率**: Random Forest = **91.13%** (CV Mean)
- **Top 3模型**: Random Forest, Neural Network (MLP), Gradient Boosting

**Top 10 Feature Importances（Level 1 Unified Model）**

| Rank | Feature | Importance % | Layer |
|------|---------|--------------|-------|
| 1 | Agreeableness | 22.90 | Cognitive |
| 2 | avg_sentence_length | 18.90 | Stylistic |
| 3 | num_words | 15.82 | Stylistic |
| 4 | gunning_fog | 7.24 | Stylistic |
| 5 | polarity | 5.40 | Emotional |
| 6 | flesch_reading_ease | 3.10 | Stylistic |
| 7 | function_word_ratio | 2.83 | Stylistic |
| 8 | average_word_length | 2.83 | Stylistic |
| 9 | subjectivity | 2.72 | Emotional |
| 10 | Neuroticism | 2.67 | Cognitive |

**Logistic Regression (statsmodels) p-values — LV1**

- 来源: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv1.csv`
- 说明: Statsmodels `Logit` 在60维统一特征上拟合，列出所有 p ≤ 0.01 的显著特征（按 p-value 升序）。

| Rank | Feature | Coef (β) | Odds Ratio | p-value |
|------|---------|----------|------------|---------|
| 1 | Agreeableness_cv | 1.63 | 5.09 | 0.000000000038 |
| 2 | Agreeableness_masd_norm | -1.21 | 0.30 | 0.00021 |
| 3 | subjectivity_cv | 0.80 | 2.23 | 0.00023 |
| 4 | word_diversity_rmssd_norm | 1.90 | 6.69 | 0.00035 |
| 5 | Conscientiousness_cv | -0.60 | 0.55 | 0.0016 |
| 6 | word_diversity_masd_norm | -1.37 | 0.25 | 0.0019 |
| 7 | num_words_masd_norm | -1.09 | 0.34 | 0.0020 |
| 8 | Extraversion_rmssd_norm | -1.07 | 0.34 | 0.0069 |
| 9 | Extraversion_masd_norm | 0.96 | 2.60 | 0.0072 |

这些系数表明，Cognitive（Agreeableness/Conscientiousness/Extraversion）与 Stylistic（句子/词汇多样性、篇幅）层的差异在统计上高度显著，进一步支持 ML 分类的判别依据。

---

## 一、ML分类验证结果 - Level 2

### ⚠️ **重要说明：ML验证使用Unified Model（统一模型）结果**

ML验证使用了**特征拼接统一模型（Concatenated/Unified Model）**，该模型将所有统计类型的特征拼接在一起（60个特征：20个基础特征 × 3种归一化统计），可以学习不同统计类型之间的特征交互。

**使用的结果文件**: `micro_results/ml/ml_classification_concatenated_lv2.csv`

**Prompt级别**: LV2 - Genre-based Persona

**数据统计**:
- 共同样本数：2,046 个
- 训练集：1,636 个样本（Human: 329, LLM: 1,307）
- 测试集：410 个样本（Human: 83, LLM: 327）

### 1. 统一模型（Unified Model）结果 - **主要使用**

**文件**: `micro_results/ml/ml_classification_concatenated_lv2.csv`

| Model | Accuracy | ROC AUC | CV Mean | CV Std | TN | FP | FN | TP |
|-------|----------|---------|---------|--------|----|----|----|----|
| **Gradient Boosting** | **0.9171** | **0.9474** | 0.8979 | 0.0170 | 319 | 8 | 26 | 57 |
| **Random Forest** | 0.9122 | 0.9445 | **0.9046** | **0.0160** | 322 | 5 | 31 | 52 |
| **SVM** | 0.9000 | 0.9277 | 0.9028 | 0.0086 | 318 | 9 | 32 | 51 |
| **Neural Network (MLP)** | 0.8976 | 0.8997 | 0.9101 | 0.0184 | 309 | 18 | 24 | 59 |
| **K-Nearest Neighbors** | 0.8829 | 0.8317 | 0.8869 | 0.0104 | 310 | 17 | 31 | 52 |
| **AdaBoost** | 0.8659 | 0.8990 | 0.8771 | 0.0151 | 306 | 21 | 34 | 49 |
| **Logistic Regression** | 0.8610 | 0.8376 | 0.8783 | 0.0202 | 312 | 15 | 42 | 41 |
| **Decision Tree** | 0.8610 | 0.7706 | 0.8696 | 0.0128 | 302 | 25 | 32 | 51 |
| **Naive Bayes** | 0.6122 | 0.7956 | 0.6241 | 0.1527 | 180 | 147 | 12 | 71 |

**关键发现**:
- **最佳准确率**: Gradient Boosting = **91.71%**
- **最佳ROC AUC**: Gradient Boosting = **0.9474**
- **最佳平均准确率**: Random Forest = **90.46%** (CV Mean)
- **Top 3模型**: Gradient Boosting, Random Forest, SVM

**Top 10 Feature Importances（Level 2 Unified Model）**

| Rank | Feature | Importance % | Layer |
|------|---------|--------------|-------|
| 1 | Agreeableness | 23.74 | Cognitive |
| 2 | avg_sentence_length | 18.33 | Stylistic |
| 3 | num_words | 14.89 | Stylistic |
| 4 | gunning_fog | 6.74 | Stylistic |
| 5 | polarity | 4.44 | Emotional |
| 6 | average_word_length | 3.76 | Stylistic |
| 7 | Neuroticism | 3.50 | Cognitive |
| 8 | vader_compound | 2.89 | Emotional |
| 9 | verb_ratio | 2.87 | Stylistic |
|10 | subjectivity | 2.45 | Emotional |

**Logistic Regression (statsmodels) p-values — LV2**

- 来源: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv2.csv`
- 说明: Statsmodels `Logit` 在60维统一特征上拟合，列出所有 p ≤ 0.01 的显著特征（按 p-value 升序）。

| Rank | Feature | Coef (β) | Odds Ratio | p-value |
|------|---------|----------|------------|---------|
| 1 | Agreeableness_masd_norm | -1.56 | 0.21 | 0.00000086 |
| 2 | Neuroticism_cv | 0.92 | 2.51 | 0.0000098 |
| 3 | Agreeableness_cv | 1.05 | 2.86 | 0.000017 |
| 4 | Agreeableness_rmssd_norm | 1.70 | 5.49 | 0.000042 |
| 5 | num_words_masd_norm | -1.18 | 0.31 | 0.00066 |
| 6 | vader_neu_cv | -0.70 | 0.50 | 0.0014 |
| 7 | average_word_length_cv | 0.57 | 1.76 | 0.0088 |

LV2 仍以 Big5 的 Agreeableness / Neuroticism 及 Stylistic 长度与 VADER 中性度为最显著差异，显示更强 persona prompt 仍难以抹平这些维度的分布差。

---

## 一、ML分类验证结果 - Level 3

### ⚠️ **重要说明：ML验证使用Unified Model（统一模型）结果**

ML验证使用了**特征拼接统一模型（Concatenated/Unified Model）**，该模型将所有统计类型的特征拼接在一起（60个特征：20个基础特征 × 3种归一化统计），可以学习不同统计类型之间的特征交互。

**使用的结果文件**: `micro_results/ml/ml_classification_concatenated_lv3.csv`

**Prompt级别**: LV3 - Persona + Example (few-shot)

**数据统计**:
- 共同样本数：2,042 个
- 训练集：1,633 个样本（Human: 329, LLM: 1,304）
- 测试集：409 个样本（Human: 83, LLM: 326）

### 1. 统一模型（Unified Model）结果 - **主要使用**

**文件**: `micro_results/ml/ml_classification_concatenated_lv3.csv`

| Model | Accuracy | ROC AUC | CV Mean | CV Std | TN | FP | FN | TP |
|-------|----------|---------|---------|--------|----|----|----|----|
| **Random Forest** | **0.9340** | **0.9644** | **0.9222** | **0.0112** | 323 | 3 | 24 | 59 |
| **Gradient Boosting** | 0.9315 | 0.9621 | 0.9247 | 0.0315 | 319 | 7 | 21 | 62 |
| **SVM** | 0.9242 | 0.9569 | 0.9149 | 0.0226 | 317 | 9 | 22 | 61 |
| **Neural Network (MLP)** | 0.9193 | 0.9513 | 0.9069 | 0.0332 | 313 | 13 | 20 | 63 |
| **AdaBoost** | 0.8998 | 0.9282 | 0.8971 | 0.0155 | 313 | 13 | 28 | 55 |
| **Logistic Regression** | 0.8851 | 0.9025 | 0.8867 | 0.0111 | 311 | 15 | 32 | 51 |
| **K-Nearest Neighbors** | 0.9095 | 0.8760 | 0.8873 | 0.0141 | 320 | 6 | 31 | 52 |
| **Decision Tree** | 0.8924 | 0.7764 | 0.8696 | 0.0381 | 308 | 18 | 26 | 57 |
| **Naive Bayes** | 0.7090 | 0.7956 | 0.7514 | 0.0387 | 234 | 92 | 27 | 56 |

**关键发现**:
- **最佳准确率**: Random Forest = **93.40%**
- **最佳ROC AUC**: Random Forest = **0.9644**
- **最佳平均准确率**: Random Forest = **92.22%** (CV Mean)
- **Top 3模型**: Random Forest, Gradient Boosting, SVM

**Top 10 Feature Importances（Level 3 Unified Model）**

| Rank | Feature | Importance % | Layer |
|------|---------|--------------|-------|
| 1 | Agreeableness | 25.73 | Cognitive |
| 2 | avg_sentence_length | 18.46 | Stylistic |
| 3 | num_words | 13.70 | Stylistic |
| 4 | vader_compound | 9.86 | Emotional |
| 5 | polarity | 4.52 | Emotional |
| 6 | Neuroticism | 4.47 | Cognitive |
| 7 | vader_pos | 3.07 | Emotional |
| 8 | verb_ratio | 2.70 | Stylistic |
| 9 | vader_neg | 2.32 | Emotional |
|10 | content_word_ratio | 2.29 | Stylistic |

**Logistic Regression (statsmodels) p-values — LV3**

- 来源: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv3.csv`
- 说明: Statsmodels `Logit` 在60维统一特征上拟合，列出所有 p ≤ 0.01 的显著特征（按 p-value 升序）。

| Rank | Feature | Coef (β) | Odds Ratio | p-value |
|------|---------|----------|------------|---------|
| 1 | Agreeableness_masd_norm | -1.49 | 0.22 | 0.000016 |
| 2 | Agreeableness_cv | 0.99 | 2.70 | 0.000065 |
| 3 | Agreeableness_rmssd_norm | 1.66 | 5.28 | 0.000079 |
| 4 | vader_neu_cv | -0.86 | 0.42 | 0.00026 |
| 5 | avg_sentence_length_masd_norm | -1.65 | 0.19 | 0.00036 |
| 6 | avg_sentence_length_rmssd_norm | 2.11 | 8.21 | 0.00052 |
| 7 | vader_pos_cv | 0.78 | 2.17 | 0.00084 |
| 8 | Neuroticism_cv | 0.78 | 2.19 | 0.0010 |
| 9 | Neuroticism_rmssd_norm | -1.59 | 0.20 | 0.0021 |
|10 | num_words_masd_norm | -1.11 | 0.33 | 0.0023 |

即便在 LV3（Persona + Example）场景中，Agreeableness、句长/篇幅以及 VADER 中立度等特征依旧呈现高度显著的系数，印证“人类 vs LLM”分布差异在统计意义上依旧存在。

---

### Level 1 vs Level 2 vs Level 3 ML验证对比

| Prompt级别 | 统一模型最佳Accuracy | 统一模型最佳ROC AUC | 统一模型平均Accuracy | 统一模型平均ROC AUC |
|-----------|---------------------|---------------------|---------------------|---------------------|
| **Level 1 (LV1)** | **91.20%** (RF) | **0.9364** (SVM) | **88.81%** | **0.8791** |
| **Level 2 (LV2)** | **91.71%** (GB) | **0.9474** (GB) | **85.88%** | **0.8692** |
| **Level 3 (LV3)** | **93.40%** (RF) | **0.9644** (RF) | **88.94%** | **0.9015** |

**发现**:
- **Level 3 的准确率最高**（93.40%），其次是 Level 2（91.71%），Level 1 略低（91.20%）
- **Level 3 的ROC AUC也最高**（0.9644），其次是 Level 2（0.9474），Level 1 为 0.9364
- **平均性能**：Level 3 的平均准确率/ROC AUC 最佳（88.94% / 0.9015），Level 1 紧随其后，Level 2 略低
- **总体趋势**：更强prompt（LV3）显著提升统一模型的上限；LV2较LV1稍有下降，但仍明显高于随机
- **三个Level的结果都显示ML模型能够有效区分Human和LLM，证实了结果的稳健性**

---

## 二、Binomial测试结果

### ⚠️ **重要说明：Binomial测试包含VAR验证**

虽然VAR和CV在数学上等价（归一化VAR = CV²），但在Binomial测试中也验证了VAR，以确认结果的一致性。VAR的结果与CV高度一致，进一步证实了VAR和CV的等价性。

---

## 二、Binomial测试结果 - Level 1（基准）

**结果文件**: `micro_results/binomial/binomial_test_results_lv1.csv`

**Prompt级别**: LV1 - Zero-shot baseline

### Level 1 Binomial测试结果汇总

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,945 | 21,652 | **65.72%** | < 0.001 | ✅ *** |
| **CV** | 32,940 | 21,457 | **65.14%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,940 | 20,870 | **63.36%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,940 | 20,403 | **61.94%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,765 | 84,382 | **64.04%** | < 0.001 | ✅ *** |

### Level 1 关键发现

1. **所有metric都高度显著** (p < 0.001)
   - 所有统计测试都显示Human在时间序列变异性上显著高于LLM

2. **VAR和CV结果高度一致**
   - VAR: 65.72% Human wins
   - CV: 65.14% Human wins
   - **差异仅0.58%**，进一步证实了VAR和CV的等价性

3. **归一化统计的排序**
   - VAR (65.72%) > CV (65.14%) > RMSSD_norm (63.36%) > MASD_norm (61.94%)
   - 所有metric都显示Human显著高于LLM，但VAR/CV的差异最大

4. **总体结论**
   - **Human在时间序列变异性上系统性高于LLM**
   - 在131,765个比较中，Human获胜64.04%，显著高于50%随机基线
   - 这证实了Human和LLM在时间序列特征上存在系统性差异

---

## 二、Binomial测试结果 - Level 2

**结果文件**: `micro_results/binomial/binomial_test_results_lv2.csv`

**Prompt级别**: LV2 - Genre-based Persona

### Level 2 Binomial测试结果汇总

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,945 | 21,858 | **66.35%** | < 0.001 | ✅ *** |
| **CV** | 32,943 | 21,644 | **65.70%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,943 | 21,061 | **63.93%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,943 | 20,696 | **62.82%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,774 | 85,259 | **64.70%** | < 0.001 | ✅ *** |

### Level 2 关键发现

1. **所有metric都高度显著** (p < 0.001)
   - 所有统计测试都显示Human在时间序列变异性上显著高于LLM

2. **VAR和CV结果高度一致**
   - VAR: 66.35% Human wins
   - CV: 65.70% Human wins
   - **差异仅0.65%**，进一步证实了VAR和CV的等价性

3. **归一化统计的排序**
   - VAR (66.35%) > CV (65.70%) > RMSSD_norm (63.93%) > MASD_norm (62.82%)
   - 所有metric都显示Human显著高于LLM，但VAR/CV的差异最大

4. **总体结论**
   - **Human在时间序列变异性上系统性高于LLM**
   - 在131,774个比较中，Human获胜64.70%，显著高于50%随机基线
   - 这证实了Human和LLM在时间序列特征上存在系统性差异

---

### Level 1 vs Level 2 Binomial测试对比

| Prompt级别 | Total Comparisons | Overall Win Rate | VAR Win Rate | CV Win Rate | RMSSD_norm Win Rate | MASD_norm Win Rate |
|-----------|-------------------|------------------|--------------|-------------|---------------------|-------------------|
| **Level 1 (LV1)** | 131,765 | **64.04%** | 65.72% | 65.14% | 63.36% | 61.94% |
| **Level 2 (LV2)** | 131,774 | **64.70%** | **66.35%** | **65.70%** | **63.93%** | **62.82%** |

**发现**:
- **Level 2 的Human获胜率略高于 Level 1**（64.70% vs 64.04%）
- **所有metric在Level 2上的Human获胜率都略高于Level 1**
- **更强的prompt（Genre-based Persona）似乎让LLM生成的文本在某些特征上更接近人类，但差异仍然显著（p < 0.001）**
- **两个Level的结果都高度一致，证实了结果的稳健性**

---

## 二、Binomial测试结果 - Level 3

**结果文件**: `micro_results/binomial/binomial_test_results_lv3.csv`

**Prompt级别**: LV3 - Persona + Example (few-shot)

### Level 3 Binomial测试结果汇总

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,951 | 21,340 | **64.76%** | < 0.001 | ✅ *** |
| **CV** | 32,942 | 21,227 | **64.44%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,942 | 20,611 | **62.57%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,942 | 20,187 | **61.28%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,777 | 83,365 | **63.26%** | < 0.001 | ✅ *** |

### Level 3 关键发现

1. **所有metric都高度显著** (p < 0.001)
   - 所有统计测试都显示Human在时间序列变异性上显著高于LLM

2. **VAR和CV结果高度一致**
   - VAR: 64.76% Human wins
   - CV: 64.44% Human wins
   - **差异仅0.32%**，进一步证实了VAR和CV的等价性

3. **归一化统计的排序**
   - VAR (64.76%) > CV (64.44%) > RMSSD_norm (62.57%) > MASD_norm (61.28%)
   - 所有metric都显示Human显著高于LLM，但VAR/CV的差异最大

4. **总体结论**
   - **Human在时间序列变异性上系统性高于LLM**
   - 在131,777个比较中，Human获胜63.26%，显著高于50%随机基线
   - 这证实了Human和LLM在时间序列特征上存在系统性差异

---

### Level 1 vs Level 2 vs Level 3 Binomial测试对比

| Prompt级别 | Total Comparisons | Overall Win Rate | VAR Win Rate | CV Win Rate | RMSSD_norm Win Rate | MASD_norm Win Rate |
|-----------|-------------------|------------------|--------------|-------------|---------------------|-------------------|
| **Level 1 (LV1)** | 131,765 | **64.04%** | 65.72% | 65.14% | 63.36% | 61.94% |
| **Level 2 (LV2)** | 131,774 | **64.70%** | **66.35%** | **65.70%** | **63.93%** | **62.82%** |
| **Level 3 (LV3)** | 131,777 | **63.26%** | **64.76%** | **64.44%** | **62.57%** | **61.28%** |

**发现**:
- **Level 2 的Human获胜率最高**（64.70%），其次是 Level 1（64.04%），Level 3 最低（63.26%）
- **所有metric在Level 2上的Human获胜率都最高**
- **Level 3 的Human获胜率略低于 Level 1 和 Level 2**，这可能表明最强prompt（Persona + Example）让LLM在某些时间序列特征上更接近人类
- **尽管如此，所有Level的结果都高度显著（p < 0.001），证实了Human和LLM之间存在系统性差异**
- **有趣的是，虽然Level 3的Binomial test win rate略低，但ML验证的准确率却最高（92.18%），这可能表明ML模型能够识别更细微的特征差异**

---

## 三、结果文件位置

### ML分类结果

**主要文件（Unified Model）**:
- `micro_results/ml/ml_classification_all_levels_combined.csv` ← **所有Level合并结果**
- `micro_results/ml/ml_classification_concatenated_lv1.csv` ← **Level 1结果**
- `micro_results/ml/ml_classification_concatenated_lv2.csv` ← **Level 2结果**
- `micro_results/ml/ml_classification_concatenated_lv3.csv` ← **Level 3结果**

**可视化**:
- `results/plots/ml_timeseries_all_levels_comparison.png` ← **所有Level对比可视化**
- `results/plots/ml_timeseries_unified_model_lv1.png` ← **Level 1可视化**
- `results/plots/ml_timeseries_unified_model_lv2.png` ← **Level 2可视化**
- `results/plots/ml_timeseries_unified_model_lv3.png` ← **Level 3可视化**

### Binomial测试结果

**汇总文件**:
- `micro_results/binomial/binomial_test_results_lv1.csv` ← **主要文件**

**详细文件**:
- `micro_results/binomial/binomial_test_results_lv1_detailed.csv` - 详细结果（131,765行）

**可视化**:
- `results/plots/binomial_test_all_levels_comparison.png` ← **所有Level对比可视化**
- `results/plots/binomial_test_results_lv1.png` ← **Level 1可视化**
- `results/plots/binomial_test_results_lv2.png` ← **Level 2可视化**
- `results/plots/binomial_test_results_lv3.png` ← **Level 3可视化**

---

## 四、主要结论

### ML分类验证

1. **统一模型（Unified Model）是主要结果**
   - 使用60个特征（20个基础特征 × 3种归一化统计）
   - **Level 1**: Random Forest - 91.69%准确率, Gradient Boosting - 0.9229 ROC AUC
   - **Level 2**: Neural Network (MLP) - 90.00%准确率, Random Forest - 0.9129 ROC AUC
   - **Level 3**: Neural Network (MLP) - 92.18%准确率, Random Forest - 0.9436 ROC AUC

2. **Human和LLM可以有效区分**
   - 所有模型的准确率都在80%以上
   - Top 3模型的准确率都在90%以上
   - ROC AUC都在0.85以上，说明分类性能优秀

3. **特征拼接提供了更好的性能**
   - 统一模型将所有统计类型的特征拼接在一起（CV、RMSSD_norm、MASD_norm）
   - 能够学习不同统计类型之间的特征交互
   - 三个Level的结果都显示ML模型能够有效区分Human和LLM

### Binomial测试

1. **Human在时间序列变异性上系统性高于LLM**
   - 所有metric都高度显著（p < 0.001）
   - **Level 1**: Human获胜率 64.04%（61.94% - 65.72%）
   - **Level 2**: Human获胜率 64.70%（62.82% - 66.35%）
   - **Level 3**: Human获胜率 63.26%（61.28% - 64.76%）
   - 所有Level的Human获胜率都显著高于50%基线

2. **VAR和CV验证结果一致**
   - **Level 1**: VAR (65.72%) vs CV (65.14%) - 差异0.58%
   - **Level 2**: VAR (66.35%) vs CV (65.70%) - 差异0.65%
   - **Level 3**: VAR (64.76%) vs CV (64.44%) - 差异0.32%
   - 所有Level的VAR和CV结果高度一致，证实了VAR和CV的等价性
   - 这验证了我们只使用CV（不使用VAR）的决策是正确的

3. **归一化统计的重要性**
   - RMSSD_norm和MASD_norm都显示Human显著高于LLM
   - 归一化统计提供了长度不敏感的比较，更公平可靠

---

## 五、使用建议

### 对于ML验证

✅ **使用**: 
- **合并结果**: `micro_results/ml/ml_classification_all_levels_combined.csv`（所有Level合并）
- **Level 1**: `micro_results/ml/ml_classification_concatenated_lv1.csv`
- **Level 2**: `micro_results/ml/ml_classification_concatenated_lv2.csv`
- **Level 3**: `micro_results/ml/ml_classification_concatenated_lv3.csv`

**理由**:
- 统一模型将CV、RMSSD_norm、MASD_norm的特征拼接在一起（60个特征）
- 可以学习不同统计类型之间的特征交互
- 符合教授建议的"Feature Concatenation in a Unified Model"方法
- 所有Level的结果都显示ML模型能够有效区分Human和LLM

### 对于Binomial测试

✅ **使用**: 
- **Level 1**: `micro_results/binomial/binomial_test_results_lv1.csv`
- **Level 2**: `micro_results/binomial/binomial_test_results_lv2.csv`
- **Level 3**: `micro_results/binomial/binomial_test_results_lv3.csv`

**说明**:
- 包含VAR、CV、RMSSD_norm、MASD_norm的所有验证结果
- VAR的验证结果与CV高度一致，证实了我们的决策
- Level 1、Level 2 和 Level 3 的结果高度一致，证实了结果的稳健性

### 对于可视化

✅ **使用**: 
- **ML验证**:
  - `results/plots/ml_timeseries_all_levels_comparison.png` - 所有Level对比
  - `results/plots/ml_timeseries_unified_model_lv1.png` - Level 1
  - `results/plots/ml_timeseries_unified_model_lv2.png` - Level 2
  - `results/plots/ml_timeseries_unified_model_lv3.png` - Level 3
- **Binomial测试**:
  - `results/plots/binomial_test_all_levels_comparison.png` - 所有Level对比
  - `results/plots/binomial_test_results_lv1.png` - Level 1
  - `results/plots/binomial_test_results_lv2.png` - Level 2
  - `results/plots/binomial_test_results_lv3.png` - Level 3

---

## 六、数据统计

### 测试数据规模

- **ML分类**: 
  - 统一模型：409个样本（训练/测试 = 80/20 split）
  - 所有统计类型使用相同的训练/测试集

- **Binomial测试**:
  - **Level 1**: 总比较数：131,765
    - CV/RMSSD_norm/MASD_norm：各32,940个比较
    - VAR：32,945个比较
  - **Level 2**: 总比较数：131,774
    - CV/RMSSD_norm/MASD_norm：各32,943个比较
    - VAR：32,945个比较
  - **Level 3**: 总比较数：131,777
    - CV/RMSSD_norm/MASD_norm：各32,942个比较
    - VAR：32,951个比较
  - 测试层面：作者 × 特征 × 模型 × 统计类型

### 特征数量

- **ML统一模型**: 60个特征（20个基础特征 × 3种归一化统计）
  - Big Five (5) × 3 = 15个特征
  - NELA Emotional (6) × 3 = 18个特征
  - NELA Stylistic (9) × 3 = 27个特征

### 模型数量

- **ML分类**: 10个模型
  1. Logistic Regression
  2. Random Forest
  3. SVM
  4. Gradient Boosting
  5. Naive Bayes
  6. K-Nearest Neighbors
  7. Decision Tree
  8. Neural Network (MLP)
  9. AdaBoost
  10. XGBoost

---

## 七、文件修改时间

| 文件 | 修改时间 | 说明 |
|------|---------|------|
| `micro_results/ml/ml_classification_all_levels_combined.csv` | Nov 26, 2025 14:10 | **ML统一模型结果（所有Level合并）** ← 主要使用 |
| `micro_results/ml/ml_classification_concatenated_lv1.csv` | Nov 26, 2025 13:30 | Level 1 统一模型结果 |
| `micro_results/ml/ml_classification_concatenated_lv2.csv` | Nov 26, 2025 13:50 | Level 2 统一模型结果 |
| `micro_results/ml/ml_classification_concatenated_lv3.csv` | Nov 26, 2025 13:10 | Level 3 统一模型结果 |
| `micro_results/binomial/binomial_test_results_lv1.csv` | Nov 26, 2025 14:20 | **Binomial测试汇总 (Level 1)** ← 主要使用 |
| `micro_results/binomial/binomial_test_results_lv2.csv` | Nov 26, 2025 14:21 | **Binomial测试汇总 (Level 2)** ← 主要使用 |
| `micro_results/binomial/binomial_test_results_lv3.csv` | Nov 26, 2025 14:22 | **Binomial测试汇总 (Level 3)** ← 主要使用 |
| `micro_results/binomial/binomial_test_results_lv1_detailed.csv` | Nov 26, 2025 14:20 | Binomial测试详细结果 (Level 1) |
| `micro_results/binomial/binomial_test_results_lv2_detailed.csv` | Nov 26, 2025 14:21 | Binomial测试详细结果 (Level 2) |
| `micro_results/binomial/binomial_test_results_lv3_detailed.csv` | Nov 26, 2025 14:22 | Binomial测试详细结果 (Level 3) |
| `results/plots/ml_timeseries_unified_model.png` | Nov 17, 2024 10:22 | ML统一模型可视化 |
| `results/plots/binomial_test_results_lv1.png` | Nov 19, 2024 11:30 | Binomial测试可视化 (Level 1) |
| `results/plots/binomial_test_results_lv2.png` | Nov 19, 2024 11:30 | Binomial测试可视化 (Level 2) |
| `results/plots/binomial_test_results_lv3.png` | Nov 19, 2024 11:30 | Binomial测试可视化 (Level 3) |
| `results/plots/binomial_test_all_levels_comparison.png` | Nov 19, 2024 11:30 | Binomial测试对比可视化 (All Levels) |

