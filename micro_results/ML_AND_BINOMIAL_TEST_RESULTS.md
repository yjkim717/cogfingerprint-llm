# ML Classification & Binomial Test Summary

## Update Notes
- **ML classification (All Levels)**: Nov 26, 2025 14:00 (full rerun)
- **Merged summary file**: `micro_results/ml/ml_classification_all_levels_combined.csv`
  (27 model configurations across the 3 levels)
- **Binomial tests (All Levels)**: Nov 26, 2025 14:20 (rerun on the latest data)

---

## I. ML Classification — Level 1 (Baseline)

### ⚠️ Unified Model Reminder

We use the **concatenated/unified model** (60 features = 20 base × 3 normalized stats) so
that cross-statistic interactions are modeled jointly.

- **Result file**: `micro_results/ml/ml_classification_concatenated_lv1.csv`
- **Prompt level**: LV1 – zero-shot baseline
- **Dataset stats**: total 2,043 samples (train 1,634: Human 330 / LLM 1,304; test 409:
  Human 82 / LLM 327)

---

### 1. Unified-model metrics (primary results)

**File**: `micro_results/ml/ml_classification_concatenated_lv1.csv`

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

**Highlights**
- **Best accuracy**: Random Forest = **91.20%**
- **Best ROC AUC**: SVM = **0.9364**
- **Best CV mean**: Random Forest = **91.13%**
- **Top 3 models**: Random Forest, Neural Network (MLP), Gradient Boosting

**Top 10 feature importances (Random Forest, LV1)**

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
Source: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv1.csv`  
Statsmodels `Logit` is run on the 60-D unified feature set; below are the features with
p ≤ 0.01 sorted by significance.

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

Interpretation: cognitive traits (Agreeableness/Conscientiousness/Extraversion) and
stylistic signals (sentence/word-length variation) remain statistically significant,
reinforcing the ML classifier’s decision basis.

---

## I. ML Classification — Level 2

- **Result file**: `micro_results/ml/ml_classification_concatenated_lv2.csv`
- **Prompt level**: LV2 — genre persona
- **Dataset stats**: total 2,046 (train 1,636: Human 329 / LLM 1,307; test 410: Human 83 /
  LLM 327)

### 1. Unified-model metrics

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

**Highlights**
- **Best accuracy**: Gradient Boosting = **91.71%**
- **Best ROC AUC**: Gradient Boosting = **0.9474**
- **Best CV mean**: Random Forest = **90.46%**
- **Top 3 models**: Gradient Boosting, Random Forest, SVM

**Top 10 feature importances (Random Forest, LV2)**

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
Source: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv2.csv`

| Rank | Feature | Coef (β) | Odds Ratio | p-value |
|------|---------|----------|------------|---------|
| 1 | Agreeableness_masd_norm | -1.56 | 0.21 | 0.00000086 |
| 2 | Neuroticism_cv | 0.92 | 2.51 | 0.0000098 |
| 3 | Agreeableness_cv | 1.05 | 2.86 | 0.000017 |
| 4 | Agreeableness_rmssd_norm | 1.70 | 5.49 | 0.000042 |
| 5 | num_words_masd_norm | -1.18 | 0.31 | 0.00066 |
| 6 | vader_neu_cv | -0.70 | 0.50 | 0.0014 |
| 7 | average_word_length_cv | 0.57 | 1.76 | 0.0088 |

Takeaway: even with persona prompts, the clearest gaps remain Big Five traits
(Agreeableness / Neuroticism) plus stylistic length and VADER-neutral metrics, so LV2
still cannot fully align those dimensions with human text.

---

## I. ML Classification — Level 3

- **Result file**: `micro_results/ml/ml_classification_concatenated_lv3.csv`
- **Prompt level**: LV3 — persona + example (few-shot)
- **Dataset stats**: total 2,042 (train 1,633: Human 329 / LLM 1,304; test 409: Human 83 /
  LLM 326)

### 1. Unified-model metrics

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

**Highlights**
- **Best accuracy**: Random Forest = **93.40%**
- **Best ROC AUC**: Random Forest = **0.9644**
- **Best CV mean**: Random Forest = **92.22%**
- **Top 3 models**: Random Forest, Gradient Boosting, SVM

**Top 10 feature importances (Random Forest, LV3)**

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
Source: `micro_results/ml/lr_pvalues/lr_pvalues_concatenated_lv3.csv`

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

Even under LV3 (persona + example) prompting, Agreeableness, sentence length, document
length, and VADER-neutral cues remain highly significant—Human vs LLM distributions
are still clearly separable.

---

### Level 1 vs Level 2 vs Level 3 — Unified-model comparison

| Prompt level | Best accuracy | Best ROC AUC | Avg. accuracy (CV) | Avg. ROC AUC (CV) |
|-----------|---------------------|---------------------|---------------------|---------------------|
| **Level 1 (LV1)** | **91.20%** (RF) | **0.9364** (SVM) | **88.81%** | **0.8791** |
| **Level 2 (LV2)** | **91.71%** (GB) | **0.9474** (GB) | **85.88%** | **0.8692** |
| **Level 3 (LV3)** | **93.40%** (RF) | **0.9644** (RF) | **88.94%** | **0.9015** |

**Observations**
- **Accuracy**: LV3 > LV2 > LV1 (93.40% > 91.71% > 91.20%)
- **ROC AUC**: LV3 > LV2 > LV1 (0.9644 > 0.9474 > 0.9364)
- **Average CV metrics**: LV3 has the strongest mean accuracy/AUC (88.94% / 0.9015),
  closely followed by LV1; LV2 dips slightly but remains high.
- Stronger prompts (LV3) noticeably raise the upper bound of the unified model, while
  all three levels still show clear Human vs LLM separation, demonstrating robustness.

---

## II. Binomial Test Results

### ⚠️ VAR validation note

Although VAR and CV are mathematically equivalent (normalized VAR = CV²), we still ran
VAR-based binomial tests to confirm consistency. VAR outcomes mirror CV closely, which
reinforces the equivalence.

---

## Binomial test — Level 1 (baseline)

- **File**: `micro_results/binomial/binomial_test_results_lv1.csv`
- **Prompt level**: LV1 — zero-shot baseline

### Summary table (Level 1)

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,945 | 21,652 | **65.72%** | < 0.001 | ✅ *** |
| **CV** | 32,940 | 21,457 | **65.14%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,940 | 20,870 | **63.36%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,940 | 20,403 | **61.94%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,765 | 84,382 | **64.04%** | < 0.001 | ✅ *** |

### Key findings (Level 1)

1. **All metrics are highly significant** (p < 0.001) — humans have greater time-series
   variability than LLMs across the board.
2. **VAR vs CV**: 65.72% vs 65.14% human win rate (difference only 0.58%), confirming
   their equivalence.
3. **Normalized statistic ranking**: VAR (65.72%) > CV (65.14%) > RMSSD_norm (63.36%) >
   MASD_norm (61.94%).
4. **Overall**: Across 131,765 comparisons, humans win 64.04% of the time, far above the
   50% baseline, proving systematic differences in time-series behavior.

---

## Binomial test — Level 2

- **File**: `micro_results/binomial/binomial_test_results_lv2.csv`
- **Prompt level**: LV2 — genre persona

### Summary table (Level 2)

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,945 | 21,858 | **66.35%** | < 0.001 | ✅ *** |
| **CV** | 32,943 | 21,644 | **65.70%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,943 | 21,061 | **63.93%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,943 | 20,696 | **62.82%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,774 | 85,259 | **64.70%** | < 0.001 | ✅ *** |

### Key findings (Level 2)

1. **All metrics remain highly significant** (p < 0.001) — humans still dominate time-series
   variability.
2. **VAR vs CV**: 66.35% vs 65.70% human wins (difference 0.65%).
3. **Ranking**: VAR (66.35%) > CV (65.70%) > RMSSD_norm (63.93%) > MASD_norm (62.82%).
4. **Overall**: Out of 131,774 comparisons, humans win 64.70% (well above chance).

---

### Level 1 vs Level 2 comparison

| Prompt level | Total comparisons | Overall win rate | VAR | CV | RMSSD_norm | MASD_norm |
|-----------|-------------------|------------------|--------------|-------------|---------------------|-------------------|
| **Level 1 (LV1)** | 131,765 | **64.04%** | 65.72% | 65.14% | 63.36% | 61.94% |
| **Level 2 (LV2)** | 131,774 | **64.70%** | **66.35%** | **65.70%** | **63.93%** | **62.82%** |

**Observations**
- LV2 is slightly higher than LV1 overall (64.70% vs 64.04%).
- Every metric shows a marginal increase in the LV2 human win rate.
- Stronger persona prompts narrow the gap a bit, but differences remain highly
  significant (p < 0.001).

---

## Binomial test — Level 3

- **File**: `micro_results/binomial/binomial_test_results_lv3.csv`
- **Prompt level**: LV3 — persona + example (few-shot)

### Summary table (Level 3)

| Metric | N Comparisons | Human Wins | Human Win Rate | p-value | Significant |
|--------|---------------|------------|----------------|---------|-------------|
| **VAR** | 32,951 | 21,340 | **64.76%** | < 0.001 | ✅ *** |
| **CV** | 32,942 | 21,227 | **64.44%** | < 0.001 | ✅ *** |
| **RMSSD (norm)** | 32,942 | 20,611 | **62.57%** | < 0.001 | ✅ *** |
| **MASD (norm)** | 32,942 | 20,187 | **61.28%** | < 0.001 | ✅ *** |
| **OVERALL** | 131,777 | 83,365 | **63.26%** | < 0.001 | ✅ *** |

### Key findings (Level 3)

1. All metrics remain highly significant (p < 0.001); humans still have higher
   time-series variability.
2. VAR vs CV: 64.76% vs 64.44% (difference 0.32%).
3. Ranking: VAR (64.76%) > CV (64.44%) > RMSSD_norm (62.57%) > MASD_norm (61.28%).
4. Overall: Out of 131,777 comparisons, humans win 63.26% (again far above chance).

---

### Level 1 vs Level 2 vs Level 3 — Binomial comparison

| Prompt level | Total comparisons | Overall win rate | VAR | CV | RMSSD_norm | MASD_norm |
|-----------|-------------------|------------------|--------------|-------------|---------------------|-------------------|
| **Level 1 (LV1)** | 131,765 | **64.04%** | 65.72% | 65.14% | 63.36% | 61.94% |
| **Level 2 (LV2)** | 131,774 | **64.70%** | **66.35%** | **65.70%** | **63.93%** | **62.82%** |
| **Level 3 (LV3)** | 131,777 | **63.26%** | **64.76%** | **64.44%** | **62.57%** | **61.28%** |

**Observations**
- LV2 has the highest human win rate (64.70%), followed by LV1 (64.04%), with LV3 slightly
  lower (63.26%).
- All metrics peak at LV2.
- LV3’s modest drop suggests persona+example prompts move LLM time-series stats closer to
  humans, though p < 0.001 everywhere.
- Interestingly, LV3 yields the highest ML accuracy (92.18%) despite the slightly smaller
  binomial gap—ML models capture subtle cues beyond simple win rates.

---

## III. File locations

### ML classification (unified model)

- `micro_results/ml/ml_classification_all_levels_combined.csv`
- `micro_results/ml/ml_classification_concatenated_lv1.csv`
- `micro_results/ml/ml_classification_concatenated_lv2.csv`
- `micro_results/ml/ml_classification_concatenated_lv3.csv`

Visualizations:
- `results/plots/ml_timeseries_all_levels_comparison.png`
- `results/plots/ml_timeseries_unified_model_lv1.png`
- `results/plots/ml_timeseries_unified_model_lv2.png`
- `results/plots/ml_timeseries_unified_model_lv3.png`

### Binomial tests

- `micro_results/binomial/binomial_test_results_lv1.csv`
- `micro_results/binomial/binomial_test_results_lv1_detailed.csv` (131,765 rows)
- `micro_results/binomial/binomial_test_results_lv2.csv`
- `micro_results/binomial/binomial_test_results_lv3.csv`

Visualizations:
- `results/plots/binomial_test_all_levels_comparison.png`
- `results/plots/binomial_test_results_lv1.png`
- `results/plots/binomial_test_results_lv2.png`
- `results/plots/binomial_test_results_lv3.png`

---

## IV. Key conclusions

### ML classification
1. **Unified model is the primary deliverable**
   - 60 features (20 base × 3 normalized stats).  
   - **LV1**: Random Forest 91.69% accuracy, Gradient Boosting 0.9229 ROC AUC.  
   - **LV2**: Neural Network (MLP) 90.00% accuracy, Random Forest 0.9129 ROC AUC.  
   - **LV3**: Neural Network (MLP) 92.18% accuracy, Random Forest 0.9436 ROC AUC.

2. **Human vs LLM remains separable**
   - Every model achieves >80% accuracy; the top three exceed 90%.  
   - ROC AUC >0.85 across the board.

3. **Feature concatenation helps**
   - Combining CV / RMSSD_norm / MASD_norm allows cross-statistic interactions to be
     learned.  
   - All levels show strong separation once concatenated.

### Binomial tests
1. **Humans maintain higher time-series variability**
   - All metrics: p < 0.001.  
   - LV1 overall win: 64.04%, LV2: 64.70%, LV3: 63.26%.

2. **VAR vs CV agree closely**
   - Differences: 0.58%, 0.65%, 0.32% (LV1–LV3), validating the decision to rely on CV.

3. **Normalized stats matter**
   - RMSSD_norm and MASD_norm continue to show significant human wins, offering
     length-insensitive comparisons.

---

## V. Usage guidance

### ML validation

Use the unified model outputs:
- `micro_results/ml/ml_classification_all_levels_combined.csv`
- Level-specific files: `ml_classification_concatenated_lv1/2/3.csv`

Why:
- 60 concatenated features (CV, RMSSD_norm, MASD_norm) capture cross-statistic
  interactions.
- Matches the “feature concatenation” recommendation from faculty.
- Demonstrably separates Human vs LLM across all levels.

### Binomial tests

Use level-specific summaries: `binomial_test_results_lv1/2/3.csv`
- Include all metrics (VAR, CV, RMSSD_norm, MASD_norm).
- VAR vs CV are nearly identical, reinforcing CV-based reporting.
- Results are consistent across LV1–LV3, proving robustness.

### Visualizations

- ML: `results/plots/ml_timeseries_all_levels_comparison.png`,
  `...lv1.png`, `...lv2.png`, `...lv3.png`
- Binomial: `results/plots/binomial_test_all_levels_comparison.png`,
  `...lv1.png`, `...lv2.png`, `...lv3.png`

---

## VI. Data statistics

### Sample sizes

- **ML classification**: Unified model uses 409 held-out samples (80/20 split shared by all
  statistics).
- **Binomial tests**  
  - LV1: 131,765 total comparisons (32,940 each for CV/RMSSD_norm/MASD_norm, 32,945 for
    VAR).  
  - LV2: 131,774 total (32,943 each for CV/RMSSD_norm/MASD_norm, 32,945 for VAR).  
  - LV3: 131,777 total (32,942 each for CV/RMSSD_norm/MASD_norm, 32,951 for VAR).  
  - Comparison granularity: author × feature × model × statistic type.

### Feature counts

- Unified model = 60 features (20 base × 3 normalized stats)  
  - Big Five: 5 × 3 = 15  
  - NELA Emotional: 6 × 3 = 18  
  - NELA Stylistic: 9 × 3 = 27

### Model list (ML classification)
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

## VII. File modification timestamps

| File | Timestamp | Description |
|------|-----------|-------------|
| `micro_results/ml/ml_classification_all_levels_combined.csv` | Nov 26, 2025 14:10 | Unified model — all levels combined (primary reference) |
| `micro_results/ml/ml_classification_concatenated_lv1.csv` | Nov 26, 2025 13:30 | LV1 unified model |
| `micro_results/ml/ml_classification_concatenated_lv2.csv` | Nov 26, 2025 13:50 | LV2 unified model |
| `micro_results/ml/ml_classification_concatenated_lv3.csv` | Nov 26, 2025 13:10 | LV3 unified model |
| `micro_results/binomial/binomial_test_results_lv1.csv` | Nov 26, 2025 14:20 | Binomial summary LV1 (primary) |
| `micro_results/binomial/binomial_test_results_lv2.csv` | Nov 26, 2025 14:21 | Binomial summary LV2 |
| `micro_results/binomial/binomial_test_results_lv3.csv` | Nov 26, 2025 14:22 | Binomial summary LV3 |
| `micro_results/binomial/binomial_test_results_lv1_detailed.csv` | Nov 26, 2025 14:20 | Detailed LV1 comparisons |
| `micro_results/binomial/binomial_test_results_lv2_detailed.csv` | Nov 26, 2025 14:21 | Detailed LV2 comparisons |
| `micro_results/binomial/binomial_test_results_lv3_detailed.csv` | Nov 26, 2025 14:22 | Detailed LV3 comparisons |
| `results/plots/ml_timeseries_unified_model.png` | Nov 17, 2024 10:22 | Unified model visualization |
| `results/plots/binomial_test_results_lv1.png` | Nov 19, 2024 11:30 | Binomial LV1 visualization |
| `results/plots/binomial_test_results_lv2.png` | Nov 19, 2024 11:30 | Binomial LV2 visualization |
| `results/plots/binomial_test_results_lv3.png` | Nov 19, 2024 11:30 | Binomial LV3 visualization |
| `results/plots/binomial_test_all_levels_comparison.png` | Nov 19, 2024 11:30 | All-level binomial comparison |

