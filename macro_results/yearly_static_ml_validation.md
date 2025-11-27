# Yearly Static ML Validation (Macro)

This note documents both the **methodology** and the **results** of the
per-year Human vs LLM classification experiments
(`macro_scripts/run_yearly_static_ml_validation.py`). The goal is to see
whether static feature separability degrades over time.

---

## Methodology

This experiment is designed as an ML-based probe for RQ2: if humans and LLMs
were converging at the population level, a classifier trained on yearly slices
should find the sources harder to separate over time. We therefore repeat the
same controlled classification task for every year (2020–2024) and domain.

1. **Data source / composition**  
   - Human samples: `macro_dataset/process/human/<domain>/combined.csv`
     (exactly 1,000 docs per domain per year).  
   - LLM samples: `macro_dataset/process/LLM/<model>/LV3/<domain>/combined_outliers_removed.csv`
     (1,000 docs per domain per model per year; four models total = 4,000 LLM docs).  
   - Each year therefore yields roughly 5,000 training samples (1k human + 4k LLM) while
     keeping the LV3 prompting configuration fixed.

2. **Feature set**  
   - Same 20 static features used across RQ2 (Big Five + NELA + length metrics).

3. **Per-year dataset construction**  
   - For each `(domain, year)` pair, take the corresponding human rows and append rows
     from all four LLM models.  
   - If any row has missing features, impute with that year/domain **median** so that no
     documents are discarded.  
   - Keep `domain`, `source`, `label` columns for analysis, but feed only the 20 static
     features to the classifier.

4. **Model / pipeline**  
   - `StandardScaler → LogisticRegression(class_weight='balanced', max_iter=2000)`  
   - For every year/domain we run an independent 70/30 stratified split (`random_state=42`)
     so train/test slices never leak.  
   - Logistic Regression is chosen because:  
     1. coefficients are interpretable and align directly with the RQ2 feature analysis;  
     2. statsmodels gives p-values for significance testing.

5. **Outputs**  
   - Accuracy & ROC AUC (`yearly_static_ml_validation_<domain>.csv`).  
   - LR coefficients as feature importance
     (`yearly_feature_importance_<domain>.csv`).  
   - LR coefficient p-values per year/domain
     (`yearly_static_ml_lr_pvalues_<domain>.csv`).  
   - Trend plots stored in `macro_results/figures/`.

---

## 1. Metrics by Domain

### Combined domains (news+blogs+academic)

- Results: `yearly_static_ml_validation_all.csv`
- Accuracy stays ~0.83–0.85; ROC AUC ~0.89–0.91 → still highly separable.

### News

- Results: `yearly_static_ml_validation_news.csv`
- Accuracy: 0.965 (2020) → 0.965 (2024); ROC AUC: 0.995 → 0.986.
- Trend: still near-perfect separability; minor fluctuations driven by
  year-specific content but no sustained decline.

### Blogs

- Results: `yearly_static_ml_validation_blogs.csv`
- Accuracy stays 0.881–0.895; ROC AUC ≈0.95–0.96 across years.
- Trend: essentially flat (±0.02) with no visible convergence.

### Academic

- Results: `yearly_static_ml_validation_academic.csv`
- Accuracy: 0.932 → 0.891; ROC AUC: 0.976 → 0.949.
- Trend: modest decline yet accuracy remains ≥0.89.

Plots for each domain are saved under `macro_results/figures/`.

---

## 2. Feature Importance (LR Coefficients)

For each domain we record LR coefficients per year:

- `yearly_feature_importance_news.csv`
- `yearly_feature_importance_blogs.csv`
- `yearly_feature_importance_academic.csv`

Columns: `year`, `feature`, `coefficient`, `abs_coefficient`. The features
with largest absolute coefficients are the strongest signals for that year.

### News — Top 5 features by |coef|

| Year | Top coefficients |
|------|------------------|
| 2020 | `flesch_reading_ease (3.486)`<br>`avg_sentence_length (2.785)`<br>`gunning_fog (-2.330)`<br>`vader_compound (-1.967)`<br>`Agreeableness (1.635)` |
| 2021 | `gunning_fog (-3.306)`<br>`avg_sentence_length (3.072)`<br>`flesch_reading_ease (2.330)`<br>`num_words (1.646)`<br>`vader_compound (-1.474)` |
| 2022 | `avg_sentence_length (2.752)`<br>`gunning_fog (-2.702)`<br>`flesch_reading_ease (2.011)`<br>`vader_compound (-1.959)`<br>`average_word_length (-1.756)` |
| 2023 | `avg_sentence_length (3.742)`<br>`flesch_reading_ease (2.477)`<br>`gunning_fog (-2.421)`<br>`vader_compound (-2.220)`<br>`Agreeableness (1.461)` |
| 2024 | `flesch_reading_ease (3.518)`<br>`avg_sentence_length (2.061)`<br>`gunning_fog (-1.813)`<br>`vader_compound (-1.682)`<br>`num_words (1.112)` |

### Blogs — Top 5 features by |coef|

| Year | Top coefficients |
|------|------------------|
| 2020 | `flesch_reading_ease (2.873)`<br>`avg_sentence_length (1.791)`<br>`Openness (1.552)`<br>`vader_neu (1.400)`<br>`Agreeableness (1.173)` |
| 2021 | `flesch_reading_ease (2.295)`<br>`Openness (1.849)`<br>`Agreeableness (1.551)`<br>`vader_neu (1.188)`<br>`avg_sentence_length (1.163)` |
| 2022 | `flesch_reading_ease (2.532)`<br>`Openness (1.783)`<br>`Agreeableness (1.472)`<br>`Extraversion (1.404)`<br>`Neuroticism (1.323)` |
| 2023 | `flesch_reading_ease (2.687)`<br>`Openness (1.890)`<br>`Agreeableness (1.698)`<br>`avg_sentence_length (1.477)`<br>`word_diversity (-1.209)` |
| 2024 | `Openness (1.874)`<br>`flesch_reading_ease (1.797)`<br>`Agreeableness (1.690)`<br>`vader_neu (1.237)`<br>`Neuroticism (1.128)` |

### Academic — Top 5 features by |coef|

| Year | Top coefficients |
|------|------------------|
| 2020 | `Agreeableness (1.546)`<br>`gunning_fog (-1.397)`<br>`flesch_reading_ease (1.364)`<br>`avg_sentence_length (1.038)`<br>`num_words (-0.775)` |
| 2021 | `flesch_reading_ease (1.658)`<br>`Agreeableness (1.417)`<br>`avg_sentence_length (1.079)`<br>`gunning_fog (-1.002)`<br>`Neuroticism (0.842)` |
| 2022 | `Agreeableness (1.597)`<br>`gunning_fog (-1.354)`<br>`avg_sentence_length (1.083)`<br>`flesch_reading_ease (1.078)`<br>`vader_neg (1.045)` |
| 2023 | `flesch_reading_ease (1.494)`<br>`Agreeableness (1.268)`<br>`gunning_fog (-1.037)`<br>`vader_neg (0.999)`<br>`num_words (-0.866)` |
| 2024 | `Agreeableness (1.952)`<br>`flesch_reading_ease (1.646)`<br>`vader_neg (1.225)`<br>`avg_sentence_length (0.990)`<br>`gunning_fog (-0.957)` |

---

## 3. Logistic Regression p-values

- Files:
  - `yearly_static_ml_lr_pvalues_news.csv`
  - `yearly_static_ml_lr_pvalues_blogs.csv`
  - `yearly_static_ml_lr_pvalues_academic.csv`
- Columns: `year`, `feature`, `coef`, `std_err`, `z_score`, `p_value`, `odds_ratio`.
- Usage:
  - Determine whether a feature’s contribution to Human-vs-LLM separation remains
    statistically significant in a given year.
  - Track convergence signals: when a feature’s p-value rises above common
    thresholds (e.g., 0.05), it implies that dimension is no longer reliably
    distinguishing Human from LLM for that domain/year.
- In practice we observe near-perfect separation for News; statsmodels reports
  convergence warnings and astronomically large coefficients/odds ratios. The
  reported p-values remain interpretable, but extreme magnitudes simply reflect
  how linearly separable the yearly dataset is.

| Domain | 2020 | 2021 | 2022 | 2023 | 2024 |
|--------|------|------|------|------|------|
| News (p ≤ 0.01 features) | 11 | 9 | 12 | 9 | 11 |
| Blogs (p ≤ 0.01 features) | 14 | 10 | 13 | 14 | 13 |
| Academic (p ≤ 0.01 features) | 9 | 9 | 6 | 9 | 10 |

- **News:** Each year still has double-digit significant features, dominated by
  length/readability metrics (`avg_sentence_length`, `gunning_fog`,
  `flesch_reading_ease`, `num_words`) with subjectivity and VADER cues reappearing
  in 2023–2024.
- **Blogs:** At least 10 features stay significant every year. Metrics such as
  `flesch_reading_ease`, `word_diversity`, `avg_sentence_length`, `num_words`, and
  `verb_ratio` remain at p ≪ 0.001, showing that stylistic differences in blogs are
  the hardest to flatten.
- **Academic:** The count briefly dipped to 6 in 2022 but bounced back to 10 in 2024.
  `avg_sentence_length`, `flesch_reading_ease`, `gunning_fog`, `subjectivity`, and
  `num_words` remain the main separators, so the small accuracy drop is driven by
  fringe features rather than core dimensions disappearing.

#### News — Top 5 significant features (by p-value)

| Year | Top features (coef, p-value) |
|------|------------------------------|
| 2020 | `avg_sentence_length (3.236, 5.7e-35)`<br>`flesch_reading_ease (4.441, 2.4e-10)`<br>`num_words (1.314, 3.5e-10)`<br>`gunning_fog (-2.585, 4.6e-7)`<br>`vader_compound (-2.648, 5.0e-6)` |
| 2021 | `avg_sentence_length (3.540, 2.9e-23)`<br>`num_words (1.857, 1.0e-11)`<br>`gunning_fog (-3.889, 5.0e-8)`<br>`word_diversity (1.696, 1.7e-6)`<br>`subjectivity (-0.763, 3.9e-5)` |
| 2022 | `avg_sentence_length (3.549, 1.8e-22)`<br>`num_words (2.051, 1.5e-13)`<br>`gunning_fog (-3.864, 1.1e-9)`<br>`word_diversity (1.773, 3.2e-7)`<br>`subjectivity (-0.959, 1.4e-5)` |
| 2023 | `avg_sentence_length (5.166, 2.6e-20)`<br>`num_words (2.062, 1.8e-17)`<br>`gunning_fog (-3.525, 2.1e-7)`<br>`word_diversity (1.626, 2.1e-7)`<br>`Agreeableness (1.485, 1.0e-6)` |
| 2024 | `avg_sentence_length (2.357, 2.1e-24)`<br>`num_words (1.432, 5.0e-13)`<br>`flesch_reading_ease (4.396, 7.3e-10)`<br>`vader_compound (-2.105, 1.3e-7)`<br>`gunning_fog (-2.062, 9.1e-5)` |

#### Blogs — Top 5 significant features

| Year | Top features (coef, p-value) |
|------|------------------------------|
| 2020 | `flesch_reading_ease (3.216, 2.9e-36)`<br>`avg_sentence_length (1.827, 3.3e-18)`<br>`Openness (1.478, 7.9e-17)`<br>`vader_compound (-0.849, 7.8e-11)`<br>`Extraversion (0.987, 1.2e-10)` |
| 2021 | `flesch_reading_ease (2.449, 2.7e-24)`<br>`Openness (1.760, 6.8e-14)`<br>`avg_sentence_length (1.175, 1.6e-12)`<br>`vader_compound (-0.811, 1.8e-12)`<br>`word_diversity (-0.681, 2.3e-10)` |
| 2022 | `flesch_reading_ease (2.728, 4.3e-24)`<br>`Openness (1.814, 7.8e-15)`<br>`Extraversion (1.413, 7.5e-14)`<br>`word_diversity (-0.843, 2.0e-12)`<br>`vader_compound (-0.821, 1.0e-11)` |
| 2023 | `flesch_reading_ease (2.865, 1.3e-31)`<br>`word_diversity (-1.186, 1.4e-24)`<br>`Openness (1.908, 1.1e-15)`<br>`avg_sentence_length (1.649, 4.8e-14)`<br>`num_words (-0.737, 4.8e-11)` |
| 2024 | `flesch_reading_ease (2.055, 2.0e-16)`<br>`Openness (1.847, 2.1e-15)`<br>`word_diversity (-0.844, 4.5e-14)`<br>`vader_compound (-0.760, 5.3e-12)`<br>`Extraversion (1.112, 5.6e-11)` |

#### Academic — Top 5 significant features

| Year | Top features (coef, p-value) |
|------|------------------------------|
| 2020 | `avg_sentence_length (1.167, 6.5e-23)`<br>`subjectivity (-0.773, 1.8e-14)`<br>`gunning_fog (-1.614, 1.1e-12)`<br>`flesch_reading_ease (1.422, 7.4e-08)`<br>`vader_neg (0.827, 1.1e-07)` |
| 2021 | `avg_sentence_length (1.117, 9.8e-19)`<br>`flesch_reading_ease (1.739, 6.0e-11)`<br>`vader_neg (0.826, 1.8e-09)`<br>`subjectivity (-0.507, 6.5e-08)`<br>`Agreeableness (1.579, 2.3e-07)` |
| 2022 | `avg_sentence_length (1.111, 2.1e-19)`<br>`subjectivity (-0.759, 1.6e-14)`<br>`gunning_fog (-1.380, 3.5e-09)`<br>`flesch_reading_ease (1.435, 3.6e-08)`<br>`Agreeableness (1.648, 1.5e-07)` |
| 2023 | `subjectivity (-0.721, 1.6e-15)`<br>`avg_sentence_length (0.811, 4.2e-15)`<br>`flesch_reading_ease (1.715, 1.8e-14)`<br>`vader_neg (0.998, 1.6e-11)`<br>`Agreeableness (1.249, 1.2e-07)` |
| 2024 | `flesch_reading_ease (1.747, 8.7e-16)`<br>`avg_sentence_length (0.869, 1.1e-12)`<br>`Agreeableness (2.052, 1.7e-11)`<br>`vader_neg (1.416, 5.8e-10)`<br>`subjectivity (-0.468, 3.4e-09)` |

---

## 4. Takeaways

- Even when every year/domain is retrained independently, accuracy still stays between
  0.88–0.97 and ROC AUC around 0.95, so macro-level human vs LLM separation is not eroding.  
- Combining LR coefficients with p-values lets us pinpoint which static features remain
  significant and track how their importance changes over time, tying directly to the
  feature-level RQ2 analysis.  
- Stylistic metrics (length, readability, lexical variety) remain the dominant signals,
  followed by a handful of Big Five traits (e.g., Conscientiousness / Agreeableness),
  indicating that drift is primarily driven by style/structure dimensions.  
- Example citation: “In 2024 Academic, `flesch_reading_ease` still has p < 1e-12, which
  shows that even though accuracy dipped slightly, that dimension continues to clearly
  separate humans and LLMs.”
---

## 5. Top 10 Features by Layer

### News — 2020
- flesch_reading_ease (stylistic, |coef|=3.810)
- avg_sentence_length (stylistic, |coef|=2.731)
- gunning_fog (stylistic, |coef|=2.586)
- average_word_length (stylistic, |coef|=1.326)
- vader_compound (emotional, |coef|=1.318)
- Conscientiousness (cognitive, |coef|=0.786)
- num_words (stylistic, |coef|=0.648)
- vader_pos (emotional, |coef|=0.539)
- polarity (emotional, |coef|=0.504)
- Agreeableness (cognitive, |coef|=0.484)

### News — 2021
- gunning_fog (stylistic, |coef|=3.453)
- avg_sentence_length (stylistic, |coef|=2.851)
- flesch_reading_ease (stylistic, |coef|=2.179)
- average_word_length (stylistic, |coef|=1.473)
- vader_compound (emotional, |coef|=1.191)
- num_words (stylistic, |coef|=1.155)
- Conscientiousness (cognitive, |coef|=0.952)
- subjectivity (emotional, |coef|=0.740)
- word_diversity (stylistic, |coef|=0.547)
- function_word_ratio (stylistic, |coef|=0.516)

### News — 2022
- gunning_fog (stylistic, |coef|=3.630)
- avg_sentence_length (stylistic, |coef|=3.244)
- flesch_reading_ease (stylistic, |coef|=1.798)
- vader_compound (emotional, |coef|=1.586)
- average_word_length (stylistic, |coef|=1.430)
- num_words (stylistic, |coef|=1.239)
- word_diversity (stylistic, |coef|=0.939)
- Conscientiousness (cognitive, |coef|=0.529)
- subjectivity (emotional, |coef|=0.523)
- function_word_ratio (stylistic, |coef|=0.429)

### News — 2023
- avg_sentence_length (stylistic, |coef|=4.389)
- flesch_reading_ease (stylistic, |coef|=3.049)
- average_word_length (stylistic, |coef|=2.260)
- vader_compound (emotional, |coef|=1.709)
- num_words (stylistic, |coef|=1.680)
- gunning_fog (stylistic, |coef|=1.611)
- word_diversity (stylistic, |coef|=1.205)
- function_word_ratio (stylistic, |coef|=1.114)
- content_word_ratio (stylistic, |coef|=1.114)
- Conscientiousness (cognitive, |coef|=0.844)

### News — 2024
- flesch_reading_ease (stylistic, |coef|=3.470)
- gunning_fog (stylistic, |coef|=2.383)
- avg_sentence_length (stylistic, |coef|=1.842)
- vader_compound (emotional, |coef|=1.647)
- content_word_ratio (stylistic, |coef|=0.880)
- function_word_ratio (stylistic, |coef|=0.880)
- num_words (stylistic, |coef|=0.818)
- subjectivity (emotional, |coef|=0.534)
- verb_ratio (stylistic, |coef|=0.525)
- Conscientiousness (cognitive, |coef|=0.431)

### Blogs — 2020
- flesch_reading_ease (stylistic, |coef|=2.298)
- Conscientiousness (cognitive, |coef|=1.674)
- avg_sentence_length (stylistic, |coef|=1.205)
- Openness (cognitive, |coef|=0.863)
- word_diversity (stylistic, |coef|=0.750)
- num_words (stylistic, |coef|=0.750)
- gunning_fog (stylistic, |coef|=0.611)
- vader_pos (emotional, |coef|=0.450)
- Neuroticism (cognitive, |coef|=0.400)
- Extraversion (cognitive, |coef|=0.354)

### Blogs — 2021
- flesch_reading_ease (stylistic, |coef|=2.100)
- avg_sentence_length (stylistic, |coef|=1.500)
- Conscientiousness (cognitive, |coef|=1.335)
- word_diversity (stylistic, |coef|=1.068)
- Openness (cognitive, |coef|=0.848)
- num_words (stylistic, |coef|=0.783)
- verb_ratio (stylistic, |coef|=0.379)
- vader_compound (emotional, |coef|=0.375)
- vader_pos (emotional, |coef|=0.362)
- Extraversion (cognitive, |coef|=0.328)

### Blogs — 2022
- flesch_reading_ease (stylistic, |coef|=1.903)
- Conscientiousness (cognitive, |coef|=1.450)
- word_diversity (stylistic, |coef|=1.402)
- num_words (stylistic, |coef|=1.082)
- avg_sentence_length (stylistic, |coef|=1.044)
- Openness (cognitive, |coef|=0.820)
- vader_pos (emotional, |coef|=0.571)
- Extraversion (cognitive, |coef|=0.372)
- verb_ratio (stylistic, |coef|=0.368)
- Neuroticism (cognitive, |coef|=0.351)

### Blogs — 2023
- flesch_reading_ease (stylistic, |coef|=2.865)
- avg_sentence_length (stylistic, |coef|=1.236)
- Conscientiousness (cognitive, |coef|=1.222)
- word_diversity (stylistic, |coef|=1.169)
- gunning_fog (stylistic, |coef|=0.969)
- num_words (stylistic, |coef|=0.894)
- Openness (cognitive, |coef|=0.666)
- vader_pos (emotional, |coef|=0.472)
- vader_compound (emotional, |coef|=0.402)
- vader_neu (emotional, |coef|=0.364)

### Blogs — 2024
- flesch_reading_ease (stylistic, |coef|=2.077)
- word_diversity (stylistic, |coef|=1.290)
- Conscientiousness (cognitive, |coef|=1.182)
- avg_sentence_length (stylistic, |coef|=0.979)
- Openness (cognitive, |coef|=0.848)
- num_words (stylistic, |coef|=0.799)
- vader_compound (emotional, |coef|=0.443)
- gunning_fog (stylistic, |coef|=0.427)
- vader_neu (emotional, |coef|=0.370)
- vader_pos (emotional, |coef|=0.333)

### Academic — 2020
- flesch_reading_ease (stylistic, |coef|=1.658)
- gunning_fog (stylistic, |coef|=1.319)
- avg_sentence_length (stylistic, |coef|=1.214)
- subjectivity (emotional, |coef|=0.826)
- num_words (stylistic, |coef|=0.740)
- function_word_ratio (stylistic, |coef|=0.732)
- content_word_ratio (stylistic, |coef|=0.732)
- Agreeableness (cognitive, |coef|=0.687)
- average_word_length (stylistic, |coef|=0.575)
- Openness (cognitive, |coef|=0.426)

### Academic — 2021
- flesch_reading_ease (stylistic, |coef|=1.718)
- avg_sentence_length (stylistic, |coef|=0.919)
- gunning_fog (stylistic, |coef|=0.874)
- subjectivity (emotional, |coef|=0.645)
- function_word_ratio (stylistic, |coef|=0.596)
- content_word_ratio (stylistic, |coef|=0.596)
- Agreeableness (cognitive, |coef|=0.592)
- num_words (stylistic, |coef|=0.560)
- Openness (cognitive, |coef|=0.495)
- vader_neg (emotional, |coef|=0.477)

### Academic — 2022
- flesch_reading_ease (stylistic, |coef|=1.421)
- gunning_fog (stylistic, |coef|=1.385)
- avg_sentence_length (stylistic, |coef|=1.095)
- Agreeableness (cognitive, |coef|=0.655)
- function_word_ratio (stylistic, |coef|=0.649)
- content_word_ratio (stylistic, |coef|=0.649)
- num_words (stylistic, |coef|=0.615)
- Openness (cognitive, |coef|=0.599)
- subjectivity (emotional, |coef|=0.552)
- vader_neg (emotional, |coef|=0.535)

### Academic — 2023
- flesch_reading_ease (stylistic, |coef|=1.444)
- gunning_fog (stylistic, |coef|=1.160)
- avg_sentence_length (stylistic, |coef|=0.974)
- vader_neg (emotional, |coef|=0.803)
- num_words (stylistic, |coef|=0.703)
- vader_pos (emotional, |coef|=0.608)
- function_word_ratio (stylistic, |coef|=0.591)
- content_word_ratio (stylistic, |coef|=0.591)
- subjectivity (emotional, |coef|=0.567)
- Agreeableness (cognitive, |coef|=0.469)

### Academic — 2024
- flesch_reading_ease (stylistic, |coef|=1.734)
- avg_sentence_length (stylistic, |coef|=1.176)
- gunning_fog (stylistic, |coef|=0.683)
- vader_neg (emotional, |coef|=0.611)
- Openness (cognitive, |coef|=0.603)
- subjectivity (emotional, |coef|=0.602)
- Extraversion (cognitive, |coef|=0.588)
- Agreeableness (cognitive, |coef|=0.562)
- vader_pos (emotional, |coef|=0.420)
- function_word_ratio (stylistic, |coef|=0.378)

