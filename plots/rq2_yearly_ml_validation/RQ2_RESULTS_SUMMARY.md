# RQ2: Yearly Accuracy Trend Analysis - Results Summary

## Overview

RQ2 tests whether Human and LLM texts **converge over time** by analyzing if classification accuracy decreases across years (2020-2024). If convergence occurs, accuracy should decrease as Human and LLM texts become more similar.

## Experimental Configuration

### Dataset Setup
- **Years**: 2020, 2021, 2022, 2023, 2024 (5 years)
- **Domains**: Academic, Blogs, News
- **Samples per year per domain**:
  - Human: 1,000 samples
  - LLM: 4,000 samples (4 models × 1,000 each: DS, G4B, G12B, LMK)
  - Total: 5,000 samples per year per domain
- **Train/Test Split**: 3,500 training / 1,500 test per year per domain
- **LLM Level**: LV3 only (highest prompt guidance)

### Feature Sets

**1. All Features (414 features)**
- 20 CE features (Cognitive Embedding)
- 10 TF-IDF features
- 384 SBERT features
- Total: 414 static features

**2. CE Only (20 features)**
- 20 CE features only
- Includes: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, and other CE features

### Model Configuration
- **Algorithm**: Logistic Regression with StandardScaler
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Evaluation**: Test accuracy on held-out test set + CV mean accuracy

## Results: All Features (414 features)

### Academic Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 97.07% | 99.47% | 97.09% | ±0.69% |
| 2021 | 96.00% | 99.10% | 96.94% | ±0.67% |
| 2022 | 97.00% | 99.12% | 96.63% | ±0.68% |
| 2023 | 97.53% | 99.05% | 97.03% | ±0.47% |
| 2024 | 95.80% | 98.69% | 96.37% | ±0.96% |

**Trend Analysis:**
- **2020 → 2024**: 97.07% → 95.80% (**-1.27%** decrease)
- **Mean Accuracy**: 96.68%
- **Std Dev**: 0.64%
- **Interpretation**: Slight decreasing trend, indicating potential convergence

### Blogs Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 98.73% | 99.83% | 98.80% | ±0.23% |
| 2021 | 98.67% | 99.79% | 98.11% | ±0.48% |
| 2022 | 99.20% | 99.92% | 98.66% | ±0.35% |
| 2023 | 98.73% | 99.91% | 98.54% | ±0.39% |
| 2024 | 98.47% | 99.79% | 98.83% | ±0.14% |

**Trend Analysis:**
- **2020 → 2024**: 98.73% → 98.47% (**-0.26%** decrease)
- **Mean Accuracy**: 98.76%
- **Std Dev**: 0.26%
- **Interpretation**: Very stable, minimal decrease (convergence not evident)

### News Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 99.47% | 99.97% | 98.91% | ±0.49% |
| 2021 | 99.53% | 99.98% | 99.29% | ±0.38% |
| 2022 | 99.27% | 99.97% | 98.94% | ±0.19% |
| 2023 | 98.93% | 99.89% | 98.83% | ±0.28% |
| 2024 | 98.73% | 99.68% | 98.00% | ±0.37% |

**Trend Analysis:**
- **2020 → 2024**: 99.47% → 98.73% (**-0.74%** decrease)
- **Mean Accuracy**: 99.19%
- **Std Dev**: 0.32%
- **Interpretation**: Slight decreasing trend, but remains above 98.7%

## Results: CE Only (20 features)

### Academic Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 92.00% | 97.39% | 93.11% | ±0.68% |
| 2021 | 93.60% | 96.96% | 93.34% | ±0.45% |
| 2022 | 92.20% | 97.10% | 93.09% | ±0.66% |
| 2023 | 90.93% | 96.90% | 91.54% | ±1.01% |
| 2024 | 88.87% | 94.85% | 89.40% | ±0.51% |

**Trend Analysis:**
- **2020 → 2024**: 92.00% → 88.87% (**-3.13%** decrease)
- **Mean Accuracy**: 91.52%
- **Std Dev**: 1.56%
- **Interpretation**: **Clear decreasing trend**, indicating convergence

### Blogs Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 89.07% | 95.47% | 90.37% | ±0.70% |
| 2021 | 87.53% | 94.48% | 89.00% | ±0.49% |
| 2022 | 88.73% | 95.35% | 90.31% | ±0.91% |
| 2023 | 90.53% | 96.14% | 88.80% | ±0.80% |
| 2024 | 89.07% | 95.44% | 88.51% | ±0.91% |

**Trend Analysis:**
- **2020 → 2024**: 89.07% → 89.07% (**0.00%** change)
- **Mean Accuracy**: 88.99%
- **Std Dev**: 1.06%
- **Interpretation**: Stable, no clear trend

### News Domain

| Year | Test Accuracy | ROC AUC | CV Mean Acc | CV Std |
|------|---------------|---------|-------------|--------|
| 2020 | 96.93% | 99.57% | 96.11% | ±0.70% |
| 2021 | 97.13% | 99.79% | 97.09% | ±0.70% |
| 2022 | 97.07% | 99.51% | 97.14% | ±0.81% |
| 2023 | 96.47% | 99.26% | 96.43% | ±0.52% |
| 2024 | 95.67% | 99.09% | 95.54% | ±0.65% |

**Trend Analysis:**
- **2020 → 2024**: 96.93% → 95.67% (**-1.26%** decrease)
- **Mean Accuracy**: 96.65%
- **Std Dev**: 0.55%
- **Interpretation**: Slight decreasing trend

## Key Findings

### 1. All Features Model (414 features)

**Overall Performance:**
- **Very High Accuracy**: All domains maintain >95% accuracy across all years
- **Academic**: 95.8% - 97.5% (mean: 96.68%)
- **Blogs**: 98.5% - 99.2% (mean: 98.76%)
- **News**: 98.7% - 99.5% (mean: 99.19%)

**Trend Patterns:**
- **Academic**: Slight decreasing trend (-1.27%)
- **Blogs**: Very stable (-0.26%, minimal change)
- **News**: Slight decreasing trend (-0.74%)

**Conclusion**: 
- Using all 414 features, accuracy remains consistently high (>95%)
- Small decreases suggest **mild convergence**, but discrimination remains strong
- News domain shows highest overall accuracy but also slight decline

### 2. CE Only Model (20 features)

**Overall Performance:**
- **Lower Accuracy**: 87.5% - 97.1% (compared to all features)
- **Academic**: 88.9% - 93.6% (mean: 91.52%)
- **Blogs**: 87.5% - 90.6% (mean: 88.99%)
- **News**: 95.7% - 97.1% (mean: 96.65%)

**Trend Patterns:**
- **Academic**: **Clear decreasing trend** (-3.13%, strongest evidence of convergence)
- **Blogs**: Stable (0.00% change, no convergence)
- **News**: Slight decreasing trend (-1.26%)

**Conclusion**:
- CE-only model shows **stronger evidence of convergence** in Academic domain
- Academic accuracy drops from 92.0% to 88.9% (3.1% decrease)
- This suggests that CE features alone may be more sensitive to temporal convergence
- News domain maintains high accuracy even with CE-only features

### 3. Comparison: All Features vs CE Only

**Performance Gap:**
- **Academic**: 
  - Gap in 2020: 97.07% - 92.00% = **5.07%**
  - Gap in 2024: 95.80% - 88.87% = **6.93%**
  - Gap increases over time, suggesting all features maintain discrimination better
  
- **Blogs**:
  - Gap in 2020: 98.73% - 89.07% = **9.66%**
  - Gap in 2024: 98.47% - 89.07% = **9.40%**
  - Gap remains stable (~9.5%)
  
- **News**:
  - Gap in 2020: 99.47% - 96.93% = **2.54%**
  - Gap in 2024: 98.73% - 95.67% = **3.06%**
  - Gap slightly increases

**Key Insight**: 
- All-features model maintains better discrimination across years
- CE-only model shows stronger convergence signal (especially in Academic)
- This suggests that **non-CE features (TF-IDF, SBERT) may help maintain discrimination** even as CE features show convergence

## Statistical Summary

### Accuracy Trends (2020 → 2024)

| Feature Set | Domain | 2020 Acc | 2024 Acc | Change | Trend |
|-------------|--------|----------|----------|--------|-------|
| **All Features** | Academic | 97.07% | 95.80% | -1.27% | ↓ Decreasing |
| **All Features** | Blogs | 98.73% | 98.47% | -0.26% | → Stable |
| **All Features** | News | 99.47% | 98.73% | -0.74% | ↓ Decreasing |
| **CE Only** | Academic | 92.00% | 88.87% | **-3.13%** | ↓↓ Strong Decrease |
| **CE Only** | Blogs | 89.07% | 89.07% | 0.00% | → Stable |
| **CE Only** | News | 96.93% | 95.67% | -1.26% | ↓ Decreasing |

### ROC AUC Trends

**All Features:**
- Academic: 99.47% → 98.69% (-0.78%)
- Blogs: 99.83% → 99.79% (-0.04%)
- News: 99.97% → 99.68% (-0.29%)

**CE Only:**
- Academic: 97.39% → 94.85% (**-2.54%**)
- Blogs: 95.47% → 95.44% (-0.03%)
- News: 99.57% → 99.09% (-0.48%)

## Interpretation

### Convergence Evidence

1. **Strongest Evidence (Academic, CE Only)**:
   - 3.13% accuracy decrease over 5 years
   - Clear downward trend suggests Human-LLM convergence in Academic domain
   - CE features become less discriminative over time

2. **Moderate Evidence (Academic & News, All Features)**:
   - Small but consistent decreases (1-1.3%)
   - Suggests mild convergence when using all features

3. **Weak/No Evidence (Blogs)**:
   - Minimal change in accuracy
   - Suggests Human-LLM texts remain distinguishable in Blogs domain

### Domain-Specific Patterns

- **Academic**: Shows strongest convergence signal, especially with CE-only features
- **News**: High accuracy maintained (>98%), but slight decreasing trend
- **Blogs**: Most stable, showing least evidence of convergence

### Feature Set Comparison

- **All Features (414)**: Better discrimination, less sensitive to convergence
- **CE Only (20)**: More sensitive to temporal changes, shows stronger convergence signal

## Output Files

**Results:**
- `macro_results/rq2_yearly_ml_validation/rq2_yearly_validation_all_features_all.csv`
- `macro_results/rq2_yearly_ml_validation/rq2_yearly_validation_ce_only_all.csv`

**Visualizations:**
- `plots/rq2_yearly_ml_validation/yearly_accuracy_trends.png`
- `plots/rq2_yearly_ml_validation/yearly_accuracy_trends_combined.png`
- `plots/rq2_yearly_ml_validation/yearly_comparison_table.png`

**Scripts:**
- Analysis: `scripts/macro/ml_classify_macro_yearly_rq2.py`
- Visualization: `scripts/macro/visualize_rq2_yearly_trends.py`

## Conclusion

RQ2 analysis reveals **domain-specific and feature-set-dependent convergence patterns**:
- **Academic domain with CE-only features** shows the strongest evidence of Human-LLM convergence (3.13% accuracy decrease)
- **All-features model** maintains high discrimination (>95%) across all domains and years
- **Blogs domain** shows minimal convergence, suggesting persistent Human-LLM differences
- Results suggest that **non-CE features help maintain discrimination** even as CE features show temporal convergence

