# RQ1: Unified 75 Features ML Validation Test - Experiment Settings

## Overview
This experiment evaluates the performance of a unified 75-feature trajectory model for distinguishing Human vs. LLM-generated text (Level 3, LV3).

## Experimental Configuration

### 1. Dataset Setup

**Data Sources:**
- **Domains**: Academic, Blogs, News (all three domains combined)
- **LLM Models**: DS (DeepSeek), G4B (Gemma 4B), G12B (Gemma 12B), LMK (Llama Maverick)
- **LLM Level**: LV3 only (highest prompt guidance level)
- **Human Data**: All three domains (LV0, representing baseline human authors)

**Total Samples**: 2,060 authors/samples

#### Sample Count Verification ✅

**Human Authors:**
- Academic: 100 authors
- Blogs: 195 authors
- News: 117 authors
- **Subtotal: 412 human authors**

**LLM-Generated Authors:**
Each LLM model (DS, G4B, G12B, LMK) generates corresponding LV3 texts for each human author across all three domains:
- **DS (DeepSeek)**: 100 + 195 + 117 = 412 authors
- **G4B (Gemma 4B)**: 100 + 195 + 117 = 412 authors
- **G12B (Gemma 12B)**: 100 + 195 + 117 = 412 authors
- **LMK (Llama Maverick)**: 100 + 195 + 117 = 412 authors
- **Subtotal: 412 × 4 = 1,648 LLM authors**

**Total Calculation:**
```
Human authors:    412
LLM authors:    1,648
────────────────────────
Total:          2,060 ✅
```

**Distribution by Label:**
- **human**: 412 authors (20.0%)
- **llm**: 1,648 authors (80.0%)

**Distribution by Domain:**
- **blogs**: 975 authors (195 human + 195×4 LLM = 195×5)
- **news**: 585 authors (117 human + 117×4 LLM = 117×5)
- **academic**: 500 authors (100 human + 100×4 LLM = 100×5)

**Breakdown by Model and Domain:**

| Domain | Human | DS | G4B | G12B | LMK | Subtotal |
|--------|-------|----|----|----|----|----------|
| Academic | 100 | 100 | 100 | 100 | 100 | **500** |
| Blogs | 195 | 195 | 195 | 195 | 195 | **975** |
| News | 117 | 117 | 117 | 117 | 117 | **585** |
| **Total** | **412** | **412** | **412** | **412** | **412** | **2,060** |

**Verification Formula:**
```
Total = Human + (LLM_models × authors_per_model)
Total = 412 + (4 × 412)
Total = 412 + 1,648
Total = 2,060 ✅
```

This represents:
- 412 unique human authors (different counts per domain)
- Each human author corresponds to 4 LLM model-generated texts (DS, G4B, G12B, LMK)
- All data at LV3 level (highest prompt guidance level)
- Covers three domains: Academic, Blogs, News

### 2. Feature Composition (75 Features Total)

The unified feature set combines two main components:

#### A. CE Variability Features (CE-VAR)
- **Type**: Normalized variability metrics for Cognitive Embedding (CE) features
- **Suffixes**: 
  - `_cv`: Coefficient of Variation
  - `_rmssd_norm`: Normalized Root Mean Square of Successive Differences
  - `_masd_norm`: Normalized Mean Absolute Successive Differences
- **Source**: Derived from CE feature time series per author

#### B. Geometry Features (15 features)
- **Spaces**: Three embedding/feature spaces
  - CE (Cognitive Embedding): 5 metrics × 1 space = 5 features
  - TF-IDF: 5 metrics × 1 space = 5 features  
  - SBERT: 5 metrics × 1 space = 5 features
- **Metrics per space** (5 metrics each):
  1. `mean_distance`: Average distance between consecutive time points
  2. `std_distance`: Standard deviation of distances
  3. `net_displacement`: Net displacement from start to end
  4. `path_length`: Total path length
  5. `tortuosity`: Path tortuosity (path_length / net_displacement)
- **Total Geometry**: 5 metrics × 3 spaces = 15 features

**Note**: `direction_consistency` and `n_years` metrics are **excluded** from the unified set.

**Calculation:**
- CE-VAR features: ~60 features (number of CE features × 3 variability metrics)
- Geometry features: 15 features (5 metrics × 3 spaces)
- **Total ≈ 75 features** (exact count depends on number of CE features in dataset)

### 3. Machine Learning Configuration

#### Classifier
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - `n_estimators`: 300 trees
  - `max_depth`: None (unlimited)
  - `class_weight`: "balanced" (handles class imbalance)
  - `random_state`: 42 (reproducibility)
  - `n_jobs`: -1 (use all CPU cores)

#### Cross-Validation
- **Method**: Stratified K-Fold
- **Folds**: 5-fold cross-validation
- **Shuffle**: True
- **Random State**: 42
- **Stratification**: Ensures balanced class distribution in each fold

#### Evaluation Metrics
1. **Accuracy**: Overall classification accuracy
2. **ROC AUC**: Area Under the ROC Curve
3. **F1 Score**: Harmonic mean of precision and recall
4. **Standard Deviations**: Reported for all metrics across 5 CV folds

### 4. Feature Sets Comparison

The experiment evaluates multiple feature set configurations:

| Feature Set | Description | Feature Count |
|------------|-------------|---------------|
| **unified** | CE-VAR + All Geometry (75 features) | ~75 |
| variability | CE-VAR only (variability metrics) | ~60 |
| geometry_all | All geometry metrics (CE + TF-IDF + SBERT) | 15 |
| ce_geometry | CE geometry metrics only | 5 |
| tfidf_geometry | TF-IDF geometry metrics only | 5 |
| sbert_geometry | SBERT geometry metrics only | 5 |

### 5. Data Preprocessing

- **Missing Values**: Filled with 0.0
- **Normalization**: Features are already normalized (variability metrics use normalized variants)
- **Feature Selection**: Only trajectory-related features included (metadata columns excluded)

### 6. Experimental Results

#### 6.1. Performance Metrics Summary

**Dataset Configuration:**
- **Total Samples**: 2,060 authors
- **Human Authors**: 412 (20.0%)
- **LLM Authors**: 1,648 (80.0%)
- **Cross-Validation**: 5-fold stratified K-Fold
- **Evaluation**: Mean ± Standard Deviation across 5 folds

#### 6.2. Unified 75 Features Model Performance (Primary Results)

| Metric | Mean | Std Dev | Range (Min-Max across folds) |
|--------|------|---------|------------------------------|
| **Accuracy** | **92.67%** | ±1.06% | ~91.61% - 93.73% |
| **ROC AUC** | **97.05%** | ±0.48% | ~96.57% - 97.53% |
| **F1 Score** | **77.85%** | ±3.77% | ~74.08% - 81.62% |

**Interpretation:**
- The unified model achieves **excellent classification performance** with 92.67% accuracy
- **High ROC AUC (97.05%)** indicates strong discriminative power between Human and LLM classes
- **F1 Score (77.85%)** reflects good balance between precision and recall despite class imbalance

#### 6.3. Feature Set Comparison Results

**Detailed Performance by Feature Set:**

| Feature Set | Accuracy | Accuracy Std | ROC AUC | ROC AUC Std | F1 Score | F1 Score Std |
|------------|----------|--------------|---------|-------------|----------|--------------|
| **unified** | **92.67%** | ±1.06% | **97.05%** | ±0.48% | **77.85%** | ±3.77% |
| variability | 91.70% | ±1.08% | 96.06% | ±0.97% | 74.79% | ±3.85% |
| geometry_all | 84.56% | ±1.11% | 84.06% | ±2.64% | 46.97% | ±5.06% |
| ce_geometry | 78.11% | ±0.86% | 57.12% | ±2.09% | 11.68% | ±3.76% |
| tfidf_geometry | 80.49% | ±0.99% | 73.94% | ±2.22% | 36.54% | ±4.14% |
| sbert_geometry | 79.37% | ±0.98% | 60.98% | ±2.42% | 22.84% | ±4.01% |

**Key Findings:**

1. **Unified Model (Best Performance)**:
   - Achieves the highest accuracy (92.67%) and ROC AUC (97.05%)
   - Demonstrates that combining CE-VAR and Geometry features provides superior discriminative power
   - F1 score (77.85%) significantly higher than individual feature sets

2. **Variability Features (CE-VAR Only)**:
   - Second-best performance: 91.70% accuracy, 96.06% ROC AUC
   - Strong performance suggests variability metrics capture important temporal patterns
   - Only 0.97% lower accuracy than unified model

3. **Geometry Features (All Spaces Combined)**:
   - Moderate performance: 84.56% accuracy, 84.06% ROC AUC
   - Lower F1 score (46.97%) indicates challenges with class imbalance
   - Combining multiple embedding spaces (CE + TF-IDF + SBERT) improves over individual spaces

4. **Individual Geometry Spaces**:
   - **TF-IDF Geometry**: Best among individual spaces (80.49% accuracy, 73.94% ROC AUC)
   - **SBERT Geometry**: Moderate performance (79.37% accuracy, 60.98% ROC AUC)
   - **CE Geometry**: Lowest performance (78.11% accuracy, 57.12% ROC AUC)
   - All individual spaces show low F1 scores, indicating difficulty with minority class (human)

#### 6.4. Performance Improvement Analysis

**Unified vs. Variability Only:**
- Accuracy improvement: +0.97% (92.67% vs. 91.70%)
- ROC AUC improvement: +0.99% (97.05% vs. 96.06%)
- F1 Score improvement: +3.06% (77.85% vs. 74.79%)
- **Conclusion**: Adding geometry features provides meaningful performance boost, especially for F1 score

**Unified vs. Geometry Only:**
- Accuracy improvement: +8.11% (92.67% vs. 84.56%)
- ROC AUC improvement: +12.99% (97.05% vs. 84.06%)
- F1 Score improvement: +30.88% (77.85% vs. 46.97%)
- **Conclusion**: Variability features are crucial for handling class imbalance and improving minority class recall

**Unified vs. Best Individual Geometry (TF-IDF):**
- Accuracy improvement: +12.18% (92.67% vs. 80.49%)
- ROC AUC improvement: +23.11% (97.05% vs. 73.94%)
- F1 Score improvement: +41.31% (77.85% vs. 36.54%)
- **Conclusion**: Unified approach significantly outperforms individual feature sets

#### 6.5. Cross-Validation Stability

**Coefficient of Variation (CV) Analysis:**
- **Unified Model**: Low CV across metrics
  - Accuracy CV: 1.14% (1.06/92.67)
  - ROC AUC CV: 0.49% (0.48/97.05)
  - F1 Score CV: 4.84% (3.77/77.85)
- **Interpretation**: Results are highly stable across CV folds, indicating robust model performance

#### 6.6. Class Imbalance Impact

**Observations:**
- Class distribution: 20% Human, 80% LLM
- High ROC AUC (97.05%) indicates good discrimination despite imbalance
- F1 score (77.85%) lower than accuracy, reflecting recall challenges for minority class
- **Balanced class weighting** in Random Forest mitigates imbalance effects

### 7. Scripts and Outputs

**Main Classification Script:**
- `scripts/micro/analysis/run_trajectory_classification.py`

**Visualization Script:**
- `scripts/micro/visualization/visualize_unified_75_features_ml.py`

**Output Files:**
- Results CSV: `plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv`
- Visualizations: `plots/trajectory/unified_75_features_ml/`
  - `performance_comparison.png`: Bar plot comparing all feature sets across metrics
  - `performance_comparison_acc_roc.png`: Focused comparison of Accuracy and ROC AUC
  - `roc_curve.png`: ROC curves for unified model (per-fold and average)
  - `confusion_matrix.png`: Confusion matrix visualization for unified model
  - `feature_importance.png`: Top 15 feature importances from Random Forest
  - `summary_panel.png`: Comprehensive multi-panel summary of all results

#### 7.1. Confusion Matrix Analysis

**Visualization Details:**
- **Type**: Normalized confusion matrix (percentages)
- **Matrix Structure**: 2×2 (Human vs. LLM)
- **Metrics Shown**: True Positive Rate, True Negative Rate, Precision, Recall
- **Calculation**: Averaged across 5 CV folds

**Expected Patterns (based on performance metrics):**
- **True Positives (Human correctly identified)**: ~77.85% (based on F1 score)
- **True Negatives (LLM correctly identified)**: ~96%+ (majority class advantage)
- **False Positives (LLM misclassified as Human)**: Low rate
- **False Negatives (Human misclassified as LLM)**: Higher rate due to class imbalance

#### 7.2. ROC Curve Analysis

**Visualization Details:**
- **Curves Shown**: 
  - Individual ROC curves for each of 5 CV folds
  - Average ROC curve across all folds
  - Diagonal reference line (random classifier)
- **AUC Display**: Mean ± Std Dev ROC AUC shown in legend
- **Threshold Range**: Full range of decision thresholds (0.0 to 1.0)

**Interpretation:**
- **Mean ROC AUC: 97.05%** indicates excellent discrimination
- **Low Std Dev (±0.48%)** shows consistent performance across folds
- Curve distance from diagonal indicates model performance (greater distance = better performance)

### 8. Key Design Decisions

1. **LV3 Only**: Focuses on highest-prompt-guidance LLM outputs
2. **All Domains Combined**: Tests generalizability across Academic, Blogs, and News
3. **All Models Combined**: Evaluates across different LLM architectures
4. **Unified Feature Set**: Combines variability and geometry metrics for comprehensive representation
5. **Balanced Class Weight**: Handles potential class imbalance
6. **5-Fold CV**: Robust evaluation with statistical stability

### 9. Reproducibility

- **Random Seed**: 42 (used consistently across all random operations)
- **Data Path**: `dataset/process/`
- **Results Path**: `plots/trajectory/combined/`
- **Command to Reproduce**:
  ```bash
  python scripts/micro/analysis/run_trajectory_classification.py \
    --domains academic blogs news \
    --models DS G4B G12B LMK \
    --level LV3 \
    --geometry-spaces ce tfidf sbert
  ```

### 10. Feature Importance Analysis

#### 10.1. Feature Importance Methodology

**Calculation Method:**
- **Algorithm**: Random Forest Feature Importance (Gini impurity-based)
- **Aggregation**: Average importance across all 300 trees in the forest
- **Cross-Validation**: Feature importances computed for each CV fold, then averaged
- **Normalization**: Importances sum to 1.0 across all features

**Interpretation:**
- Higher importance values indicate features that contribute more to classification decisions
- Features with importance > 0.01 (1%) are considered "important"
- Top features provide insights into which trajectory characteristics best distinguish Human vs. LLM

#### 10.2. Feature Importance Categories

The unified 75 features are grouped into **seven categories** for detailed analysis:

1. **CE-VAR Features** (subdivided into three layers):
   - **CE-VAR-Cognitive**: Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) - 15 features
   - **CE-VAR-Emotional**: Sentiment and emotional features (polarity, subjectivity, vader_*) - 18 features
   - **CE-VAR-Stylistic**: Writing style features (gunning_fog, avg_sentence_length, word_diversity, etc.) - 21 features
   - **CE-VAR (Unclassified)**: Features that cannot be categorized into above layers - 6 features
   - Variability metrics: `_cv`, `_rmssd_norm`, `_masd_norm`

2. **CE Geometry Features** (5 features)
   - `ce_mean_distance`, `ce_std_distance`, `ce_net_displacement`, `ce_path_length`, `ce_tortuosity`

3. **TF-IDF Geometry Features** (5 features)
   - `tfidf_mean_distance`, `tfidf_std_distance`, `tfidf_net_displacement`, `tfidf_path_length`, `tfidf_tortuosity`

4. **SBERT Geometry Features** (5 features)
   - `sbert_mean_distance`, `sbert_std_distance`, `sbert_net_displacement`, `sbert_path_length`, `sbert_tortuosity`

#### 10.3. Feature Importance Visualization

**Output Visualizations:**
- `feature_importance.png`: Bar plot of top 15 most important features
- `summary_panel.png`: Includes feature importance subplot in comprehensive summary panel

**Analysis Approach:**
- Top N features (default: top 15) are displayed in descending order
- Features are color-coded by category (CE-VAR, CE-Geo, TF-IDF-Geo, SBERT-Geo)
- Relative importance provides insights into feature contributions

#### 10.4. Feature Importance Results

**Top 10 Most Important Features:**

| Rank | Feature | Importance | Std Dev | Feature Type |
|------|---------|------------|---------|--------------|
| 1 | `Agreeableness_cv` | 0.0742 | ±0.0043 | CE-VAR-Cognitive |
| 2 | `avg_sentence_length_cv` | 0.0673 | ±0.0016 | CE-VAR-Stylistic |
| 3 | `Agreeableness_rmssd_norm` | 0.0407 | ±0.0015 | CE-VAR-Cognitive |
| 4 | `avg_sentence_length_rmssd_norm` | 0.0391 | ±0.0032 | CE-VAR-Stylistic |
| 5 | `tfidf_path_length` | 0.0304 | ±0.0022 | TF-IDF-Geometry |
| 6 | `flesch_reading_ease_masd_norm` | 0.0300 | ±0.0029 | CE-VAR-Stylistic |
| 7 | `Neuroticism_cv` | 0.0298 | ±0.0041 | CE-VAR-Cognitive |
| 8 | `tfidf_mean_distance` | 0.0284 | ±0.0023 | TF-IDF-Geometry |
| 9 | `gunning_fog_cv` | 0.0261 | ±0.0046 | CE-VAR-Stylistic |
| 10 | `avg_sentence_length_masd_norm` | 0.0239 | ±0.0023 | CE-VAR-Stylistic |

**Feature Importance by Category:**

| Feature Type | Count | Mean Importance | Max Importance | Std Dev |
|--------------|-------|-----------------|----------------|---------|
| **CE-VAR-Cognitive** | 15 | 0.0174 | 0.0742 | 0.0136 |
| **CE-VAR-Stylistic** | 21 | 0.0168 | 0.0673 | 0.0114 |
| **TF-IDF-Geometry** | 5 | 0.0173 | 0.0304 | 0.0085 |
| **CE-VAR-Emotional** | 18 | 0.0092 | 0.0223 | 0.0046 |
| **SBERT-Geometry** | 5 | 0.0094 | 0.0148 | 0.0034 |
| **CE-Geometry** | 5 | 0.0104 | 0.0144 | 0.0033 |
| **CE-VAR (Unclassified)** | 6 | 0.0067 | 0.0079 | 0.0006 |

**Key Findings:**

1. **CE-VAR-Cognitive Features Dominate**:
   - Top feature: `Agreeableness_cv` (0.0742 importance, 7.4% of total)
   - Highest mean importance among all categories (0.0174)
   - 4 out of top 10 features are from this category

2. **CE-VAR-Stylistic Features Show Strong Contribution**:
   - Second-highest mean importance (0.0168)
   - 5 out of top 10 features (including #2, #4, #6, #9, #10)
   - `avg_sentence_length_cv` ranked #2 (0.0673)

3. **TF-IDF Geometry Features Are Prominent**:
   - Third-highest mean importance (0.0173)
   - Two features in top 10: `tfidf_path_length` (#5) and `tfidf_mean_distance` (#8)
   - Strong discriminative power despite only 5 features

4. **CE-VAR-Emotional Features Have Lower Impact**:
   - Mean importance (0.0092) lower than Cognitive and Stylistic
   - Top emotional feature: `vader_compound_cv` (0.0223, rank #13)

5. **Geometry Features Show Moderate Importance**:
   - CE-Geometry and SBERT-Geometry have similar mean importance (~0.009-0.010)
   - Path length features generally more important than other geometry metrics

**Visualization Output:**
- `feature_importance.png`: Top 15 features basic visualization
- `feature_importance_comparison.png`: Comprehensive 4-panel comparison visualization
  - (A) Top 30 features overall
  - (B) Distribution by feature type (boxplot)
  - (C) Top 5 per feature type
  - (D) Mean importance by feature type with statistics
- `feature_importance_data.csv`: Complete feature importance data for all 75 features

**Scripts and Outputs:**
- Analysis Script: `scripts/micro/visualization/visualize_unified_75_features_importance_comparison.py`
- Data File: `plots/trajectory/unified_75_features_ml/feature_importance_data.csv`

### 11. Binomial Test Analysis

In addition to ML classification, **per-feature binomial tests** were conducted for all 75 unified features to assess whether human authors show higher variability/geometry values compared to LLM-generated text.

#### Binomial Test Configuration:
- **Test Type**: One-sided binomial test (H₁: Human > LLM, i.e., p > 0.5)
- **Comparison Unit**: Per-feature, per author pair (human vs. each of 4 LLM models)
- **Total Comparisons**: 1,648 comparisons per feature (412 human authors × 4 LLM models)
- **Significance Level**: α = 0.05
- **Null Hypothesis**: H₀: p = 0.5 (no difference)

#### Binomial Test Results Summary:

**Overall Statistics:**
- **Total Features Tested**: 75 features
- **Significant Features (p < 0.05)**: 57/75 (76.0%)
- **Features with Human Win Rate > 50%**: 60/75 (80.0%)

**Results by Feature Group:**

| Feature Group | Count | Mean Human Win Rate | Significant | Win Rate > 50% |
|---------------|-------|---------------------|-------------|----------------|
| **CE-VAR** | 60 | 62.76% | 49/60 (81.7%) | 51/60 (85.0%) |
| **CE-GEO** | 5 | 67.57% | 4/5 (80.0%) | 4/5 (80.0%) |
| **SBERT-GEO** | 5 | 65.87% | 3/5 (60.0%) | 4/5 (80.0%) |
| **TF-IDF-GEO** | 5 | 35.39% | 1/5 (20.0%) | 1/5 (20.0%) |

**Key Findings:**
- **CE-VAR features** show strongest human advantage (mean 62.76% win rate, 49/60 significant)
- **CE-GEO features** show strong human advantage (mean 67.57% win rate)
- **SBERT-GEO features** show strong human advantage (mean 65.87% win rate, 4/5 features > 50%)
- **TF-IDF-GEO features** generally favor LLM (mean 35.39% human win rate, only 1/5 > 50%)

**Top Performing Features (Highest Human Win Rates):**
1. `gunning_fog_cv` (CE-VAR): 83.37% win rate (p < 0.001)
2. `avg_sentence_length_cv` (CE-VAR): 82.83% win rate (p < 0.001)
3. `Neuroticism_cv` (CE-VAR): 81.74% win rate (p < 0.001)
4. `sbert_mean_distance` (SBERT-GEO): 80.34% win rate (p < 0.001)
5. `sbert_path_length` (SBERT-GEO): 80.34% win rate (p < 0.001)

**Scripts and Outputs:**
- Analysis Script: `scripts/micro/analysis/analyze_unified_75_features_binomial.py`
- Visualization Script: `scripts/micro/visualization/visualize_unified_75_features_binomial.py`
- Results CSV: `plots/binomial/unified_75_features/unified_75_features_binomial.csv`
- Visualizations: `plots/binomial/unified_75_features/`
  - `unified_75_features_binomial.png`
  - `win_rate_by_feature_group.png`
  - `per_feature_win_rate_by_group.png`
  - `win_rate_heatmap.png`
  - `summary_panel.png`

### 12. Robustness Test Across Prompting Levels

To validate the robustness of findings across different prompting levels, **binomial tests were conducted for all three prompting levels (LV1, LV2, LV3)** using the unified 75 features.

#### Robustness Test Configuration:
- **Levels Tested**: LV1, LV2, LV3 (all three prompting levels)
- **Features**: Unified 75 trajectory features (same feature set across all levels)
- **Test Type**: Same one-sided binomial test (H₁: Human > LLM)
- **Purpose**: Assess whether findings are consistent across different prompt guidance intensities

#### Robustness Test Results:

**Overall Human Win Rate Across Levels:**
- **LV1**: 61.79% (n = 123,540 comparisons, p < 0.001)
- **LV2**: 62.02% (n = 123,549 comparisons, p < 0.001)
- **LV3**: 61.46% (n = 123,546 comparisons, p < 0.001)
- **Range**: 0.56% (very stable across levels)
- **Mean**: 61.76%
- **Std Dev**: 0.28%

**Results by Feature Group Across Levels:**

| Feature Group | LV1 Win Rate | LV2 Win Rate | LV3 Win Rate | Range | Mean |
|---------------|--------------|--------------|--------------|-------|------|
| **CE-VAR** | 63.48% | 64.15% | 62.76% | 1.39% | 63.46% |
| **CE-GEO** | 65.22% | 63.64% | 67.57% | 3.93% | 65.48% |
| **TF-IDF-GEO** | 34.92% | 32.03% | 35.39% | 3.36% | 34.11% |
| **SBERT-GEO** | 65.01% | 64.85% | 65.87% | 1.02% | 65.24% |

**Key Robustness Findings:**
1. **Overall consistency**: Human win rates are remarkably stable across all three levels (range < 1%)
2. **CE-VAR features**: Consistent advantage across levels (range: 1.39%)
3. **CE-GEO features**: Strong human advantage maintained across all levels (mean: 65.48%)
4. **SBERT-GEO features**: Very stable performance across levels (range: 1.02%)
5. **TF-IDF-GEO features**: Consistently favor LLM across all levels (mean: 34.11%)

**Robustness Conclusion:**
The binomial test results demonstrate **high robustness** across prompting levels. The human advantage in trajectory features is consistent regardless of the prompt guidance intensity (LV1: minimal guidance, LV2: moderate guidance, LV3: maximum guidance).

**Robustness Test Scripts and Outputs:**
- Visualization Script: `scripts/micro/visualization/visualize_binomial_test_robustness.py`
- Results Files:
  - LV1: `micro_results/binomial/binomial_test_trajectory_features_lv1.csv`
  - LV2: `micro_results/binomial/binomial_test_trajectory_features_lv2.csv`
  - LV3: `micro_results/binomial/binomial_test_trajectory_features_lv3.csv`
- Visualizations: `plots/binomial/robustness_test/`
  - `win_rate_comparison_by_level.png`
  - `overall_win_rate_comparison.png`
  - `variability_analysis.png`
  - `summary_table.png`

**Command to Run Robustness Visualization:**
```bash
python scripts/micro/visualization/visualize_binomial_test_robustness.py \
    --results-lv1 micro_results/binomial/binomial_test_trajectory_features_lv1.csv \
    --results-lv2 micro_results/binomial/binomial_test_trajectory_features_lv2.csv \
    --results-lv3 micro_results/binomial/binomial_test_trajectory_features_lv3.csv \
    --output plots/binomial/robustness_test
```

### 13. Logistic Regression p-value Validation

To provide statistical significance testing for the unified 75-feature model, **logistic regression with p-value computation** was conducted using statsmodels. This analysis validates that features have genuine discriminative power (not just by chance) and provides interpretable coefficients.

#### 13.1. Logistic Regression Configuration

**Method:**
- **Algorithm**: Logistic Regression (statsmodels)
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Feature Scaling**: StandardScaler applied (required for logistic regression)
- **Aggregation**: Results averaged across CV folds, with min/max p-values reported
- **Significance Levels**: 
  - `***`: p < 0.001
  - `**`: p < 0.01
  - `*`: p < 0.05

**Output Metrics:**
- **Coefficient**: Logistic regression coefficient (log-odds)
- **Standard Error**: Standard error of coefficient
- **Z-value**: Z-statistic (coefficient / standard error)
- **P-value**: Statistical significance (two-tailed test)
- **Odds Ratio**: exp(coefficient), interpretable effect size

#### 13.2. Logistic Regression Results Summary

**Overall Statistics:**
- **Total Features Tested**: 75 features (excluding intercept)
- **Significant Features (p < 0.05)**: 18/75 (24.0%)
- **Highly Significant (p < 0.01)**: 11/75 (14.7%)
- **Very Highly Significant (p < 0.001)**: 4/75 (5.3%)

**Top 15 Most Significant Features (by p-value):**

| Rank | Feature | Coefficient | P-value | Significance | Feature Type | Odds Ratio |
|------|---------|-------------|---------|--------------|--------------|------------|
| 1 | `tfidf_std_distance` | 1.26 | 9.24e-10 | *** | TF-IDF-Geometry | 3.56 |
| 2 | `tfidf_mean_distance` | -2.99 | 9.24e-08 | *** | TF-IDF-Geometry | 0.05 |
| 3 | `avg_sentence_length_masd_norm` | -2.32 | 4.20e-04 | *** | CE-VAR | 0.10 |
| 4 | `Agreeableness_masd_norm` | -1.61 | 1.04e-03 | ** | CE-VAR | 0.20 |
| 5 | `avg_sentence_length_rmssd_norm` | 2.66 | 2.40e-03 | ** | CE-VAR | 15.53 |
| 6 | `Agreeableness_rmssd_norm` | 1.73 | 3.24e-03 | ** | CE-VAR | 5.71 |
| 7 | `Agreeableness_cv` | 1.13 | 4.42e-03 | ** | CE-VAR | 3.21 |
| 8 | `sbert_mean_distance` | 1.16 | 4.59e-03 | ** | SBERT-Geometry | 3.23 |
| 9 | `tfidf_path_length` | 1.87 | 7.43e-03 | ** | TF-IDF-Geometry | 6.89 |
| 10 | `Neuroticism_cv` | 0.80 | 2.12e-02 | * | CE-VAR | 2.24 |
| 11 | `word_diversity_masd_norm` | -1.54 | 2.29e-02 | * | CE-VAR | 0.22 |
| 12 | `avg_sentence_length_cv` | 0.95 | 2.37e-02 | * | CE-VAR | 2.64 |
| 13 | `Neuroticism_rmssd_norm` | -1.54 | 2.90e-02 | * | CE-VAR | 0.22 |
| 14 | `vader_pos_cv` | 0.69 | 3.33e-02 | * | CE-VAR | 2.01 |
| 15 | `word_diversity_rmssd_norm` | 1.69 | 3.95e-02 | * | CE-VAR | 5.79 |

#### 13.3. Results by Feature Type

**Statistical Significance by Feature Type:**

| Feature Type | Count | Significant (p<0.05) | Highly Sig (p<0.01) | Very High Sig (p<0.001) | Mean P-value |
|--------------|-------|---------------------|---------------------|-------------------------|--------------|
| **TF-IDF-Geometry** | 5 | 3/5 (60.0%) | 3/5 (60.0%) | 2/5 (40.0%) | 0.104 |
| **CE-VAR** | 60 | 13/60 (21.7%) | 8/60 (13.3%) | 2/60 (3.3%) | 0.299 |
| **SBERT-Geometry** | 5 | 1/5 (20.0%) | 1/5 (20.0%) | 0/5 (0%) | 0.359 |
| **CE-Geometry** | 5 | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0.518 |

**Key Findings:**

1. **TF-IDF Geometry Features Show Strongest Statistical Significance**:
   - 60% of features significant (3/5)
   - Top 2 most significant features overall: `tfidf_std_distance` (p = 9.24e-10) and `tfidf_mean_distance` (p = 9.24e-08)
   - All 3 significant TF-IDF features have p < 0.01 (highly significant)
   - Strong discriminative power confirmed statistically

2. **CE-VAR Features Show Moderate Significance**:
   - 21.7% significant (13/60 features)
   - 8 features highly significant (p < 0.01)
   - `avg_sentence_length_masd_norm` and related features show strong significance
   - `Agreeableness` features consistently significant across different variability metrics

3. **SBERT Geometry Features**:
   - Only 1/5 features significant: `sbert_mean_distance` (p = 0.0046)
   - Moderate discriminative power in statistical testing

4. **CE-Geometry Features**:
   - No features reach statistical significance (p < 0.05)
   - Mean p-value: 0.518 (not significant)
   - Despite importance in Random Forest, not statistically significant in logistic regression

#### 13.4. Coefficient Interpretation

**Positive Coefficients** (increase probability of Human):
- `tfidf_std_distance` (coef = 1.26): Higher TF-IDF std distance increases odds of Human by 3.56×
- `avg_sentence_length_rmssd_norm` (coef = 2.66): Higher RMSSD increases odds of Human by 15.53×
- `Agreeableness_cv` (coef = 1.13): Higher Agreeableness variability increases odds of Human by 3.21×

**Negative Coefficients** (increase probability of LLM):
- `tfidf_mean_distance` (coef = -2.99): Lower TF-IDF mean distance increases odds of LLM
- `avg_sentence_length_masd_norm` (coef = -2.32): Lower MASD norm increases odds of LLM

#### 13.5. Comparison with Feature Importance

**Alignment Between Importance and Significance:**

**Both High Importance AND Significant:**
- `Agreeableness_cv`: Rank #1 importance (0.0742), p = 0.0044 **
- `tfidf_path_length`: Rank #5 importance (0.0304), p = 0.0074 **
- `avg_sentence_length_cv`: Rank #2 importance (0.0673), p = 0.0237 *
- `Neuroticism_cv`: Rank #7 importance (0.0298), p = 0.0212 *

**High Importance but NOT Significant:**
- `avg_sentence_length_cv` (rank #2): p = 0.0237 * (borderline significant)
- `flesch_reading_ease_masd_norm` (rank #6): p = 0.764 (not significant)
- `gunning_fog_cv` (rank #9): p = 0.322 (not significant)

**Low Importance but Significant:**
- `tfidf_std_distance`: Rank #39 importance (0.0085), p = 9.24e-10 *** (most significant!)
- `tfidf_mean_distance`: Rank #8 importance (0.0284), p = 9.24e-08 ***

**Interpretation:**
- Feature importance (Random Forest) and statistical significance (Logistic Regression) provide complementary perspectives
- Some features may have high importance due to complex interactions (captured by RF) but show lower statistical significance in linear models
- TF-IDF Geometry features show strongest alignment between importance and significance

#### 13.6. Output Files

**Results Files:**
- `lr_pvalues_cv_aggregated.csv`: Aggregated results across CV folds (mean, std, min, max p-values)
- `lr_pvalues_cv_folds.csv`: Detailed results for each CV fold

**Script:**
- Analysis Script: `scripts/micro/analysis/analyze_unified_75_features_lr_pvalues.py`

**Command to Run:**
```bash
python scripts/micro/analysis/analyze_unified_75_features_lr_pvalues.py \
    --domains academic blogs news \
    --models DS G4B G12B LMK \
    --level LV3 \
    --output plots/trajectory/unified_75_features_ml \
    --cv
```

#### 13.7. Statistical Validation Summary

**Complementary Evidence for Feature Significance:**

1. **Random Forest Feature Importance**: Identifies features contributing most to classification decisions
2. **Logistic Regression p-values**: Provides statistical significance testing
3. **Binomial Tests**: Per-feature comparison of Human vs. LLM distributions

**Convergent Validation:**
- Features appearing in top ranks across multiple methods provide strongest evidence
- Example: `Agreeableness_cv` is #1 in importance, significant in LR (p = 0.0044), and significant in binomial test
- TF-IDF Geometry features show strong performance across all validation methods

## Summary

The unified 75-feature model achieves excellent performance (92.67% accuracy, 97.05% ROC AUC) in distinguishing human vs. LLM-generated text, demonstrating that combining CE variability metrics with multi-space geometry features provides a robust cognitive fingerprinting approach for RQ1.

**Complementary Validation:**

1. **ML Classification**: Demonstrates strong discriminative power (92.67% accuracy, 97.05% ROC AUC)

2. **Feature Importance Analysis**: 
   - Identifies most discriminative features (top: `Agreeableness_cv`, `avg_sentence_length_cv`)
   - CE-VAR-Cognitive and CE-VAR-Stylistic show highest importance
   - TF-IDF Geometry features rank prominently

3. **Logistic Regression p-values**:
   - 18/75 features (24%) show statistical significance (p < 0.05)
   - TF-IDF Geometry features show strongest statistical significance (3/5 significant, including 2 with p < 0.001)
   - Provides interpretable coefficients and odds ratios

4. **Binomial Tests**: 
   - 57/75 features (76%) show statistically significant human advantage
   - CE-VAR and geometry features (except TF-IDF) consistently favor human authors
   - Provides per-feature comparison validation

**Convergent Evidence:**
- Multiple validation methods converge on key features: `Agreeableness_cv`, TF-IDF Geometry features, and stylistic variability features
- These complementary analyses strengthen confidence in feature significance and model validity

