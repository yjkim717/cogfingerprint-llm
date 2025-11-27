# Macro Static Classification Experiment

## Objective
- Quantify how well the 20 static features (Big5 + NELA merged + length metrics) can distinguish macro human vs. LLM texts.
- Reproduce a macro-scale analogue of the micro time-series ML validation (which achieved ~90% accuracy) using only static fingerprints.

## Data & Features
- Sources: `macro_dataset/process/human/<domain>/combined.csv` (original human files) and `macro_dataset/process/LLM/<model>/LV3/<domain>/combined_outliers_removed.csv`.
- Total rows: 74,995 (15k human + 59,995 LLM). Domain/model breakdown is stored alongside the ML summary.

```36:90:results/macro_static_classification/macro_static_classification_summary.json
  "dataset_overview": {
    "sample_counts": [
      {
        "label": "human",
        "model_target": "HUMAN",
        "domain": "academic",
        "count": 5000
      },
      ...
      {
        "label": "llm",
        "model_target": "LMK",
        "domain": "news",
        "count": 5000
      }
    ],
    "total_rows": 74995
  },
```

- Feature columns: 5 Big5 + 15 NELA / stylistic metrics (see `ml_classify_macro_static.py`).
- Train/test split: 2020–2023 for training, 2024 as the hold-out test year. 5-fold stratified CV on the training portion.

## Modeling Setup
- Script: `ml_classify_macro_static.py`
- Pipeline: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression` (default multi-class).
- Two classification tasks:
  1. Binary (`label`: human vs. llm).
  2. Multi-class (`model_target`: HUMAN, DS, G4B, G12B, LMK).

## Results (2024 Test Year)

### Binary (Human vs. LLM)
- Test accuracy: **88.8%** (CV mean 89.7 ± 0.24%).
- Human precision/recall/F1: 0.79 / 0.60 / 0.68 (most errors are human→LLM due to class imbalance).
- LLM precision/recall/F1: 0.91 / 0.96 / 0.93.
- Confusion (rows = true, cols = pred):
  - Human → Human: 1,793; Human → LLM: 1,207.
  - LLM → Human: 467; LLM → LLM: 11,532.

```131:174:results/macro_static_classification/macro_static_classification_summary.json
  "binary_results": {
    "train_samples": 59996,
    "test_samples": 14999,
    "test_accuracy": 0.8883925595039669,
    ...
    "confusion_matrix": [
      [
        1793,
        1207
      ],
      [
        467,
        11532
      ]
    ],
    "cv_mean_accuracy": 0.8974764660943968,
    "cv_std_accuracy": 0.0024204505321761987
  },
```

### Multi-class (HUMAN + four LLM providers)
- Test accuracy: **52.7%** (CV mean 53.9 ± 0.78%).
- Human class remains the easiest (precision 0.65 / recall 0.70), but the three Google/Gemma models overlap heavily; LMK is moderately separable.
- Confusion matrix highlights:
  - HUMAN rows: 2,086 correct, false positives mostly predicted as DS/G12B/LMK (~30% combined).
  - DS misclassified as HUMAN 14.5% of the time, etc.

```175:270:results/macro_static_classification/macro_static_classification_summary.json
  "multiclass_results": {
    "classes": [
      "DS",
      "G12B",
      "G4B",
      "HUMAN",
      "LMK"
    ],
    "test_accuracy": 0.5269684645643042,
    ...
    "confusion_matrix": [
      [
        1536,
        446,
        306,
        435,
        276
      ],
      ...
    ],
    "cv_mean_accuracy": 0.5387857363113592,
    "cv_std_accuracy": 0.0077983621152652805
  }
```

## Observations
1. Static features alone already reach ~89% binary accuracy—close to the micro time-series benchmark, but human recall is notably lower (0.60) because many human samples resemble LLM statistics.
2. Multi-class accuracy is only ~53%, implying the four macro LLM providers are difficult to separate using aggregated static metrics; stylistic fingerprints converge substantially at LV3.
3. Domain balancing (1000 samples/year/domain) plus the year-based hold-out reduces temporal leakage, but the binary classifier still benefits from the large LLM sample size.

## Next Steps / Ideas
- Address class imbalance (e.g., class-weighted logistic regression or focal loss) to boost human recall.
- Experiment with tree ensembles (XGBoost/LightGBM) and feature-selection to see if non-linear interactions improve multi-class accuracy.
- Incorporate domain and year meta-features (one-hot) to stabilize per-domain decision boundaries.
- Compare with time-series signatures to quantify the exact delta between static vs. sequential representation power on macro data.

