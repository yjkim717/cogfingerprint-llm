# Trajectory Modeling Plan & Progress

This note tracks the unified trajectory modeling design (local variability + global geometry) and the concrete progress toward each milestone.

## Design Snapshot

1. **State Representation**
   - Per-author annual vectors in three complementary spaces:
     - CE space: 20 handcrafted features (Big5, NELA, stylistic metrics).
     - TF-IDF+SVD space: 10-dimensional vectors (`tfidf_author_yearly_vectors.csv`).
     - SBERT space: 384-dimensional vectors (`sbert_author_yearly_vectors.csv`).

2. **Local Variability**
   - CE space: existing `CV`, `RMSSD_norm`, `MASD_norm`.
   - TF-IDF/SBERT: yearly Euclidean steps (`mean_distance`, `std_distance`).

3. **Global Geometry**
   - Net displacement `D = ||f_last - f_first||`.
   - Path length `L = Σ ||f_{t+1} - f_t||`.
   - Tortuosity `τ = L / D` (fallback 1 when D≈0).
   - Directional consistency `C = mean cos(v_t, v_{t+1})`.

4. **Unified Trajectory Representation**
   - Concatenate CE variability (≈60 dims) + TF-IDF/SBERT geometries (2 spaces × 4 dims, optionally CE geometry too).
   - Use for statistical tests, trajectory-level classification, PCA/UMAP visualization.

## Progress Checklist

| Step | Status | Notes |
| --- | --- | --- |
| Aggregate per-sample TF-IDF vectors | ✅ | `scripts/micro/tfidf/extract_tfidf_vectors.py` writes `tfidf_vectors.csv`. |
| Aggregate per-sample SBERT vectors | ✅ | `scripts/micro/sbert/extract_sbert_vectors.py` writes `sbert_vectors.csv`. |
| Average to per-author per-year TF-IDF vectors | ✅ | `scripts/micro/tfidf/aggregate_tfidf_yearly.py` outputs `tfidf_author_yearly_vectors.csv`. |
| Average to per-author per-year SBERT vectors | ✅ | `scripts/micro/sbert/aggregate_sbert_yearly.py` outputs `sbert_author_yearly_vectors.csv`. |
| Compute TF-IDF/SBERT yearly drift + geometry | ✅ | `scripts/micro/trajectory/compute_embedding_trajectory_features.py` → `*_trajectory_features.csv`. |
| Compute CE-space geometry | ✅ | `scripts/micro/trajectory/compute_ce_trajectory_features.py` → `ce_trajectory_features.csv`. |
| Build unified trajectory feature table | ✅ | `scripts/micro/trajectory/build_combined_trajectory_features.py` → `trajectory_features_combined.csv`. |
| Statistical comparison (Human vs LLM) | ✅ | `scripts/micro/analysis/analyze_trajectory_features.py` → boxplots, stats, PCA (`plots/trajectory/`). |
| Trajectory-level classification | ✅ | `scripts/micro/analysis/run_trajectory_classification.py` (Human vs LV3). |

### Key Metrics (Human vs LV3, combined domains)

| Feature Set | Accuracy | ROC AUC | Notes |
| --- | --- | --- | --- |
| Variability (CE CV/RMSSD_norm/MASD_norm) | 0.917 ± 0.011 | 0.961 ± 0.010 | Baseline from `author_timeseries_stats_merged.csv`. |
| Geometry – CE only | 0.792 ± 0.008 | 0.569 ± 0.018 | Global drift in CE space (`ce_*` metrics). |
| Geometry – TF-IDF only | 0.815 ± 0.007 | 0.767 ± 0.028 | TF-IDF net/path/τ/C alone. |
| Geometry – SBERT only | 0.795 ± 0.008 | 0.620 ± 0.024 | SBERT net/path/τ/C alone. |
| Geometry – All spaces | 0.845 ± 0.014 | 0.846 ± 0.027 | All embedding geometries combined (no CE variability). |
| Unified (Variability + Geometry, concatenated channel-wise) | 0.942 ± 0.011 | 0.983 ± 0.005 | Concatenation of CE variability + CE/TF-IDF/SBERT geometry from `trajectory_features_combined.csv`. |

*(Source: `plots/trajectory/combined/classification_results_academic_blogs_news_LV3.csv`.)*

**ML Validation Design**

- **Task**: Binary Human vs. LLM classification (LV3 only), combining academic + blogs + news samples.
- **Splits**: 5-fold Stratified CV, RandomForest (300 trees, balanced class weights); metrics reported as mean ± std.
- **Feature Sets**:
  - `variability`: 60 CE local statistics (`*_cv`, `*_rmssd_norm`, `*_masd_norm`), mirroring `ml_classify_author_by_timeseries.py`.
  - `geometry` variants: CE / TF-IDF / SBERT `mean_distance`, `std_distance`, `net_displacement`, `path_length`, `tortuosity`, `direction_consistency`.
  - `unified`: concatenation of variability + all geometry features (~72 dims).
- **Outputs**: CSV summaries stored under `plots/trajectory/combined/`; “unified” set concatenates variability + CE/TF-IDF/SBERT geometry columns.

## Upcoming Tasks

1. Optional: additional visualizations (UMAP, provider-level drift) if needed for RQ2.
2. Document final RQ1 results (stats + classification + PCA) in the paper draft.

