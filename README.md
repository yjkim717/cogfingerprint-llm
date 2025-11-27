# LLM Dataset Generation Pipeline — cogfingerprint-llm 
**Cognitive Fingerprinting Dataset Generator — Multi-Level Human→LLM Parallel Text Pipeline**

## Overview
This repository builds a **3-Level LLM dataset pipeline** that mirrors human cognitive trajectories in text writing.  
For each *Human* text (Academic / Blogs / News), the pipeline generates **Level 1–3 LLM counterparts** using  
incremental prompt guidance (zero-shot → persona → persona + few-shot examples).

Each file produces:
-  **5 Keywords**
-  **1-Sentence Summary**
-  **Word Count**
-  **LLM-Generated Texts (LV1–LV3)**

---

## Dataset Structure

```
cogfingerprint-llm/
 ├── dataset/                     # Micro dataset (human + generated LLM txt)
 │   ├── human/<genre>/<field>/...
 │   └── llm/<model>/<level>/<genre>/...
 │
 ├── dataset/process/             # Micro features & stats
 │   ├── human/<domain>/big5.csv
 │   ├── human/<domain>/nela_merged.csv
 │   ├── human/<domain>/combined_merged.csv
 │   └── LLM/<MODEL>/<LV>/<domain>/...
 │
 ├── macro_dataset/               # Macro dataset mirrors the same structure
 │   ├── human/<domain>/...
 │   └── LLM/<MODEL>/<LV>/<domain>/...
 │
 ├── scripts/
 │   ├── micro/                   # Micro pipelines (feature extraction, ML, binomial, etc.)
 │   ├── macro/                   # Macro pipelines (static classification, RQ2, etc.)
 │   ├── generation/
 │   │   ├── micro/               # CLI entry points per genre (Academic / Blogs / News)
 │   │   └── macro/               # Macro-level generation CLIs
 │   ├── visualization/           # Plotting helpers
 │   └── tools/                   # Misc utilities (e.g., quick tests)
 │
 ├── docs/                        # Methodology + cleanup notes
 ├── utils/                       # Shared helpers (API, prompts, metadata parsing, etc.)
 ├── macro_results/               # Aggregated macro experiment outputs
 ├── micro_results/               # Aggregated micro ML/Binomial outputs
 ├── config.py
 ├── main.py (legacy aggregator)
 └── main_refactored.py (legacy aggregator)
```

---

##  Architecture

```
cogfingerprint-llm/
│
├── scripts/
│   ├── generation/micro/        # Genre-specific CLI (generate_*_cli.py)
│   ├── generation/macro/        # Macro-level CLI (generate_macro_*_cli.py)
│   ├── micro/                   # batch_analyze_metrics, remove_outliers, ML/Binomial, etc.
│   └── macro/                   # analyze_macro_metrics, ml_classify_macro_static, etc.
│
├── utils/
│   ├── api_utils.py            # Unified LLM API handler (DeepSeek / Gemma / Llama / OpenRouter)
│   ├── extract_utils.py        # Keyword, summary, word-count extractor via JSON parsing
│   ├── prompt_utils.py         # Prompt templates for all genres & levels
│   ├── file_utils.py           # Metadata parsing & filename builders
│   └── topic_utils.py          # (Optional) Topic extractor using OpenAI GPT-4o-mini
│
├── Datasets/
│   ├── Human/...               # Human source corpus by genre / subfield / year
│   └── LLM/...                 # Generated parallel texts (LV1–3)
│
└── config.py                   # API key management
```

---

##  Setup

### 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install requests openai
```

### 2. API Keys

Create a `.env` file or export keys directly:

```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxx"
export DEEPSEEK_API_KEY="sk-deepseek-xxxxxxxxxxxx"
export OPENAI_API_KEY="sk-openai-xxxxxxxxxxxx"
```

### 3. Select Provider
Default = `DEEPSEEK`  
You can override by setting:

```bash
export LLM_PROVIDER="GEMMA_4B"
# or GEMMA_12B / LLAMA_MAVRICK / DEEPSEEK
```

---

##  Extraction Levels

| Level | Description | Prompt Style |
|--------|--------------|--------------|
| LV1 | Zero-shot baseline | Simple extraction without persona |
| LV2 | Genre-based persona | Adds stylistic guidance per genre |
| LV3 | Persona + few-shot examples | Adds contextual examples for stronger fidelity |

---

##  Model Support

| Provider | Model ID | Endpoint | Notes |
|-----------|-----------|-----------|--------|
| **DeepSeek** | `deepseek-chat` | `https://api.deepseek.com/v1` | Default fast model |
| **Gemma-4B** | `google/gemma-3-4b-it` | OpenRouter | Paid tier only |
| **Gemma-12B** | `google/gemma-3-12b-it` | OpenRouter | Stronger reasoning |
| **Llama-4 Maverick** | `meta-llama/llama-4-maverick` | OpenRouter | Used for LV1–3 fallback |

---

##  Run Pipeline

1. **Generate LLM texts per genre** (micro dataset):
   ```bash
   # Academic / Blogs / News share the same flags (model, levels, workers, etc.)
   python scripts/generation/micro/generate_academic_cli.py --model GEMMA_12B --levels 1 2 3
   python scripts/generation/micro/generate_blogs_cli.py --model DEEPSEEK --levels 1
   python scripts/generation/micro/generate_news_cli.py --model LLAMA_MAVRICK --levels 1 2 3
   ```

2. **Generate macro-level LLM texts** (macro dataset):
   ```bash
   python scripts/generation/macro/generate_macro_news_cli.py --model GEMMA_12B --levels 1 2 3
   # ... similarly for Blogs / Academic
   ```

3. **Batch feature extraction (Big5 + merged NELA)**
   ```bash
   python scripts/micro/batch_analyze_metrics.py                 # runs across all micro domains/models/levels
   python scripts/macro/analyze_macro_metrics.py --target llm ... # macro dataset per-domain
   ```

4. **Outlier removal → time series stats**
   ```bash
   python scripts/micro/remove_outliers_from_combined_merged.py --input dataset/process/LLM/G12B/LV1/news/combined_merged.csv
   python scripts/micro/generate_timeseries_stats_from_outliers_removed.py --target llm --models G12B GEMMA_12B
   ```

5. **ML validation + Binomial tests**
   ```bash
   python scripts/micro/ml_classify_author_by_timeseries.py --level 1 --outliers-removed
   python scripts/micro/binomial_test_human_vs_llm.py --level 1 --stat cv
   ```

> `main.py` / `main_refactored.py` are legacy wrappers that batch all genres/levels. Use the CLI scripts above for new runs.

---

## Generator CLI Usage

### Micro dataset

| Flag | Meaning | Default |
|------|---------|---------|
| `--model {DEEPSEEK,GEMMA_4B,GEMMA_12B,LLAMA_MAVRICK}` | LLM provider/model tag | value from `.env` or `DEEPSEEK` |
| `--levels ...` | List of levels to run (space separated) | `1 2 3` |
| `--human-dir` | Source directory (`dataset/human`) | `dataset/human` |
| `--llm-dir` | Output directory (`dataset/llm`) | `dataset/llm` |
| `--workers` | Concurrent worker threads | `5` |
| `--verbose` | Show per-file progress | off |

Examples:

```bash
# Academic LV1-3 using Gemma 12B
python scripts/generation/micro/generate_academic_cli.py --model GEMMA_12B --levels 1 2 3

# Blogs LV1 only, verbose logging
python scripts/generation/micro/generate_blogs_cli.py --model DEEPSEEK --levels 1 --verbose

# News LV2 with more workers
python scripts/generation/micro/generate_news_cli.py --model LLAMA_MAVRICK --levels 2 --workers 10
```

### Macro dataset

Macro CLIs mirror the same flags but default to `macro_dataset/human` / `macro_dataset/llm`:

```bash
python scripts/generation/macro/generate_macro_news_cli.py --model GEMMA_12B --levels 1 2 3 --workers 8
```

### Handling moderated files

If Meta moderation replaced a News sample with the placeholder phrase, rerun:

```bash
python scripts/generation/micro/rerun_meta_flagged.py
```

It reads `logs/fallback_samples.txt`, maps back to the human source, and regenerates the affected LLM outputs.

---

##  Robustness Features

| Category | Behavior |
|-----------|-----------|
| **Empty Response** | Skips safely, creates placeholder JSON |
| **Meta Moderation** | Detects “requires moderation” → logs and skips |
| **429 Rate Limit** | Retries with exponential backoff (up to 5 times) |
| **Invalid JSON** | Auto-repairs escape sequences & braces |
| **Truncated JSON** | Adds closing braces automatically |
| **Fallback** | Returns safe default dict without breaking pipeline |

All handled within `utils/api_utils.py → chat_json()`:

```python
if json_str.count("{") > json_str.count("}"):
    print("[INFO] Detected truncated JSON → auto-fixing with closing brace.")
    json_str += "}"
```

---

##  Prompt Design

**Level 1 (Zero-Shot)**  
> “You are a careful text analyst. Return 5 keywords, 1 sentence summary, and word count.”

**Level 2 (Persona)**  
> Adds genre-specific persona like researcher / blogger / journalist.

**Level 3 (Few-Shot)**  
> Adds prior examples (“Input → Output”) to simulate stylistic learning.

All prompt templates live in `utils/prompt_utils.py`.

---

## Output Naming

### Micro dataset

```
<Genre>_<SUBFIELD>_<AuthorID>_<Year>_<ItemIndex>_<ModelTag>_LV<Level>.txt
```

- `AuthorID` is the batch portion of the human filename.
- SUBFIELD casing: Blogs & Academic → uppercase, News → lowercase.
- Model tags: `DS`, `G4B`, `G12B`, `LMK`.

Examples:
```
Blogs_LIFESTYLE_03_2021_05_DS_LV2.txt
News_world_01_2014_09_LMK_LV1.txt
```

### Macro dataset

```
<Genre>_<Subfield>_<Year>_<Index>_<ModelTag>_LV<Level>.txt
```

Examples:
```
Academic_BIOLOGY_2023_068_G12B_LV1.txt
News_world_2024_200_DS_LV2.txt
```

---

##  Notes

- Models like `Llama-4 Maverick` may occasionally trigger Meta moderation → skipped gracefully.  
- OpenRouter credits are consumed per request; monitor your quota.
- DeepSeek and Gemma APIs are rate-limited (~60 req/min); the backoff handles automatic waiting.

---

##  Authors

**YeoJin Jenny Kim**  
Graduate Researcher, Northeastern University  
Seattle, WA  
yejinjenny717@gmail.com  

**Zhanwei Cao**  
Graduate Researcher, Northeastern University  
Seattle, WA  
chanweicao@gmail.com  
