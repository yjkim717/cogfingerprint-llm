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
 ├── Datasets/
 │   ├── Human/
 │   │   ├── Academic/
 │   │   │   ├── Chem/
 │   │   │   │   └── 2021/
 │   │   │   │       └── Academic_CHEMISTRY_2021_01.txt
 │   │   ├── Blogs/
 │   │   └── News/
 │   │
 │   └── LLM/
 │       ├── Academic/
 │       │   ├── Chem/
 │       │   │   └── 2021/
 │       │   │       └── Academic_Chem_DS_2021_01.txt
 │       ├── Blogs/
 │       └── News/
 │
 ├── utils/
 │   ├── api_utils.py
 │   ├── extract_utils.py
 │   ├── prompt_utils.py
 │   ├── file_utils.py
 │   └── __init__.py
 │
 ├── config.py
 ├── main.py
 └── test_pipeline.py
```

---

##  Architecture

```
cogfingerprint-llm/
│
├── main.py                     # Entry point for multi-level extraction & generation
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

```bash
python main.py
```

The script:
1. Iterates through all **Human** files  
2. Runs extraction (keywords / summary / count) via `chat_json`  
3. Saves JSON-parsed output  
4. Automatically builds corresponding LLM filenames  
5. Generates Level 1–3 texts under `/Datasets/LLM/...`

Example output:
```
✅ Saved Level 1 file → Academic_bio_LMK_LV1_2024_05.txt
✅ Saved Level 2 file → Academic_bio_LMK_LV2_2024_05.txt
✅ Saved Level 3 file → Academic_bio_LMK_LV3_2024_05.txt
```

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

Automatically generated filenames include:
```
<Genre>_<Subfield>_<ModelTag>_LV<Level>_<Year>_<Index>.txt
```

Example:
```
Academic_Bio_LMK_LV3_2021_07.txt
```


---

##  Notes

- Models like `Llama-4 Maverick` may occasionally trigger Meta moderation → skipped gracefully.  
- OpenRouter credits are consumed per request; monitor your quota.
- DeepSeek and Gemma APIs are rate-limited (~60 req/min); the backoff handles automatic waiting.

---

##  Author

**YeoJin Jenny Kim**  
Graduate Researcher, Northeastern University  
 Seattle, WA  
 yejinjenny717@gmail.com  
