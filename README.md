# LLM Dataset Generation Pipeline — cogfingerprint-llm (DeepSeek Edition)

## Overview
This repository provides a reproducible pipeline for constructing a parallel Human–LLM dataset for authorship and cognitive-fingerprinting analysis.  
Each human-authored text is automatically paired with a DeepSeek-generated counterpart, enabling 1:1 controlled comparisons in stylistic, cognitive, and temporal studies.

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

**Naming Convention**
- Human file → `<Genre>_<Subfield>_<Year>_<Index>.txt`
- LLM file → `<Genre>_<Subfield>_DS_<Year>_<Index>.txt`  
  (e.g., `Academic_Chem_DS_2021_01.txt`)

---

## Pipeline Flow

### 1. Metadata and Text Extraction
- Traverse all `.txt` files under `Datasets/Human/`
- Parse metadata: `genre`, `subfield`, `year`, `index`
- Load text content for processing

### 2. Keyword and Summary Extraction
Each Human text is processed via DeepSeek-Chat using a minimal JSON-only instruction.  
Returned data example:

```json
{
  "keywords": ["asymmetric catalysis", "chiral metal complexes", "organocatalysis", "enantioselectivity", "hybrid catalysts"],
  "summary": "Hybrid chiral catalysts combine metal and organic functions for enantioselective transformations.",
  "word_count": 320
}
```

### 3. Prompt Generation
Keywords and summaries are used to form leakage-free prompts guiding DeepSeek to produce new, conceptually aligned texts.

Example (Academic style):

```
Here is a summary of a human-written academic text:

Keywords: asymmetric catalysis, chiral metal complexes, organocatalysis, enantioselectivity, hybrid catalysts
Summary: Hybrid chiral catalysts combine metal and organic functions for enantioselective transformations.

Now, write a new academic-style abstract inspired by these ideas.
Do not copy or paraphrase the original text — write an original, conceptually similar discussion in Chem.
Use a formal academic tone (~320 words). Year context: 2021.
```

### 4. LLM Generation and File Saving
- The DeepSeek-Chat model generates a new text.
- The result is saved under the parallel `Datasets/LLM/` directory.
- File naming preserves 1:1 mapping between Human and LLM versions.

---

## Key Features

| Feature | Description |
|----------|-------------|
| Leakage-Free Generation | Human content is never passed to the LLM |
| Keyword-Based Prompting | Domain-specific keywords guide contextual alignment |
| Dynamic Length Control | Output length matches the original human text |
| Genre-Aware Prompts | Distinct templates for Academic, Blogs, and News |
| Reproducible Mapping | 1:1 file alignment between Human and LLM |
| Scalable Design | Easily extendable to new genres or subfields |

---

## Configuration (config.py)

```python
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"   # cheapest and fastest
LLM_TEMPERATURE = 0.7
MAX_TOKENS = 1500

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

GENRE_STRUCTURE = {
    "Academic": {
        "path": "Academic",
        "subfields": ["bio", "Chem", "CS", "med", "phy"]
    },
    "Blogs": {
        "path": "Blogs",
        "subfields": ["Twitter", "BlueSkys", "TruthSocialMedia"]
    },
    "News": {
        "path": "News",
        "subfields": ["CNN", "FOX"]
    }
}
```

---

## Usage

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install requests python-dotenv
```

Create a `.env` file in the root directory:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Single-File Test
```bash
python test_pipeline.py
```
- Outputs preview in terminal  
- Saves generated file in `Datasets/LLM/...`

### 3. Full Pipeline
```bash
python main.py
```
Processes all Human files and generates DeepSeek outputs.

---

## Research Context
This dataset supports work in:
- Authorship Attribution — identifying stylistic distinctions between Human and LLM texts  
- Cognitive Fingerprinting — mapping cognitive-emotional trajectories in text evolution  
- Temporal Linguistic Analysis — comparing multi-year stylistic trends  

---

## Example Result

| Type | File | Excerpt |
|------|------|----------|
| Human | `Academic_CHEMISTRY_2021_01.txt` | “Asymmetric catalysis holds a prominent position among the important developments in chemistry…” |
| LLM (DeepSeek) | `Academic_Chem_DS_2021_01.txt` | “The paradigm of asymmetric catalysis has been historically delineated along the lines of metal-centered and organocatalytic activation modes…” |

---

## Future Work
- Add Llama-based LM support (in addition to DeepSeek) for cross-model authorship comparison.  
- Implement auto balance check before DeepSeek API calls.  
- Extend prompt templates for additional genres (e.g., Technical Reports, Essays).  
- Enable configurable multi-LLM benchmarking.

---

## Citation

```
Kim, Y. J. (2025). cogfingerprint-llm: Constructing Parallel Human–LLM Datasets
for Authorship and Cognitive Trajectory Analysis. Northeastern University.
```

---

## Current Status
- DeepSeek API integrated  
- Academic folder renamed  
- Output suffix changed to `_DS_`  
- Tested successfully on Academic-Chem 2021 sample  
- Llama LM integration planned
