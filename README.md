# LLM Dataset Generation Pipeline — *cogfingerprint-llm*

## Overview
This repository provides a reproducible pipeline for constructing a **parallel Human–LLM dataset** for authorship analysis.  
The goal is to generate **LLM-written texts** that mirror the structure of **Human-written texts**, while avoiding direct content leakage.  
The pipeline ensures a **1:1 mapping** between Human and LLM files, enabling controlled experiments in authorship detection, cognitive fingerprinting, and trajectory analysis.

---

## Dataset Structure
The dataset is organized hierarchically by **Genre → Year → File Index**.  
Each Human file has a corresponding LLM-generated file:

```
cogfingerprint-llm/
 ├── Dataset/
 │   ├── Human/
 │   │   ├── News/
 │   │   │   └── 2021/
 │   │   │       └── News_2021_01.txt
 │   │   ├── Blogs/
 │   │   │   └── 2023/
 │   │   │       └── Blogs_2023_05.txt
 │   │   ├── Academic/
 │   │   │   └── 2024/
 │   │   │       └── Academic_2024_12.txt
 │   │   └── Literary_works/
 │   │       └── 2022/
 │   │           └── Literary_works_2022_07.txt
 │   └── LLM/
 │       ├── News/
 │       │   └── 2021/
 │       │       └── News_llm_2021_01.txt
 │       ├── Blogs/
 │       │   └── 2023/
 │       │       └── Blogs_llm_2023_05.txt
 │       ├── Academic/
 │       │   └── 2024/
 │       │       └── Academic_llm_2024_12.txt
 │       └── Literary_works/
 │           └── 2022/
 │               └── Literary_works_llm_2022_07c.txt
```

**Naming Convention:**  
- Human file → `<Genre>_<Year>_<Index>.txt`  
- LLM file → `<Genre>_llm_<Year>_<Index>.txt`

---

## Pipeline Design

### 1. Metadata Extraction
- Traverse `Dataset/Human/` recursively.  
- Extract **Genre, Year, Index, Topic** from filename.  
- Measure Human text **word count** (used as target length).  

### 2. Prompt Generation
- Use **all metadata** (genre, year, index, topic, length) to create prompts.  
- **Human text is never copied.** Only the topic label and word count are used.  

**Example:**  
- **Academic**  
  ```
  Write an academic-style essay on AI ethics (2024). 
  Use a formal tone and produce approximately 600 words.
  ```


### 3. LLM API Generation
- Prompts are sent to an LLM API (e.g., GPT-4).  
- The output is returned as text and stored in the parallel `Dataset/LLM/` folder.  

### 4. File Saving
- Generated files follow the strict naming convention.  
- Human ↔ LLM files maintain **1:1 mapping** for clean comparison.  

---

## Project Structure
```
cogfingerprint-llm/
├── Dataset/              # Human & LLM datasets
├── utils/                # Utility functions
│   ├── file_utils.py     # word count, filename parsing
│   └── prompt_utils.py   # prompt generation
├── config.py             # paths, styles, API keys
└── main.py               # entry point for the pipeline
```

---

## Key Features
- **Leakage-Free Generation**: Human text content is never exposed to the LLM.  
- **Dynamic Length Control**: Word count of Human text determines target length for LLM generation.  
- **Genre-Specific Styles**: Prompts adapt to academic, news, blogs, or literary works.  
- **Reproducible & Scalable**: Extendable to new genres, years, or LLMs.  

---

## Research Context
This pipeline is designed to support research on:  
- **Authorship Attribution**: Distinguishing Human vs. LLM writing.  
- **Cognitive Fingerprinting**: Capturing dynamic stylistic and emotional trajectories.  
- **Temporal Analysis**: Comparing how writing evolves across years in Human vs. LLM texts.  

---

## Next Steps
- Implement `parse_filename()` to extract metadata.  
- Integrate OpenAI API calls for text generation.  
- Conduct experiments on **variability, coherence, and separability** of trajectories.  
