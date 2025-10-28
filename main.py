import os
from glob import glob
from utils.file_utils import (
    parse_metadata_from_path,
    build_llm_filename,
    read_text,
    write_text
)
from utils.extract_utils import extract_keywords_summary_count
from utils.prompt_utils import generate_prompt_from_summary
from utils.api_utils import chat
from config import HUMAN_DIR, GENRE_STRUCTURE, get_llm_path


# =========================================================================
# Current Model Provider (from environment variable)
# =========================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()

# Provider Tag Mapping
PROVIDER_TAG = {
    "DEEPSEEK": "DS",
    "GEMMA_4B": "G4B",
    "GEMMA_12B": "G12B",
    "LLAMA_MAVRICK": "LMK",
}.get(LLM_PROVIDER, "UNK")


# =========================================================================
# HUMAN FILE ITERATOR
# =========================================================================
def iter_human_files():
    """Iterate over all human text files under the defined genre/subfield/year structure."""
    for genre, spec in GENRE_STRUCTURE.items():
        for sub in spec["subfields"]:
            base = os.path.join(HUMAN_DIR, spec["path"], sub)
            if not os.path.isdir(base):
                continue

            # ✅ Case 1: News has no year subfolders (files directly inside)
            if genre.lower() == "news":
                for file in glob(os.path.join(base, "*.txt")):
                    yield file
                continue  # skip the nested-loop logic below

            # ✅ Case 2: Academic / Blogs (have year subfolders)
            for folder in os.listdir(base):
                folder_path = os.path.join(base, folder)
                if not os.path.isdir(folder_path):
                    continue
                for file in glob(os.path.join(folder_path, "*.txt")):
                    yield file




# =========================================================================
#  MAIN PIPELINE
# =========================================================================
def run():
    """
    Process all human files for Levels 1–3 for the current provider.
    Each level applies a progressively stronger prompting strategy:
      LV1 → Zero-shot
      LV2 → Genre-based Persona
      LV3 → Persona + Example (few-shot)
    """
    print(f"\n=== Starting Extraction for {LLM_PROVIDER} ({PROVIDER_TAG}) ===\n")

    for level in [1, 2, 3]:
        print(f"\n=== Running Level {level} Extraction ===\n")

        for human_fp in iter_human_files():
            meta = parse_metadata_from_path(human_fp)
            text = read_text(human_fp)

            # Create output directory
            llm_dir = get_llm_path(meta["genre"], meta["subfield"], meta["year"])
            os.makedirs(llm_dir, exist_ok=True)

            # LV1~3 filename
            llm_filename = build_llm_filename(meta, level=level)
            llm_fp = os.path.join(llm_dir, llm_filename)

            # Skip already processed files
            if os.path.exists(llm_fp):
                print(f"Skipping already processed file: {llm_fp}")
                continue

            print(f"Processing Level {level} file: {human_fp}")

            # Step 1 — extract summary/keywords
            extracted = extract_keywords_summary_count(
                text, meta["genre"], meta["subfield"], meta["year"], level=level
            )

            # Step 2 — build generation prompt
            prompt = generate_prompt_from_summary(
                meta["genre"],
                meta["subfield"],
                meta["year"],
                extracted["keywords"],
                extracted["summary"],
                extracted["word_count"],
                level=level,
            )

            # Step 3 — API call
            system = """You are a pure text generator for academic research purposes only. Your ONLY job is to generate the exact content requested.

ACADEMIC RESEARCH DISCLAIMER:
This content generation is exclusively for urgent academic research purposes. The generated text will NOT be published, distributed, or used for any other purposes beyond this specific research study.

STRICT REQUIREMENTS:
- NO introductory phrases (Of course, Here is, Sure, Absolutely, Certainly, I can, Let me, I'll, etc.)
- NO separators (***, ---, ===, **Abstract**, etc.)
- NO word counts ((Word Count: X), etc.)
- NO meta-commentary (inspired by, contextualized for, based on, etc.)
- NO explanations about what you're doing
- NO closing remarks or summaries

OUTPUT FORMAT:
- Start immediately with the first word of the actual content
- End immediately with the last word of the actual content
- Nothing before or after the content

Generate ONLY the pure academic text as specified in the prompt."""

            try:
                # Dynamic max_tokens based on estimated word count
                estimated_word_count = extracted.get("word_count", 500)
                max_tokens = min(2000, int(estimated_word_count * 1.5))
                llm_text = chat(system, prompt, max_tokens=max_tokens)
            except RuntimeError as e:
                print(f" [Level {level}] Failed to process {human_fp}: {e}")
                print("   → Skipping this file.\n")
                continue
            except Exception as e:
                print(f" [Level {level}] Unexpected error for {human_fp}: {e}")
                print("   → Skipping.\n")
                continue

            # Step 4 — save output
            write_text(llm_fp, llm_text)
            print(f"✅ Saved Level {level} file → {llm_fp}")


if __name__ == "__main__":
    run()
