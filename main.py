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
from config import GENRE_STRUCTURE, get_llm_path

# Updated paths for dataset
CLEANED_HUMAN_DIR = "dataset/human"
CLEANED_LLM_DIR = "dataset/llm"


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
    """
    Iterate over all human text files from dataset/human.
    
    All genres now use unified format: {GENRE}_{SUBFIELD}_{AUTHOR_ID}_{YEAR}_{ITEM_INDEX}.txt
    Files can be in root directory or subdirectories - we recursively find all .txt files.
    """
    # Map genre names (used in code) to directory names (actual folder names)
    genre_map = {
        "Academic": "academic",
        "Blogs": "blogs",
        "News": "news"
    }
    
    for genre, genre_dir_name in genre_map.items():
        genre_dir = os.path.join(CLEANED_HUMAN_DIR, genre_dir_name)
        if not os.path.isdir(genre_dir):
            continue
        
        # Unified approach: recursively find all .txt files in the genre directory
        # Since all files now use unified format, we don't need to worry about directory structure
        # Files can be in root directory (Blogs, News) or subdirectories (Academic)
        for root, dirs, files in os.walk(genre_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    yield file_path




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

            # Build output directory structure in dataset/llm
            # Unified format: all files go to genre root directory
            # Since all files use unified format with all metadata in filename,
            # we don't need subdirectory structure for output
            genre = meta["genre"]
            genre_dir_name = genre.lower()  # All genres use lowercase directory names
            llm_dir = os.path.join(CLEANED_LLM_DIR, genre_dir_name)
            
            # Extract metadata for extraction and prompt generation
            subfield = meta["subfield"]
            year = meta["year"]
            
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
            system = """You are a PURE PLAIN TEXT generator for academic research. Your ONLY task is to output the raw text content.

CRITICAL: PURE PLAIN TEXT ONLY
This output will be directly analyzed as-is. Any formatting, metadata, or commentary will corrupt the research data.

ABSOLUTELY FORBIDDEN:
- NO "Here is", "Of course", "Sure", "Absolutely", "Certainly", "I can", "Let me", "I'll"
- NO separators: ***, ---, ===, **Abstract**, #, ===
- NO word counts: (Word Count: X)
- NO meta-commentary: "inspired by", "contextualized for", "based on"
- NO explanations about what you're doing
- NO "The following text...", "Let me provide...", "I will now..."
- NO closing remarks: "I hope this helps", "In summary", "Best regards"
- NO markdown formatting: **bold**, *italic*, # headings
- NO bullet points or lists with symbols
- NO quotation marks around the text
- NO box drawing characters: ┌─┐│└┘
- NO emojis or special characters

START DIRECTLY WITH THE FIRST WORD OF YOUR RESPONSE
END DIRECTLY WITH THE LAST WORD OF YOUR RESPONSE
NOTHING BEFORE, NOTHING AFTER

Generate ONLY the raw plain text content as specified."""

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
