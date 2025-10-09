# main.py

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


def iter_human_files():
    """
    Iterate over all human text files under the defined genre and subfield structure.
    """
    for genre, spec in GENRE_STRUCTURE.items():
        for sub in spec["subfields"]:
            base = os.path.join(HUMAN_DIR, spec["path"], sub)
            if not os.path.isdir(base):
                continue
            for year in os.listdir(base):
                folder = os.path.join(base, year)
                for file in glob(os.path.join(folder, "*.txt")):
                    yield file


def run():
    """
    Main pipeline to process new human files and generate corresponding LLM outputs.
    If the LLM file already exists, it is skipped automatically.
    """
    for human_fp in iter_human_files():
        meta = parse_metadata_from_path(human_fp)
        llm_dir = get_llm_path(meta["genre"], meta["subfield"], meta["year"])
        llm_filename = build_llm_filename(meta)
        llm_fp = os.path.join(llm_dir, llm_filename)

        # Skip files that have already been processed
        if os.path.exists(llm_fp):
            print(f"Skipping already processed file: {llm_fp}")
            continue

        text = read_text(human_fp)
        print(f"Processing new file: {human_fp}")

        # 1. Extract keywords and summary
        extracted = extract_keywords_summary_count(
            text, meta["genre"], meta["subfield"], meta["year"]
        )

        # 2. Build the generation prompt
        prompt = generate_prompt_from_summary(
            meta["genre"],
            meta["subfield"],
            meta["year"],
            extracted["keywords"],
            extracted["summary"],
            extracted["word_count"]
        )

        # 3. Generate new LLM text
        system = "You are a helpful assistant generating original text for research."
        llm_text = chat(system, prompt, max_tokens=1500)

        # 4. Save the generated result
        write_text(llm_fp, llm_text)
        print(f"✅Saved new LLM file → {llm_fp}")


if __name__ == "__main__":
    run()
