import os
from config import HUMAN_DIR, LLM_DIR
from utils.file_utils import list_human_files, parse_filename, get_word_count, get_llm_save_path
from utils.topic_utils import extract_topic_with_llm
from utils.prompt_utils import generate_prompt
from utils.llm_utils import generate_text_from_prompt

def main():
    print("Processing Human dataset...\n")

    files = list_human_files(HUMAN_DIR)

    for file_path in files:
        filename = os.path.basename(file_path)

        try:
            # Extract metadata (Genre, Year, Index)
            metadata = parse_filename(filename)

            # Load full text
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Count words
            word_count = get_word_count(file_path)

            # Extract topic using LLM
            topic = extract_topic_with_llm(text)

            # Generate a prompt for LLM
            prompt = generate_prompt(metadata, topic, word_count)

            # Call LLM to generate text
            llm_output = generate_text_from_prompt(prompt)

            # Save output
            save_path = get_llm_save_path(file_path, HUMAN_DIR, LLM_DIR)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(llm_output)

            print(f"{filename} → Saved LLM output to {save_path}")

        except ValueError as e:
            print(f"Skipping file (bad format): {filename}")
            print(str(e))

if __name__ == "__main__":
    main()
