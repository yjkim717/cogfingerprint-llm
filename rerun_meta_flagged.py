import os
import re
from glob import glob
from utils.file_utils import read_text, write_text
from utils.extract_utils import extract_keywords_summary_count
from utils.prompt_utils import generate_prompt_from_summary
from utils.api_utils import chat

FLAG_PHRASE = "Llama 4 Maverick: Meta moderation flagged this input"
LLM_DIR = os.path.join("Datasets", "LLM")
HUMAN_DIR = os.path.join("Datasets", "Human")

def iter_flagged_files():
    flagged = []
    for root, _, files in os.walk(LLM_DIR):
        for f in files:
            if f.endswith(".txt"):
                fp = os.path.join(root, f)
                try:
                    if FLAG_PHRASE in read_text(fp):
                        flagged.append(fp)
                except Exception:
                    pass
    return flagged

def parse_llm_news_meta(llm_fp: str):
    # ex) Datasets/LLM/News/6_years/2013/News_6_years_LMK_LV1_2013_05.txt
    m_dir = re.search(r"/News/([^/]+)/(\d{4})/", llm_fp.replace("\\", "/"))
    period = m_dir.group(1) if m_dir else None
    year_from_dir = m_dir.group(2) if m_dir else None

    name = os.path.basename(llm_fp)
    m_name = re.match(r"^News_([\w_]+)_[A-Z0-9]+_LV(\d)_(\d{4})_(\d{2})\.txt$", name)
    if not m_name:
        return None

    period2 = m_name.group(1)
    level = int(m_name.group(2))
    year_from_name = m_name.group(3)
    idx = m_name.group(4)

    period = period or period2
    year = year_from_dir or year_from_name
    return {"period": period, "year": year, "idx": idx, "level": level}


def find_human_news_file(meta):
    # ex) Human/News/5_years/5_years_25_2019_05.txt
    base = os.path.join(HUMAN_DIR, "News", meta["period"])
    pat1 = os.path.join(base, f'{meta["period"]}_*_{meta["year"]}_{meta["idx"]}.txt')
    pat2 = os.path.join(base, meta["year"], f'{meta["period"]}_*_{meta["year"]}_{meta["idx"]}.txt')

    matches = glob(pat1)
    if not matches:
        matches = glob(pat2)
    return matches[0] if matches else None

def rerun_flagged_file(llm_fp: str):
    if "/News/" not in llm_fp.replace("\\", "/"):
        print(f"[INFO] Skipping non-News file → {llm_fp}")
        return

    meta = parse_llm_news_meta(llm_fp)
    if not meta:
        print(f"[WARN] Unrecognized LLM filename pattern → {llm_fp}")
        return

    human_fp = find_human_news_file(meta)
    if not human_fp or not os.path.exists(human_fp):
        print(f"[WARN] No matching human file found → {llm_fp}")
        return

    text = read_text(human_fp)
    level = meta["level"]

    print(f"\nRetrying {llm_fp} (Level {level}) using Human → {os.path.relpath(human_fp)}")

    extracted = extract_keywords_summary_count(
        text, "News", meta["period"], meta["year"], level=level
    )

    prompt = (
        "This content is for academic research and will not be published.\n\n" +
        generate_prompt_from_summary(
            "News",
            meta["period"],
            meta["year"],
            extracted["keywords"],
            extracted["summary"],
            extracted["word_count"],
            level=level,
        )
    )

    system = "You are a helpful assistant generating original text for research."

    try:
        llm_text = chat(system, prompt, max_tokens=700)
        write_text(llm_fp, llm_text)
        print(f"Successfully re-generated: {llm_fp}")

    except Exception as e:
        print(f"[ERROR] Failed to regenerate {llm_fp}: {e}")

def run():
    print(f"\nSearching flagged LLM files in: {os.path.abspath(LLM_DIR)}\n")
    flagged = iter_flagged_files()
    total_txt = sum(1 for _r, _d, files in os.walk(LLM_DIR) for f in files if f.endswith(".txt"))
    print(f"Found {len(flagged)} flagged files out of {total_txt} total LLM text files.\n")

    for fp in flagged:
        rerun_flagged_file(fp)

    print(f"\nRerun completed for {len(flagged)} flagged files.\n")

if __name__ == "__main__":
    run()
