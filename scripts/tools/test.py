# test_pipeline.py
import os
from utils.file_utils import read_text, parse_metadata_from_path, build_llm_filename, write_text
from utils.extract_utils import extract_keywords_summary_count
from utils.prompt_utils import generate_prompt_from_summary
from utils.api_utils import chat
from config import get_llm_path

# ✅ 1. 테스트용 Human 파일 경로 지정
TEST_FILE = "Datasets/Human/Academic/Chem/2021/Academic_CHEMISTRY_2021_01.txt"

def run_test():
    if not os.path.exists(TEST_FILE):
        print(f"❌ 파일을 찾을 수 없습니다: {TEST_FILE}")
        return

    print(f"📄 Testing file: {TEST_FILE}")
    text = read_text(TEST_FILE)
    meta = parse_metadata_from_path(TEST_FILE)

    # 2️⃣ Keyword + Summary 추출
    print("\n🧠 Extracting keywords and summary...")
    extracted = extract_keywords_summary_count(text, meta["genre"], meta["subfield"], meta["year"])
    print(f"Keywords: {extracted['keywords']}")
    print(f"Summary: {extracted['summary']}")
    print(f"Word Count: {extracted['word_count']}")

    # 3️⃣ Prompt 생성
    prompt = generate_prompt_from_summary(
        meta["genre"], meta["subfield"], meta["year"],
        extracted["keywords"], extracted["summary"], extracted["word_count"]
    )

    # 4️⃣ DeepSeek으로 새 글 생성
    print("\n✍️ Generating new LLM text...")
    system_prompt = "You are a helpful assistant generating original academic text for research."
    llm_output = chat(system_prompt, prompt, max_tokens=1200)

    # 5️⃣ 결과 저장 (선택)
    out_dir = get_llm_path(meta["genre"], meta["subfield"], meta["year"])
    out_path = os.path.join(out_dir, build_llm_filename(meta))
    write_text(out_path, llm_output)
    print(f"\n✅ Generated file saved to: {out_path}")

    # 6️⃣ 콘솔 출력 (미리보기)
    print("\n--- 🔹 LLM Output (Preview) 🔹 ---")
    print(llm_output[:1000])  # 처음 1000자만 미리보기

if __name__ == "__main__":
    run_test()
