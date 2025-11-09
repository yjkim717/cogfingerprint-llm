#!/usr/bin/env python3
"""
Generate LLM texts for News dataset only.
Supports all providers: DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK
"""

import os
import sys
from glob import glob

# CRITICAL: Save LLM_PROVIDER BEFORE loading .env file
# Command line arguments should take precedence over .env file
saved_llm_provider = os.getenv("LLM_PROVIDER")

# Load .env file if available
# Use override=True to ensure .env values are loaded (especially for API keys)
# But we'll restore LLM_PROVIDER afterwards if it was set before
try:
    from dotenv import load_dotenv

    # Load .env with override=True to ensure API keys are loaded correctly
    load_dotenv(override=True)
except ImportError:
    # python-dotenv not installed, try manual loading
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Always set API keys and other config from .env
                    # (We'll restore LLM_PROVIDER afterwards if needed)
                    os.environ[key] = value

# ALWAYS restore LLM_PROVIDER if it was set before loading .env
# This ensures command line/env vars take precedence over .env file
if saved_llm_provider:
    os.environ["LLM_PROVIDER"] = saved_llm_provider
    print(f"✅ Using LLM_PROVIDER: {saved_llm_provider} (command line/environment)")
elif not os.getenv("LLM_PROVIDER"):
    # Default to DEEPSEEK if nothing is set
    os.environ["LLM_PROVIDER"] = "DEEPSEEK"
    print(f"✅ Using default LLM_PROVIDER: DEEPSEEK")
from utils.file_utils import (
    parse_metadata_from_path,
    build_llm_filename,
    read_text,
    write_text,
)
from utils.extract_utils import extract_keywords_summary_count
from utils.prompt_utils import generate_prompt_from_summary
from utils.api_utils import chat

# Paths
CLEANED_HUMAN_DIR = "cleaned_datasets/cleaned_human"
CLEANED_LLM_DIR = "cleaned_datasets/cleaned_llm"
# Use dataset/human/news as input source
HUMAN_NEWS_DIR = "dataset/human/news"
# Output directory for LLM generated news files
LLM_NEWS_DIR = "dataset/llm/news"
# Test output directory for new generation
TEST_LLM_DIR = "dataset/llm/news_test"

# Provider tag mapping
PROVIDER_TAG_MAP = {
    "DEEPSEEK": "DS",
    "GEMMA_4B": "G4B",
    "GEMMA_12B": "G12B",
    "LLAMA_MAVRICK": "LMK",
}


def get_provider_info():
    """Get provider and tag dynamically from environment at runtime."""
    provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    tag = PROVIDER_TAG_MAP.get(provider, "UNK")
    return provider, tag


# System prompt for pure plain text generation
SYSTEM_PROMPT = """You are a PURE PLAIN TEXT generator for academic research. Your ONLY task is to output the raw text content.

CRITICAL: PURE PLAIN TEXT ONLY
This output will be directly analyzed as-is. Any formatting, metadata, or commentary will corrupt the research data.

WORD COUNT REQUIREMENT:
- You MUST strictly adhere to the word count range specified in the user prompt
- The word count is CRITICAL and non-negotiable
- Count words carefully before finishing your response
- If you are approaching the upper limit, conclude your article appropriately
- Do NOT exceed the specified word count range

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

Generate ONLY the raw plain text content as specified, within the required word count range."""


def clean_text(text):
    """Remove leading '>' or other unwanted characters from generated text"""
    text = text.strip()
    # Remove leading '>'
    if text.startswith(">"):
        text = text[1:].strip()
    return text


def process_news_files(levels=[1, 2, 3]):
    """
    Process all News files for specified levels.

    Args:
        levels: List of levels to process (default: [1, 2, 3])
    """
    # Force set LLM_PROVIDER to DEEPSEEK if not explicitly set
    # This ensures we use DEEPSEEK by default
    if not os.getenv("LLM_PROVIDER"):
        os.environ["LLM_PROVIDER"] = "DEEPSEEK"

    # Get provider info dynamically at runtime
    llm_provider, provider_tag = get_provider_info()

    # Double-check: verify the provider is correct
    actual_provider = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    if actual_provider != llm_provider:
        print(
            f"⚠️  Warning: Provider mismatch! Environment: {actual_provider}, Function returned: {llm_provider}"
        )
        llm_provider = actual_provider
        provider_tag = PROVIDER_TAG_MAP.get(llm_provider, "UNK")

    print(f"\n{'='*80}")
    print(f"News Generation with {llm_provider} ({provider_tag})")
    print(f"Levels: {levels}")
    print(f"Environment LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'NOT SET')}")
    print(f"{'='*80}\n")

    # Get all news files from dataset/human/news
    news_dir = HUMAN_NEWS_DIR
    if not os.path.isdir(news_dir):
        print(f"❌ News directory not found: {news_dir}")
        return

    news_files = sorted(glob(os.path.join(news_dir, "*.txt")))
    total_files = len(news_files)

    print(f"📊 Found {total_files} News files to process\n")

    # Create output directory
    # Set TEST_OUTPUT=True in environment to use test directory
    use_test_dir = os.getenv("TEST_OUTPUT", "false").lower() == "true"
    if use_test_dir:
        llm_dir = TEST_LLM_DIR
    else:
        llm_dir = LLM_NEWS_DIR
    os.makedirs(llm_dir, exist_ok=True)

    # Check existing files for progress tracking
    existing_files = set()
    if os.path.isdir(llm_dir):
        for f in os.listdir(llm_dir):
            existing_files.add(f)

    print(f"📂 Output directory: {llm_dir}")
    if existing_files:
        print(f"✅ Found {len(existing_files)} existing files (will be skipped)")
    print()

    processed = 0
    skipped = 0
    failed = 0

    # Statistics for length accuracy
    length_stats = {
        "within_5pct": 0,
        "within_10pct": 0,
        "total_with_length": 0,
        "length_diffs": [],
    }

    for level in levels:
        print(f"\n{'='*80}")
        print(f"Level {level} Processing")
        print(f"{'='*80}\n")

        level_processed = 0
        level_skipped = 0
        level_failed = 0
        level_length_stats = {
            "within_5pct": 0,
            "within_10pct": 0,
            "total_with_length": 0,
        }

        for idx, human_fp in enumerate(news_files, 1):
            try:
                meta = parse_metadata_from_path(human_fp)
                text = read_text(human_fp)

                # Build output filename
                llm_filename = build_llm_filename(meta, level=level)
                llm_fp = os.path.join(llm_dir, llm_filename)

                # Skip if already exists
                if os.path.exists(llm_fp):
                    level_skipped += 1
                    if idx % 100 == 0:  # Show progress every 100 files
                        progress = (idx / total_files) * 100
                        print(
                            f"  [{idx}/{total_files}] ({progress:.1f}%) Skipped (already exists)"
                        )
                    continue

                progress = (idx / total_files) * 100
                print(
                    f"  [{idx}/{total_files}] ({progress:.1f}%) Processing: {os.path.basename(human_fp)}"
                )

                # Step 1: Extract
                # Normalize genre to ensure compatibility (parse returns lowercase, but prompts expect capitalized)
                genre = meta["genre"].capitalize() if meta.get("genre") else "News"
                extracted = extract_keywords_summary_count(
                    text, genre, meta["subfield"], meta["year"], level=level
                )

                # Step 2: Calculate actual human text word count
                actual_human_word_count = len(text.split())

                # Step 3: Build prompt with target word count (use actual human length)
                # Prompt will guide LLM to generate text of similar length
                prompt = generate_prompt_from_summary(
                    genre,
                    meta["subfield"],
                    meta["year"],
                    extracted["keywords"],
                    extracted["summary"],
                    actual_human_word_count,  # Use actual human text length
                    level=level,
                )

                # Step 4: Generate with max_tokens calculated for ±5% tolerance
                # Target: actual_human_word_count ± 5%
                # Convert words to tokens: approximately 1 word = 1.33 tokens for English
                # Set max_tokens to accommodate upper bound (105% of target) with minimal buffer
                target_word_count = actual_human_word_count
                upper_bound_words = int(target_word_count * 1.05)  # +5% tolerance
                # Convert to tokens: words * 1.33
                # Use smaller buffer (10% instead of 20%) to enforce stricter length control
                max_tokens = int(upper_bound_words * 1.33 * 1.1)
                # Remove hard cap of 2000, but set reasonable maximum (e.g., 15000 tokens)
                max_tokens = min(max_tokens, 15000)
                # Ensure minimum tokens for very short texts
                max_tokens = max(max_tokens, 500)

                llm_text = chat(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)

                # Clean the generated text
                llm_text = clean_text(llm_text)

                # Step 5: Verify length and save
                llm_word_count = len(llm_text.split())
                length_diff_pct = (
                    (
                        (llm_word_count - actual_human_word_count)
                        / actual_human_word_count
                        * 100
                    )
                    if actual_human_word_count > 0
                    else 0
                )
                abs_diff_pct = abs(length_diff_pct)

                write_text(llm_fp, llm_text)
                level_processed += 1

                # Print detailed length information
                status_icon = (
                    "✅" if abs_diff_pct <= 5 else "⚠️" if abs_diff_pct <= 10 else "❌"
                )
                print(f"     {status_icon} Generated:")
                print(f"        Input sample: {actual_human_word_count} words")
                print(f"        Generated sample: {llm_word_count} words")
                print(
                    f"        Difference: {length_diff_pct:+.2f}% ({abs_diff_pct:.2f}% absolute)"
                )
                if abs_diff_pct > 5:
                    print(f"        ⚠️  Warning: Difference exceeds ±5% tolerance")

                # Update statistics
                level_length_stats["total_with_length"] += 1
                length_stats["total_with_length"] += 1
                if abs_diff_pct <= 5:
                    level_length_stats["within_5pct"] += 1
                    length_stats["within_5pct"] += 1
                if abs_diff_pct <= 10:
                    level_length_stats["within_10pct"] += 1
                    length_stats["within_10pct"] += 1
                length_stats["length_diffs"].append(abs_diff_pct)

            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user")
                print(
                    f"   Processed: {level_processed}, Skipped: {level_skipped}, Failed: {level_failed}"
                )
                sys.exit(0)

            except Exception as e:
                level_failed += 1
                print(f"     ❌ Error: {e}")
                continue

        print(f"\n📊 Level {level} Summary:")
        print(f"   ✅ Processed: {level_processed}")
        print(f"   ⏭️  Skipped: {level_skipped}")
        print(f"   ❌ Failed: {level_failed}")
        if level_length_stats["total_with_length"] > 0:
            pct_5 = (
                level_length_stats["within_5pct"]
                / level_length_stats["total_with_length"]
            ) * 100
            pct_10 = (
                level_length_stats["within_10pct"]
                / level_length_stats["total_with_length"]
            ) * 100
            print(f"   📏 Length Accuracy:")
            print(
                f"      Within ±5%: {level_length_stats['within_5pct']}/{level_length_stats['total_with_length']} ({pct_5:.2f}%)"
            )
            print(
                f"      Within ±10%: {level_length_stats['within_10pct']}/{level_length_stats['total_with_length']} ({pct_10:.2f}%)"
            )

        processed += level_processed
        skipped += level_skipped
        failed += level_failed

    print(f"\n{'='*80}")
    print(f"Final Summary")
    print(f"{'='*80}")
    print(f"✅ Total Processed: {processed}")
    print(f"⏭️  Total Skipped: {skipped}")
    print(f"❌ Total Failed: {failed}")
    print(f"📁 Output directory: {llm_dir}")
    print()
    if length_stats["total_with_length"] > 0:
        pct_5 = (length_stats["within_5pct"] / length_stats["total_with_length"]) * 100
        pct_10 = (
            length_stats["within_10pct"] / length_stats["total_with_length"]
        ) * 100
        avg_diff = sum(length_stats["length_diffs"]) / len(length_stats["length_diffs"])
        median_diff = sorted(length_stats["length_diffs"])[
            len(length_stats["length_diffs"]) // 2
        ]
        print(f"📏 Length Accuracy Summary:")
        print(f"   Total files with length data: {length_stats['total_with_length']}")
        print(f"   Within ±5%: {length_stats['within_5pct']} ({pct_5:.2f}%)")
        print(f"   Within ±10%: {length_stats['within_10pct']} ({pct_10:.2f}%)")
        print(f"   Average absolute difference: {avg_diff:.2f}%")
        print(f"   Median absolute difference: {median_diff:.2f}%")
    print(f"{'='*80}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate LLM texts for News dataset")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Levels to process (default: 1 2 3)",
    )

    args = parser.parse_args()

    process_news_files(levels=args.levels)


if __name__ == "__main__":
    main()
