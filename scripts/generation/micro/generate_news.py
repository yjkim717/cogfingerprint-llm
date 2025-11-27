#!/usr/bin/env python3
"""
Generate LLM texts for News dataset only.
Supports all providers: DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK
"""

import os
import sys
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

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


# Global locks for thread-safe operations
stats_lock = Lock()
cache_lock = Lock()
processing_lock = Lock()  # Lock for tracking files being processed


def process_single_file(
    human_fp,
    level,
    llm_dir,
    total_files,
    extracted_cache,
    processing_files_set,
    verbose=False,
):
    """
    Process a single news file with thread-safe duplicate prevention.

    Args:
        human_fp: Path to human news file
        level: Processing level (1, 2, or 3)
        llm_dir: Output directory for LLM files
        total_files: Total number of files (for progress tracking)
        extracted_cache: Dictionary to cache extraction results
        processing_files_set: Set to track files being processed (thread-safe)
        verbose: Whether to print detailed logs

    Returns:
        Tuple of (success: bool, stats: dict, error: str or None)
    """
    try:
        meta = parse_metadata_from_path(human_fp)

        # Build output filename
        llm_filename = build_llm_filename(meta, level=level)
        llm_fp = os.path.join(llm_dir, llm_filename)

        # Thread-safe check: skip if file already exists or is being processed
        with processing_lock:
            # Check if file already exists (another thread might have just created it)
            if os.path.exists(llm_fp):
                return (True, {"skipped": True}, None)

            # Check if another thread is already processing this file
            if llm_fp in processing_files_set:
                return (True, {"skipped": True, "reason": "already_processing"}, None)

            # Mark this file as being processed
            processing_files_set.add(llm_fp)

        try:
            # Read the text file
            text = read_text(human_fp)

            # Double-check after acquiring lock (file might have been created by another thread)
            if os.path.exists(llm_fp):
                return (
                    True,
                    {"skipped": True, "reason": "created_by_another_thread"},
                    None,
                )

            # Step 1: Extract (with caching)
            genre = meta["genre"].capitalize() if meta.get("genre") else "News"
            cache_key = f"{human_fp}_level{level}"

            # Thread-safe cache access (read is safe, write needs lock)
            if cache_key in extracted_cache:
                extracted = extracted_cache[cache_key]
            else:
                # Perform extraction (this is the expensive operation)
                extracted = extract_keywords_summary_count(
                    text, genre, meta["subfield"], meta["year"], level=level
                )
                # Thread-safe cache write
                with cache_lock:
                    # Double-check pattern: another thread might have added it
                    if cache_key not in extracted_cache:
                        extracted_cache[cache_key] = extracted
                    else:
                        # Use the cached version if another thread added it
                        extracted = extracted_cache[cache_key]

            # Step 2: Calculate actual human text word count
            actual_human_word_count = len(text.split())

            # Step 3: Build prompt
            prompt = generate_prompt_from_summary(
                genre,
                meta["subfield"],
                meta["year"],
                extracted["keywords"],
                extracted["summary"],
                actual_human_word_count,
                level=level,
            )

            # Step 4: Generate with max_tokens calculated for ±5% tolerance
            target_word_count = actual_human_word_count
            upper_bound_words = int(target_word_count * 1.05)
            max_tokens = int(upper_bound_words * 1.33 * 1.1)
            max_tokens = min(max_tokens, 15000)
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

            # Final check before writing (atomic operation)
            if not os.path.exists(llm_fp):
                write_text(llm_fp, llm_text)
            else:
                # File was created by another thread, skip writing
                return (
                    True,
                    {"skipped": True, "reason": "created_during_processing"},
                    None,
                )

            stats = {
                "skipped": False,
                "word_count": actual_human_word_count,
                "llm_word_count": llm_word_count,
                "abs_diff_pct": abs_diff_pct,
            }
            return (True, stats, None)

        finally:
            # Always remove from processing set, even if an error occurred
            with processing_lock:
                processing_files_set.discard(llm_fp)

    except Exception as e:
        # Make sure to remove from processing set on error
        with processing_lock:
            processing_files_set.discard(llm_fp)
        return (False, {}, str(e))


def process_news_files(levels=[1, 2, 3], max_workers=5, verbose=False):
    """
    Process all News files for specified levels with concurrent processing.

    Args:
        levels: List of levels to process (default: [1, 2, 3])
        max_workers: Maximum number of concurrent threads (default: 5)
        verbose: Whether to print detailed logs (default: False)
    """
    # Force set LLM_PROVIDER to DEEPSEEK if not explicitly set
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
    print(f"Max Workers (Concurrency): {max_workers}")
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

    # Cache for extraction results (shared across levels)
    extracted_cache = {}

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

        # Filter out files that already exist
        files_to_process = []
        for human_fp in news_files:
            meta = parse_metadata_from_path(human_fp)
            llm_filename = build_llm_filename(meta, level=level)
            llm_fp = os.path.join(llm_dir, llm_filename)
            if not os.path.exists(llm_fp):
                files_to_process.append(human_fp)
            else:
                level_skipped += 1

        remaining_files = len(files_to_process)
        print(f"📋 Files to process: {remaining_files} (skipped: {level_skipped})\n")

        if remaining_files == 0:
            print("⏭️  All files already processed for this level.\n")
            processed += level_processed
            skipped += level_skipped
            failed += level_failed
            continue

        # Process files concurrently
        start_time = time.time()
        completed = 0

        # Create a set to track files being processed (shared across threads)
        level_processing_files = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_single_file,
                    human_fp,
                    level,
                    llm_dir,
                    total_files,
                    extracted_cache,
                    level_processing_files,  # Pass the processing set
                    verbose,
                ): human_fp
                for human_fp in files_to_process
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                human_fp = future_to_file[future]
                completed += 1

                try:
                    success, stats, error = future.result()

                    with stats_lock:
                        if success:
                            if stats.get("skipped"):
                                level_skipped += 1
                            else:
                                level_processed += 1
                                abs_diff_pct = stats.get("abs_diff_pct", 0)

                                level_length_stats["total_with_length"] += 1
                                length_stats["total_with_length"] += 1
                                if abs_diff_pct <= 5:
                                    level_length_stats["within_5pct"] += 1
                                    length_stats["within_5pct"] += 1
                                if abs_diff_pct <= 10:
                                    level_length_stats["within_10pct"] += 1
                                    length_stats["within_10pct"] += 1
                                length_stats["length_diffs"].append(abs_diff_pct)

                                if verbose:
                                    status_icon = (
                                        "✅"
                                        if abs_diff_pct <= 5
                                        else "⚠️" if abs_diff_pct <= 10 else "❌"
                                    )
                                    print(
                                        f"  [{completed}/{remaining_files}] {status_icon} {os.path.basename(human_fp)}"
                                    )
                        else:
                            level_failed += 1
                            if verbose:
                                print(
                                    f"  ❌ Error processing {os.path.basename(human_fp)}: {error}"
                                )

                    # Progress update every 10 files or at the end
                    if completed % 10 == 0 or completed == remaining_files:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = remaining_files - completed
                        eta = remaining / rate if rate > 0 else 0
                        progress_pct = (completed / remaining_files) * 100
                        print(
                            f"  Progress: [{completed}/{remaining_files}] ({progress_pct:.1f}%) | "
                            f"Rate: {rate:.2f} files/s | ETA: {eta/60:.1f} min"
                        )

                except Exception as e:
                    with stats_lock:
                        level_failed += 1
                    if verbose:
                        print(f"  ❌ Unexpected error: {e}")

        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Level {level} completed in {elapsed_time/60:.1f} minutes")

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
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5). Increase for faster processing, but be aware of API rate limits.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs for each file",
    )

    args = parser.parse_args()

    process_news_files(
        levels=args.levels, max_workers=args.workers, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
