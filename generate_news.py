#!/usr/bin/env python3
"""
Generate LLM texts for News dataset only.
Supports all providers: DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK
"""

import os
import sys
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

# Paths
CLEANED_HUMAN_DIR = "cleaned_datasets/cleaned_human"
CLEANED_LLM_DIR = "cleaned_datasets/cleaned_llm"

# Get provider from environment
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()

PROVIDER_TAG = {
    "DEEPSEEK": "DS",
    "GEMMA_4B": "G4B",
    "GEMMA_12B": "G12B",
    "LLAMA_MAVRICK": "LMK",
}.get(LLM_PROVIDER, "UNK")

# System prompt for pure plain text generation
SYSTEM_PROMPT = """You are a PURE PLAIN TEXT generator for academic research. Your ONLY task is to output the raw text content.

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


def clean_text(text):
    """Remove leading '>' or other unwanted characters from generated text"""
    text = text.strip()
    # Remove leading '>'
    if text.startswith('>'):
        text = text[1:].strip()
    return text


def process_news_files(levels=[1, 2, 3]):
    """
    Process all News files for specified levels.
    
    Args:
        levels: List of levels to process (default: [1, 2, 3])
    """
    print(f"\n{'='*80}")
    print(f"News Generation with {LLM_PROVIDER} ({PROVIDER_TAG})")
    print(f"Levels: {levels}")
    print(f"{'='*80}\n")
    
    # Get all news files
    news_dir = os.path.join(CLEANED_HUMAN_DIR, "News", "filtered_years")
    if not os.path.isdir(news_dir):
        print(f"❌ News directory not found: {news_dir}")
        return
    
    news_files = sorted(glob(os.path.join(news_dir, "*.txt")))
    total_files = len(news_files)
    
    print(f"📊 Found {total_files} News files to process\n")
    
    # Create output directory
    llm_dir = os.path.join(CLEANED_LLM_DIR, "News")
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
    
    for level in levels:
        print(f"\n{'='*80}")
        print(f"Level {level} Processing")
        print(f"{'='*80}\n")
        
        level_processed = 0
        level_skipped = 0
        level_failed = 0
        
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
                        print(f"  [{idx}/{total_files}] ({progress:.1f}%) Skipped (already exists)")
                    continue
                
                progress = (idx / total_files) * 100
                print(f"  [{idx}/{total_files}] ({progress:.1f}%) Processing: {os.path.basename(human_fp)}")
                
                # Step 1: Extract
                extracted = extract_keywords_summary_count(
                    text, meta["genre"], meta["subfield"], meta["year"], level=level
                )
                
                # Step 2: Build prompt
                prompt = generate_prompt_from_summary(
                    meta["genre"],
                    meta["subfield"],
                    meta["year"],
                    extracted["keywords"],
                    extracted["summary"],
                    extracted["word_count"],
                    level=level,
                )
                
                # Step 3: Generate
                estimated_word_count = extracted.get("word_count", 500)
                max_tokens = min(2000, int(estimated_word_count * 1.5))
                llm_text = chat(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
                
                # Clean the generated text
                llm_text = clean_text(llm_text)
                
                # Step 4: Save
                write_text(llm_fp, llm_text)
                level_processed += 1
                print(f"     ✅ Generated ({len(llm_text)} chars)")
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user")
                print(f"   Processed: {level_processed}, Skipped: {level_skipped}, Failed: {level_failed}")
                sys.exit(0)
                
            except Exception as e:
                level_failed += 1
                print(f"     ❌ Error: {e}")
                continue
        
        print(f"\n📊 Level {level} Summary:")
        print(f"   ✅ Processed: {level_processed}")
        print(f"   ⏭️  Skipped: {level_skipped}")
        print(f"   ❌ Failed: {level_failed}")
        
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
        help="Levels to process (default: 1 2 3)"
    )
    
    args = parser.parse_args()
    
    process_news_files(levels=args.levels)


if __name__ == "__main__":
    main()

