"""
Base generator class for LLM text generation.

This module provides a base class that handles common logic for generating
LLM texts from human-written texts. Subclasses can override specific methods
to customize behavior for different genres (Academic, News, Blogs).

This module automatically loads .env file to get API keys and configuration.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Iterator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

# Load .env file if available (before importing other modules that need API keys)
# This ensures API keys are loaded before api_utils is imported
_env_loaded = False

def _load_env_file():
    """Load .env file if available."""
    global _env_loaded
    if _env_loaded:
        return
    
    try:
        from dotenv import load_dotenv
        # Load .env with override=True to ensure API keys are loaded correctly
        load_dotenv(override=True)
        _env_loaded = True
    except ImportError:
        # python-dotenv not installed, try manual loading
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Set API keys and other config from .env
                        if key not in os.environ or os.environ[key] != value:
                            os.environ[key] = value
            _env_loaded = True

# Load .env file on module import
_load_env_file()

from utils.file_utils import (
    parse_metadata_from_path,
    build_llm_filename,
    read_text,
    write_text
)
from utils.extract_utils import extract_keywords_summary_count
from utils.prompt_utils import generate_prompt_from_summary
from utils.api_utils import chat

# Global locks for thread-safe operations
_stats_lock = Lock()
_cache_lock = Lock()
_processing_lock = Lock()


def _get_timestamp():
    """Get formatted timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds):
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def _format_progress_bar(current, total, width=30):
    """Create a progress bar string."""
    if total == 0:
        return "[" + "=" * width + "]"
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}]"


class BaseGenerator(ABC):
    """
    Base class for generating LLM texts from human-written texts.
    
    This class provides common functionality that all genre-specific generators
    can use. Subclasses should implement genre-specific logic if needed.
    """
    
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
    
    def __init__(self, human_dir: str, llm_dir: str, provider: str = "DEEPSEEK"):
        """
        Initialize the generator.
        
        Args:
            human_dir: Directory containing human-written texts
            llm_dir: Directory for output LLM-generated texts
            provider: LLM provider (DEEPSEEK, GEMMA_4B, GEMMA_12B, LLAMA_MAVRICK)
        """
        self.human_dir = human_dir
        self.llm_dir = llm_dir
        self.provider = provider.upper()
        
        # Set environment variable for provider
        os.environ["LLM_PROVIDER"] = self.provider
    
    @property
    @abstractmethod
    def genre(self) -> str:
        """
        Return the genre name this generator handles.
        
        Returns:
            Genre name (e.g., "Academic", "News", "Blogs")
        """
        pass
    
    def iter_human_files(self) -> Iterator[str]:
        """
        Iterate over all human text files for this genre.
        
        Returns:
            Iterator of file paths
        """
        genre_dir = os.path.join(self.human_dir, self.genre.lower())
        if not os.path.isdir(genre_dir):
            return
        
        # Recursively find all .txt files
        for root, dirs, files in os.walk(genre_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    yield file_path
    
    def get_output_dir(self, meta: Dict[str, str]) -> str:
        """
        Get the output directory for a given file.
        
        By default, all files go to genre root directory.
        Subclasses can override this to use subdirectory structures.
        
        Args:
            meta: Metadata dictionary from parse_metadata_from_path
            
        Returns:
            Output directory path
        """
        genre_dir_name = self.genre.lower()
        return os.path.join(self.llm_dir, genre_dir_name)
    
    def get_output_filename(self, meta: Dict[str, str], level: int) -> str:
        """
        Get the output filename for a given file.
        
        By default, uses build_llm_filename from file_utils with this generator's provider.
        This ensures the filename contains the correct model tag for this generator instance.
        Subclasses can override this to use custom filename formats.
        
        Args:
            meta: Metadata dictionary from parse_metadata_from_path
            level: Generation level (1, 2, or 3)
            
        Returns:
            Output filename (includes model tag and level)
        """
        # Pass self.provider to ensure filename uses the correct model tag
        # This is critical for thread safety: each generator instance has its own provider
        return build_llm_filename(meta, level=level, provider=self.provider)
    
    def extract_metadata(self, human_fp: str) -> Dict[str, str]:
        """
        Extract metadata from human file path.
        
        Args:
            human_fp: Path to human-written text file
            
        Returns:
            Metadata dictionary
        """
        return parse_metadata_from_path(human_fp)
    
    def extract_keywords_summary(self, text: str, meta: Dict[str, str], level: int) -> Dict:
        """
        Extract keywords, summary, and word count from text.
        
        Args:
            text: Human-written text content
            meta: Metadata dictionary
            level: Extraction level (1, 2, or 3)
            
        Returns:
            Dictionary with keywords, summary, and word_count
        """
        # Convert genre to capitalized form (e.g., "blogs" -> "Blogs")
        # prompt_utils expects capitalized genre names
        genre = meta["genre"].capitalize()
        return extract_keywords_summary_count(
            text,
            genre,
            meta["subfield"],
            int(meta["year"]),
            level=level
        )
    
    def generate_prompt(self, meta: Dict[str, str], extracted: Dict, level: int) -> str:
        """
        Generate prompt for LLM text generation.
        
        By default, uses generate_prompt_from_summary from prompt_utils.
        Subclasses can override this to use custom prompt generation.
        
        Args:
            meta: Metadata dictionary
            extracted: Extracted keywords, summary, and word_count
            level: Generation level (1, 2, or 3)
            
        Returns:
            Generation prompt string
        """
        # Convert genre to capitalized form (e.g., "blogs" -> "Blogs")
        # prompt_utils expects capitalized genre names
        genre = meta["genre"].capitalize()
        return generate_prompt_from_summary(
            genre,
            meta["subfield"],
            int(meta["year"]),
            extracted["keywords"],
            extracted["summary"],
            extracted["word_count"],
            level=level,
        )
    
    def calculate_max_tokens(self, word_count: int) -> int:
        """
        Calculate max_tokens based on word count.
        
        By default, uses: min(2000, int(word_count * 1.5))
        Subclasses can override this to use different calculations.
        
        Args:
            word_count: Target word count
            
        Returns:
            Maximum tokens for API call
        """
        return min(2000, int(word_count * 1.5))
    
    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Generate LLM text using API call.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            max_tokens: Maximum tokens for generation
            
        Returns:
            Generated text
        """
        return chat(system_prompt, user_prompt, max_tokens=max_tokens)
    
    def process_file(self, human_fp: str, level: int) -> bool:
        """
        Process a single human file and generate LLM text.
        
        Args:
            human_fp: Path to human-written text file
            level: Generation level (1, 2, or 3)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Extract metadata
            meta = self.extract_metadata(human_fp)
            
            # Step 2: Read text
            text = read_text(human_fp)
            
            # Step 3: Get output path
            output_dir = self.get_output_dir(meta)
            output_filename = self.get_output_filename(meta, level)
            output_path = os.path.join(output_dir, output_filename)
            
            # Step 4: Skip if already processed
            if os.path.exists(output_path):
                print(f"Skipping already processed file: {output_path}")
                return True
            
            print(f"Processing Level {level} file: {human_fp}")
            
            # Step 5: Extract keywords and summary
            extracted = self.extract_keywords_summary(text, meta, level)
            
            # Step 6: Generate prompt
            prompt = self.generate_prompt(meta, extracted, level)
            
            # Step 7: Generate text
            max_tokens = self.calculate_max_tokens(extracted.get("word_count", 500))
            llm_text = self.generate_text(self.SYSTEM_PROMPT, prompt, max_tokens)
            
            # Step 8: Save output
            os.makedirs(output_dir, exist_ok=True)
            write_text(output_path, llm_text)
            print(f"✅ Saved Level {level} file → {output_path}")
            
            return True
            
        except RuntimeError as e:
            print(f" [Level {level}] Failed to process {human_fp}: {e}")
            print("   → Skipping this file.\n")
            return False
        except Exception as e:
            print(f" [Level {level}] Unexpected error for {human_fp}: {e}")
            print("   → Skipping.\n")
            return False
    
    def _process_single_file_threadsafe(
        self,
        human_fp: str,
        level: int,
        extracted_cache: Dict,
        processing_files_set: set,
        verbose: bool = False,
    ):
        """
        Thread-safe version of process_file for concurrent processing.
        
        Args:
            human_fp: Path to human-written text file
            level: Generation level (1, 2, or 3)
            extracted_cache: Dictionary to cache extraction results (thread-safe)
            processing_files_set: Set to track files being processed (thread-safe)
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of (success: bool, stats: dict, error: str or None)
        """
        try:
            # Step 1: Extract metadata
            meta = self.extract_metadata(human_fp)
            human_filename = os.path.basename(human_fp)
            
            # Step 2: Get output path
            output_dir = self.get_output_dir(meta)
            output_filename = self.get_output_filename(meta, level)
            output_path = os.path.join(output_dir, output_filename)
            
            # Thread-safe check: skip if file already exists or is being processed
            with _processing_lock:
                # Check if file already exists (another thread might have just created it)
                if os.path.exists(output_path):
                    return (True, {"skipped": True}, None)
                
                # Check if another thread is already processing this file
                if output_path in processing_files_set:
                    return (True, {"skipped": True, "reason": "already_processing"}, None)
                
                # Mark this file as being processed
                processing_files_set.add(output_path)
            
            try:
                # Step 3: Read text
                text = read_text(human_fp)
                
                # Double-check after acquiring lock (file might have been created by another thread)
                if os.path.exists(output_path):
                    return (True, {"skipped": True, "reason": "created_by_another_thread"}, None)
                
                # Step 4: Extract keywords and summary (with caching)
                cache_key = f"{human_fp}_level{level}"
                
                # Thread-safe cache access (read is safe, write needs lock)
                if cache_key in extracted_cache:
                    extracted = extracted_cache[cache_key]
                else:
                    # Perform extraction (this is the expensive operation)
                    extracted = self.extract_keywords_summary(text, meta, level)
                    # Thread-safe cache write
                    with _cache_lock:
                        # Double-check pattern: another thread might have added it
                        if cache_key not in extracted_cache:
                            extracted_cache[cache_key] = extracted
                        else:
                            # Use the cached version if another thread added it
                            extracted = extracted_cache[cache_key]
                
                # Step 5: Generate prompt
                prompt = self.generate_prompt(meta, extracted, level)
                
                # Step 6: Generate text
                max_tokens = self.calculate_max_tokens(extracted.get("word_count", 500))
                llm_text = self.generate_text(self.SYSTEM_PROMPT, prompt, max_tokens)
                
                # Step 7: Save output (thread-safe)
                # Final check before writing (thread-safe atomic operation)
                with _processing_lock:
                    if os.path.exists(output_path):
                        # File was created by another thread, skip writing
                        return (True, {"skipped": True, "reason": "created_during_processing"}, None)
                    
                    # Create output directory and write file while holding the lock
                    os.makedirs(output_dir, exist_ok=True)
                    write_text(output_path, llm_text)
                
                if verbose:
                    print(f"  ✅ [{_get_timestamp()}] {human_filename} → {output_filename}")
                
                return (True, {"skipped": False}, None)
                
            finally:
                # Always remove from processing set, even if an error occurred
                with _processing_lock:
                    processing_files_set.discard(output_path)
                    
        except Exception as e:
            # Make sure to remove from processing set on error
            error_msg = str(e)
            try:
                with _processing_lock:
                    if 'output_path' in locals():
                        processing_files_set.discard(output_path)
            except:
                pass
            return (False, {}, error_msg)
    
    def run(self, levels: list = [1, 2, 3], max_workers: int = 5, verbose: bool = False):
        """
        Run the generation process for all files and levels with concurrent processing.
        
        Args:
            levels: List of levels to process (default: [1, 2, 3])
            max_workers: Maximum number of concurrent threads (default: 5)
            verbose: Whether to print detailed logs (default: False)
        """
        provider_tag = {
            "DEEPSEEK": "DS",
            "GEMMA_4B": "G4B",
            "GEMMA_12B": "G12B",
            "LLAMA_MAVRICK": "LMK",
        }.get(self.provider, "UNK")
        
        start_time_total = time.time()
        timestamp_start = _get_timestamp()
        
        print(f"\n{'='*80}")
        print(f"🚀 {self.genre} Generation Pipeline")
        print(f"{'='*80}")
        print(f"📅 Started: {timestamp_start}")
        print(f"🤖 Model: {self.provider} ({provider_tag})")
        print(f"📊 Levels: {levels}")
        print(f"⚙️  Workers: {max_workers}")
        print(f"{'='*80}\n")
        
        # Get all files
        human_files = list(self.iter_human_files())
        total_files = len(human_files)
        
        if total_files == 0:
            print(f"❌ No files found in {os.path.join(self.human_dir, self.genre.lower())}")
            return
        
        print(f"📂 Input Directory: {os.path.join(self.human_dir, self.genre.lower())}")
        print(f"📊 Total Files: {total_files}")
        print()
        
        # Cache for extraction results (shared across levels)
        extracted_cache = {}
        
        # Track overall progress across all levels
        level_index = 0
        total_levels = len(levels)
        processed = 0
        skipped = 0
        failed = 0
        
        for level in levels:
            level_index += 1
            level_start_time = time.time()
            level_timestamp = _get_timestamp()
            
            print(f"\n{'='*80}")
            print(f"📋 Level {level} Processing ({level_index}/{total_levels})")
            print(f"{'='*80}")
            print(f"⏰ Started: {level_timestamp}")
            print()
            
            level_processed = 0
            level_skipped = 0
            level_failed = 0
            
            # Filter out files that already exist
            files_to_process = []
            for human_fp in human_files:
                meta = self.extract_metadata(human_fp)
                output_dir = self.get_output_dir(meta)
                output_filename = self.get_output_filename(meta, level)
                output_path = os.path.join(output_dir, output_filename)
                if not os.path.exists(output_path):
                    files_to_process.append(human_fp)
                else:
                    level_skipped += 1
            
            remaining_files = len(files_to_process)
            print(f"📋 Files Status:")
            print(f"   • To Process: {remaining_files}")
            print(f"   • Already Exist: {level_skipped}")
            print(f"   • Total: {total_files}")
            print()
            
            if remaining_files == 0:
                print(f"⏭️  All files already processed for Level {level}. Skipping...\n")
                processed += level_processed
                skipped += level_skipped
                failed += level_failed
                continue
            
            # Process files concurrently
            start_time = time.time()
            completed = 0
            last_progress_time = start_time
            
            # Create a set to track files being processed (shared across threads)
            level_processing_files = set()
            
            print(f"🚀 Starting concurrent processing with {max_workers} workers...\n")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(
                        self._process_single_file_threadsafe,
                        human_fp,
                        level,
                        extracted_cache,
                        level_processing_files,
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
                        
                        with _stats_lock:
                            if success:
                                if stats.get("skipped"):
                                    level_skipped += 1
                                else:
                                    level_processed += 1
                            else:
                                level_failed += 1
                                if verbose:
                                    error_msg = error[:100] if error else "Unknown error"
                                    print(
                                        f"  [{_get_timestamp()}] ❌ ERROR [{completed}/{remaining_files}] "
                                        f"{os.path.basename(human_fp)}: {error_msg}"
                                    )
                        
                        # Progress update every 10 files or at the end, or every 30 seconds
                        current_time = time.time()
                        should_update = (
                            completed % 10 == 0 
                            or completed == remaining_files
                            or (current_time - last_progress_time) >= 30
                        )
                        
                        if should_update:
                            last_progress_time = current_time
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            remaining = remaining_files - completed
                            eta = remaining / rate if rate > 0 else 0
                            progress_pct = (completed / remaining_files) * 100
                            
                            # Calculate overall progress across all levels
                            overall_progress = (
                                ((level_index - 1) * total_files + completed) 
                                / (total_levels * total_files) * 100
                            )
                            
                            progress_bar = _format_progress_bar(completed, remaining_files, width=20)
                            
                            print(
                                f"\n  [{_get_timestamp()}] 📊 Level {level} Progress: "
                                f"{progress_bar} {completed}/{remaining_files} ({progress_pct:.1f}%)"
                            )
                            print(
                                f"      ⚡ Rate: {rate:.2f} files/s | "
                                f"⏱️  Elapsed: {_format_duration(elapsed)} | "
                                f"⏳ ETA: {_format_duration(eta)}"
                            )
                            print(
                                f"      📈 Overall Progress: {overall_progress:.1f}% | "
                                f"✅ Processed: {level_processed} | "
                                f"❌ Failed: {level_failed} | "
                                f"⏭️  Skipped: {level_skipped}"
                            )
                            print()
                    
                    except Exception as e:
                        with _stats_lock:
                            level_failed += 1
                        file_name = os.path.basename(human_fp) if human_fp else "unknown"
                        print(
                            f"  [{_get_timestamp()}] ❌ EXCEPTION [{completed}/{remaining_files}] "
                            f"{file_name}: {str(e)[:100]}"
                        )
            
            elapsed_time = time.time() - level_start_time
            level_end_timestamp = _get_timestamp()
            print(f"\n{'='*80}")
            print(f"✅ Level {level} Completed")
            print(f"{'='*80}")
            print(f"⏰ Started: {level_timestamp}")
            print(f"⏰ Ended: {level_end_timestamp}")
            print(f"⏱️  Duration: {_format_duration(elapsed_time)}")
            print()
            
            print(f"📊 Level {level} Statistics:")
            print(f"   ✅ Successfully Processed: {level_processed}")
            print(f"   ⏭️  Skipped (already exist): {level_skipped}")
            print(f"   ❌ Failed: {level_failed}")
            print(f"   📁 Total Files: {total_files}")
            print()
            
            processed += level_processed
            skipped += level_skipped
            failed += level_failed
        
        # Final summary
        total_elapsed_time = time.time() - start_time_total
        timestamp_end = _get_timestamp()
        
        print(f"\n{'='*80}")
        print(f"🎉 Pipeline Completed")
        print(f"{'='*80}")
        print(f"📅 Started: {timestamp_start}")
        print(f"📅 Ended: {timestamp_end}")
        print(f"⏱️  Total Duration: {_format_duration(total_elapsed_time)}")
        print(f"🤖 Model: {self.provider} ({provider_tag})")
        print(f"📊 Levels Processed: {levels}")
        print()
        print(f"📈 Final Statistics:")
        print(f"   ✅ Total Processed: {processed}")
        print(f"   ⏭️  Total Skipped: {skipped}")
        print(f"   ❌ Total Failed: {failed}")
        print(f"   📁 Output Directory: {os.path.join(self.llm_dir, self.genre.lower())}")
        print()
        
        # Calculate average processing rate
        if total_elapsed_time > 0:
            avg_rate = processed / total_elapsed_time
            print(f"⚡ Performance Metrics:")
            print(f"   • Average Rate: {avg_rate:.2f} files/second")
            print(f"   • Total Files: {total_files * len(levels)} (all levels)")
            print()
        
        print(f"{'='*80}\n")

