"""
Refactored main script using generator classes.

This script demonstrates how to use the generator classes to process
different genres. It provides a more scalable and maintainable approach
compared to the original main.py.
"""

import os
import argparse

# CRITICAL: Load .env file BEFORE importing generators
# This ensures API keys and other configuration are loaded from .env
saved_llm_provider = os.getenv("LLM_PROVIDER")

# Load .env file if available
# Use override=True to ensure .env values are loaded (especially for API keys)
# But we'll restore LLM_PROVIDER afterwards if it was set before
try:
    from dotenv import load_dotenv
    # Load .env with override=True to ensure API keys are loaded correctly
    load_dotenv(override=True)
    print("✅ Loaded .env file")
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
                    value = value.strip().strip('"').strip("'")
                    # Always set API keys and other config from .env
                    # (We'll restore LLM_PROVIDER afterwards if needed)
                    os.environ[key] = value
        print("✅ Loaded .env file (manual)")
    else:
        print("⚠️  .env file not found, using environment variables only")

# ALWAYS restore LLM_PROVIDER if it was set before loading .env
# This ensures command line/env vars take precedence over .env file
if saved_llm_provider:
    os.environ["LLM_PROVIDER"] = saved_llm_provider
    print(f"✅ Using LLM_PROVIDER: {saved_llm_provider} (command line/environment)")

from generators import AcademicGenerator, NewsGenerator, BlogsGenerator

# Default paths
DEFAULT_HUMAN_DIR = "dataset/human"
DEFAULT_LLM_DIR = "dataset/llm"
DEFAULT_PROVIDER = "DEEPSEEK"


def main():
    """Main entry point for the refactored generator."""
    parser = argparse.ArgumentParser(
        description="Generate LLM texts from human-written texts using generator classes"
    )
    parser.add_argument(
        "--genre",
        type=str,
        choices=["academic", "news", "blogs", "all"],
        default="all",
        help="Genre to process (default: all)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["DEEPSEEK", "GEMMA_4B", "GEMMA_12B", "LLAMA_MAVRICK"],
        default=DEFAULT_PROVIDER,
        help="LLM provider to use (default: DEEPSEEK)"
    )
    parser.add_argument(
        "--human-dir",
        type=str,
        default=DEFAULT_HUMAN_DIR,
        help=f"Directory containing human-written texts (default: {DEFAULT_HUMAN_DIR})"
    )
    parser.add_argument(
        "--llm-dir",
        type=str,
        default=DEFAULT_LLM_DIR,
        help=f"Directory for output LLM-generated texts (default: {DEFAULT_LLM_DIR})"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Generation levels to process (default: 1 2 3)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5). Increase for faster processing, but be aware of API rate limits."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs for each file"
    )
    
    args = parser.parse_args()
    
    # Set LLM_PROVIDER if provided via command line (takes precedence over .env)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        print(f"✅ Using LLM_PROVIDER: {args.provider} (command line)")
    elif not os.getenv("LLM_PROVIDER"):
        # Use provider from .env or default
        provider_from_env = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        os.environ["LLM_PROVIDER"] = provider_from_env
        print(f"✅ Using LLM_PROVIDER: {provider_from_env} (.env or default)")
    
    # Create generators based on genre
    generators = []
    
    if args.genre in ["academic", "all"]:
        generators.append(AcademicGenerator(args.human_dir, args.llm_dir, args.provider))
    
    if args.genre in ["news", "all"]:
        generators.append(NewsGenerator(args.human_dir, args.llm_dir, args.provider))
    
    if args.genre in ["blogs", "all"]:
        generators.append(BlogsGenerator(args.human_dir, args.llm_dir, args.provider))
    
    # Run each generator
    for generator in generators:
        generator.run(levels=args.levels, max_workers=args.workers, verbose=args.verbose)


if __name__ == "__main__":
    main()

