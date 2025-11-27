#!/usr/bin/env python3
"""
CLI script for generating News texts using NewsGenerator.

This is a standalone script that can be run directly from the project root.
Usage:
    python generate_news_cli.py --model GEMMA_12B --levels 1
    python generate_news_cli.py --model DEEPSEEK --levels 1 2 3
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure we're in the project root
_project_root = Path(__file__).parent
os.chdir(_project_root)
sys.path.insert(0, str(_project_root))

# Load .env file first
saved_llm_provider = os.getenv("LLM_PROVIDER")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✅ Loaded .env file")
except ImportError:
    # Manual loading
    env_file = _project_root / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("✅ Loaded .env file (manual)")

# Restore LLM_PROVIDER if it was set before
if saved_llm_provider:
    os.environ["LLM_PROVIDER"] = saved_llm_provider

# Now import the generator
from generators import NewsGenerator


def main():
    """Main entry point for News generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate LLM texts for News dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Level 1 with DEEPSEEK
  python generate_news_cli.py --model DEEPSEEK --levels 1
  
  # Generate all levels with GEMMA_12B
  python generate_news_cli.py --model GEMMA_12B --levels 1 2 3
  
  # Generate Level 2 with GEMMA_4B
  python generate_news_cli.py --model GEMMA_4B --levels 2
  
  # Generate with LLAMA_MAVRICK
  python generate_news_cli.py --model LLAMA_MAVRICK --levels 1
  
  # Generate with 10 workers for faster processing
  python generate_news_cli.py --model GEMMA_12B --levels 1 --workers 10
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["DEEPSEEK", "GEMMA_4B", "GEMMA_12B", "LLAMA_MAVRICK"],
        default=None,
        help="LLM model to use (default: from .env LLM_PROVIDER or DEEPSEEK)"
    )
    
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Generation levels to process (default: 1 2 3). Can specify multiple: --levels 1 2 3"
    )
    
    parser.add_argument(
        "--human-dir",
        type=str,
        default="dataset/human",
        help="Directory containing human-written texts (default: dataset/human)"
    )
    
    parser.add_argument(
        "--llm-dir",
        type=str,
        default="dataset/llm",
        help="Directory for output LLM-generated texts (default: dataset/llm)"
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
    
    # Determine provider
    if args.model:
        provider = args.model
        os.environ["LLM_PROVIDER"] = provider
        print(f"✅ Using model: {provider} (command line)")
    elif os.getenv("LLM_PROVIDER"):
        provider = os.getenv("LLM_PROVIDER")
        print(f"✅ Using model: {provider} (.env)")
    else:
        provider = "DEEPSEEK"
        os.environ["LLM_PROVIDER"] = provider
        print(f"✅ Using model: {provider} (default)")
    
    print(f"✅ Levels: {args.levels}")
    print(f"✅ Human dir: {args.human_dir}")
    print(f"✅ LLM dir: {args.llm_dir}")
    print(f"✅ Workers: {args.workers}")
    print(f"✅ Verbose: {args.verbose}")
    print()
    
    # Validate directories
    human_dir = Path(args.human_dir)
    if not human_dir.exists():
        print(f"❌ Error: Human directory not found: {human_dir}")
        sys.exit(1)
    
    news_input_dir = human_dir / "news"
    if not news_input_dir.exists():
        print(f"❌ Error: News input directory not found: {news_input_dir}")
        sys.exit(1)
    
    news_files = list(news_input_dir.glob("*.txt"))
    if not news_files:
        print(f"⚠️  Warning: No .txt files found in {news_input_dir}")
    else:
        print(f"✅ Found {len(news_files)} input files")
    
    print()
    print("=" * 70)
    print("Starting generation...")
    print("=" * 70)
    print()
    
    # Create generator
    generator = NewsGenerator(
        human_dir=str(human_dir),
        llm_dir=args.llm_dir,
        provider=provider
    )
    
    # Run generation
    try:
        generator.run(levels=args.levels, max_workers=args.workers, verbose=args.verbose)
    except KeyboardInterrupt:
        print()
        print("⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

