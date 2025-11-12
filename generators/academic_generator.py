"""
Academic text generator.

This module provides a generator specifically for Academic texts.
It inherits from BaseGenerator and can override methods for Academic-specific logic.

This module can also be run as a standalone script to generate Academic texts.
"""

import os
import sys
import argparse
from typing import Dict
from pathlib import Path

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    _parent_dir = Path(__file__).parent.parent
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))

# Handle both module import and direct script execution
if __name__ == "__main__":
    # When running as script, import after path setup
    from generators.base_generator import BaseGenerator
else:
    # When importing as module, use relative import
    from .base_generator import BaseGenerator


class AcademicGenerator(BaseGenerator):
    """
    Generator for Academic texts.
    
    Currently uses all default behavior from BaseGenerator.
    Can be extended with Academic-specific logic if needed.
    """
    
    @property
    def genre(self) -> str:
        """Return the genre name."""
        return "Academic"
    
    # Example: If Academic needs special prompt generation, override here:
    # def generate_prompt(self, meta: Dict[str, str], extracted: Dict, level: int) -> str:
    #     # Custom prompt generation for Academic
    #     pass


def main():
    """Main entry point for Academic generator script."""
    # Load .env file first
    saved_llm_provider = os.getenv("LLM_PROVIDER")
    
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        print("✅ Loaded .env file")
    except ImportError:
        # Manual loading
        env_file = Path(__file__).parent.parent / ".env"
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
    
    parser = argparse.ArgumentParser(
        description="Generate LLM texts for Academic dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Level 1 with DEEPSEEK
  python generators/academic_generator.py --model DEEPSEEK --levels 1
  
  # Generate all levels with GEMMA_12B
  python generators/academic_generator.py --model GEMMA_12B --levels 1 2 3
  
  # Generate Level 2 with GEMMA_4B
  python generators/academic_generator.py --model GEMMA_4B --levels 2
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
        help="Generation levels to process (default: 1 2 3)"
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
    
    # Create generator
    generator = AcademicGenerator(
        human_dir=args.human_dir,
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

