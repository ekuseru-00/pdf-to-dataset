#!/usr/bin/env python3
"""
Example usage script demonstrating the enhanced PDF processing pipeline.
This shows how to use the script with various options.
"""

import subprocess
import sys

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║  PDF to Dataset - Enhanced Pipeline Usage Examples         ║
╔════════════════════════════════════════════════════════════╝
""")

    examples = [
        {
            "title": "1. Basic Text-Only Processing",
            "description": "Process PDF extracting only text content with default settings",
            "command": [
                "python3", "history_to_dataset.py",
                "input.pdf",
                "output.jsonl",
                "--model-name", "llama3.1"
            ]
        },
        {
            "title": "2. Enhanced with Vision (Recommended)",
            "description": "Process both text and images using vision-capable model",
            "command": [
                "python3", "history_to_dataset.py",
                "input.pdf",
                "output.jsonl",
                "--model-name", "llama3.1",
                "--enable-vision",
                "--vision-model", "qwen2.5:14b"
            ]
        },
        {
            "title": "3. Fast Testing (Limited Pages)",
            "description": "Quick test with limited pages and workers",
            "command": [
                "python3", "history_to_dataset.py",
                "input.pdf",
                "output.jsonl",
                "--model-name", "llama3.1",
                "--enable-vision",
                "--vision-model", "qwen2.5:14b",
                "--max-image-pages", "10",
                "--max-workers", "2"
            ]
        },
        {
            "title": "4. High-Performance Processing",
            "description": "Maximum throughput with parallel workers",
            "command": [
                "python3", "history_to_dataset.py",
                "input.pdf",
                "output.jsonl",
                "--model-name", "llama3.1",
                "--enable-vision",
                "--vision-model", "qwen2.5:14b",
                "--max-workers", "8"
            ]
        },
        {
            "title": "5. Resume from Checkpoint",
            "description": "Continue processing from where it left off",
            "command": [
                "python3", "history_to_dataset.py",
                "input.pdf",
                "output.jsonl",
                "--model-name", "llama3.1",
                "--start-chunk", "25"
            ]
        }
    ]

    for example in examples:
        print_section(example["title"])
        print(f"Description: {example['description']}\n")
        print("Command:")
        print("  " + " \\\n    ".join(example["command"]))
        print()

    print_section("Key Improvements Summary")
    print("""
✓ Fixed deduplication logic - properly removes duplicates
✓ Increased output - 12 → 30 pairs per chunk (150% increase)
✓ Reduced restrictions - minimum answer length 100 → 60 chars
✓ Less filtering - exclude patterns reduced from 17 to 6
✓ Smarter detection - accepts keywords OR entities OR specs
✓ Vision support - analyze diagrams, charts, and drawings
✓ Multi-modal - combine text and image-based Q&A
✓ Source tracking - know which pairs came from text vs images

Expected Results:
  Old system: ~53 Q&A pairs for 400-page book
  New system: 500+ Q&A pairs (10x improvement)
""")

    print_section("Quick Start")
    print("""
1. Ensure Ollama is running:
   ollama serve

2. Pull required models:
   ollama pull llama3.1
   ollama pull qwen2.5:14b  # For vision features

3. Install dependencies:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

4. Run with vision enabled (recommended):
   python3 history_to_dataset.py input.pdf output.jsonl \\
       --model-name llama3.1 \\
       --enable-vision \\
       --vision-model qwen2.5:14b

5. Check output:
   head -n 5 output.jsonl
   python3 -c "import jsonlines; print(f'Total Q&A pairs: {sum(1 for _ in jsonlines.open(\"output.jsonl\"))}')"
""")

    print_section("Custom Keywords")
    print("""
Create keywords.txt to customize domain keywords:

steel
welding
fabrication
metallurgy
heat treatment
forging
casting
...

The system will use these keywords for content filtering.
""")

    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
