# PDF to Dataset - Enhanced Pipeline

Convert technical PDF documents into high-quality Q&A datasets for fine-tuning language models, with support for both text and image analysis.

## Key Features

### 🎯 Enhanced Q&A Generation
- **Fixed deduplication logic** - Properly removes duplicate Q&A pairs while preserving unique content
- **Increased output** - Generate up to 30 Q&A pairs per chunk (previously 12)
- **Reduced restrictions** - Minimum answer length reduced from 100 to 60 characters
- **Less aggressive filtering** - Reduced exclude patterns from 17 to 6 essential ones
- **Smarter content detection** - Accepts Q&A with keywords, technical entities, or technical specifications

### 🖼️ Vision-Based Analysis (New!)
- **Image extraction** - Convert PDF pages to images using pdf2image
- **Vision model support** - Use qwen2.5:14b or other vision-capable models to analyze diagrams, charts, and technical drawings
- **Multi-modal Q&A** - Generate Q&A pairs from both text and visual content
- **Source tracking** - Each Q&A pair tagged with source (text/image)

### 📊 Expected Results
For a 400-page technical book:
- **Old system**: ~53 Q&A pairs
- **New system**: 500+ Q&A pairs (10x improvement)

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# For image processing (optional but recommended)
# On Ubuntu/Debian:
sudo apt-get install poppler-utils

# On macOS:
brew install poppler
```

## Requirements

### Core Dependencies
- Python 3.7+
- Ollama server running locally (http://localhost:11434)
- Model for text-based Q&A (e.g., llama3.1, mistral)

### Optional Dependencies (for vision features)
- Vision-capable model (e.g., qwen2.5:14b)
- pdf2image library
- poppler-utils (system package)

## Usage

### Basic Usage (Text-Only)

```bash
python history_to_dataset.py input.pdf output.jsonl --model-name llama3.1
```

### Enhanced Usage (Text + Images)

```bash
python history_to_dataset.py input.pdf output.jsonl \
    --model-name llama3.1 \
    --enable-vision \
    --vision-model qwen2.5:14b \
    --max-image-pages 50
```

### Command-Line Options

```
positional arguments:
  pdf_path              Path to the PDF file
  output_path           Path for the output JSONL file

optional arguments:
  --start-chunk N       Starting chunk index (default: 0)
  --model-name MODEL    Text-based Q&A model (default: llama3.1)
  --max-workers N       Number of parallel workers (default: 4)
  --enable-vision       Enable image extraction and vision-based Q&A
  --vision-model MODEL  Vision model name (default: qwen2.5:14b)
  --max-image-pages N   Max pages to extract images from (default: all)
```

## Configuration

### Custom Keywords

Create a `keywords.txt` file to define custom domain keywords:

```
steel
welding
fabrication
metallurgy
heat treatment
...
```

### Checkpointing

The system automatically saves progress to `checkpoint.json`. If interrupted, resume with:

```bash
python history_to_dataset.py input.pdf output.jsonl --start-chunk 25
```

## Output Format

Each Q&A pair in the JSONL output includes:

```json
{
  "instruction": "What are the key properties of ASTM A36 steel?",
  "input": "ASTM A36 is a structural steel with minimum yield strength of 36,000 psi...",
  "output": "ASTM A36 is a widely used structural carbon steel that has a minimum yield strength of 36,000 psi (250 MPa)...",
  "source": "text"
}
```

For image-based Q&A, additional fields:
```json
{
  "instruction": "What welding process is illustrated in the diagram?",
  "input": "Diagram showing welding torch, wire feed, and shielding gas setup",
  "output": "The diagram illustrates the MIG welding process...",
  "source": "image",
  "page": 42
}
```

## Architecture Changes

### Deduplication Fix
**Before**: Buggy logic that incorrectly added indices, keeping too many duplicates
```python
# Old buggy code
if similarity_matrix[i][j] > 0.9:
    continue  # Skip but still add j later
keep_indices.append(j)
```

**After**: Proper duplicate tracking with skip set
```python
# Fixed code
if similarity_matrix[i][j] > 0.9:
    skip_indices.add(j)  # Mark as duplicate, never add
```

### Increased Limits
- Pair limit per chunk: 12 → 30 (150% increase)
- Minimum answer length: 100 → 60 characters (40% reduction)
- Exclude patterns: 17 → 6 (65% reduction)

### Enhanced Filtering
**Old**: Required strict steel engineering keywords
**New**: Accepts content with:
- Steel engineering keywords, OR
- Technical entities (ORG, PRODUCT, QUANTITY, etc.), OR
- Technical specifications (measurements with units)

### Vision Pipeline
1. Extract PDF pages as images (pdf2image)
2. Send images to vision model with specialized prompt
3. Parse Q&A pairs from vision model response
4. Tag with source="image" and page number
5. Merge with text-based Q&A pairs

## Performance Tips

1. **Use parallel workers**: Increase `--max-workers` based on CPU cores
2. **Limit image pages**: Use `--max-image-pages` for faster testing
3. **Monitor resources**: System metrics logged every 10 chunks
4. **Checkpointing**: Resume from interruptions without losing progress

## Troubleshooting

### "pdf2image not available"
Install poppler-utils system package and pdf2image Python package.

### "Ollama server not available"
Ensure Ollama is running: `ollama serve`

### Low Q&A output
- Check if PDF has extractable text (not scanned images)
- Verify steel engineering keywords match your domain
- Enable vision mode to capture diagram content
- Review exclude patterns in code

## Testing

Run the improvement test suite:

```bash
python test_improvements.py
```

This validates:
- Deduplication logic correctness
- Answer length requirements
- Pair limit increases
- Filtering improvements

## License

See repository license.

## Contributing

Contributions welcome! Key areas:
- Support for additional vision models
- OCR fallback for text extraction from images
- Additional domain keyword sets
- Performance optimizations
