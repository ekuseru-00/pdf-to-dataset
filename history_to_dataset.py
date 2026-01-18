import jsonlines
import requests
import re
import json
import time
import logging
from tqdm import tqdm
import PyPDF2
from datetime import datetime, timedelta
import psutil
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import textacy.preprocessing as preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import base64
from io import BytesIO

# Try to import pdf2image, fallback to None if not available
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Image extraction will be disabled.")

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Image processing will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy for named entity recognition and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Compiled regex pattern for technical numbers/specs (performance optimization)
TECHNICAL_NUMBERS_PATTERN = re.compile(r'\d+\s*(mm|cm|m|kg|ton|psi|mpa|gpa|°c|°f|%)', re.IGNORECASE)

# Default steel engineering keywords - users can modify this set in the config section
DEFAULT_STEEL_ENGINEERING_KEYWORDS = {
    # Materials
    'steel', 'alloy', 'carbon steel', 'stainless steel', 'mild steel', 'high-strength steel', 
    'tool steel', 'galvanized steel',
    # Processes
    'welding', 'forging', 'casting', 'machining', 'grinding', 'heat treatment', 
    'quenching', 'tempering', 'annealing', 'rolling',
    # Techniques
    'arc welding', 'mig welding', 'tig welding', 'plasma cutting', 'flame cutting', 
    'shearing', 'bending', 'stamping',
    # Standards
    'astm', 'api', 'aws', 'iso', 'en', 'yield strength', 'tensile strength', 
    'hardness', 'ductility',
    # Equipment
    'furnace', 'lathe', 'press', 'hammer', 'anvil', 'forge', 'kiln', 'reactor',
    # Properties
    'brittleness', 'elasticity', 'plasticity', 'corrosion resistance', 'fatigue', 
    'fatigue resistance',
    # Applications
    'structural', 'construction', 'pipeline', 'automotive', 'aerospace', 'marine', 
    'pressure vessel'
}

def load_custom_keywords(keywords_file="keywords.txt"):
    """Load custom steel engineering keywords from a file if it exists."""
    if os.path.exists(keywords_file):
        try:
            with open(keywords_file, 'r', encoding='utf-8') as f:
                custom_keywords = set(line.strip().lower() for line in f if line.strip())
            print(f"Loaded {len(custom_keywords)} custom keywords from {keywords_file}")
            logger.info(f"Loaded {len(custom_keywords)} custom keywords from {keywords_file}")
            return custom_keywords
        except Exception as e:
            print(f"Error loading keywords file: {e}. Using default keywords.")
            logger.warning(f"Error loading keywords file: {e}. Using default keywords.")
    return DEFAULT_STEEL_ENGINEERING_KEYWORDS

def check_ollama_health():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.ok:
            print("Ollama server is running")
            logger.info("Ollama server is running")
            return True
        else:
            print(f"Ollama server error: {response.text}")
            logger.error(f"Ollama server responded with error: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Ollama server not reachable: {str(e)}")
        logger.error(f"Ollama server not reachable: {str(e)}")
        return False

def log_system_metrics(chunk_index):
    """Log CPU, memory, and GPU usage every 10 chunks."""
    if chunk_index % 10 != 0:
        return
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    mem_usage = mem.used / (1024 ** 3)  # GB
    mem_total = mem.total / (1024 ** 3)  # GB
    print(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    logger.info(f"System metrics: CPU {cpu_percent:.1f}%, Memory {mem_usage:.1f}/{mem_total:.1f} GB")
    try:
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(device)
        gpu_usage = gpu_mem.used / (1024 ** 3)  # GB
        gpu_total = gpu_mem.total / (1024 ** 3)  # GB
        print(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
        logger.info(f"GPU metrics: VRAM {gpu_usage:.1f}/{gpu_total:.1f} GB")
    except Exception as e:
        print(f"Could not retrieve GPU metrics: {str(e)}")
        logger.warning(f"Could not retrieve GPU metrics: {str(e)}")

def read_pdf_text(pdf_path):
    """Read text from a PDF file."""
    start_time = time.time()
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        print(f"Read PDF: {len(text)} characters in {time.time() - start_time:.2f}s")
        logger.info(f"Read PDF: {len(text)} characters in {time.time() - start_time:.2f}s")
        
        # Check for steel engineering keywords
        steel_keywords = load_custom_keywords()
        found_keywords = [k for k in steel_keywords if k in text.lower()]
        print(f"Steel engineering keywords found: {found_keywords[:10]}...")  # Show first 10
        logger.info(f"Steel engineering keywords found: {found_keywords[:10]}...")
        if not found_keywords:
            print("Warning: PDF text may lack sufficient steel engineering content for Q&A generation")
            logger.warning("PDF text may lack sufficient steel engineering content for Q&A generation")
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {str(e)}")
        logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

def extract_pdf_images(pdf_path, dpi=150, max_pages=None):
    """Extract images from PDF by converting pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image extraction (default 150)
        max_pages: Maximum number of pages to process (None for all)
    
    Returns:
        List of tuples (page_number, PIL.Image)
    """
    if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
        print("Image extraction disabled: pdf2image or PIL not available")
        logger.warning("Image extraction disabled: pdf2image or PIL not available")
        return []
    
    start_time = time.time()
    try:
        # Get total page count
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        print(f"Extracting images from {total_pages} pages at {dpi} DPI...")
        logger.info(f"Extracting images from {total_pages} pages at {dpi} DPI...")
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=dpi, last_page=total_pages)
        
        page_images = [(i + 1, img) for i, img in enumerate(images)]
        
        print(f"Extracted {len(page_images)} images in {time.time() - start_time:.2f}s")
        logger.info(f"Extracted {len(page_images)} images in {time.time() - start_time:.2f}s")
        
        return page_images
    except Exception as e:
        print(f"Error extracting images from PDF {pdf_path}: {str(e)}")
        logger.error(f"Error extracting images from PDF {pdf_path}: {str(e)}")
        return []

def encode_image_to_base64(image):
    """Encode a PIL Image to base64 string for API transmission."""
    if not PIL_AVAILABLE:
        return None
    
    try:
        buffered = BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {str(e)}")
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

def generate_qa_from_image(page_number, image, model_name="qwen2.5:14b", max_retries=3):
    """Generate Q&A pairs from an image using vision-capable model.
    
    Args:
        page_number: Page number of the image
        image: PIL.Image object
        model_name: Name of vision-capable model (default qwen2.5:14b)
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of Q&A pair dictionaries with 'source' field set to 'image'
    """
    print(f"Analyzing image from page {page_number} for Q&A generation...")
    logger.info(f"Analyzing image from page {page_number} for Q&A generation...")
    
    # Encode image to base64
    img_base64 = encode_image_to_base64(image)
    if not img_base64:
        print(f"Failed to encode image from page {page_number}")
        logger.error(f"Failed to encode image from page {page_number}")
        return []
    
    start_time = time.time()
    prompt = """
You are a steel engineering and fabrication expert analyzing technical diagrams, charts, metallurgical structures, and fabrication drawings. Generate 1–20 high-quality question-answer pairs from this image for a fine-tuned question-answering model. Your output MUST be a valid JSON array containing objects with the fields "instruction" (the question), "input" (a brief description of what's shown in the image), and "output" (the detailed answer).

Focus on: diagrams (process flows, welding procedures, assembly instructions), charts (material properties, stress-strain curves, phase diagrams), metallurgical structures (grain structures, microstructures, defects), technical drawings (fabrication details, dimensions, specifications), equipment layouts, safety procedures, and visual technical content.

For each Q&A pair:
- "instruction": Ask about specific elements visible in the image (e.g., "What welding technique is shown?", "What are the key dimensions?", "What type of microstructure is visible?")
- "input": Describe relevant parts of the image (e.g., "Diagram showing MIG welding setup with labeled components")
- "output": Provide detailed technical explanations (at least 60 characters) based on what's visible in the image

Generate as many relevant Q&A pairs as possible based on the image content. If the image has limited technical content, generate at least 1-2 pairs.

Example output:
[
  {"instruction": "What welding process is illustrated in the diagram?", "input": "Diagram showing welding torch, wire feed, and shielding gas setup", "output": "The diagram illustrates the MIG (Metal Inert Gas) welding process, also known as GMAW (Gas Metal Arc Welding). The setup shows a continuous wire electrode being fed through a welding gun, with an inert gas (typically argon or CO2 mixture) shielding the weld pool from atmospheric contamination."},
  {"instruction": "What are the critical dimensions shown?", "input": "Technical drawing with measurements and tolerances", "output": "The drawing specifies critical dimensions including a base plate thickness of 12mm, weld bead width of 8-10mm, and penetration depth of 6mm minimum. Tolerances are indicated as ±0.5mm for all dimensional measurements."}
]
"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=180
            )
            
            if not response.ok:
                print(f"Vision model request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                logger.error(f"Vision model request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                time.sleep(5)
                continue
            
            raw = response.json()["response"]
            with open("raw_responses.log", "a") as f:
                f.write(f"Image page {page_number} response at {datetime.now()}:\n{raw}\n{'='*50}\n")
            
            print(f"Image analysis response time: {time.time() - start_time:.2f}s")
            logger.info(f"Image analysis response time: {time.time() - start_time:.2f}s")
            
            # Parse JSON response
            qa_pairs = parse_image_qa_response(raw, page_number)
            return qa_pairs
            
        except Exception as e:
            print(f"Error processing image (attempt {attempt+1}/{max_retries}): {str(e)}")
            logger.error(f"Error processing image (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to process image from page {page_number} after {max_retries} attempts")
                logger.error(f"Failed to process image from page {page_number} after {max_retries} attempts")
                return []
            time.sleep(5)
    
    return []

def parse_image_qa_response(text, page_number):
    """Parse Q&A pairs from vision model response and mark them as image-sourced."""
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title.*published', 
        r'who.*supervisor', r'what.*table of contents',
        r'what.*abbreviations.*list', r'who.*editor'
    ]
    
    try:
        # Try to extract JSON array
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        
        json_str = text[start:end]
        json_str = fix_json_string(json_str)
        parsed = json.loads(json_str)
        
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("Invalid JSON structure")
        
        filtered_pairs = []
        for item in parsed:
            question = normalize_text(item.get("instruction", ""))
            answer = normalize_text(item.get("output", ""))
            input_text = item.get("input", f"Image from page {page_number}")
            
            # Apply same filtering as text-based Q&A
            if (not any(re.search(pattern, question) for pattern in exclude_patterns) and 
                len(answer) >= 60 and len(question) >= 10):
                filtered_pairs.append({
                    "instruction": question.capitalize(),
                    "input": input_text,
                    "output": answer,
                    "source": "image",
                    "page": page_number
                })
        
        print(f"Parsed {len(filtered_pairs)} image Q&A pairs from page {page_number}")
        logger.info(f"Parsed {len(filtered_pairs)} image Q&A pairs from page {page_number}")
        return filtered_pairs
        
    except Exception as e:
        print(f"Failed to parse image Q&A response: {str(e)}")
        logger.warning(f"Failed to parse image Q&A response: {str(e)}")
        return []

def chunk_text(text, max_chars=800, overlap=200, min_chunks=3):
    """Split text into semantically meaningful chunks."""
    steel_keywords = load_custom_keywords()
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_length = 0
    
    # Try splitting by paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_doc = nlp(para)
        # More lenient: Accept if has steel keywords OR technical entities OR just has reasonable content
        has_steel_content = (
            any(keyword in para.lower() for keyword in steel_keywords) or 
            any(ent.label_ in {'ORG', 'PRODUCT', 'QUANTITY', 'CARDINAL', 'PERCENT', 'MONEY', 'GPE', 'DATE'} for ent in para_doc.ents) or
            len(para) > 100  # Accept longer paragraphs that might have technical content
        )
        if not has_steel_content:
            print(f"Discarded paragraph (no steel engineering content): {para[:100]}...")
            logger.debug(f"Discarded paragraph (no steel engineering content): {para[:100]}...")
            continue
        para_length = len(para)
        
        if current_length + para_length <= max_chars:
            current_chunk += para + "\n\n"
            current_length += para_length + 2
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para[-overlap:] if len(para) > overlap else para
            current_length = len(current_chunk)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If too few chunks, split by sentences
    if len(chunks) < min_chunks and text:
        chunks = []
        current_chunk = ""
        current_length = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_length = len(sent_text)
            # More lenient: Accept if has steel keywords OR technical entities OR reasonable length
            has_steel_content = (
                any(keyword in sent_text.lower() for keyword in steel_keywords) or 
                any(ent.label_ in {'ORG', 'PRODUCT', 'QUANTITY', 'CARDINAL', 'PERCENT', 'MONEY', 'GPE', 'DATE'} for ent in sent.ents) or
                len(sent_text) > 50  # Accept longer sentences
            )
            if not has_steel_content:
                print(f"Discarded sentence (no steel engineering content): {sent_text[:100]}...")
                logger.debug(f"Discarded sentence (no steel engineering content): {sent_text[:100]}...")
                continue
            if current_length + sent_length <= max_chars:
                current_chunk += sent_text + " "
                current_length += sent_length + 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text[-overlap:] if len(sent_text) > overlap else sent_text
                current_length = len(current_chunk)
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    print(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    logger.info(f"Split text into {len(chunks)} chunks (text length: {len(text)} chars)")
    return chunks

def normalize_text(text):
    """Normalize text to handle case sensitivity and clean up."""
    return preprocessing.normalize.unicode(text.lower())

def is_steel_engineering_qa(question, answer):
    """Check if Q&A pair is steel engineering related using named entity recognition or keywords."""
    doc_q = nlp(question)
    doc_a = nlp(answer)
    steel_keywords = load_custom_keywords()
    
    # More lenient: Accept if has technical entities OR keywords OR technical numbers/specs
    has_entities = any(ent.label_ in {'ORG', 'PRODUCT', 'QUANTITY', 'CARDINAL', 'PERCENT', 'MONEY', 'GPE', 'DATE'} for ent in doc_q.ents + doc_a.ents)
    has_keywords = any(keyword in question.lower() or keyword in answer.lower() for keyword in steel_keywords)
    has_technical_numbers = bool(TECHNICAL_NUMBERS_PATTERN.search(answer.lower()))
    
    return has_entities or has_keywords or has_technical_numbers

def deduplicate_qa_pairs(pairs):
    """Remove duplicate Q&A pairs based on semantic similarity."""
    if not pairs:
        return pairs
    
    texts = [pair["instruction"] + " " + pair["output"] for pair in pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Fix: Keep track of which indices to keep, avoiding duplicates
    keep_indices = set()
    skip_indices = set()
    
    for i in range(len(pairs)):
        if i in skip_indices:
            continue
        keep_indices.add(i)
        for j in range(i + 1, len(pairs)):
            if similarity_matrix[i][j] > 0.9:  # Threshold for near-duplicates
                skip_indices.add(j)  # Mark as duplicate
    
    deduped_pairs = [pairs[i] for i in sorted(keep_indices)]
    print(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    logger.info(f"Deduplicated {len(pairs)} to {len(deduped_pairs)} Q&A pairs")
    return deduped_pairs[:30]  # Increased limit from 12 to 30 pairs

def fix_json_string(json_str):
    """Attempt to fix common JSON errors (trailing commas, invalid escapes)."""
    try:
        # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix invalid escapes
        json_str = re.sub(r'\\[^\\bfnrtu"]', r'\\', json_str)
        return json_str
    except Exception as e:
        print(f"Failed to fix JSON string: {str(e)}")
        logger.error(f"Failed to fix JSON string: {str(e)}")
        return json_str

def extract_relevant_input(chunk, question, full_text):
    """Extract relevant sentences from the chunk or full text for the input field."""
    steel_keywords = load_custom_keywords()
    doc = nlp(chunk)
    relevant_sentences = []
    question_keywords = set(normalize_text(question).split()) & steel_keywords
    
    # First, try to find relevant sentences in the chunk
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
           any(ent.label_ in {'ORG', 'PRODUCT', 'QUANTITY', 'CARDINAL', 'PERCENT', 'MONEY'} for ent in sent.ents):
            relevant_sentences.append(sent_text)
    
    # If no relevant sentences found, search the full text
    if not relevant_sentences:
        doc_full = nlp(full_text)
        for sent in doc_full.sents:
            sent_text = sent.text.strip()
            if any(keyword in normalize_text(sent_text) for keyword in question_keywords) or \
               any(ent.label_ in {'ORG', 'PRODUCT', 'QUANTITY', 'CARDINAL', 'PERCENT', 'MONEY'} for ent in sent.ents):
                relevant_sentences.append(sent_text)
                if len(' '.join(relevant_sentences)) >= 800:
                    break
    
    input_text = ' '.join(relevant_sentences)[:800]
    if not input_text:
        input_text = chunk[:800]  # Fallback to original chunk if no relevant sentences found
    return input_text

def extract_json_array(text, chunk, full_text):
    """Extract JSON array from text, handling malformed cases."""
    # Reduced and less restrictive exclude patterns
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title.*published', 
        r'who.*supervisor', r'what.*table of contents',
        r'what.*abbreviations.*list', r'who.*editor'
    ]
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found")
        json_str = text[start:end]
        json_str = fix_json_string(json_str)
        parsed = json.loads(json_str)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("Invalid JSON structure: not a list of dictionaries")
        
        filtered_pairs = []
        for item in parsed:
            question = normalize_text(item["instruction"])
            answer = normalize_text(item["output"])
            # Reduced minimum answer length from 100 to 60 characters
            if (not any(re.search(pattern, question) for pattern in exclude_patterns) and 
                len(answer) >= 60 and is_steel_engineering_qa(question, answer)):
                item["input"] = extract_relevant_input(chunk, question, full_text)
                item["instruction"] = question.capitalize()
                item["output"] = answer
                item["source"] = "text"  # Mark source as text
                filtered_pairs.append(item)
            else:
                print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-steel engineering or too short)")
                logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-steel engineering or too short)")
        
        print(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        logger.info(f"Parsed {len(filtered_pairs)} valid JSON Q&A pairs")
        return deduplicate_qa_pairs(filtered_pairs)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        logger.warning(f"JSON parsing failed: {str(e)}. Non-JSON response:\n{text[:500]}...")
        return parse_qa_pairs(text, chunk, full_text)

def parse_qa_pairs(text, chunk, full_text):
    """Parse Q: A: pairs from non-JSON response, focusing on steel engineering content."""
    qa_pairs = []
    qa_pattern = re.compile(r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)', re.DOTALL)
    matches = qa_pattern.findall(text)
    
    # Reduced and less restrictive exclude patterns
    exclude_patterns = [
        r'who.*wrote', r'who.*authored', r'what.*title.*published', 
        r'who.*supervisor', r'what.*table of contents',
        r'what.*abbreviations.*list', r'who.*editor'
    ]
    
    for question, answer in matches:
        question = normalize_text(question.strip())
        answer = normalize_text(answer.strip())
        # Reduced minimum answer length from 100 to 60 characters
        if (not any(re.search(pattern, question) for pattern in exclude_patterns) and 
            len(answer) >= 60 and is_steel_engineering_qa(question, answer)):
            qa_pairs.append({
                "instruction": question.capitalize(),
                "input": extract_relevant_input(chunk, question, full_text),
                "output": answer,
                "source": "text"  # Mark source as text
            })
            print(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
            logger.info(f"Accepted Q&A pair: Q: {question.capitalize()}, A: {answer[:50]}...")
        else:
            print(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-steel engineering or too short)")
            logger.debug(f"Filtered out Q&A pair: Q: {question}, A: {answer[:50]}... (non-steel engineering or too short)")
    
    if not qa_pairs:
        print(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
        logger.warning(f"No valid Q&A pairs found in output for chunk:\n{chunk[:500]}...")
    else:
        print(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs via fallback")
    
    return deduplicate_qa_pairs(qa_pairs)

def generate_questions_answers(chunk, full_text, model_name="llama3.1", max_retries=5):
    """Generate 1–30 Q&A pairs about steel engineering materials, processes, and fabrication techniques."""
    print(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    logger.info(f"Processing chunk (first 200 chars): {chunk[:200]}...")
    start_time = time.time()
    prompt = f"""
You are a steel engineering and fabrication expert tasked with generating 1–30 high-quality question-answer pairs from a given text passage for a fine-tuned question-answering model. Your output MUST be a valid JSON array containing 1–30 objects, each with the fields "instruction" (the question), "input" (the specific sentences or phrases from the passage that directly relate to the question and answer), and "output" (the answer). Do NOT include any text outside the JSON array (e.g., explanations, headings, Q: A: pairs). Non-JSON output will be discarded.

Focus on questions about steel engineering and fabrication topics including: materials (e.g., steel types, alloys, carbon steel, stainless steel), processes (e.g., welding, forging, casting, machining, heat treatment), techniques (e.g., arc welding, MIG welding, TIG welding, plasma cutting), standards (e.g., ASTM, API, AWS, ISO, material properties), equipment (e.g., furnace, lathe, press, forge), material properties (e.g., hardness, tensile strength, ductility, corrosion resistance), applications (e.g., structural, construction, pipeline, automotive, aerospace), and technical specifications. For the "input" field, include only the sentences or phrases from the passage that directly support the question and answer, ensuring the input is complete and meaningful. Answers must be at least 60 characters, include specific details (e.g., specifications, standards, procedures, material compositions, property values), and explicitly cite sources mentioned in the passage or note if no source is provided. Generate as many relevant Q&A pairs as possible from the content, aiming for 20-30 pairs for content-rich passages. If the passage is short or lacks sufficient content, generate at least 1 high-quality pair.

Ensure answers are detailed, accurate, and contextually rich, drawing directly from the passage. Verify technical specifications and use industry standards or technical sources cited in the passage for accuracy. If the passage lacks specific details, do NOT generate Q&A pairs based on external knowledge.

Example output:
[
  {{"instruction": "What are the key properties of ASTM A36 steel?", "input": "ASTM A36 is a structural steel with a minimum yield strength of 36,000 psi and tensile strength of 58,000-80,000 psi.", "output": "ASTM A36 is a widely used structural carbon steel that has a minimum yield strength of 36,000 psi (250 MPa) and a tensile strength ranging from 58,000 to 80,000 psi (400-550 MPa). This combination of properties makes it ideal for construction and structural applications where high strength and good weldability are required."}},
  {{"instruction": "How does quenching affect steel hardness?", "input": "Quenching rapidly cools heated steel in water or oil, transforming its microstructure to increase hardness and strength.", "output": "Quenching is a heat treatment process that involves rapidly cooling heated steel by immersing it in water, oil, or other quenching media. This rapid cooling transforms the steel's microstructure from austenite to martensite, significantly increasing its hardness and strength. The rate of cooling and choice of quenching medium are critical factors that determine the final properties of the steel."}}
]

Generate 1–30 high-quality question-answer pairs based on the following passage:
\"\"\"{chunk}\"\"\"
"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.5
                },
                timeout=180
            )
            if not response.ok:
                print(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                logger.error(f"LLaMA request failed (attempt {attempt+1}/{max_retries}): {response.text}")
                time.sleep(5)
                continue

            raw = response.json()["response"]
            with open("raw_responses.log", "a") as f:
                f.write(f"Chunk response at {datetime.now()}:\n{raw}\n{'='*50}\n")
            print(f"Response time: {time.time() - start_time:.2f}s")
            logger.info(f"Response time: {time.time() - start_time:.2f}s")
            return extract_json_array(raw, chunk, full_text)
        except Exception as e:
            print(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            logger.error(f"Error processing chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                logger.error(f"Failed to process chunk after {max_retries} attempts:\n{chunk[:500]}...")
                return []
            time.sleep(5)
    return []

def process_chunk(i, chunk, full_text, model_name):
    """Helper function to process a single chunk and return results."""
    log_system_metrics(i)
    items = generate_questions_answers(chunk, full_text, model_name)
    return i, chunk, items

def generate_dataset_from_pdf(pdf_path, output_path, model_name="llama3.1", start_chunk=0, temp_path=None, checkpoint_path="checkpoint.json", max_workers=4, enable_vision=True, vision_model="qwen2.5:14b", max_image_pages=None):
    """Generate a dataset from a PDF and save to JSONL, processing chunks in parallel.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path for the output JSONL file
        model_name: Model for text-based Q&A generation
        start_chunk: Starting chunk index
        temp_path: Path for temporary checkpoint file
        checkpoint_path: Path for checkpoint JSON
        max_workers: Number of parallel workers
        enable_vision: Enable image extraction and vision-based Q&A (default True)
        vision_model: Model name for vision-based Q&A (default qwen2.5:14b)
        max_image_pages: Maximum number of pages to extract images from (None for all)
    """
    if not check_ollama_health():
        print("Aborting: Ollama server not available")
        logger.error("Aborting: Ollama server not available")
        return

    if temp_path is None:
        bookname = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_path = f"temp_{bookname}.jsonl"
    
    start_time = time.time()
    
    # Extract text from PDF
    full_text = read_pdf_text(pdf_path)
    print(f"First 500 chars of full text: {full_text[:500]}")
    logger.info(f"First 500 chars of full text: {full_text[:500]}")
    if len(full_text) < 200:
        print(f"Warning: Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")
        logger.warning(f"Input text is too short ({len(full_text)} chars). Consider augmenting with additional sources.")
    
    # Extract images from PDF if enabled
    page_images = []
    if enable_vision:
        print("Image extraction enabled - processing PDF pages as images...")
        logger.info("Image extraction enabled - processing PDF pages as images...")
        page_images = extract_pdf_images(pdf_path, dpi=150, max_pages=max_image_pages)
        print(f"Extracted {len(page_images)} page images for vision analysis")
        logger.info(f"Extracted {len(page_images)} page images for vision analysis")
    
    chunks = chunk_text(full_text)
    
    if start_chunk < 0 or start_chunk >= len(chunks):
        print(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        logger.error(f"Invalid start_chunk {start_chunk}; must be between 0 and {len(chunks)-1}")
        return

    all_data = []
    stats = {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0}
    failed_chunks = []
    lock = threading.Lock()

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        all_data = checkpoint.get("data", [])
        stats = checkpoint.get("stats", {"successful_chunks": 0, "failed_chunks": 0, "total_pairs": 0})
        failed_chunks = checkpoint.get("failed_chunks", [])
        checkpoint_last = checkpoint.get("last_chunk", -1)
        if checkpoint_last >= start_chunk:
            start_chunk = checkpoint_last + 1
            print(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
            logger.info(f"Resumed from checkpoint: {len(all_data)} pairs, starting at chunk {start_chunk}")
        else:
            print(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")
            logger.info(f"Ignoring checkpoint (last_chunk {checkpoint_last} < start_chunk {start_chunk})")

    try:
        with tqdm(total=len(chunks), initial=start_chunk, desc="Processing chunks", unit="chunk") as pbar:
            for batch_start in range(start_chunk, len(chunks), max_workers):
                batch = [(i, chunk) for i, chunk in enumerate(chunks[batch_start:batch_start + max_workers], start=batch_start)]
                futures = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for i, chunk in batch:
                        futures.append(executor.submit(process_chunk, i, chunk, full_text, model_name))
                    
                    for future in as_completed(futures):
                        try:
                            i, chunk, items = future.result()
                            with lock:
                                if items:
                                    all_data.extend(items)
                                    stats["successful_chunks"] += 1
                                    stats["total_pairs"] += len(items)
                                    print(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    logger.info(f"Chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                                    if stats["successful_chunks"] % 3 == 0:
                                        with jsonlines.open(temp_path, mode='w') as writer:
                                            writer.write_all(all_data)
                                        print(f"Saved partial results to {temp_path}")
                                        logger.info(f"Saved partial results to {temp_path}")
                                else:
                                    stats["failed_chunks"] += 1
                                    failed_chunks.append((i, chunk))
                                    print(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                    logger.warning(f"Chunk {i}/{len(chunks)}: No Q&A pairs generated, added to failed_chunks")
                                
                                with open(checkpoint_path, 'w') as f:
                                    json.dump({
                                        "data": all_data,
                                        "stats": stats,
                                        "last_chunk": i,
                                        "failed_chunks": failed_chunks
                                    }, f)
                                
                                elapsed = time.time() - start_time
                                processed_chunks = i - start_chunk + 1 if i >= start_chunk else 1
                                avg_time_per_chunk = elapsed / processed_chunks
                                remaining_chunks = len(chunks) - i
                                eta_seconds = remaining_chunks * avg_time_per_chunk
                                eta = str(timedelta(seconds=int(eta_seconds)))
                                pbar.set_postfix({"ETA": eta, "Pairs": stats["total_pairs"]})
                                pbar.update(1)
                        except Exception as e:
                            print(f"Error processing future for chunk {i}: {str(e)}")
                            logger.error(f"Error processing future for chunk {i}: {str(e)}")
                
                time.sleep(2)

        if failed_chunks:
            print(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            logger.info(f"Retrying {len(failed_chunks)} failed chunks sequentially")
            for i, chunk in failed_chunks:
                log_system_metrics(i)
                print(f"Retrying chunk {i}/{len(chunks)}")
                logger.info(f"Retrying chunk {i}/{len(chunks)}")
                items = generate_questions_answers(chunk, full_text, model_name)
                with lock:
                    if items:
                        all_data.extend(items)
                        stats["successful_chunks"] += 1
                        stats["total_pairs"] += len(items)
                        print(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        logger.info(f"Retry chunk {i}/{len(chunks)}: Generated {len(items)} Q&A pairs")
                        if stats["successful_chunks"] % 3 == 0:
                            with jsonlines.open(temp_path, mode='w') as writer:
                                writer.write_all(all_data)
                            print(f"Saved partial results to {temp_path}")
                            logger.info(f"Saved partial results to {temp_path}")
                    else:
                        print(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                        logger.warning(f"Retry chunk {i}/{len(chunks)}: No Q&A pairs generated:\n{chunk[:500]}...")
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            "data": all_data,
                            "stats": stats,
                            "last_chunk": i,
                            "failed_chunks": failed_chunks
                        }, f)

        # Process images for vision-based Q&A generation
        if enable_vision and page_images:
            print(f"\nProcessing {len(page_images)} images for vision-based Q&A generation...")
            logger.info(f"Processing {len(page_images)} images for vision-based Q&A generation...")
            
            image_stats = {"successful_images": 0, "failed_images": 0, "total_image_pairs": 0}
            
            for page_num, image in tqdm(page_images, desc="Processing images", unit="image"):
                try:
                    image_qa_pairs = generate_qa_from_image(page_num, image, model_name=vision_model)
                    
                    with lock:
                        if image_qa_pairs:
                            all_data.extend(image_qa_pairs)
                            image_stats["successful_images"] += 1
                            image_stats["total_image_pairs"] += len(image_qa_pairs)
                            stats["total_pairs"] += len(image_qa_pairs)
                            print(f"Page {page_num}: Generated {len(image_qa_pairs)} image-based Q&A pairs")
                            logger.info(f"Page {page_num}: Generated {len(image_qa_pairs)} image-based Q&A pairs")
                        else:
                            image_stats["failed_images"] += 1
                            print(f"Page {page_num}: No image-based Q&A pairs generated")
                            logger.warning(f"Page {page_num}: No image-based Q&A pairs generated")
                        
                        # Save periodically
                        if image_stats["successful_images"] % 5 == 0:
                            with jsonlines.open(temp_path, mode='w') as writer:
                                writer.write_all(all_data)
                            print(f"Saved partial results including image Q&A to {temp_path}")
                            logger.info(f"Saved partial results including image Q&A to {temp_path}")
                
                except Exception as e:
                    print(f"Error processing image from page {page_num}: {str(e)}")
                    logger.error(f"Error processing image from page {page_num}: {str(e)}")
                    image_stats["failed_images"] += 1
                
                time.sleep(1)  # Brief pause between image processing
            
            print(f"Image processing complete: {image_stats['successful_images']} successful, "
                  f"{image_stats['failed_images']} failed, {image_stats['total_image_pairs']} total image Q&A pairs")
            logger.info(f"Image processing complete: {image_stats['successful_images']} successful, "
                        f"{image_stats['failed_images']} failed, {image_stats['total_image_pairs']} total image Q&A pairs")

    except Exception as e:
        print(f"Processing interrupted: {str(e)}. Saving partial results")
        logger.error(f"Processing interrupted: {str(e)}. Saving partial results")
        with lock:
            if all_data:
                with jsonlines.open(temp_path, mode='w') as writer:
                    writer.write_all(all_data)
                print(f"Saved partial results to {temp_path}")
                logger.info(f"Saved partial results to {temp_path}")

    print("Generated Q&A pairs (sample of up to 10):")
    logger.info("Generated Q&A pairs (sample of up to 10):")
    for i, item in enumerate(all_data[:10], 1):
        print(f"Pair {i}:")
        print(f"  Instruction: {item['instruction']}")
        print(f"  Input: {item['input'][:200]}...")
        print(f"  Output: {item['output'][:100]}...")
        print(f"  Input length: {len(item['input'])} chars")
        logger.info(f"Pair {i}:")
        logger.info(f"  Instruction: {item['instruction']}")
        logger.info(f"  Input: {item['input'][:200]}...")
        logger.info(f"  Output: {item['output'][:100]}...")
        logger.info(f"  Input length: {len(item['input'])} chars")

    if all_data:
        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(all_data)
        print(f"Dataset written to {output_path} with {len(all_data)} items")
        logger.info(f"Dataset written to {output_path} with {len(all_data)} items")
    else:
        print(f"No data written to {output_path}: No Q&A pairs generated")
        logger.warning(f"No data written to {output_path}: No Q&A pairs generated")

    elapsed_time = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Processing complete in {elapsed_time}")
    print(f"Summary: {stats['successful_chunks']} successful chunks, "
          f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")
    logger.info(f"Processing complete in {elapsed_time}")
    logger.info(f"Summary: {stats['successful_chunks']} successful chunks, "
                f"{stats['failed_chunks']} failed chunks, {stats['total_pairs']} total Q&A pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A dataset from PDF with optional vision-based image analysis")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("output_path", help="Path for the output JSONL file")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting chunk index (0-based, default 0)")
    parser.add_argument("--model-name", type=str, default="llama3.1", help="Ollama model name for text-based Q&A (e.g., llama3.1, mistral)")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers (default 4)")
    parser.add_argument("--enable-vision", action="store_true", default=True, help="Enable image extraction and vision-based Q&A generation (default: enabled)")
    parser.add_argument("--vision-model", type=str, default="qwen2.5:14b", help="Ollama vision model name (default qwen2.5:14b)")
    parser.add_argument("--max-image-pages", type=int, default=None, help="Maximum number of pages to extract images from (default: all pages)")
    args = parser.parse_args()

    generate_dataset_from_pdf(
        args.pdf_path, 
        args.output_path, 
        model_name=args.model_name, 
        start_chunk=args.start_chunk, 
        max_workers=args.max_workers,
        enable_vision=args.enable_vision,
        vision_model=args.vision_model,
        max_image_pages=args.max_image_pages
    )
