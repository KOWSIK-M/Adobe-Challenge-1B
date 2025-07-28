 # main.py

import os
import json
import fitz  # PyMuPDF
import re
import logging
import string
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer, util

# ================================
# CONFIGURATION CONSTANTS & LOGGING
# ================================

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SCORE_THRESHOLD = 0.0
MAX_SECTIONS_OUTPUT = 5

MAX_SECTION_CONTENT_CHARS = 3000
MIN_SECTION_CONTENT_CHARS = 20

LOG_FILE = "persona_pdf_analyzer.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    filemode='w',
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ================================
# UTILITY FUNCTIONS
# ================================

def safe_execute(func, *args, default=None, **kwargs):
    """
    Helper to safely execute a function with args/kwargs.
    Logs exception and returns default if error occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {str(e)}")
        logging.debug(traceback.format_exc())
        return default

def clean_text(text: str) -> str:
    """
    Cleans text by removing line breaks, extra spaces, control chars, and unwanted characters.
    """
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable chars
    text = ''.join(ch for ch in text if ch in string.printable)
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Lowercase and remove punctuation for normalization.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.strip()
    return text

def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to max_length preserving whole words if possible.
    """
    if not text or len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space]
    return truncated

def is_valid_filename(filename: str) -> bool:
    """
    Check if a filename is valid for reading.
    """
    if not filename or len(filename.strip()) == 0:
        return False
    invalid_chars = set('/\\:*?"<>|')
    return not any((c in invalid_chars) for c in filename)

def safe_open_json(filepath: Path) -> Optional[Dict]:
    """
    Safe load JSON file from given path.
    """
    if not filepath.exists():
        logging.error(f"JSON file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON {filepath}: {str(e)}")
        logging.debug(traceback.format_exc())
        return None

# ================================
# PDF TEXT EXTRACTION AND SECTION PARSING
# ================================

def extract_text_elements_from_pdf(pdf_path: Path) -> List[List[Dict[str, Any]]]:
    """
    Extract text spans from each page as elements (text, font size, font name).
    Returns a list of pages, each containing list of elements.
    """
    logging.info(f"Extracting text elements from PDF: {pdf_path}")
    doc = safe_execute(fitz.open, pdf_path, default=None)
    if doc is None:
        logging.error(f"Failed to open PDF: {pdf_path}")
        return []

    all_pages_elements = []
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        page_elements = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                span_text = ""
                size = None
                font = None
                for span in line["spans"]:
                    # Collect all text spans in line
                    span_text += span.get("text", "")
                    size = span.get("size", size)
                    font = span.get("font", font)
                cleaned_text = clean_text(span_text)
                if cleaned_text:
                    page_elements.append({
                        "text": cleaned_text,
                        "size": size,
                        "font": font,
                        "page_number": page.number + 1
                    })
        all_pages_elements.append(page_elements)

    logging.debug(f"Extracted elements from {len(all_pages_elements)} pages.")
    return all_pages_elements

def estimate_body_font_properties(pages_elements: List[List[Dict]]) -> Tuple[float, str]:
    """
    Estimate most common font size and font name across body text.
    """
    sizes, fonts = [], []
    for page in pages_elements:
        for el in page:
            try:
                sizes.append(round(el.get('size', 0), 2))
                fonts.append(el.get('font', ''))
            except Exception:
                continue

    if not sizes:
        default_size = 12.0
        logging.warning(f"No sizes found. Using default font size: {default_size}")
        return default_size, "Times-Roman"

    body_font_size = max(set(sizes), key=sizes.count)
    body_font_name = max(set(fonts), key=fonts.count)
    logging.info(f"Estimated body font size: {body_font_size}, font name: {body_font_name}")
    return body_font_size, body_font_name

def is_heading(element: Dict[str, Any], body_font_size: float, body_font_name: str) -> bool:
    """
    Heuristic: Heading if font size is significantly larger or font different and not body.
    """
    try:
        size = element.get('size', 0)
        font = element.get('font', '')
        if size is None:
            return False
        if size > body_font_size + 1.5:
            return True
        if font != body_font_name and size >= body_font_size:
            return True
        return False
    except Exception as e:
        logging.error(f"Error in heading detection: {e}")
        return False

def parse_sections_from_elements(pages_elements: List[List[Dict]]) -> List[Dict[str, Any]]:
    """
    Using extracted text elements, detect headings and collect sections.
    Returns list of dicts with keys: title, content, page
    """
    logging.info("Parsing sections from text elements.")
    body_font_size, body_font_name = estimate_body_font_properties(pages_elements)
    sections = []
    current_section = {"title": None, "content": "", "page": None}

    for page in pages_elements:
        for el in page:
            if is_heading(el, body_font_size, body_font_name):
                if current_section["title"]:
                    # Append previous section
                    current_section["content"] = current_section["content"].strip()
                    sections.append(current_section)
                current_section = {
                    "title": el.get("text", "").strip(),
                    "content": "",
                    "page": el.get("page_number", 1)
                }
            else:
                if current_section["title"]:
                    current_section["content"] += " " + el.get("text", "")

    if current_section["title"]:
        current_section["content"] = current_section["content"].strip()
        sections.append(current_section)

    # Filter out short or empty sections
    filtered_sections = []
    for s in sections:
        if not s["title"]:
            continue
        s["content"] = truncate_text(s["content"], MAX_SECTION_CONTENT_CHARS)
        if len(s["content"]) >= MIN_SECTION_CONTENT_CHARS:
            filtered_sections.append(s)
        else:
            logging.debug(f"Ignored short section '{s['title'][:30]}' length={len(s['content'])}")

    logging.info(f"Parsed {len(filtered_sections)} valid sections.")
    return filtered_sections

def parse_pdf_sections(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Full pipeline for PDF -> sections extraction
    """
    try:
        elements = extract_text_elements_from_pdf(pdf_path)
        if not elements:
            logging.warning(f"No elements extracted from {pdf_path}")
            return []
        return parse_sections_from_elements(elements)
    except Exception as e:
        logging.error(f"Failed to parse PDF sections: {e}")
        logging.debug(traceback.format_exc())
        return []

# ================================
# DIETARY FILTERS & PREFERENCES
# ================================

class DietaryFilter:
    """
    Extended dietary filter class.
    """

    def __init__(self):
        self.non_veg_keywords = [
            "chicken", "beef", "pork", "fish", "bacon", "lamb",
            "shrimp", "ham", "meat", "mutton", "gelatin", "anchovy"
        ]
        self.gluten_keywords = [
            "bread", "wheat", "barley", "rye", "flour", "pasta",
            "tortilla", "bun", "roll", "noodle", "cracker", "couscous"
        ]
        self.lactose_keywords = [
            "milk", "cheese", "butter", "cream", "yogurt", "casein"
        ]
        self.soy_keywords = ["soy", "soybean", "tofu"]
        self.nut_keywords = ["almond", "walnut", "cashew", "pecan", "hazelnut", "pistachio"]
        self.eggs_keywords = ["egg", "eggs", "albumin"]
        self.fish_shellfish_keywords = ["crab", "lobster", "shrimp", "prawn", "clam", "mussel", "oyster"]

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        norm_text = normalize_text(text)
        return any(kw in norm_text for kw in keywords)

    def is_vegetarian(self, text: str) -> bool:
        return not self._contains_any(text, self.non_veg_keywords + self.fish_shellfish_keywords + self.eggs_keywords)

    def is_gluten_free(self, text: str) -> bool:
        return not self._contains_any(text, self.gluten_keywords)

    def is_lactose_free(self, text: str) -> bool:
        return not self._contains_any(text, self.lactose_keywords)

    def is_soy_free(self, text: str) -> bool:
        return not self._contains_any(text, self.soy_keywords)

    def is_nut_free(self, text: str) -> bool:
        return not self._contains_any(text, self.nut_keywords)

    def passes_all(self, text: str) -> bool:
        """
        Check if text passes all dietary filters.
        Extend this based on persona dietary preferences.
        """
        return (
            self.is_vegetarian(text)
            and self.is_gluten_free(text)
            and self.is_lactose_free(text)
            and self.is_soy_free(text)
            and self.is_nut_free(text)
        )

# ================================
# SEMANTIC RANKING WITH EMBEDDINGS
# ================================

class SectionRanker:
    """
    Rank sections by semantic similarity to persona + task.
    """

    def __init__(self, model_name: str, min_score_threshold: float):
        logging.info(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.min_score = min_score_threshold

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_tensor=True)

    def rank_sections(self, persona_embed, sections: List[Dict]) -> List[Dict]:
        texts = [f"{s['title']} - {s['content']}" for s in sections]
        embeddings = self.encode(texts)
        scores = util.cos_sim(persona_embed, embeddings)[0]

        ranked_sections = []
        for idx, score in enumerate(scores):
            if score < self.min_score:
                logging.debug(f"Skipping section {sections[idx]['title']} due to low score {score:.3f}")
                continue
            ranked_sections.append({"section": sections[idx], "score": float(score)})

        ranked_sections.sort(key=lambda x: x["score"], reverse=True)
        logging.info(f"Ranked {len(ranked_sections)} sections with score above threshold.")
        return ranked_sections

# ================================
# PERSONA & TASK ANALYSIS
# ================================

def compose_persona_text(persona_data: Dict) -> str:
    """
    Compose a text string from persona role and job to be done, for semantic embedding.
    """
    role = persona_data.get("persona", {}).get("role", "")
    task = persona_data.get("job_to_be_done", {}).get("task", "")
    preferences = persona_data.get("persona", {}).get("preferences", [])
    prefs_text = ", ".join(preferences) if preferences else ""
    composed_text = f"{role}. Task: {task}. Preferences: {prefs_text}"
    logging.debug(f"Composed persona text: {composed_text}")
    return composed_text

# ================================
# OUTPUT FORMATTER
# ================================

def format_output(
    input_documents: List[str],
    persona_role: str,
    job_task: str,
    sections_ranked: List[Dict],
    max_output: int = MAX_SECTIONS_OUTPUT
) -> Dict:
    """
    Format the final output JSON structure.
    """
    extracted = []
    refined = []

    top_sections = sections_ranked[:max_output]

    for idx, sec in enumerate(top_sections):
        sec_data = sec["section"]
        extracted.append({
            "document": sec_data["document"],
            "section_title": sec_data["section_title"],
            "importance_rank": idx + 1,
            "page_number": sec_data["page_number"]
        })
        refined.append({
            "document": sec_data["document"],
            "refined_text": sec_data["refined_text"],
            "page_number": sec_data["page_number"]
        })

    output = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": extracted,
        "subsection_analysis": refined
    }
    return output

# ================================
# MAIN PIPELINE FUNCTION
# ================================

def main():
    print("🚀 Starting persona-driven PDF analysis...")
    logging.info("=== STARTING NEW RUN ===")

    # Load persona.json
    persona_json_path = INPUT_DIR / "persona.json"
    persona_data = safe_open_json(persona_json_path)
    if persona_data is None:
        print("❌ Could not load persona.json. Exiting.")
        return

    persona_text = compose_persona_text(persona_data)
    pdf_files = [doc["filename"] for doc in persona_data.get("documents", []) if is_valid_filename(doc.get("filename", ""))]

    if not pdf_files:
        print("❌ No valid PDF files listed in persona.json.")
        return

    # Initialize components
    dietary_filter = DietaryFilter()
    ranker = SectionRanker(MODEL_NAME, MIN_SCORE_THRESHOLD)
    persona_embedding = ranker.encode([persona_text])[0]

    all_ranked_sections = []

    # Process each PDF
    for pdf_file in pdf_files:
        file_path = INPUT_DIR / pdf_file
        if not file_path.exists():
            logging.warning(f"File missing: {pdf_file}")
            print(f"⚠️ File not found: {pdf_file}")
            continue

        print(f"📄 Processing {pdf_file}...")
        sections = parse_pdf_sections(file_path)
        if not sections:
            logging.warning(f"No sections found in {pdf_file}")
            print(f"⚠️ No sections found in {pdf_file}. Skipping.")
            continue

        ranked_sections = ranker.rank_sections(persona_embedding, sections)
        logging.info(f"{len(ranked_sections)} sections passed ranking threshold in {pdf_file}.")

        # Filter by dietary preferences & extend section data for output
        for r in ranked_sections:
            sec = r["section"]
            score = r["score"]
            full_text = f"{sec['title']} {sec['content']}"
            if dietary_filter.passes_all(full_text):
                all_ranked_sections.append({
                    "document": pdf_file,
                    "section_title": sec["title"],
                    "refined_text": f"{sec['title']} Ingredients and Instructions: {sec['content']}",
                    "page_number": sec["page"],
                    "score": score
                })

    if not all_ranked_sections:
        print("⚠️ No relevant sections found after filtering.")
        logging.warning("No relevant sections found in any document after dietary filtering.")
        return

    # Sort all ranked sections globally and output top N
    all_ranked_sections.sort(key=lambda x: x["score"], reverse=True)
    top_sections = all_ranked_sections[:MAX_SECTIONS_OUTPUT]

    # Format output
    output_data = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona_data.get("persona", {}).get("role", ""),
            "job_to_be_done": persona_data.get("job_to_be_done", {}).get("task", ""),
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": sec["section_title"],
                "importance_rank": idx + 1,
                "page_number": sec["page_number"]
            }
            for idx, sec in enumerate(top_sections)
        ],
        "subsection_analysis": [
            {
                "document": sec["document"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page_number"]
            }
            for sec in top_sections
        ]
    }

    # Save output
    output_file = OUTPUT_DIR / "ranked_output.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Output saved to {output_file}")
        print(f"✅ Completed! Output saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
        print(f"❌ Failed to save output: {e}")

if __name__ == "__main__":
    main()
