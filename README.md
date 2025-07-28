# 🧠 Adobe Hackathon Round 2 – Persona-Driven Document Intelligence

This project is a winning-level solution for **Challenge 1B** of the Adobe India Hackathon 2025. It intelligently extracts and ranks the most relevant sections from a collection of PDFs based on a **persona** and their **job-to-be-done**, with optional filtering (e.g., dietary preferences), and outputs structured summaries in the required format.

---

## 🚀 Features

- ✅ **Persona-based semantic ranking** using SentenceTransformer
- ✅ **PDF parsing** with heading detection and section segmentation
- ✅ **Dietary and preference filtering** (e.g., vegetarian, gluten-free)
- ✅ **Heuristic-based heading extraction**
- ✅ Outputs fully structured `ranked_output.json`
- ✅ Fast, offline, and 100% CPU-compatible (<200MB)
- ✅ Clean, modular pipeline design for easy extension

---

## 📦 Directory Structure

```
.
├── input/
│   ├── persona.json
│   └── *.pdf
├── output/
│   └── ranked_output.json
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Build (Docker)

```bash
docker build --platform linux/amd64 -t persona_pdf_analyzer:challenge1b .
```

---

## ▶️ Run

```bash
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none persona_pdf_analyzer:challenge1b
```

---

## 📥 Input Format

### persona.json
```json
{
  "persona": {
    "role": "Travel Planner",
    "preferences": ["vegetarian", "gluten-free"]
  },
  "job_to_be_done": {
    "task": "Plan a healthy culinary trip across South of France"
  },
  "documents": [
    { "filename": "South of France - Cuisine.pdf" },
    { "filename": "South of France - Culture.pdf" }
  ]
}
```

---

## 📤 Output Format

### output/ranked_output.json

```json
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "...",
    "processing_timestamp": "..."
  },
  "extracted_sections": [
    {
      "document": "...",
      "section_title": "...",
      "importance_rank": 1,
      "page_number": ...
    }
  ],
  "subsection_analysis": [
    {
      "document": "...",
      "refined_text": "...",
      "page_number": ...
    }
  ]
}
```

---

## 🧠 How It Works (Under the Hood)

1. **PDF Parsing** with PyMuPDF to extract heading-based sections.
2. **Persona Embedding** using SentenceTransformer (`all-MiniLM-L6-v2`)
3. **Semantic Similarity Scoring** using cosine similarity.
4. **Optional Filtering** based on keywords (e.g., meat, nuts, gluten).
5. **Global Section Ranking** across all PDFs.
6. **Top N Sections** saved with summaries and metadata.

---

## 📌 Constraints Met

- ✅ Runs locally, no API/web calls
- ✅ < 10s for <50 pages
- ✅ CPU-only, <200MB model
- ✅ Structured JSON output format

---

## 🤝 Credits

Inspired by real-world document intelligence needs in content personalization and AI search systems.

---
