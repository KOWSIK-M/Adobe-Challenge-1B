🎯 Objective
The goal is to build a persona-driven intelligent document analysis system that:
1. Extracts sections from a set of PDFs.
2. Ranks them based on relevance to a given persona and task.
3. Filters sections based on preferences (e.g., dietary filters).
4. Outputs a structured JSON with the most relevant and refined content.

📂 Input
1. persona.json: Contains:
  Persona description (role, preferences)
  Job to be done
  List of document filenames.
2. 3–10 related PDFs.

⚙️ Pipeline Overview
1. Load Persona Definition
2. Parse and Clean Each PDF
3. Identify Sections using Heading Heuristics
4. Filter Based on Preferences
5. Rank Sections via Semantic Similarity
6. Format and Output Top Sections

🔍 Detailed Breakdown

1. Persona Encoding
Combines:
   Role (e.g., Travel Planner)
   Task (e.g., Plan a vegetarian trip)
   Preferences (e.g., vegetarian, nut-free)
Embeds the resulting text using SentenceTransformer (all-MiniLM-L6-v2).

2. PDF Text Extraction
Uses PyMuPDF to extract:
   Font size
   Font name
   Text spans
Detects headings heuristically:
   Font size > body size + 1.5
   Or different font family at body size

3. Section Parsing
Each heading starts a new section.
Content is accumulated until the next heading.
Filters:
   Minimum characters
   Max content size truncated

4. Preference Filtering
Applies dietary filters (e.g., vegetarian, nut-free, gluten-free).
Filters out sections that mention forbidden ingredients.

5. Semantic Ranking
Concatenates title + content of each section.
Encodes using SentenceTransformer.
Computes cosine similarity with persona embedding.
Sections below a similarity threshold are discarded.

6. Global Ranking & Output
Ranked sections from all PDFs are:
   Sorted by similarity score.
   Filtered by dietary preferences.
   Top 5 selected and structured into:
    extracted_sections: with title, doc, page, and rank.
    subsection_analysis: refined content.

🧪 Model & Thresholds
Model : sentence-transformers/all-MiniLM-L6-v2
Output: Max 5 top sections (MAX_SECTIONS_OUTPUT)
Section constraints:
    Max 3000 chars
    Min 20 chars
    Score threshold: 0.0 (but sorted, so only top few are kept)

🧾 Output
A final ranked_output.json with:
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job_to_be_done": "...",
    "processing_timestamp": "..."
  },
  "extracted_sections": [...],
  "subsection_analysis": [...]
}

✅ Strengths
Modular, reusable pipeline.
Heuristic + embedding hybrid approach.
Persona-awareness with dietary and semantic filtering.
Lightweight model suitable for <200MB CPU-only constraints.

📦 Files Involved
main.py → Main pipeline controller.
persona.json → Input configuration file.
*.pdf → Input documents.
ranked_output.json → Final result.