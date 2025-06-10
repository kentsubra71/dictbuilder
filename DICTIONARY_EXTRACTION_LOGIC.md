# Dictionary Extraction Logic

This document describes the exact logic used to extract and align English-Spanish sentence pairs from aligned textbook PDFs for the translation dictionary.

## 1. PDF Text Extraction
- **Library:** `pdfplumber` is used to extract words and their bounding boxes from each page of the PDF.

## 2. Page Segmentation
- For each page, words are grouped into lines based on their vertical (y) position, with a small tolerance to account for minor misalignments.
- Each line is reconstructed by sorting words by their horizontal (x) position and joining them with spaces.

## 3. Section Classification
- Each line is classified as one of:
  - **Header:** If its y-position is in the top 10% of the page.
  - **Footer:** If its y-position is in the bottom 10% of the page.
  - **Body:** Otherwise (middle 80% of the page).

## 4. Extraction Logic by Section
- **Headers and Footers:**
  - Each header/footer line is kept as a separate entry in the output (preserving visual grouping).
- **Body:**
  - All body lines are joined into a single string.
  - Whitespace is normalized (all `\r`, `\n`, `\t` replaced with spaces, multiple spaces collapsed).
  - The body text is first split on bullet points (`•`). Each bullet point (including the bullet) is kept as a separate entry.
  - Each non-bullet chunk is further split into sentences using `nltk.sent_tokenize` (which handles English and Spanish sentence boundaries).
  - Each resulting sentence is stripped and added as a separate entry.

## 5. Dictionary Building and Filtering
- **Alignment:**
  - For each page, English and Spanish entries are aligned using a hybrid approach:
    - 1:1 matching by index (first English entry to first Spanish entry, etc.), with fallback to fuzzy and embedding-based matching for unmatched entries.
- **Filtering:**
  - Only pairs where both sides are meaningful are included. A pair is excluded if:
    - Either side is just a bullet, single letter, or very short non-alphanumeric string.
    - Either side is primarily numbers or symbols (e.g., `_1 _1 _3 _3 _1 _1`).
    - Either side is just whitespace or contains no alphabetic characters.
    - Both sides are identical after normalization (case and accent insensitive).
- **Post-processing:**
  - All whitespace is cleaned from both keys and values.
  - Pairs that are identical after cleaning are removed.

## 6. Output
- The final dictionary is saved as a JSON file, mapping English entries to their Spanish counterparts.

---

**This logic ensures that:**
- Headers and footers are preserved as lines.
- Body content is split into logical sentences, not broken by visual line wraps.
- Bullet points are preserved as separate entries.
- Only meaningful, non-numeric, non-identical pairs are included in the dictionary. 