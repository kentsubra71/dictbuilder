import pypdfium2
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import process, fuzz
import unicodedata
from sentence_transformers import SentenceTransformer, util

ENGLISH_PDF = 'sample/unit1_grade6_english.pdf'
SPANISH_PDF = 'sample/unit1_grade6_spanish.pdf'
OUTPUT_JSON = 'translation_dictionary.json'

nltk.download('punkt')


def extract_text_by_page_lines(pdf_path):
    pdf = pypdfium2.PdfDocument(pdf_path)
    pages_lines = []
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_textpage().get_text_range()
        # Split by lines, strip whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        pages_lines.append(lines)
    return pages_lines


def split_sentences(pages_text, language='english'):
    sentences = []
    for page in pages_text:
        sentences.extend(sent_tokenize(page, language=language))
    return [s.strip() for s in sentences if s.strip()]


def should_ignore(line):
    # Add more patterns as needed
    ignore_patterns = [
        r'copyright',
        r'mcgraw hill',
        r'all rights reserved',
        r'page \d+',
        r'^\s*$',  # empty lines
    ]
    line_lower = line.lower()
    for pat in ignore_patterns:
        if re.search(pat, line_lower):
            return True
    return False


def normalize(text):
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text


def is_length_similar(a, b, min_ratio=0.5, max_ratio=2.0):
    len_a = len(a)
    len_b = len(b)
    if len_b == 0:
        return False
    ratio = len_a / len_b
    return min_ratio <= ratio <= max_ratio


def is_numbers_symbols_only(s):
    # Returns True if s contains no alphabetic characters
    return not re.search(r'[a-zA-Z]', s)


def build_dictionary_hybrid(english_pages_lines, spanish_pages_lines, threshold=0.7, max_window=4, fuzzy_threshold=90):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    dictionary = {}
    num_pages = min(len(english_pages_lines), len(spanish_pages_lines))
    for page_idx in range(num_pages):
        print(f"\n[Progress] Processing page {page_idx+1}/{num_pages}...")
        en_lines = english_pages_lines[page_idx]
        es_lines = spanish_pages_lines[page_idx]
        en_used = set()
        es_used = set()

        # Precompute all English line embeddings
        en_embeddings = model.encode(en_lines, convert_to_tensor=True)
        # Precompute all possible Spanish chunks and their embeddings
        es_chunks = []
        es_chunk_indices = []  # (start, window_size)
        for w in range(1, max_window+1):
            for j in range(len(es_lines) - w + 1):
                chunk = ' '.join(es_lines[j:j+w])
                es_chunks.append(chunk)
                es_chunk_indices.append((j, w))
        es_embeddings = model.encode(es_chunks, convert_to_tensor=True)

        # English to Spanish matching
        for i, en_line in enumerate(en_lines):
            if i % 10 == 0:
                print(f"  [EN->ES] Page {page_idx+1}: Line {i+1}/{len(en_lines)}")
            if i in en_used:
                continue
            best_j, best_w, best_chunk, best_type = None, None, None, None
            # 1. Try normalized exact match
            for idx, (j, w) in enumerate(es_chunk_indices):
                if any(x in es_used for x in range(j, j+w)):
                    continue
                chunk = es_chunks[idx]
                if normalize(en_line) == normalize(chunk):
                    best_j, best_w, best_chunk, best_type = j, w, chunk, 'exact'
                    break
            # 2. Try fuzzy match
            if best_chunk is None:
                for idx, (j, w) in enumerate(es_chunk_indices):
                    if any(x in es_used for x in range(j, j+w)):
                        continue
                    chunk = es_chunks[idx]
                    if is_length_similar(en_line, chunk):
                        score = fuzz.ratio(en_line, chunk)
                        if score >= fuzzy_threshold:
                            best_j, best_w, best_chunk, best_type = j, w, chunk, 'fuzzy'
                            break
            # 3. Use embeddings as fallback
            if best_chunk is None:
                en_emb = en_embeddings[i]
                scores = util.cos_sim(en_emb, es_embeddings)[0].cpu().numpy()
                for idx, (j, w) in enumerate(es_chunk_indices):
                    if any(x in es_used for x in range(j, j+w)):
                        scores[idx] = -1
                    elif not is_length_similar(en_line, es_chunks[idx]):
                        scores[idx] = -1
                best_idx = scores.argmax()
                best_score = scores[best_idx]
                if best_score >= threshold:
                    best_j, best_w = es_chunk_indices[best_idx]
                    best_chunk = es_chunks[best_idx]
                    best_type = 'embedding'
            # Save match if found
            if best_chunk is not None:
                if should_ignore(en_line) or should_ignore(best_chunk):
                    continue
                dictionary[en_line] = best_chunk
                for idx in range(best_j, best_j+best_w):
                    es_used.add(idx)
                en_used.add(i)

        # Precompute all Spanish line embeddings
        es_embeddings_lines = model.encode(es_lines, convert_to_tensor=True)
        # Precompute all possible English chunks and their embeddings
        en_chunks = []
        en_chunk_indices = []  # (start, window_size)
        for w in range(1, max_window+1):
            for i in range(len(en_lines) - w + 1):
                chunk = ' '.join(en_lines[i:i+w])
                en_chunks.append(chunk)
                en_chunk_indices.append((i, w))
        en_embeddings_chunks = model.encode(en_chunks, convert_to_tensor=True)

        # Spanish to English matching (for unmatched Spanish lines)
        for j, es_line in enumerate(es_lines):
            if j in es_used:
                continue
            if j % 10 == 0:
                print(f"  [ES->EN] Page {page_idx+1}: Line {j+1}/{len(es_lines)}")
            best_i, best_w, best_chunk, best_type = None, None, None, None
            # 1. Try normalized exact match
            for idx, (i, w) in enumerate(en_chunk_indices):
                if any(x in en_used for x in range(i, i+w)):
                    continue
                chunk = en_chunks[idx]
                if normalize(es_line) == normalize(chunk):
                    best_i, best_w, best_chunk, best_type = i, w, chunk, 'exact'
                    break
            # 2. Try fuzzy match
            if best_chunk is None:
                for idx, (i, w) in enumerate(en_chunk_indices):
                    if any(x in en_used for x in range(i, i+w)):
                        continue
                    chunk = en_chunks[idx]
                    if is_length_similar(es_line, chunk):
                        score = fuzz.ratio(es_line, chunk)
                        if score >= fuzzy_threshold:
                            best_i, best_w, best_chunk, best_type = i, w, chunk, 'fuzzy'
                            break
            # 3. Use embeddings as fallback
            if best_chunk is None:
                es_emb = es_embeddings_lines[j]
                scores = util.cos_sim(es_emb, en_embeddings_chunks)[0].cpu().numpy()
                for idx, (i, w) in enumerate(en_chunk_indices):
                    if any(x in en_used for x in range(i, i+w)):
                        scores[idx] = -1
                    elif not is_length_similar(es_line, en_chunks[idx]):
                        scores[idx] = -1
                best_idx = scores.argmax()
                best_score = scores[best_idx]
                if best_score >= threshold:
                    best_i, best_w = en_chunk_indices[best_idx]
                    best_chunk = en_chunks[best_idx]
                    best_type = 'embedding'
            # Save match if found
            if best_chunk is not None:
                if should_ignore(es_line) or should_ignore(best_chunk):
                    continue
                if best_chunk not in dictionary:
                    dictionary[best_chunk] = es_line
                for idx in range(best_i, best_i+best_w):
                    en_used.add(idx)
                es_used.add(j)
    return dictionary


def main():
    print('Extracting English PDF by page/line...')
    english_pages_lines = extract_text_by_page_lines(ENGLISH_PDF)
    print('Extracting Spanish PDF by page/line...')
    spanish_pages_lines = extract_text_by_page_lines(SPANISH_PDF)

    # Log all lines for page 24 (index 23)
    page_idx = 23
    print(f'\n--- English lines on page 24 ---')
    for i, line in enumerate(english_pages_lines[page_idx]):
        print(f'EN[{i}]: {repr(line)}')
    print(f'\n--- Spanish lines on page 24 ---')
    for i, line in enumerate(spanish_pages_lines[page_idx]):
        print(f'ES[{i}]: {repr(line)}')
    print(f'--- End of page 24 lines ---\n')

    print(f'Aligning {len(english_pages_lines)} English pages and {len(spanish_pages_lines)} Spanish pages with hybrid alignment.')
    dictionary = build_dictionary_hybrid(english_pages_lines, spanish_pages_lines)

    # Filter out pairs that are identical and only numbers/symbols/whitespace
    filtered_dictionary = {
        k: v for k, v in dictionary.items()
        if not (normalize(k) == normalize(v) and is_numbers_symbols_only(k) and is_numbers_symbols_only(v))
    }

    print(f'Saving dictionary with {len(filtered_dictionary)} entries to {OUTPUT_JSON}...')
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(filtered_dictionary, f, ensure_ascii=False, indent=2)
    print('Done!')


if __name__ == '__main__':
    main() 