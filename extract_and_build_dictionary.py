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


def extract_sentences_by_page(pdf_path, language='english'):
    pdf = pypdfium2.PdfDocument(pdf_path)
    pages_sentences = []
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_textpage().get_text_range()
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Split on bullet points first
        bullet_chunks = re.split(r'(•)', text)
        sentences = []
        i = 0
        while i < len(bullet_chunks):
            if bullet_chunks[i] == '•':
                # Combine bullet with following text
                if i + 1 < len(bullet_chunks):
                    chunk = '•' + bullet_chunks[i + 1]
                    sentences.append(chunk.strip())
                    i += 2
                else:
                    i += 1
            else:
                # Tokenize non-bullet text
                sentences.extend([s.strip() for s in sent_tokenize(bullet_chunks[i], language=language) if s.strip()])
                i += 1
        pages_sentences.append([s for s in sentences if s])
    return pages_sentences


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
    # Remove all spaces and underscores for the check
    s = s.replace(' ', '').replace('_', '')
    return not re.search(r'[a-zA-Z]', s)


def build_dictionary_1to1_with_fallback(english_pages_sentences, spanish_pages_sentences, threshold=0.7, fuzzy_threshold=90):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    dictionary = {}
    num_pages = min(len(english_pages_sentences), len(spanish_pages_sentences))
    for page_idx in range(num_pages):
        en_sentences = english_pages_sentences[page_idx]
        es_sentences = spanish_pages_sentences[page_idx]
        en_used = set()
        es_used = set()
        # 1:1 matching by index
        for i in range(min(len(en_sentences), len(es_sentences))):
            en_sentence = en_sentences[i]
            es_sentence = es_sentences[i]
            # Try exact match
            if normalize(en_sentence) == normalize(es_sentence):
                dictionary[en_sentence] = es_sentence
                en_used.add(i)
                es_used.add(i)
                continue
            # Try fuzzy match
            if is_length_similar(en_sentence, es_sentence):
                score = fuzz.ratio(en_sentence, es_sentence)
                if score >= fuzzy_threshold:
                    dictionary[en_sentence] = es_sentence
                    en_used.add(i)
                    es_used.add(i)
                    continue
            # Try embeddings
            en_emb = model.encode(en_sentence, convert_to_tensor=True)
            es_emb = model.encode(es_sentence, convert_to_tensor=True)
            score = util.cos_sim(en_emb, es_emb)[0][0].item()
            if score >= threshold:
                dictionary[en_sentence] = es_sentence
                en_used.add(i)
                es_used.add(i)
        # Fallback: try to match any remaining unmatched English sentences to any unmatched Spanish sentences using hybrid logic
        unmatched_en = [i for i in range(len(en_sentences)) if i not in en_used]
        unmatched_es = [i for i in range(len(es_sentences)) if i not in es_used]
        if unmatched_en and unmatched_es:
            en_embs = model.encode([en_sentences[i] for i in unmatched_en], convert_to_tensor=True)
            es_embs = model.encode([es_sentences[j] for j in unmatched_es], convert_to_tensor=True)
            sim_matrix = util.cos_sim(en_embs, es_embs).cpu().numpy()
            for idx_en, i in enumerate(unmatched_en):
                best_idx_es = sim_matrix[idx_en].argmax()
                best_score = sim_matrix[idx_en][best_idx_es]
                j = unmatched_es[best_idx_es]
                if best_score >= threshold:
                    dictionary[en_sentences[i]] = es_sentences[j]
                    en_used.add(i)
                    es_used.add(j)
    return dictionary


def clean_whitespace(s):
    # Replace all \r, \n, \t with a single space, then collapse multiple spaces
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r' +', ' ', s)
    return s.strip()


def main():
    print('Extracting English PDF by page/sentence...')
    english_pages_sentences = extract_sentences_by_page(ENGLISH_PDF, language='english')
    print('Extracting Spanish PDF by page/sentence...')
    spanish_pages_sentences = extract_sentences_by_page(SPANISH_PDF, language='spanish')

    # Debug: Print raw and split Spanish text for page 2 (index 1)
    pdf = pypdfium2.PdfDocument(SPANISH_PDF)
    page = pdf[1]
    raw_text = page.get_textpage().get_text_range().replace('\n', ' ').replace('\r', ' ')
    print('\n--- Raw Spanish text on page 2 ---')
    print(repr(raw_text))
    print('\n--- Spanish sentences on page 2 ---')
    for i, line in enumerate(spanish_pages_sentences[1]):
        print(f'ES[{i}]: {repr(line)}')
    print(f'--- End of page 2 sentences ---\n')

    # Log all sentences for page 24 (index 23)
    page_idx = 23
    print(f'\n--- English sentences on page 24 ---')
    for i, line in enumerate(english_pages_sentences[page_idx]):
        print(f'EN[{i}]: {repr(line)}')
    print(f'\n--- Spanish sentences on page 24 ---')
    for i, line in enumerate(spanish_pages_sentences[page_idx]):
        print(f'ES[{i}]: {repr(line)}')
    print(f'--- End of page 24 sentences ---\n')

    print(f'Aligning {len(english_pages_sentences)} English pages and {len(spanish_pages_sentences)} Spanish pages with 1:1 matching.')
    dictionary = build_dictionary_1to1_with_fallback(english_pages_sentences, spanish_pages_sentences)

    # Filter out pairs that are identical and only numbers/symbols/whitespace
    filtered_dictionary = {
        k: v for k, v in dictionary.items()
        if not (normalize(k) == normalize(v) and is_numbers_symbols_only(k) and is_numbers_symbols_only(v))
    }

    # Clean up \n, \r, \t from both keys and values
    cleaned_dictionary = {clean_whitespace(k): clean_whitespace(v) for k, v in filtered_dictionary.items()}

    # Remove pairs that are identical after cleaning
    final_dictionary = {k: v for k, v in cleaned_dictionary.items() if k != v}

    print(f'Saving dictionary with {len(final_dictionary)} entries to {OUTPUT_JSON}...')
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dictionary, f, ensure_ascii=False, indent=2)
    print('Done!')


if __name__ == '__main__':
    main() 