import re
import platform
import shutil
import numpy as np
from PIL import Image
import pytesseract
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Auto-detect Tesseract path based on OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif platform.system() == "Darwin":  # macOS
    # Apple Silicon or Intel Mac
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
    elif shutil.which("/opt/homebrew/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    else:
        pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
# Linux: usually in PATH, no need to set explicitly

def extract_text_from_pdf(file_content):
    """
    Extract text from PDF - tries text extraction first, falls back to OCR.
    """
    from io import BytesIO

    pdf_stream = BytesIO(file_content)
    reader = PdfReader(pdf_stream)
    print(f"PDF pages: {len(reader.pages)}")

    # First, try extracting embedded text
    text = ""
    for i, p in enumerate(reader.pages):
        page_text = p.extract_text()
        if page_text:
            text += page_text + " "

    # If we got meaningful text, return it
    if text.strip() and len(text.strip()) > 50:
        print(f"Extracted embedded text: {len(text)} chars")
        return text.strip()

    # Otherwise, use OCR on PDF pages
    print("No embedded text found, using OCR...")
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_content)
        print(f"Converted PDF to {len(images)} images")

        ocr_text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang="eng+ind")
            print(f"OCR Page {i+1}: {len(page_text)} chars")
            ocr_text += page_text + " "

        return ocr_text.strip()
    except ImportError:
        print("pdf2image not installed. Install with: pip install pdf2image")
        print("Also install poppler: brew install poppler (macOS)")
        return ""
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


def extract_text(text_input, file):
    if text_input and text_input.strip():
        return text_input

    if file and file.filename:
        name = file.filename.lower()
        print(f"Processing file: {name}")

        if name.endswith(".pdf"):
            try:
                from io import BytesIO
                file_content = file.read()
                print(f"PDF file size: {len(file_content)} bytes")
                return extract_text_from_pdf(file_content)
            except Exception as e:
                print(f"PDF extraction error: {e}")
                import traceback
                traceback.print_exc()
                return ""

        if name.endswith((".png", ".jpg", ".jpeg")):
            try:
                from io import BytesIO
                file_content = file.read()
                img_stream = BytesIO(file_content)
                img = Image.open(img_stream)
                return pytesseract.image_to_string(img, lang="eng+ind")
            except Exception as e:
                print(f"Image OCR error: {e}")
                import traceback
                traceback.print_exc()
                return ""

    print(f"No file or unsupported format")
    return ""

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

def text_similarity(t1, t2):
    """Basic TF-IDF cosine similarity."""
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([t1, t2])
    return cosine_similarity(matrix[0], matrix[1])[0][0]


def find_matching_phrases(t1, t2, n=3):
    """Find matching n-gram phrases between two texts."""
    def get_ngrams(text, n):
        words = text.split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    ngrams1 = get_ngrams(t1, n)
    ngrams2 = get_ngrams(t2, n)
    matches = ngrams1 & ngrams2
    return list(matches)[:10]  # Return top 10 matches


def plagiarism_detection(t1, t2):
    """
    Comprehensive plagiarism detection with detailed breakdown.
    """
    words1 = t1.split()
    words2 = t2.split()

    if not words1 or not words2:
        return None

    # 1. TF-IDF Cosine Similarity (overall semantic similarity)
    tfidf_score = text_similarity(t1, t2) * 100

    # 2. Jaccard Similarity (word overlap)
    set1 = set(words1)
    set2 = set(words2)
    jaccard = len(set1 & set2) / len(set1 | set2) * 100 if (set1 | set2) else 0

    # 3. N-gram similarity (phrase matching)
    trigrams1 = set(' '.join(words1[i:i+3]) for i in range(len(words1)-2))
    trigrams2 = set(' '.join(words2[i:i+3]) for i in range(len(words2)-2))
    ngram_sim = len(trigrams1 & trigrams2) / max(len(trigrams1 | trigrams2), 1) * 100

    # 4. Find matching phrases
    matching_phrases = find_matching_phrases(t1, t2, 4)

    # 5. Longest Common Subsequence ratio (simplified)
    common_words = set1 & set2
    lcs_ratio = len(common_words) / max(len(set1), len(set2)) * 100 if max(len(set1), len(set2)) > 0 else 0

    # Calculate weighted overall score
    overall_score = round(
        tfidf_score * 0.4 +      # Semantic similarity
        jaccard * 0.25 +          # Word overlap
        ngram_sim * 0.25 +        # Phrase matching
        lcs_ratio * 0.1,          # Common vocabulary
        1
    )

    # Interpretation
    if overall_score >= 80:
        interpretation = "High Similarity - Likely Plagiarism"
        risk_level = "high"
    elif overall_score >= 60:
        interpretation = "Moderate Similarity - Review Recommended"
        risk_level = "medium"
    elif overall_score >= 40:
        interpretation = "Low Similarity - Some Common Content"
        risk_level = "low"
    else:
        interpretation = "Minimal Similarity - Likely Original"
        risk_level = "safe"

    return {
        "overall_score": overall_score,
        "interpretation": interpretation,
        "risk_level": risk_level,
        "breakdown": {
            "semantic_similarity": {
                "score": round(tfidf_score, 1),
                "label": "Semantic Similarity (TF-IDF)",
                "description": "How similar the meaning and context are"
            },
            "word_overlap": {
                "score": round(jaccard, 1),
                "label": "Word Overlap (Jaccard)",
                "description": "Percentage of shared unique words"
            },
            "phrase_matching": {
                "score": round(ngram_sim, 1),
                "label": "Phrase Matching (N-gram)",
                "description": "Similar word sequences found"
            },
            "vocabulary_overlap": {
                "score": round(lcs_ratio, 1),
                "label": "Vocabulary Overlap",
                "description": "Common vocabulary ratio"
            }
        },
        "matching_phrases": matching_phrases,
        "stats": {
            "text1": {"words": len(words1), "unique_words": len(set1)},
            "text2": {"words": len(words2), "unique_words": len(set2)},
            "common_words": len(common_words)
        }
    }

def writing_style_similarity(t1, t2):
    """
    Compare writing styles and return detailed breakdown.
    """
    def extract_features(text):
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        word_count = len(words)
        unique_words = len(set(words))
        sentence_count = len(sentences) if sentences else 1

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0,
            "vocabulary_richness": unique_words / word_count if word_count > 0 else 0,
            "comma_ratio": text.count(",") / word_count if word_count > 0 else 0,
            "semicolon_ratio": text.count(";") / word_count if word_count > 0 else 0,
            "question_ratio": text.count("?") / sentence_count if sentence_count > 0 else 0,
            "exclamation_ratio": text.count("!") / sentence_count if sentence_count > 0 else 0,
        }

    def compare_feature(v1, v2):
        """Calculate similarity between two values (0-100)."""
        if v1 == 0 and v2 == 0:
            return 100.0
        max_val = max(abs(v1), abs(v2))
        if max_val == 0:
            return 100.0
        diff = abs(v1 - v2) / max_val
        return max(0, (1 - diff) * 100)

    f1 = extract_features(t1)
    f2 = extract_features(t2)

    breakdown = {
        "sentence_structure": {
            "score": round(compare_feature(f1["avg_sentence_length"], f2["avg_sentence_length"]), 1),
            "text1": round(f1["avg_sentence_length"], 1),
            "text2": round(f2["avg_sentence_length"], 1),
            "label": "Avg words per sentence"
        },
        "word_complexity": {
            "score": round(compare_feature(f1["avg_word_length"], f2["avg_word_length"]), 1),
            "text1": round(f1["avg_word_length"], 2),
            "text2": round(f2["avg_word_length"], 2),
            "label": "Avg word length"
        },
        "vocabulary_richness": {
            "score": round(compare_feature(f1["vocabulary_richness"], f2["vocabulary_richness"]), 1),
            "text1": round(f1["vocabulary_richness"] * 100, 1),
            "text2": round(f2["vocabulary_richness"] * 100, 1),
            "label": "Unique words %"
        },
        "punctuation_style": {
            "score": round((
                compare_feature(f1["comma_ratio"], f2["comma_ratio"]) +
                compare_feature(f1["semicolon_ratio"], f2["semicolon_ratio"])
            ) / 2, 1),
            "text1": round(f1["comma_ratio"] * 100, 2),
            "text2": round(f2["comma_ratio"] * 100, 2),
            "label": "Comma usage %"
        },
        "expression_style": {
            "score": round((
                compare_feature(f1["question_ratio"], f2["question_ratio"]) +
                compare_feature(f1["exclamation_ratio"], f2["exclamation_ratio"])
            ) / 2, 1),
            "text1": round(f1["question_ratio"] * 100, 1),
            "text2": round(f2["question_ratio"] * 100, 1),
            "label": "Question sentences %"
        }
    }

    overall_score = round(np.mean([b["score"] for b in breakdown.values()]), 1)

    if overall_score >= 80:
        interpretation = "Very Similar Style"
    elif overall_score >= 60:
        interpretation = "Somewhat Similar"
    elif overall_score >= 40:
        interpretation = "Moderately Different"
    else:
        interpretation = "Very Different Style"

    return {
        "overall_score": overall_score,
        "breakdown": breakdown,
        "interpretation": interpretation,
        "stats": {
            "text1": {"words": f1["word_count"], "sentences": f1["sentence_count"]},
            "text2": {"words": f2["word_count"], "sentences": f2["sentence_count"]}
        }
    }
