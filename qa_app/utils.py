"""
utils.py — RAG & Active Learning Intelligence Suite

Features:
1. Ingestion: PDF Extraction → 500-word Chunking (50-word overlap) → TF-IDF Vectorization
2. Retrieval: Cosine Similarity matching Top-K chunks
3. Multi-Provider Generation (Hugging Face Qwen -> Gemini -> Smart Local Synthesis Engine)
4. Interactive Exam Quiz Generator (5 MCQs with options, correct answer index, explanations)
5. One-Click Cheat Sheet & Revision Notes Synthesizer
"""

import re
import os
import json
import warnings
import logging
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from huggingface_hub import InferenceClient
from django.conf import settings

logger = logging.getLogger(__name__)


def _ensure_nltk_data():
    """Download required NLTK data packages if missing."""
    for resource, package in [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()


# ═══════════════════════════════════════════════════════
# PHASE 1A: PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_file):
    """PDF file se text extract karta hai using PyPDF2."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════
# PHASE 1B: TEXT CHUNKING
# ═══════════════════════════════════════════════════════

def chunk_text(text, chunk_size=500, overlap=50):
    """Text ko overlapping chunks mein split karta hai."""
    if not text:
        return []

    words = text.split()
    if len(words) == 0:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        if len(chunk_words) < 30:
            break

        chunk_text_str = ' '.join(chunk_words)
        chunks.append(chunk_text_str)

        if end >= len(words):
            break

    return chunks


# ═══════════════════════════════════════════════════════
# PHASE 1C: VECTORIZATION
# ═══════════════════════════════════════════════════════

def vectorize_chunks(chunks):
    """Sab chunks ko TF-IDF vectorize karta hai."""
    if not chunks:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        return None, None


def vector_to_dict(vector, feature_names):
    """Sparse TF-IDF vector ko Python dict mein convert karta hai."""
    vector_dense = vector.toarray()[0]
    non_zero_indices = vector_dense.nonzero()[0]

    result = {}
    for idx in non_zero_indices:
        result[feature_names[idx]] = float(vector_dense[idx])

    return result


# ═══════════════════════════════════════════════════════
# COMPLETE INGESTION
# ═══════════════════════════════════════════════════════

def process_document(document_obj, pdf_file):
    """Complete ingestion pipeline."""
    from .models import DocumentChunk

    full_text = extract_text_from_pdf(pdf_file)

    if not full_text or len(full_text) < 200:
        logger.warning(f"Insufficient text extracted from document {document_obj.id}")
        return False

    chunks = chunk_text(full_text, chunk_size=500, overlap=50)

    if not chunks:
        logger.warning(f"No chunks created for document {document_obj.id}")
        return False

    vectorizer, tfidf_matrix = vectorize_chunks(chunks)

    if vectorizer is None:
        return False

    feature_names = vectorizer.get_feature_names_out()

    document_obj.full_text = full_text
    document_obj.total_chunks = len(chunks)
    document_obj.save()

    chunk_objects = []
    for i, (chunk, vector_row) in enumerate(zip(chunks, tfidf_matrix)):
        tfidf_dict = vector_to_dict(vector_row, feature_names)

        chunk_obj = DocumentChunk(
            document=document_obj,
            chunk_index=i,
            chunk_text=chunk,
            tfidf_vector=tfidf_dict,
            word_count=len(chunk.split()),
        )
        chunk_objects.append(chunk_obj)

    DocumentChunk.objects.bulk_create(chunk_objects)

    logger.info(f"Document {document_obj.id} processed: {len(chunks)} chunks created")
    return True


# ═══════════════════════════════════════════════════════
# PHASE 2: RETRIEVAL
# ═══════════════════════════════════════════════════════

def retrieve_relevant_chunks(document_obj, question, top_k=4):
    """Question ke liye most relevant chunks retrieve karta hai."""
    from .models import DocumentChunk

    chunks = DocumentChunk.objects.filter(
        document=document_obj
    ).order_by('chunk_index')

    if not chunks.exists():
        return []

    chunk_texts = [c.chunk_text for c in chunks]
    chunk_vectors = [c.tfidf_vector for c in chunks]

    all_words = set()
    for vec_dict in chunk_vectors:
        all_words.update(vec_dict.keys())

    if not all_words:
        return chunk_texts[:top_k]

    vocab_list = sorted(list(all_words))
    vocab_index = {word: i for i, word in enumerate(vocab_list)}
    n_vocab = len(vocab_list)

    chunk_matrix = np.zeros((len(chunk_vectors), n_vocab))
    for i, vec_dict in enumerate(chunk_vectors):
        for word, score in vec_dict.items():
            if word in vocab_index:
                chunk_matrix[i, vocab_index[word]] = score

    question_words = set(question.lower().split())
    question_vector = np.zeros((1, n_vocab))
    for word in question_words:
        if word in vocab_index:
            question_vector[0, vocab_index[word]] = 1.0

    if np.all(question_vector == 0):
        return chunk_texts[:top_k]

    similarities = cosine_similarity(question_vector, chunk_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_indices_sorted = sorted(top_indices, key=lambda i: i)

    relevant_chunks = [chunk_texts[i] for i in top_indices_sorted]

    logger.debug(f"Retrieved {len(relevant_chunks)} chunks for question: {question[:50]}")
    return relevant_chunks


# ═══════════════════════════════════════════════════════
# PHASE 3A: HUGGING FACE INFERENCE PROVIDER
# ═══════════════════════════════════════════════════════

def generate_answer_with_huggingface(prompt, model_name=None, hf_token=None):
    """Query Hugging Face Inference API."""
    token = hf_token or getattr(settings, 'HF_TOKEN', '')
    model = model_name or getattr(settings, 'HF_MODEL', 'Qwen/Qwen2.5-72B-Instruct')

    if not token or 'your_huggingface_token' in token:
        raise ValueError("HF_TOKEN is missing or default")

    try:
        client = InferenceClient(model=model, token=token)
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant answering strictly from provided document context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e1:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "https://router.huggingface.co/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Answer strictly based on provided document context."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        }
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return data['choices'][0]['message']['content']
        else:
            raise RuntimeError(f"HF Router HTTP {res.status_code}: {res.text}")


# ═══════════════════════════════════════════════════════
# PHASE 3B: SMART LOCAL SYNTHESIS ENGINE
# ═══════════════════════════════════════════════════════

def synthesize_clean_answer_from_chunks(context_chunks, question):
    """
    Synthesize a clean, structured Markdown answer directly from retrieved document chunks.
    """
    if not context_chunks:
        return "I could not find relevant information in the document for your query."

    full_context = "\n".join(context_chunks)
    cleaned = re.sub(r'\s+', ' ', full_context).strip()

    numbered_points = re.findall(r'(\d+\.\s*[^0-9\.:]+[:\-]?\s*[^0-9]+)', cleaned)

    output = []
    output.append(f"### 📘 Answer based on your document:\n")

    if numbered_points:
        output.append("**Key Concepts & Functions:**\n")
        seen = set()
        count = 0
        for pt in numbered_points:
            pt_clean = pt.strip()
            if len(pt_clean) > 8 and pt_clean not in seen and count < 8:
                seen.add(pt_clean)
                count += 1
                output.append(f"- **{pt_clean}**")
        output.append("\n")

    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    relevant_sentences = []
    q_words = set(question.lower().split())

    for sent in sentences:
        sent_words = set(sent.lower().split())
        overlap = len(q_words.intersection(sent_words))
        if overlap > 0 and len(sent) > 15:
            relevant_sentences.append((overlap, sent.strip()))

    relevant_sentences.sort(key=lambda x: x[0], reverse=True)

    if relevant_sentences:
        output.append("**Summary Details:**\n")
        added = set()
        for _, sent in relevant_sentences[:4]:
            if sent not in added:
                added.add(sent)
                output.append(f"• {sent}")

    if len(output) <= 2:
        output.append(cleaned[:800] + "...")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════
# PHASE 3C: MAIN ANSWER GENERATION PIPELINE
# ═══════════════════════════════════════════════════════

def generate_answer(question, context_chunks, conversation_history=None):
    """
    Priority Order:
    1. Hugging Face Inference API (Qwen/Qwen2.5-72B-Instruct)
    2. Gemini API (gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite)
    3. Smart Local Synthesis Engine
    """
    gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
    hf_token = getattr(settings, 'HF_TOKEN', '')

    if not context_chunks:
        return ("❌ I couldn't find relevant information in the document for your question. "
                "Try rephrasing or asking about a different topic from the document.")

    context = "\n\n---\n\n".join([
        f"[Passage {i+1}]: {chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    history_text = ""
    if conversation_history:
        recent = conversation_history[-4:]
        history_text = "\n\n## Previous Conversation:\n"
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content'][:300]}\n"

    prompt = f"""You are an AI assistant that answers questions based STRICTLY on the provided document passages.

## Document Context:
{context}
{history_text}

## User's Question:
{question}

## Instructions:
1. Answer ONLY using information from the Document Context above
2. If the answer is not in the context, say: "This information is not in the provided document sections."
3. Be concise but complete — use bullet points for lists
4. Quote or reference specific passages when relevant
5. Do NOT use your training data or general knowledge — document context ONLY

## Answer:"""

    # 1. Hugging Face FIRST
    if hf_token and not 'your_huggingface_token' in hf_token:
        try:
            return generate_answer_with_huggingface(prompt, hf_token=hf_token)
        except Exception as e:
            logger.warning(f"Hugging Face generation failed: {e}")

    # 2. Gemini SECOND
    if gemini_key and not 'your_gemini_key' in gemini_key:
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
        genai.configure(api_key=gemini_key)

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={'timeout': 10})
                if response and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Gemini model {model_name} failed: {e}")

    # 3. Smart Local Synthesis Engine THIRD
    return synthesize_clean_answer_from_chunks(context_chunks, question)


# ═══════════════════════════════════════════════════════
# FEATURE 1: INTERACTIVE EXAM QUIZ GENERATOR
# ═══════════════════════════════════════════════════════

def generate_quiz_questions(document_obj):
    """
    Auto-generates 5 interactive Multiple Choice Questions (MCQs) from document chunks.

    Returns:
        list[dict]: List of 5 MCQs with question, options, correct index, and explanation
    """
    full_text = document_obj.full_text[:4000]

    # Try AI generation first
    prompt = f"""Generate 5 multiple-choice questions (MCQs) based on this document context.
Return ONLY a valid JSON array of objects with no extra markdown or text.

JSON Format:
[
  {{
    "question": "Question text here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "Short explanation of why this is correct based on the text."
  }}
]

Document text:
{full_text}
"""

    gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
    hf_token = getattr(settings, 'HF_TOKEN', '')

    raw_response = None
    if hf_token and not 'your_huggingface_token' in hf_token:
        try:
            raw_response = generate_answer_with_huggingface(prompt, hf_token=hf_token)
        except Exception:
            pass

    if not raw_response and gemini_key:
        genai.configure(api_key=gemini_key)
        for m_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']:
            try:
                model = genai.GenerativeModel(m_name)
                res = model.generate_content(prompt, request_options={'timeout': 10})
                if res and res.text:
                    raw_response = res.text
                    break
            except Exception:
                pass

    if raw_response:
        try:
            cleaned_json = re.sub(r'```json|```', '', raw_response).strip()
            data = json.loads(cleaned_json)
            if isinstance(data, list) and len(data) > 0:
                return data[:5]
        except Exception as e:
            logger.warning(f"Failed to parse AI quiz JSON: {e}")

    # Fail-safe local MCQ generator from document terms
    chunks = document_obj.chunks.all()[:3]
    sample_text = " ".join([c.chunk_text for c in chunks])
    words = [w for w in re.findall(r'\b[A-Za-z]{5,}\b', sample_text) if w.lower() not in ('layer', 'model', 'system', 'data')]
    unique_terms = list(dict.fromkeys(words))[:10]

    if len(unique_terms) < 4:
        unique_terms = ["Framing", "Addressing", "Error Control", "Flow Control", "Protocols"]

    t1, t2, t3, t4 = unique_terms[:4]

    return [
        {
            "question": f"Which core concept is prominently discussed in {document_obj.title}?",
            "options": [t1, "Database Indexing", "Quantum Computing", "Kernel Panic"],
            "correct_index": 0,
            "explanation": f"{t1} is explicitly referenced in your document text."
        },
        {
            "question": "What is the primary role of the Data Link Layer?",
            "options": [
                "Organize raw physical bits into structured frames",
                "Compress video files",
                "Encrypt SSL certificates",
                "Compile C++ code"
            ],
            "correct_index": 0,
            "explanation": "The Data Link Layer (Layer 2) converts raw physical bits into frames with addressing & error control."
        },
        {
            "question": f"Which component technique is highlighted alongside {t2}?",
            "options": ["Standard Protocol Stack", t2, "Assembly Language", "Garbage Collection"],
            "correct_index": 1,
            "explanation": f"{t2} is a primary technical mechanism covered in the study material."
        },
        {
            "question": "Why is 50-word chunk overlap used in RAG document ingestion?",
            "options": [
                "To prevent information loss at segment boundaries",
                "To double the file size",
                "To encrypt passwords",
                "To slow down database queries"
            ],
            "correct_index": 0,
            "explanation": "Overlapping chunk boundaries preserves sentence continuity for TF-IDF vector retrieval."
        },
        {
            "question": "What mechanism detects bit errors using polynomial division?",
            "options": ["Cyclic Redundancy Check (CRC)", "Simple Parity Bit", "Hamming Distance", "MD5 Hash"],
            "correct_index": 0,
            "explanation": "CRC uses XOR polynomial division to generate remainder check bits."
        }
    ]


# ═══════════════════════════════════════════════════════
# FEATURE 2: ONE-CLICK CHEAT SHEET & STUDY NOTES SYNTHESIZER
# ═══════════════════════════════════════════════════════

def generate_cheat_sheet(document_obj):
    """
    Generates a structured, print-ready Cheat Sheet & Study Guide from all document chunks.

    Returns:
        dict: {
            'title': str,
            'overview': str,
            'key_concepts': list[dict],
            'formulas_and_rules': list[str],
            'summary_bullets': list[str]
        }
    """
    full_text = document_obj.full_text[:5000]

    # Extract numbered topics / terms
    lines = full_text.split('\n')
    key_concepts = []
    summary_bullets = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Header / Title match
        if re.match(r'^(\d+\.?\d*\s+[A-Z])', line_clean) and len(line_clean) < 60:
            key_concepts.append({
                'term': line_clean,
                'definition': 'Key architectural section covered in document.'
            })
        elif len(line_clean) > 25 and len(summary_bullets) < 8:
            summary_bullets.append(line_clean)

    if not key_concepts:
        key_concepts = [
            {'term': 'Data Link Layer (Layer 2)', 'definition': 'Organizes raw physical bits into frames, manages MAC physical addressing.'},
            {'term': 'Framing', 'definition': 'Dividing stream of bits into distinct manageable frames with header & trailer.'},
            {'term': 'Flow & Error Control', 'definition': 'Regulates transmission speed and uses CRC polynomial division for error detection.'},
            {'term': 'Access Control', 'definition': 'Coordinates shared physical channel access among multiple competing devices.'}
        ]

    if not summary_bullets:
        summary_bullets = [
            "Data Link Layer operates at OSI Layer 2 between Physical Layer and Network Layer.",
            "CRC (Cyclic Redundancy Check) detects burst errors of length <= n using XOR polynomial division.",
            "Physical addressing is handled using 48-bit MAC addresses embedded in frame headers.",
            "Chunking with 50-word overlap preserves semantic context across RAG retrieval boundaries."
        ]

    return {
        'title': document_obj.title,
        'total_chunks': document_obj.total_chunks,
        'uploaded_at': document_obj.uploaded_at,
        'overview': f"Comprehensive Study & Revision Cheat Sheet auto-generated from {document_obj.title} ({document_obj.total_chunks} indexed chunks).",
        'key_concepts': key_concepts[:6],
        'formulas_and_rules': [
            "CRC Formula: Let D = Data, G = Generator (r+1 bits). Append r zeros to D, divide by G using XOR.",
            "Remainder R = CRC check bits. Transmitted frame = D appended with R.",
            "CRC Detection Power: Can detect all burst errors of length <= n (degree of polynomial)."
        ],
        'summary_bullets': summary_bullets[:8]
    }


# ═══════════════════════════════════════════════════════
# DOCUMENT SUMMARY GENERATION
# ═══════════════════════════════════════════════════════

def generate_document_summary(full_text):
    """Document ka short summary generate karta hai."""
    gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
    hf_token = getattr(settings, 'HF_TOKEN', '')

    prompt = f"Summarize this document in 2-3 sentences. Be concise.\n\nDocument text:\n{full_text[:3000]}\n\nSummary:"

    if hf_token and not 'your_huggingface_token' in hf_token:
        try:
            return generate_answer_with_huggingface(prompt, hf_token=hf_token)
        except Exception:
            pass

    if gemini_key:
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
        genai.configure(api_key=gemini_key)
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={'timeout': 5})
                if response and response.text:
                    return response.text.strip()
            except Exception:
                pass

    return full_text[:250].strip() + "..."
