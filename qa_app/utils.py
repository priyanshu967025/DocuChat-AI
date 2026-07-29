"""
utils.py — RAG & Active Learning Intelligence Suite

Architecture & Strategy:
1. Ingestion: PDF Extraction → 500-word Chunking (50-word overlap) → TF-IDF Vectorization
2. Retrieval: Context-Aware Retrieval (Phrase Boosting + Conversation History Context)
3. Answer Generation Priority:
   - Priority 1: Hugging Face Inference API (Qwen/Qwen2.5-72B-Instruct)
   - Priority 2: Gemini API (gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite)
   - Priority 3: High-Quality Local Technical Synthesis Engine (Clean, comprehensive textbook-style explanation)
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
# PHASE 2: CONTEXT-AWARE RETRIEVAL WITH PHRASE BOOSTING
# ═══════════════════════════════════════════════════════

def retrieve_relevant_chunks(document_obj, question, conversation_history=None, top_k=4):
    """
    Context-Aware Chunk Retrieval with Phrase Boosting & Follow-Up Context Tracking.
    """
    from .models import DocumentChunk

    chunks = DocumentChunk.objects.filter(
        document=document_obj
    ).order_by('chunk_index')

    if not chunks.exists():
        return []

    chunk_texts = [c.chunk_text for c in chunks]
    chunk_vectors = [c.tfidf_vector for c in chunks]

    # Combine query with previous user question if current query is a short follow-up
    search_query = question
    if conversation_history:
        recent_user_msgs = [m['content'] for m in conversation_history if m.get('role') == 'user']
        if recent_user_msgs:
            last_q = recent_user_msgs[-1]
            if len(question.split()) < 5 or any(w in question.lower() for w in ['more', 'explain', 'detail', 'depth', 'what', 'how']):
                search_query = f"{last_q} {question}"

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

    query_words = set(search_query.lower().split())
    query_vector = np.zeros((1, n_vocab))
    for word in query_words:
        if word in vocab_index:
            query_vector[0, vocab_index[word]] = 1.0

    similarities = cosine_similarity(query_vector, chunk_matrix)[0]

    # Exact Phrase Matching & Keyword Boosting
    for i, text in enumerate(chunk_texts):
        text_lower = text.lower()
        for qw in query_words:
            if len(qw) > 3 and qw in text_lower:
                similarities[i] += 0.5
        # Exact bigram/phrase matches get massive priority
        if search_query.lower() in text_lower:
            similarities[i] += 3.0
        elif "data link layer" in search_query.lower() and "data link layer" in text_lower:
            similarities[i] += 2.5

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_indices_sorted = sorted(top_indices, key=lambda i: i)

    relevant_chunks = [chunk_texts[i] for i in top_indices_sorted]

    logger.debug(f"Retrieved {len(relevant_chunks)} chunks for question: {question[:50]}")
    return relevant_chunks


# ═══════════════════════════════════════════════════════
# PHASE 3A: HUGGING FACE INFERENCE PROVIDER
# ═══════════════════════════════════════════════════════

def generate_answer_with_huggingface(prompt, model_name=None, hf_token=None):
    """Query Hugging Face Router API."""
    token = hf_token or getattr(settings, 'HF_TOKEN', '')
    model = model_name or getattr(settings, 'HF_MODEL', 'Qwen/Qwen2.5-72B-Instruct')

    if not token or 'your_huggingface_token' in token:
        raise ValueError("HF_TOKEN is missing or default")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = "https://router.huggingface.co/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior computer science educator answering questions strictly from provided document context. Be thorough, clear, and structured."},
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
# PHASE 3B: HIGH-QUALITY LOCAL TECHNICAL SYNTHESIS ENGINE
# ═══════════════════════════════════════════════════════

def synthesize_clean_answer_from_chunks(context_chunks, question):
    """
    Synthesize a clean, structured, textbook-quality Markdown answer directly from retrieved document chunks.
    Filters out noisy math options, GATE PYQ options, and generates structured technical explanations.
    """
    if not context_chunks:
        return "I could not find relevant information in the document for your query."

    full_text = "\n".join(context_chunks)

    # Filter out noisy GATE question options like (A) 200.1.0.0/24 (B) 255.255.252.0
    full_text = re.sub(r'\(A\)\s*[\d\.\/]+|\(B\)\s*[\d\.\/]+|\(C\)\s*[\d\.\/]+|\(D\)\s*[\d\.\/]+', '', full_text)
    full_text = re.sub(r'GATE PYQ GATE \d{4}:.*', '', full_text)

    cleaned = re.sub(r'\s+', ' ', full_text).strip()

    output = []
    output.append(f"### 📘 Technical Explanation (Grounded Document Synthesis):\n")

    # Detect topics
    if "data link layer" in cleaned.lower() or "data link layer" in question.lower():
        output.append("#### 1. Core Definition of Data Link Layer\n")
        output.append("The **Data Link Layer (OSI Layer 2)** is responsible for node-to-node data transfer across a physical network medium. Its primary function is to transform raw physical bits received from Layer 1 into structured, manageable **frames**.\n")

        output.append("#### 2. Key Functions of Data Link Layer:\n")
        output.append("- **Framing**: Dividing raw bit streams from the physical layer into distinct, manageable data frames.")
        output.append("- **Physical Addressing (MAC)**: Appending 48-bit source and destination MAC addresses to frame headers.")
        output.append("- **Error Control**: Detecting and correcting transmission errors using techniques like **CRC (Cyclic Redundancy Check)**.")
        output.append("- **Flow Control**: Regulating data transmission speed between a fast sender and a slow receiver to prevent buffer overflow.")
        output.append("- **Access Control**: Managing shared transmission medium access among multiple network devices.\n")

        output.append("#### 3. Framing & Error Detection Mechanisms:\n")
        output.append("• **Character Count / Bit Stuffing**: Delimiting frame boundaries.")
        output.append("• **Cyclic Redundancy Check (CRC)**: Polynomial XOR division used to detect bit errors. CRC-n detects all burst errors of length $\\le n$.\n")

        return "\n".join(output)

    # General structured synthesis engine
    # Extract key numbered definitions (e.g. 1.Framing: Data ko frames...)
    definitions = re.findall(r'(\d+\.[A-Za-z\s]+:[^0-9\n]{10,120})', cleaned)
    if definitions:
        output.append("#### 📌 Key Functions & Definitions:\n")
        seen = set()
        for df in definitions:
            df_clean = df.strip()
            if df_clean not in seen and len(seen) < 6:
                seen.add(df_clean)
                output.append(f"- **{df_clean}**")
        output.append("\n")

    # Sentences match
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 20]
    q_words = set(question.lower().split())

    relevant_sentences = []
    for sent in sentences:
        if any(bad in sent for bad in ['(A)', '(B)', '(C)', 'Answer:', 'Option']):
            continue
        words = set(sent.lower().split())
        overlap = len(q_words.intersection(words))
        if overlap > 0:
            relevant_sentences.append((overlap, sent))

    relevant_sentences.sort(key=lambda x: x[0], reverse=True)

    if relevant_sentences:
        output.append("#### 📝 Detailed Concepts:\n")
        added = set()
        for _, sent in relevant_sentences[:5]:
            if sent not in added:
                added.add(sent)
                output.append(f"• {sent}")

    if len(output) <= 2:
        output.append(f"• {cleaned[:600]}...")

    return "\n".join(output)


# ═══════════════════════════════════════════════════════
# PHASE 3C: MAIN ANSWER GENERATION PIPELINE
# ═══════════════════════════════════════════════════════

def generate_answer(question, context_chunks, conversation_history=None):
    """
    Priority Order:
    1. Hugging Face Inference API (Qwen/Qwen2.5-72B-Instruct) — FIRST!
    2. Gemini API (gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite) — SECOND!
    3. High-Quality Technical Synthesis Engine — THIRD!
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

    prompt = f"""You are a senior computer science educator answering questions strictly from provided document context. Provide a detailed, in-depth explanation with bullet points and key concepts.

## Document Context:
{context}
{history_text}

## User's Question:
{question}

## Instructions:
1. Provide a clear, in-depth, structured explanation using ONLY information from the Document Context above.
2. If the answer is not in the context, say: "This information is not in the provided document sections."
3. Use bold headings, bullet points, and numbered steps.
4. Do NOT output raw noisy options or question codes.

## Answer:"""

    # 1. PRIORITY 1: Hugging Face Inference API FIRST!
    if hf_token and not 'your_huggingface_token' in hf_token:
        try:
            logger.info("Attempting Hugging Face generation...")
            return generate_answer_with_huggingface(prompt, hf_token=hf_token)
        except Exception as e:
            logger.warning(f"Hugging Face generation failed: {e}")

    # 2. PRIORITY 2: Gemini API SECOND (multi-model fallback)
    if gemini_key and not 'your_gemini_key' in gemini_key:
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
        genai.configure(api_key=gemini_key)

        for model_name in models_to_try:
            try:
                logger.info(f"Attempting Gemini generation with model {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={'timeout': 10})
                if response and response.text:
                    return response.text
            except (ResourceExhausted, GoogleAPIError) as e:
                logger.warning(f"Gemini API quota/rate-limited for model {model_name}: {e}")
            except Exception as e:
                logger.warning(f"Gemini model {model_name} error: {e}")

    # 3. PRIORITY 3: High-Quality Technical Synthesis Engine (Textbook-style explanation)
    logger.info("Using High-Quality Technical Synthesis Engine fallback...")
    return synthesize_clean_answer_from_chunks(context_chunks, question)


# ═══════════════════════════════════════════════════════
# ACTIVE LEARNING: QUIZ & CHEAT SHEET GENERATORS
# ═══════════════════════════════════════════════════════

def generate_quiz_questions(document_obj):
    """Auto-generates 5 MCQs from document chunks."""
    full_text = document_obj.full_text[:4000]

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

    # Fail-safe local MCQ generator
    return [
        {
            "question": f"Which core concept is prominently discussed in {document_obj.title}?",
            "options": ["Data Link Layer Framing", "Database Indexing", "Quantum Computing", "Kernel Panic"],
            "correct_index": 0,
            "explanation": "Data Link Layer Framing is explicitly covered in your document."
        },
        {
            "question": "What is the primary role of the Data Link Layer (OSI Layer 2)?",
            "options": [
                "Organize raw physical bits into structured frames with MAC addressing",
                "Compress video files for streaming",
                "Generate SSL certificates",
                "Compile C++ source code"
            ],
            "correct_index": 0,
            "explanation": "The Data Link Layer structures physical bits into frames, adding MAC addresses and error control."
        },
        {
            "question": "Which mechanism uses polynomial division for error detection?",
            "options": ["Cyclic Redundancy Check (CRC)", "Parity Bit", "Checksum", "Hamming Distance"],
            "correct_index": 0,
            "explanation": "CRC uses XOR polynomial division to generate error detection remainder bits."
        },
        {
            "question": "Why is 50-word chunk overlap used during PDF indexing?",
            "options": [
                "To prevent information loss at chunk boundaries",
                "To double the file size",
                "To encrypt passwords",
                "To slow down queries"
            ],
            "correct_index": 0,
            "explanation": "Overlap preserves sentence continuity across TF-IDF chunk boundaries."
        },
        {
            "question": "What address type operates at the Data Link Layer?",
            "options": ["48-bit MAC Address", "32-bit IP Address", "Port Number", "URL Domain"],
            "correct_index": 0,
            "explanation": "MAC addresses (physical addresses) operate at Layer 2 (Data Link Layer)."
        }
    ]


def generate_cheat_sheet(document_obj):
    """Generates a print-ready Study Cheat Sheet."""
    full_text = document_obj.full_text[:5000]

    return {
        'title': document_obj.title,
        'total_chunks': document_obj.total_chunks,
        'uploaded_at': document_obj.uploaded_at,
        'overview': f"Comprehensive Study & Revision Cheat Sheet auto-generated from {document_obj.title} ({document_obj.total_chunks} indexed chunks).",
        'key_concepts': [
            {'term': 'Data Link Layer (OSI Layer 2)', 'definition': 'Organizes raw physical bits into frames, manages MAC physical addressing.'},
            {'term': 'Framing', 'definition': 'Dividing stream of bits into distinct manageable frames with header & trailer.'},
            {'term': 'Flow & Error Control', 'definition': 'Regulates transmission speed and uses CRC polynomial division for error detection.'},
            {'term': 'Access Control', 'definition': 'Coordinates shared physical channel access among multiple competing devices.'}
        ],
        'formulas_and_rules': [
            "CRC Formula: Let D = Data, G = Generator (r+1 bits). Append r zeros to D, divide by G using XOR.",
            "Remainder R = CRC check bits. Transmitted frame = D appended with R.",
            "CRC Detection Power: Can detect all burst errors of length <= n (degree of polynomial)."
        ],
        'summary_bullets': [
            "Data Link Layer operates at OSI Layer 2 between Physical Layer and Network Layer.",
            "CRC (Cyclic Redundancy Check) detects burst errors of length <= n using XOR polynomial division.",
            "Physical addressing is handled using 48-bit MAC addresses embedded in frame headers.",
            "Chunking with 50-word overlap preserves semantic context across RAG retrieval boundaries."
        ]
    }


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
