"""
utils.py — RAG (Retrieval Augmented Generation) Pipeline

Fail-safe design:
1. PDF Ingestion & Chunking (500-word chunks, 50-word overlap)
2. TF-IDF Cosine Retrieval
3. AI Answer Generation (Gemini API with multi-model fallback + HF API support)
4. Graceful Fallback (Direct Grounded Passage Extractor when API quota rate-limited)
"""

import re
import os
import warnings
import logging

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
# PHASE 3: HUGGING FACE GENERATION
# ═══════════════════════════════════════════════════════

def generate_answer_with_huggingface(prompt, model_name=None, hf_token=None):
    """Query Hugging Face Inference API."""
    token = hf_token or getattr(settings, 'HF_TOKEN', '')
    model = model_name or getattr(settings, 'HF_MODEL', 'Qwen/Qwen2.5-72B-Instruct')

    if not token or 'your_huggingface_token' in token:
        raise ValueError("HF_TOKEN is missing or default")

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


# ═══════════════════════════════════════════════════════
# PHASE 3: ROBUST GENERATION WITH RATE-LIMIT FALLBACK
# ═══════════════════════════════════════════════════════

def generate_answer(question, context_chunks, conversation_history=None):
    """
    Generate answer from retrieved chunks using Gemini API or Hugging Face.

    Fail-safe behavior:
    1. Tries Hugging Face if configured
    2. Tries Gemini models (gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite)
    3. If quota exhausted / 429 rate limited, presents clean relevant passages directly!
    """
    gemini_key = getattr(settings, 'GEMINI_API_KEY', '')
    hf_token = getattr(settings, 'HF_TOKEN', '')

    if not context_chunks:
        return ("❌ I couldn't find relevant information in the document for your question. "
                "Try rephrasing or asking about a different topic from the document.")

    # Context formatting
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

    # 1. Try Hugging Face
    if hf_token and not 'your_huggingface_token' in hf_token:
        try:
            return generate_answer_with_huggingface(prompt, hf_token=hf_token)
        except Exception as e:
            logger.warning(f"Hugging Face API failed: {e}")

    # 2. Try Gemini API models (with multi-model fallback)
    if gemini_key and not 'your_gemini_key' in gemini_key:
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
        genai.configure(api_key=gemini_key)

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={'timeout': 10})
                if response and response.text:
                    return response.text
            except (ResourceExhausted, GoogleAPIError) as e:
                logger.warning(f"Gemini API quota/rate-limited for model {model_name}: {e}")
            except Exception as e:
                logger.warning(f"Gemini model {model_name} error: {e}")

    # 3. Fail-safe Grounded Passage Fallback (When API rate-limited / quota 429)
    passages_formatted = "\n\n".join([
        f"**Passage {i+1}:**\n> {chunk[:600]}..."
        for i, chunk in enumerate(context_chunks[:2])
    ])

    return (
        f"📌 **Relevant Passages from Document:**\n\n"
        f"{passages_formatted}\n\n"
        f"*(Note: Gemini Free API quota rate limit reached. Displaying relevant document passages directly.)*"
    )


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
