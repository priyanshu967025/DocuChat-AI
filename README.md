# 💬 DocuChat AI — Chat with Your PDFs

A Retrieval Augmented Generation (RAG) web application that lets you upload PDF
documents and have AI-powered conversations grounded in the document content.
Built with Django, scikit-learn, and Google Gemini API.

## 🔗 Live Demo
[Add link here when deployed]

## 📸 Screenshots
[Add screenshots]

---

## ✨ Features

- **RAG Architecture:** Document-grounded Q&A — no hallucination
- **PDF Processing:** Upload any text-based PDF, auto-extracted and indexed
- **Smart Chunking:** 500-word chunks with 50-word overlap for retrieval accuracy
- **TF-IDF Retrieval:** Vectorized chunks searched via cosine similarity
- **Gemini AI Answers:** Contextual answers using only document content
- **Multi-Conversation:** Multiple chat sessions per document
- **Chat History:** Persistent conversation history with multi-turn context
- **Auto Summary:** AI-generated document summaries on upload
- **User Auth:** Register/Login with session-based authentication
- **Real-time Chat:** AJAX-based messaging without page reloads
- **Responsive UI:** Dark theme with sidebar navigation

---

## 🏗️ System Architecture

```
           ┌──────────────────────────────────────────────┐
           │           RAG PIPELINE                       │
           │                                              │
UPLOAD     │  PDF ──▶ Extract ──▶ Chunk ──▶ Vectorize     │
PHASE      │                      (500w)    (TF-IDF)      │
           │                      overlap    per chunk     │
           │                       50w                     │
           │                                  ▼            │
           │                          Store in DB          │
           │                     (Document + Chunks)       │
           │                                              │
           ├──────────────────────────────────────────────┤
           │                                              │
QUERY      │  Question ──▶ Vectorize ──▶ Cosine Sim      │
PHASE      │                             vs all chunks    │
           │                                  ▼            │
           │                          Top 4 chunks        │
           │                                  ▼            │
           │                    Prompt: Context + Question │
           │                                  ▼            │
           │                          Gemini API          │
           │                                  ▼            │
           │                     Grounded Answer!         │
           └──────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | Django 6.x | Web framework, Auth, ORM |
| NLP | NLTK | Text preprocessing |
| ML | scikit-learn | TF-IDF vectorization + Cosine similarity retrieval |
| Gen AI | Google Gemini API | Answer generation (RAG-grounded) |
| PDF | PyPDF2 | Document text extraction |
| Frontend | Bootstrap 5 | Responsive dark theme UI |
| Real-time | AJAX (Fetch API) | Chat without page reload |
| Database | SQLite | Document, chunk, conversation, message storage |

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/docuchat-ai.git
cd docuchat-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your free key: https://aistudio.google.com/

### 5. Run migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Start the server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` 🚀

---

## 🧠 How RAG Works in This Project

### 1. Document Indexing (One-time per upload)
```
PDF → PyPDF2 extracts text → Split into 500-word chunks (50-word overlap)
→ Each chunk vectorized with TF-IDF → Stored as JSON dict in database
```

### 2. Question Answering (Per query)
```
User question → Vectorize → Cosine similarity against all chunk vectors
→ Top 4 chunks retrieved → Prompt: "Context: [chunks] Question: [question]
   Answer ONLY from context." → Gemini API → Grounded answer!
```

### 3. Why Chunking with Overlap?
Without overlap, information at chunk boundaries gets split across two chunks.
A 50-word overlap ensures continuous ideas remain searchable.

### 4. Why TF-IDF + Cosine (not embeddings)?
TF-IDF is interpretable, lightweight, and requires no API calls for vectorization.
For production, this can be upgraded to dense embeddings (Sentence-BERT).

---

## 📁 Project Structure
```
doc_qa_project/
├── doc_qa_project/          # Django settings
│   ├── settings.py
│   └── urls.py
├── qa_app/                  # Main app
│   ├── models.py            # Document, Chunk, Conversation, Message
│   ├── views.py             # Auth + Document + Chat views
│   ├── utils.py             # RAG pipeline (chunk, vectorize, retrieve, generate)
│   ├── urls.py
│   └── admin.py
├── templates/               # HTML templates
│   ├── base.html            # Dark theme layout
│   ├── landing.html
│   ├── register.html
│   ├── login.html
│   ├── documents.html       # Document list + upload modal
│   └── chat.html            # Chat UI with sidebar
├── requirements.txt
└── README.md
```

---

## 📊 Database Schema
```
USER ──< DOCUMENT ──< DOCUMENT_CHUNK
     ──< CONVERSATION ──< CHAT_MESSAGE
```

- **Document**: PDF metadata + full extracted text + auto-summary
- **DocumentChunk**: 500-word text + TF-IDF vector (JSON)
- **Conversation**: User × Document chat session
- **ChatMessage**: Individual messages (user/assistant) + retrieved chunks

---

## 🔒 Security

- CSRF protection on all forms
- PBKDF2 password hashing
- API keys in `.env` (never in source code)
- User data isolation — users see only their own documents
- File type (PDF only) and size (10MB) validation
- Input length validation on all user inputs

---

## 📊 Limitations & Future Improvements

| Current | Future |
|---|---|
| TF-IDF (sparse vectors) | Dense embeddings (Sentence-BERT, pgvector) |
| SQLite | PostgreSQL + vector extension |
| Synchronous processing | Celery + Redis async pipeline |
| Text-only PDFs | OCR for scanned PDFs (Tesseract) |
| English only | Multilingual support |
| Single model | Model selection (GPT-4, Claude, etc.) |
| No streaming | SSE/WebSocket streaming responses |

---

## 👨‍💻 Author

**Priyanshu Mishra**
- NIT Hamirpur
- [GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

*Built as a CV project for on-campus placements | July 2026*
