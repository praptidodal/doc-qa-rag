# Document Q&A Assistant

A production-ready, minimal Retrieval-Augmented Generation (RAG) assistant built with Streamlit, ChromaDB, Sentence-Transformers, and Groq.

---

## Architecture & Pipeline

```
PDF / TXT files
      │
      ▼
 extract_pages()          ← pypdf for PDFs, UTF-8 decode for TXT
      │
      ▼
  chunk_text()            ← Configurable chunk size (200–1000 chars, 50 overlap)
      │
      ▼
all-MiniLM-L6-v2          ← sentence-transformers embedding model
      │
      ▼
ChromaDB (Persistent)     ← Cosine-distance vector store in chroma_db/
      │
      ▼  (user question)
 rewrite_query()          ← Context-aware follow-up query rewriting (uses history)
      │
      ▼
 retrieve_chunks()        ← Top-K cosine search + DISTANCE_THRESHOLD filtering
      │
      ▼
 generate_answer()        ← Groq LLM (qwen/qwen3.8-27b) with grounded context
      │
      ▼
  Streamlit UI            ← Answer + match percentages in Sources + chat history
```

---

## Features

1. **Multi-Format Ingestion:** Upload multiple PDFs and TXTs; extracts text per page using `pypdf`.
2. **Persistent Vector Storage:** Saved in `chroma_db/` so your indexed documents survive app restarts.
3. **Smart Deduplication:** Automatically detects and skips files that are already indexed.
4. **Follow-Up Question Rewriting:** Conversational memory detects follow-ups (e.g. *"tell me more about it"*) and rewrites them into standalone queries before retrieval.
5. **Hallucination Shield:** Cosine distance thresholding (`DISTANCE_THRESHOLD=0.72`) skips the LLM call entirely when a question has no relevant chunks.
6. **Relevance Percentages:** Each retrieved source displays an exact match confidence score (e.g., `82% match`).
7. **5-Bullet Document Summarizer:** Automatically analyzes the first chunks of all uploaded files to produce an instant executive summary.
8. **Interactive Controls:** Sidebar sliders for chunk size (200–1000) and Top-K retrieved chunks (1–8).

---

## Quick Start

### 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd doc-qa-rag
```

### 2 — Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add your Groq API key

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY from https://console.groq.com/keys
```

### 5 — Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Swappable to any Groq chat model |
| `DISTANCE_THRESHOLD` | `0.72` | Cosine distance cut-off for grounding |

---

## Design Notes & Interview Talking Points

- **Provider Isolation:** The LLM call is isolated in one function (`generate_answer`), allowing seamless provider switches (Gemini → Groq) without touching chunking or vector search.
- **Query Disambiguation:** Resolving pronouns before retrieval prevents vector search failures on multi-turn conversations.
- **Fail-Fast Retrieval:** Filtering queries with high cosine distances avoids unnecessary API costs and prevents hallucinations on out-of-domain questions.
- **Stateless Model, Stateful Store:** Embeddings are persisted in local ChromaDB storage, while the chat state is maintained via Streamlit `session_state`.
