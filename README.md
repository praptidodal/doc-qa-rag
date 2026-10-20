# Document Q&A Assistant

A minimal Retrieval-Augmented Generation (RAG) application built with Streamlit, ChromaDB, sentence-transformers, and Groq.

---

## Pipeline

```
PDF / TXT files
      │
      ▼
 extract_pages()          ← pypdf for PDFs, UTF-8 decode for TXT
      │
      ▼
  chunk_text()            ← 500-char chunks, 50-char overlap
      │
      ▼
all-MiniLM-L6-v2          ← sentence-transformers embedding model
      │
      ▼
   ChromaDB               ← in-memory vector store (rebuilt on each upload)
      │
      ▼  (user question)
 retrieve_chunks()        ← embed question → top-4 cosine-similar chunks
      │
      ▼
 generate_answer()        ← Groq (llama-3.3-70b-versatile) with context-only prompt
      │
      ▼
  Streamlit UI            ← answer + Sources expander + chat history
```

---

## Quick Start

### 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd doc-qa-rag
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the ~90 MB `all-MiniLM-L6-v2` model automatically.

### 4 — Add your Groq API key

```bash
cp .env.example .env
# open .env and replace "your_key_here" with your actual key
```

Get a free key at <https://console.groq.com/keys>.

### 5 — Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Usage

1. **Sidebar → Upload Documents** — select one or more PDF/TXT files.
2. Click **⚙️ Process documents** — the app chunks and indexes them.
3. Type a question in the chat box at the bottom.
4. The answer appears above; expand **🔍 Sources** to see the exact chunks used.

---

## Key Configuration (top of `app.py`)

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `500` | Characters per chunk — smaller = more precise retrieval |
| `CHUNK_OVERLAP` | `50` | Shared chars between chunks — prevents context loss at boundaries |
| `TOP_K` | `4` | Chunks sent to the LLM — more = richer context but longer prompt |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Swappable to any Groq-supported model (e.g. `llama-3.1-8b-instant`) |

---

## Design Notes & Limitations

- **Provider isolation** — The LLM call is isolated in one function, so the provider can be swapped (Gemini → Groq) without touching retrieval.
- **In-memory only** — the vector store is lost when the app restarts; re-upload to re-index.
- **Scanned PDFs** — images-only PDFs yield no text; use OCR pre-processing first.
- **Context window** — very large documents produce many chunks; only the top 4 reach the LLM.
- **No authentication** — not suitable for multi-user production deployment as-is.
- **Language** — `all-MiniLM-L6-v2` is optimised for English; multilingual accuracy may vary.

