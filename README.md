# Document Q&A Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://praptidodal-doc-qa-rag-app-6obiu1.streamlit.app/)

A production-ready, minimal Retrieval-Augmented Generation (RAG) assistant built with Python, Streamlit, ChromaDB, Sentence-Transformers, and Groq.

**Live Demo:** [https://praptidodal-doc-qa-rag-app-6obiu1.streamlit.app/](https://praptidodal-doc-qa-rag-app-6obiu1.streamlit.app/)

---

## Overview

The Document Q&A Assistant allows users to upload PDF and TXT documents, chunk and index them into a local vector database, and ask questions through a chat interface. Answers are generated strictly from the uploaded content using an LLM.

---

## Pipeline

```
PDF / TXT files
      │
      ▼
 extract_pages()          Extract text per page (pypdf for PDFs, UTF-8 for TXT)
      │
      ▼
  chunk_text()            Character chunking (configurable size 200–1000, 50 overlap)
      │
      ▼
all-MiniLM-L6-v2          sentence-transformers embedding model (384 dimensions)
      │
      ▼
ChromaDB (Persistent)     Cosine-distance index saved locally in chroma_db/
      │
      ▼ (user question)
 rewrite_query()          Reformulates conversational follow-ups into standalone queries
      │
      ▼
 retrieve_chunks()        Top-K cosine nearest-neighbors + DISTANCE_THRESHOLD filtering
      │
      ▼
 generate_answer()        Groq (qwen/qwen3.8-27b) generates grounded answers
      │
      ▼
  Streamlit UI            Displays answer, match percentages in Sources, and history
```

---

## Setup & Run

### 1. Clone or navigate to the repository

```bash
cd doc-qa-rag
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
LLM_MODEL=qwen/qwen3.8-27b
DISTANCE_THRESHOLD=0.72
```

> Get a free Groq API key at <https://console.groq.com/keys>.

### 5. Start the application

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## Features

- **Document Ingestion & Deduplication:** Uploads PDF/TXT files, extracts text page-by-page, and automatically skips re-indexing files that are already stored.
- **Persistent Local Index:** Embeddings and chunks persist in `chroma_db/`, surviving application restarts.
- **Selective Management:** Choose specific indexed files to summarize or remove from the index.
- **Context-Aware Query Rewriting:** Conversational pronouns (e.g., *"tell me more about it"*) are rewritten into standalone questions before retrieval using the recent chat history.
- **Relevance Confidence Scores:** Each retrieved chunk displays a match percentage computed directly from its cosine distance.
- **5-Bullet Summarizer:** Synthesizes the opening sections of selected documents into 5 concise bullet points.
- **Tuning Controls:** Live sidebar sliders for chunk size (200–1000) and Top-K retrieval (1–8).

---

## Design Notes & Provider Swapping

- **Gemini to Groq Migration:** The LLM integration is completely encapsulated inside the `generate_answer()` and `rewrite_query()` functions. When switching providers (such as moving from Gemini to Groq), chunking, vector indexing, embedding, and retrieval remain completely untouched.
- **Model Config in Environment:** Model identifiers are read from `LLM_MODEL` in `.env`, allowing model deprecations or upgrades to be handled with a one-line config change rather than code refactoring.

---

## Limitations

- **Scanned Documents:** Only text-based PDFs are parsed. Scanned or image-only documents require OCR preprocessing before upload.
- **Local Single-User Storage:** The persistent ChromaDB directory is designed for local single-user use and is not set up for distributed multi-tenant concurrency.
- **Context Truncation:** Large files produce multiple chunks; only the top-K most similar chunks are passed to the LLM.
