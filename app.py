"""
Document Q&A Assistant — RAG pipeline using ChromaDB + Groq
=============================================================
Pipeline:
  1. Upload PDF/TXT  →  extract_pages()
  2. Split text      →  chunk_text()
  3. Embed chunks    →  get_embedding_model() + ChromaDB
  4. User question   →  embed question, retrieve top-4 chunks
  5. Answer          →  Groq API with context-only prompt
"""

import os
import io

import streamlit as st
from dotenv import load_dotenv
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()                          # reads GROQ_API_KEY from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # characters shared between neighbouring chunks
TOP_K              = 4      # number of chunks to retrieve
EMBED_MODEL        = "all-MiniLM-L6-v2"
LLM_MODEL          = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")  # override in .env
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.72"))  # cosine distance threshold
COLLECTION         = "doc_qa"
NOT_FOUND_MESSAGE  = "I couldn't find that in the uploaded documents."

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pages(uploaded_file) -> list[tuple[int, str]]:
    """
    Return a list of (page_number, text) tuples from an uploaded file.

    • PDF  → use pypdf, one tuple per page (1-indexed).
    • TXT  → decode as UTF-8, returned as a single page (page 1).
    """
    pages: list[tuple[int, str]] = []
    raw = uploaded_file.read()

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    else:
        # TXT — treat whole file as page 1
        text = raw.decode("utf-8", errors="replace")
        if text.strip():
            pages.append((1, text))

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

    Each chunk is `size` characters long; the next chunk starts
    `size - overlap` characters after the previous one begins,
    so consecutive chunks share `overlap` characters.
    """
    chunks: list[str] = []
    start = 0
    step  = size - overlap

    while start < len(text):
        chunks.append(text[start : start + size])
        start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBEDDING MODEL  (cached — loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def get_embedding_model() -> SentenceTransformer:
    """Download / load all-MiniLM-L6-v2 once and cache it in the session."""
    return SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(files) -> tuple[chromadb.Collection, int]:
    """
    Embed all chunks from *files* and store them in an in-memory ChromaDB
    collection.  The collection is rebuilt fresh on every call so re-uploading
    replaces the previous index.

    Returns (collection, total_chunk_count).
    """
    model  = get_embedding_model()
    client = chromadb.Client()          # purely in-memory, nothing persisted

    # Delete and recreate the collection so each "Process" starts clean
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks: list[str]  = []
    all_ids:    list[str]  = []
    all_meta:   list[dict] = []

    for f in files:
        pages = extract_pages(f)

        if not pages:
            st.warning(
                f"⚠️ **{f.name}** — no extractable text found "
                "(scanned PDF or empty file; skipping)."
            )
            continue

        for page_num, page_text in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page_text)):
                uid = f"{f.name}::p{page_num}::c{chunk_idx}"
                all_chunks.append(chunk)
                all_ids.append(uid)
                all_meta.append({"file": f.name, "page": page_num})

    if all_chunks:
        # Encode in one batch for efficiency
        embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()
        collection.add(
            ids        = all_ids,
            documents  = all_chunks,
            embeddings = embeddings,
            metadatas  = all_meta,
        )

    return collection, len(all_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL + ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(question: str, collection: chromadb.Collection) -> tuple[list[dict], list[float]]:
    """
    Embed *question* and return the top-K most similar chunks from ChromaDB
    along with their cosine distances.
    """
    model = get_embedding_model()
    q_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings = [q_embedding],
        n_results        = min(TOP_K, collection.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    chunks = []
    distances = results.get("distances", [[]])[0]

    print(f"\n[Retrieval] Question: '{question}'")
    print(f"[Retrieval] Distances (lower = closer match): {[round(d, 4) for d in distances]}")
    if distances:
        print(f"[Retrieval] Best distance: {distances[0]:.4f} (threshold: {DISTANCE_THRESHOLD})")

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": doc,
            "file": meta["file"],
            "page": meta["page"],
            "distance": dist
        })

    return chunks, distances


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Call the Groq API with a strict context-only prompt.
    Returns the model's text answer (or a friendly error string).
    """
    if not GROQ_API_KEY:
        return (
            "❌ **GROQ_API_KEY is not set.**  "
            "Copy `.env.example` to `.env` and add your key from console.groq.com."
        )

    # Build the context block from retrieved chunks
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['file']}, page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer cannot be found in the context, reply exactly: "I couldn't find that in the uploaded documents."
Do NOT use any outside knowledge.

Context:
{context_text}

Question: {question}

Answer:"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"❌ Groq API error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")
    st.title("📄 Document Q&A Assistant")
    st.caption("Upload documents, then ask questions — answers come only from your files.")

    # ── Session state initialisation ─────────────────────────────────────────
    if "collection"   not in st.session_state:
        st.session_state.collection = None   # ChromaDB collection
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role, content, sources}

    # ── Sidebar — file upload & processing ───────────────────────────────────
    with st.sidebar:
        st.header("📂 Upload Documents")
        uploaded_files = st.file_uploader(
            "PDF or TXT files",
            type        = ["pdf", "txt"],
            accept_multiple_files = True,
        )

        if st.button("⚙️ Process documents", disabled=not uploaded_files):
            with st.spinner("Extracting text, chunking, and embedding …"):
                collection, n_chunks = build_vector_store(uploaded_files)
                st.session_state.collection   = collection
                st.session_state.chat_history = []   # reset chat on new docs
            if n_chunks:
                st.success(f"✅ Indexed **{n_chunks}** chunks from {len(uploaded_files)} file(s).")
            else:
                st.error("No text could be extracted from the uploaded files.")

        # Show API key status
        st.divider()
        if GROQ_API_KEY:
            st.success("🔑 Groq API key loaded", icon="✅")
        else:
            st.warning("🔑 GROQ_API_KEY not found in `.env`")

    # ── Chat history ─────────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            # Re-display source expanders for assistant messages (only if not a refusal)
            if entry["role"] == "assistant" and entry.get("sources") and entry.get("content", "").strip() != NOT_FOUND_MESSAGE:
                with st.expander("🔍 Sources"):
                    for s in entry["sources"]:
                        st.markdown(f"**{s['file']}** — page {s['page']}")
                        st.text(s["text"])
                        st.divider()

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your documents …")

    if question:
        # Guard: documents must be indexed first
        if st.session_state.collection is None:
            st.warning("⬅️ Please upload and process documents first.")
            st.stop()

        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer …"):
                chunks, distances = retrieve_chunks(question, st.session_state.collection)

                # If best chunk distance is above threshold, skip LLM call
                if distances and distances[0] > DISTANCE_THRESHOLD:
                    print(f"[Threshold] Best distance {distances[0]:.4f} > {DISTANCE_THRESHOLD}. Skipping LLM call.")
                    answer = NOT_FOUND_MESSAGE
                else:
                    answer = generate_answer(question, chunks)

            st.markdown(answer)

            # Source expander: only show if answer is NOT the refusal message
            is_refusal = (answer.strip() == NOT_FOUND_MESSAGE)
            if chunks and not is_refusal:
                with st.expander("🔍 Sources"):
                    for c in chunks:
                        st.markdown(f"**{c['file']}** — page {c['page']}")
                        st.text(c["text"])
                        st.divider()

        # Persist to history
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "sources": [] if is_refusal else chunks,
        })


if __name__ == "__main__":
    main()
