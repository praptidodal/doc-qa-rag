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

@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client storing data in chroma_db/."""
    return chromadb.PersistentClient(path="chroma_db")


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the persistent cosine-distance collection."""
    return client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def get_indexed_files(collection: chromadb.Collection) -> list[str]:
    """Return a sorted list of unique file names already indexed in ChromaDB."""
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    metas = results.get("metadatas") or []
    return sorted(list({m["file"] for m in metas if m and "file" in m}))


def add_files_to_vector_store(files, collection: chromadb.Collection, chunk_size: int = CHUNK_SIZE) -> tuple[int, int]:
    """
    Extract, chunk, and embed only new files not yet in ChromaDB.
    Returns (new_chunks_count, total_chunks_count).
    """
    model = get_embedding_model()
    already_indexed = set(get_indexed_files(collection))

    all_chunks: list[str]  = []
    all_ids:    list[str]  = []
    all_meta:   list[dict] = []
    skipped_files: list[str] = []

    for f in files:
        if f.name in already_indexed:
            skipped_files.append(f.name)
            continue

        pages = extract_pages(f)
        if not pages:
            st.warning(
                f"⚠️ **{f.name}** — no extractable text found "
                "(scanned PDF or empty file; skipping)."
            )
            continue

        for page_num, page_text in pages:
            for chunk_idx, chunk in enumerate(chunk_text(page_text, size=chunk_size, overlap=CHUNK_OVERLAP)):
                uid = f"{f.name}::p{page_num}::c{chunk_idx}"
                all_chunks.append(chunk)
                all_ids.append(uid)
                all_meta.append({"file": f.name, "page": page_num})

    if skipped_files:
        st.info(f"Skipped {len(skipped_files)} already-indexed file(s): {', '.join(skipped_files)}")

    if all_chunks:
        embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()
        collection.add(
            ids        = all_ids,
            documents  = all_chunks,
            embeddings = embeddings,
            metadatas  = all_meta,
        )

    return len(all_chunks), collection.count()


def clear_vector_store(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Delete the persistent collection and recreate an empty one."""
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL + ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks(question: str, collection: chromadb.Collection, top_k: int = TOP_K) -> tuple[list[dict], list[float]]:
    """
    Embed *question* and return the top-K most similar chunks from ChromaDB
    along with their cosine distances.
    """
    model = get_embedding_model()
    q_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings = [q_embedding],
        n_results        = min(top_k, collection.count()),
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
            "distance": dist,
            "match_pct": max(0, min(100, round((1.0 - dist) * 100))),
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


def rewrite_query(question: str, history: list[dict]) -> str:
    """
    If chat history exists, call the LLM with the last 3 messages plus the new question
    to rewrite it as a standalone question that needs no previous context.
    If there is no history or rewrite fails, return question unchanged.
    """
    if not history or not GROQ_API_KEY:
        return question

    # Use the last 3 messages for conversational context
    context_msgs = history[-3:]
    conversation_snippet = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in context_msgs
    )

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone, self-contained question that can be understood without the conversation history. Do NOT answer the question, only return the rewritten question.

Conversation History:
{conversation_snippet}

Follow-up Question: {question}

Standalone Question:"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        if rewritten:
            print(f"[Query Rewrite] Original: '{question}' -> Standalone: '{rewritten}'")
            return rewritten
        return question
    except Exception as exc:
        print(f"[Query Rewrite] Error: {exc}, using original question.")
        return question


def summarize_documents(collection: chromadb.Collection, target_files: list[str] | None = None) -> str:
    """
    Retrieve the first 6 chunks of each document (or target_files) in the collection
    and ask the LLM to generate a concise summary with exactly 5 bullet points.
    """
    if collection.count() == 0:
        return "No documents available to summarize."
    if not GROQ_API_KEY:
        return "❌ **GROQ_API_KEY is not set.**"

    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    target_set = set(target_files) if target_files else None

    file_chunks: dict[str, list[str]] = {}
    for doc, meta in zip(docs, metas):
        fname = meta.get("file", "Document")
        if target_set is not None and fname not in target_set:
            continue
        file_chunks.setdefault(fname, []).append(doc)

    if not file_chunks:
        return "No chunks found for the selected document(s)."

    combined_context = []
    for fname, chunks in file_chunks.items():
        combined_context.append(f"=== {fname} ===\n" + "\n\n".join(chunks[:6]))

    prompt = f"""You are a helpful assistant. Provide a concise summary of the document(s) below in exactly 5 bullet points. Ground every point strictly in the provided text.

Context:
{"\n\n".join(combined_context)}

Summary:"""

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

    # ── Persistent collection & session state initialisation ──────────────────
    chroma_client = get_chroma_client()
    collection    = get_collection(chroma_client)

    if "collection" not in st.session_state:
        st.session_state.collection = collection
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role, content, sources, searched_for}

    # ── Sidebar — file upload, persistent index, & controls ──────────────────
    with st.sidebar:
        st.header("📂 Documents")

        # Settings sliders
        st.subheader("⚙️ Settings")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200,
            max_value=1000,
            value=st.session_state.get("indexed_chunk_size", 500),
            step=50,
        )
        top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=4, step=1)

        if "indexed_chunk_size" in st.session_state and chunk_size != st.session_state.indexed_chunk_size:
            st.caption("⚠️ Changing chunk size requires clicking 'Process' (or Clear all & re-process) to re-chunk.")

        st.divider()

        # Manage already indexed files
        st.subheader("📚 Indexed Files")
        indexed_files = get_indexed_files(st.session_state.collection)
        if indexed_files:
            selected_files = st.multiselect(
                "Choose file(s) to summarize or remove:",
                options=indexed_files,
                default=indexed_files,
                help="Select which indexed files to summarize or delete.",
            )

            col_sum, col_del = st.columns(2)
            with col_sum:
                sum_btn = st.button("📋 Summarize", disabled=(len(selected_files) == 0), help="Summarize selected files")
            with col_del:
                del_btn = st.button("🗑️ Remove", disabled=(len(selected_files) == 0), help="Delete selected files from index")

            if sum_btn:
                with st.spinner(f"Generating 5-bullet summary for {len(selected_files)} file(s) …"):
                    st.session_state.doc_summary = summarize_documents(st.session_state.collection, target_files=selected_files)
                st.rerun()

            if del_btn:
                with st.spinner(f"Removing {len(selected_files)} file(s) …"):
                    for fname in selected_files:
                        st.session_state.collection.delete(where={"file": fname})
                if "doc_summary" in st.session_state:
                    del st.session_state.doc_summary
                st.success(f"Removed {len(selected_files)} file(s) from index!")
                st.rerun()
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        st.markdown("**Add Documents**")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        col_proc, col_clear = st.columns(2)
        with col_proc:
            process_btn = st.button("⚙️ Process", disabled=not uploaded_files)
        with col_clear:
            clear_btn = st.button("🗑️ Clear all", disabled=(st.session_state.collection.count() == 0))

        if process_btn:
            with st.spinner("Extracting, chunking, and embedding new files …"):
                new_chunks, total_chunks = add_files_to_vector_store(
                    uploaded_files,
                    st.session_state.collection,
                    chunk_size=chunk_size,
                )
                st.session_state.indexed_chunk_size = chunk_size
            if new_chunks > 0:
                st.success(f"✅ Added **{new_chunks}** new chunks! (Total: {total_chunks})")
                st.rerun()
            else:
                st.info("Uploaded files are already indexed.")

        if clear_btn:
            collection = clear_vector_store(chroma_client)
            st.session_state.collection   = collection
            st.session_state.chat_history = []
            if "doc_summary" in st.session_state:
                del st.session_state.doc_summary
            st.success("Cleared all indexed documents!")
            st.rerun()

        # Show API key status
        st.divider()
        if GROQ_API_KEY:
            st.success("🔑 Groq API key loaded", icon="✅")
        else:
            st.warning("🔑 GROQ_API_KEY not found in `.env`")

    # ── Document Summary (if generated) ──────────────────────────────────────
    if st.session_state.get("doc_summary"):
        with st.expander("📋 Document Summary (5 Key Points)", expanded=True):
            st.markdown(st.session_state.doc_summary)
            if st.button("Dismiss summary"):
                del st.session_state.doc_summary
                st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            # Display rewritten search query if applicable
            if entry["role"] == "assistant" and entry.get("searched_for"):
                st.markdown(f"<small style='color: grey;'>Searched for: {entry['searched_for']}</small>", unsafe_allow_html=True)
            # Re-display source expanders with match percentage
            if entry["role"] == "assistant" and entry.get("sources") and entry.get("content", "").strip() != NOT_FOUND_MESSAGE:
                with st.expander("🔍 Sources"):
                    for s in entry["sources"]:
                        match_info = f" · **{s.get('match_pct', 0)}% match**" if "match_pct" in s else ""
                        st.markdown(f"**{s['file']}** — page {s['page']}{match_info}")
                        st.text(s["text"])
                        st.divider()

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your documents …")

    if question:
        # Guard: documents must be indexed first
        if st.session_state.collection is None or st.session_state.collection.count() == 0:
            st.warning("⬅️ Please upload and process documents first.")
            st.stop()

        # Capture history before appending current user message
        prior_history = list(st.session_state.chat_history)

        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        with st.chat_message("user"):
            st.markdown(question)

        # If there is prior chat history, rewrite follow-up question into a standalone query for retrieval
        search_query = rewrite_query(question, prior_history) if prior_history else question

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer …"):
                chunks, distances = retrieve_chunks(search_query, st.session_state.collection, top_k=top_k)

                # If best chunk distance is above threshold, skip LLM call
                if distances and distances[0] > DISTANCE_THRESHOLD:
                    print(f"[Threshold] Best distance {distances[0]:.4f} > {DISTANCE_THRESHOLD}. Skipping LLM call.")
                    answer = NOT_FOUND_MESSAGE
                else:
                    answer = generate_answer(question, chunks)

            st.markdown(answer)

            # Show rewritten query in small grey text
            if search_query != question:
                st.markdown(f"<small style='color: grey;'>Searched for: {search_query}</small>", unsafe_allow_html=True)

            # Source expander: only show if answer is NOT the refusal message
            is_refusal = (answer.strip() == NOT_FOUND_MESSAGE)
            if chunks and not is_refusal:
                with st.expander("🔍 Sources"):
                    for c in chunks:
                        match_info = f" · **{c.get('match_pct', 0)}% match**" if "match_pct" in c else ""
                        st.markdown(f"**{c['file']}** — page {c['page']}{match_info}")
                        st.text(c["text"])
                        st.divider()

        # Persist to history
        st.session_state.chat_history.append({
            "role":         "assistant",
            "content":      answer,
            "sources":      [] if is_refusal else chunks,
            "searched_for": search_query if search_query != question else None,
        })


if __name__ == "__main__":
    main()
