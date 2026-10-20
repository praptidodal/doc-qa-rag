import io
import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from environment variable or Streamlit Cloud secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
LLM_MODEL = get_secret("LLM_MODEL", "qwen/qwen3.8-27b")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "doc_qa"
REFUSAL_MESSAGE = "I couldn't find that in the uploaded documents."


def extract_pages(uploaded_file) -> list[tuple[int, str]]:
    """Extract (page_number, text) tuples from PDF or TXT files."""
    pages = []
    raw = uploaded_file.read()

    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    else:
        text = raw.decode("utf-8", errors="replace")
        if text.strip():
            pages.append((1, text))

    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        chunks.append(text[start : start + size])
        start += step

    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path="chroma_db")


def get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def get_indexed_files(collection: chromadb.Collection) -> list[str]:
    """Return sorted unique file names stored in the collection."""
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    metas = results.get("metadatas") or []
    return sorted(list({m["file"] for m in metas if m and "file" in m}))


def add_files_to_vector_store(
    files,
    collection: chromadb.Collection,
    chunk_size: int = CHUNK_SIZE
) -> tuple[int, int]:
    """Extract, chunk, and embed new files into ChromaDB."""
    model = get_embedding_model()
    already_indexed = set(get_indexed_files(collection))

    all_chunks = []
    all_ids = []
    all_meta = []
    skipped_files = []

    for f in files:
        if f.name in already_indexed:
            skipped_files.append(f.name)
            continue

        pages = extract_pages(f)
        if not pages:
            st.warning(f"File '{f.name}' has no extractable text.")
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
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_meta,
        )

    return len(all_chunks), collection.count()


def clear_vector_store(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Delete and re-create the collection."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def retrieve_chunks(
    question: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K
) -> tuple[list[dict], list[float]]:
    """Retrieve top-K similar chunks and cosine distances from ChromaDB."""
    model = get_embedding_model()
    q_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    distances = results.get("distances", [[]])[0]
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
    """Query Groq using retrieved chunks as strict context."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not set. Copy .env.example to .env and configure your key."

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
        return f"Groq API error: {exc}"


def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite follow-up question into a standalone query using recent history."""
    if not history or not GROQ_API_KEY:
        return question

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
        return rewritten if rewritten else question
    except Exception:
        return question


def summarize_documents(collection: chromadb.Collection, target_files: list[str] | None = None) -> str:
    """Summarize the first 6 chunks of target documents into 5 bullet points."""
    if collection.count() == 0:
        return "No documents available to summarize."
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not set."

    results = collection.get(include=["documents", "metadatas"])
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    target_set = set(target_files) if target_files else None
    file_chunks = {}
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
        return f"Groq API error: {exc}"


def main() -> None:
    st.set_page_config(page_title="Document Q&A Assistant")
    st.title("Document Q&A Assistant")
    st.caption("Upload documents, then ask questions grounded in your files.")

    chroma_client = get_chroma_client()
    collection = get_collection(chroma_client)

    if "collection" not in st.session_state:
        st.session_state.collection = collection
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    with st.sidebar:
        st.header("Documents")

        if st.session_state.get("flash_message"):
            msg_type, msg_text = st.session_state.flash_message
            if msg_type == "success":
                st.success(msg_text)
            elif msg_type == "info":
                st.info(msg_text)
            elif msg_type == "warning":
                st.warning(msg_text)
            del st.session_state.flash_message

        st.subheader("Settings")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200,
            max_value=1000,
            value=st.session_state.get("indexed_chunk_size", 500),
            step=50,
        )
        top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=4, step=1)

        if "indexed_chunk_size" in st.session_state and chunk_size != st.session_state.indexed_chunk_size:
            st.caption("Changing chunk size requires clicking Process to re-chunk.")

        st.divider()

        st.subheader("Indexed Files")
        indexed_files = get_indexed_files(st.session_state.collection)
        if indexed_files:
            selected_files = st.multiselect(
                "Select files to summarize or remove:",
                options=indexed_files,
                default=indexed_files,
                key=f"indexed_select_{st.session_state.uploader_key}_{len(indexed_files)}",
            )

            col_sum, col_del = st.columns(2)
            with col_sum:
                sum_btn = st.button("Summarize", disabled=(len(selected_files) == 0))
            with col_del:
                del_btn = st.button("Remove", disabled=(len(selected_files) == 0))

            if sum_btn:
                with st.spinner("Generating summary..."):
                    st.session_state.doc_summary = summarize_documents(
                        st.session_state.collection,
                        target_files=selected_files
                    )
                st.rerun()

            if del_btn:
                with st.spinner("Removing selected files..."):
                    for fname in selected_files:
                        st.session_state.collection.delete(where={"file": fname})
                if "doc_summary" in st.session_state:
                    del st.session_state.doc_summary
                st.session_state.flash_message = ("success", f"Removed {len(selected_files)} file(s).")
                st.rerun()
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        st.markdown("**Add Documents**")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        col_proc, col_clear = st.columns(2)
        with col_proc:
            process_btn = st.button("Process", disabled=not uploaded_files)
        with col_clear:
            clear_btn = st.button("Clear all", disabled=(st.session_state.collection.count() == 0))

        if process_btn:
            with st.spinner("Processing documents..."):
                new_chunks, total_chunks = add_files_to_vector_store(
                    uploaded_files,
                    st.session_state.collection,
                    chunk_size=chunk_size,
                )
                st.session_state.indexed_chunk_size = chunk_size

            st.session_state.uploader_key += 1

            if new_chunks > 0:
                st.session_state.flash_message = ("success", f"Added {new_chunks} new chunks. Total: {total_chunks}")
            else:
                st.session_state.flash_message = ("info", "Uploaded files were already indexed.")
            st.rerun()

        if clear_btn:
            collection = clear_vector_store(chroma_client)
            st.session_state.collection = collection
            st.session_state.chat_history = []
            st.session_state.uploader_key += 1
            if "doc_summary" in st.session_state:
                del st.session_state.doc_summary
            st.session_state.flash_message = ("success", "Cleared all indexed documents.")
            st.rerun()

        st.divider()
        if GROQ_API_KEY:
            st.success("Groq API key loaded")
        else:
            st.warning("GROQ_API_KEY not found in .env")

    if st.session_state.get("doc_summary"):
        with st.expander("Document Summary", expanded=True):
            st.markdown(st.session_state.doc_summary)
            if st.button("Dismiss summary"):
                del st.session_state.doc_summary
                st.rerun()

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("searched_for"):
                st.markdown(f"<small style='color: gray;'>Searched for: {entry['searched_for']}</small>", unsafe_allow_html=True)
            if entry["role"] == "assistant" and entry.get("sources") and entry.get("content", "").strip() != REFUSAL_MESSAGE:
                with st.expander("Sources"):
                    for s in entry["sources"]:
                        match_info = f" | {s.get('match_pct', 0)}% match" if "match_pct" in s else ""
                        st.markdown(f"**{s['file']}** — page {s['page']}{match_info}")
                        st.text(s["text"])
                        st.divider()

    question = st.chat_input("Ask a question about your documents...")

    if question:
        if st.session_state.collection is None or st.session_state.collection.count() == 0:
            st.warning("Please upload and process documents first.")
            st.stop()

        prior_history = list(st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        with st.chat_message("user"):
            st.markdown(question)

        search_query = rewrite_query(question, prior_history) if prior_history else question

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                chunks, _ = retrieve_chunks(search_query, st.session_state.collection, top_k=top_k)
                answer = generate_answer(question, chunks)

            st.markdown(answer)

            if search_query != question:
                st.markdown(f"<small style='color: gray;'>Searched for: {search_query}</small>", unsafe_allow_html=True)

            is_refusal = (answer.strip() == REFUSAL_MESSAGE)
            if chunks and not is_refusal:
                with st.expander("Sources"):
                    for c in chunks:
                        match_info = f" | {c.get('match_pct', 0)}% match" if "match_pct" in c else ""
                        st.markdown(f"**{c['file']}** — page {c['page']}{match_info}")
                        st.text(c["text"])
                        st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": [] if is_refusal else chunks,
            "searched_for": search_query if search_query != question else None,
        })


if __name__ == "__main__":
    main()
