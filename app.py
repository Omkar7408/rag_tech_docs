import hashlib
import os
import re
import tempfile
from io import BytesIO

import chromadb
import fitz  # PyMuPDF
import ollama
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
LLM_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db_uploaded"
COLLECTION_NAME = "uploaded_doc"


# -----------------------------
# Page setup and styling
# -----------------------------
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .main { padding-top: 1.5rem; }
    .doc-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        margin: 0 0.25rem 0.4rem 0;
        border-radius: 999px;
        background: #f1f5f9;
        color: #334155;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Models
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


model = load_model()


# -----------------------------
# Text helpers
# -----------------------------
def clean_text(text):
    """Normalize extracted text while preserving paragraph breaks."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_text(table):
    """Convert an extracted PDF table into readable pipe-separated rows."""
    rows = []
    for row in table or []:
        cleaned_cells = [
            clean_text(str(cell)) if cell is not None else ""
            for cell in row
        ]
        if any(cleaned_cells):
            rows.append(" | ".join(cleaned_cells))
    return "\n".join(rows)


def ocr_bytes(image_bytes):
    """OCR an image byte string and return extracted text."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            return clean_text(pytesseract.image_to_string(image))
    except Exception as exc:
        print(f"OCR error: {exc}")
        return ""


# -----------------------------
# PDF extraction / chunking
# -----------------------------
def build_chunks(path):
    """
    Extract page text, tables, OCR from scanned pages and embedded images,
    plus clickable HTTP/HTTPS links. Returns (chunks, number_of_tables).
    """
    chunks = []
    n_tables = 0

    with fitz.open(path) as fdoc, pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(fdoc):
            page_num = page_index + 1
            page_text = clean_text(page.get_text("text"))

            # OCR scanned pages that contain little or no selectable text.
            if len(page_text) < 40:
                try:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    rendered_image = Image.open(
                        BytesIO(pix.tobytes("png"))
                    ).convert("RGB")
                    ocr_text = clean_text(
                        pytesseract.image_to_string(rendered_image)
                    )
                    if len(ocr_text) > len(page_text):
                        page_text = ocr_text
                except Exception as exc:
                    print(f"Page {page_num} OCR failed: {exc}")

            # Extract tables with pdfplumber.
            table_texts = []
            try:
                tables = pdf.pages[page_index].extract_tables() or []
                n_tables += len(tables)
                for table in tables:
                    text = table_to_text(table)
                    if text:
                        table_texts.append(text)
            except Exception as exc:
                print(f"Table extraction failed on page {page_num}: {exc}")

            combined_text = page_text
            if table_texts:
                table_block = "\n\n".join(table_texts)
                combined_text = (
                    f"{combined_text}\n\n[TABLES ON THIS PAGE]\n{table_block}"
                    if combined_text
                    else f"[TABLES ON THIS PAGE]\n{table_block}"
                )

            if combined_text.strip():
                chunks.append(
                    {
                        "page": page_num,
                        "content": combined_text,
                        "content_type": "page_text",
                        "url": "",
                    }
                )

            # OCR text in embedded image objects.
            seen_xrefs = set()
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    extracted_image = fdoc.extract_image(xref)
                    image_bytes = extracted_image["image"]

                    with Image.open(BytesIO(image_bytes)) as image:
                        if image.width < 80 or image.height < 50:
                            continue

                    image_text = ocr_bytes(image_bytes)
                    if image_text:
                        chunks.append(
                            {
                                "page": page_num,
                                "content": (
                                    f"[OCR TEXT FROM EMBEDDED IMAGE ON PAGE {page_num}]\n"
                                    f"{image_text}"
                                ),
                                "content_type": "image_ocr",
                                "url": "",
                            }
                        )
                except Exception as exc:
                    print(
                        f"Embedded image {image_index} on page "
                        f"{page_num} failed: {exc}"
                    )

            # Extract hyperlinks and any text located inside each link rectangle.
            try:
                for link in page.get_links():
                    url = link.get("uri", "")
                    if not isinstance(url, str) or not url.startswith(
                        ("https://", "http://")
                    ):
                        continue

                    link_text = ""
                    link_rect = link.get("from")
                    if link_rect:
                        try:
                            link_text = clean_text(
                                page.get_text("text", clip=link_rect)
                            )
                        except Exception:
                            pass

                    chunks.append(
                        {
                            "page": page_num,
                            "content": (
                                f"[PDF HYPERLINK ON PAGE {page_num}]\n"
                                f"Link text: {link_text or 'Not available'}\n"
                                f"URL: {url}"
                            ),
                            "content_type": "hyperlink",
                            "url": url,
                        }
                    )
            except Exception as exc:
                print(f"Hyperlink extraction failed on page {page_num}: {exc}")

    return chunks, n_tables


# -----------------------------
# ChromaDB indexing
# -----------------------------
def index_pdf(path):
    chunks, n_tables = build_chunks(path)
    if not chunks:
        return None, 0, n_tables

    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # This app indexes one active PDF at a time, matching the original workflow.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = []
    metadatas = []
    for index, chunk in enumerate(chunks):
        stable_text = (
            f"{index}|{chunk['page']}|{chunk['content_type']}|"
            f"{chunk['content']}"
        )
        ids.append(hashlib.sha256(stable_text.encode("utf-8")).hexdigest())
        metadatas.append(
            {
                "page": int(chunk["page"]),
                "content_type": chunk["content_type"],
                "url": chunk["url"] or "",
            }
        )

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas,
    )
    return collection, len(chunks), n_tables


# -----------------------------
# Answer generation
# -----------------------------
def stream_answer(context, question):
    prompt = f"""You are a helpful assistant answering questions about an uploaded PDF.

Use only the supplied context to answer. If the answer is not supported by the
context, say that you could not find it in the PDF. Do not invent facts.
When useful, refer to the source page numbers. URLs in the context are references;
do not claim you opened or verified those websites.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt,
        stream=True,
    )
    for part in response:
        text = part.get("response", "")
        if text:
            yield text


# -----------------------------
# Source rendering
# -----------------------------
def render_sources(sources):
    if not sources:
        st.caption("No source records were returned.")
        return

    for source in sources:
        page = source.get("page", "?")
        content_type = source.get("content_type", "page_text")
        source_text = source.get("text", "")
        url = source.get("url", "")

        st.markdown(
            f'<span class="badge">Page {page}</span>'
            f'<span class="badge">{content_type}</span>',
            unsafe_allow_html=True,
        )
        st.text(source_text[:800] + ("…" if len(source_text) > 800 else ""))

        if url:
            st.markdown(f"[Open referenced link]({url})")


# -----------------------------
# Session state
# -----------------------------
if "collection" not in st.session_state:
    st.session_state.collection = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0
if "n_tables" not in st.session_state:
    st.session_state.n_tables = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar: upload and settings
# -----------------------------
with st.sidebar:
    st.header("📄 PDF RAG Assistant")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Text, tables, scanned pages, embedded-image text, and PDF links are indexed.",
    )

    if uploaded_file is not None:
        is_new_upload = (
            st.session_state.filename != uploaded_file.name
            or st.session_state.collection is None
        )
        if is_new_upload:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                with st.spinner("Extracting PDF text, tables, images, and links…"):
                    collection, n_chunks, n_tables = index_pdf(temp_path)

                if collection is None:
                    st.error(
                        "No extractable text, tables, OCR text, or hyperlinks "
                        "were found in this PDF."
                    )
                    st.session_state.collection = None
                    st.session_state.filename = None
                else:
                    st.session_state.collection = collection
                    st.session_state.filename = uploaded_file.name
                    st.session_state.n_chunks = n_chunks
                    st.session_state.n_tables = n_tables
                    st.session_state.messages = []
                    st.success("PDF indexed successfully.")
            except Exception as exc:
                st.error(f"Could not index this PDF: {exc}")
                st.caption(
                    "Check that Ollama is running only when asking questions, "
                    "and that Tesseract OCR is installed and available on PATH."
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e:
                        print(f"Could not remove temporary PDF: {e}")

    if st.session_state.collection is not None:
        st.divider()
        st.subheader("Document info")
        st.markdown(
            f'<span class="doc-pill">{st.session_state.filename}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Records indexed", st.session_state.n_chunks)
        st.metric("Tables found", st.session_state.n_tables)

        top_k = st.slider(
            "Retrieved records",
            min_value=1,
            max_value=10,
            value=4,
            help="How many matching text, table, OCR, or link records to retrieve.",
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        top_k = 4
        st.caption("Upload a PDF to start.")


# -----------------------------
# Main chat interface
# -----------------------------
st.title("Ask your PDF")
st.caption(
    "Ask questions about page text, tables, scanned pages, text in embedded "
    "images, and hyperlinks extracted from your PDF."
)

if st.session_state.collection is None:
    st.info("Upload a PDF in the sidebar to begin.")
else:
    # Restore previous chat messages and their sources.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    render_sources(message["sources"])

    question = st.chat_input("Ask a question about your PDF…")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        collection = st.session_state.collection
        query_embedding = model.encode([question], show_progress_bar=False)

        with st.chat_message("assistant"):
            try:
                result = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=min(top_k, collection.count()),
                    include=["documents", "metadatas", "distances"],
                )

                documents = result["documents"][0] if result["documents"] else []
                metadatas = result["metadatas"][0] if result["metadatas"] else []

                sources = []
                context_parts = []

                for document, metadata in zip(documents, metadatas):
                    metadata = metadata or {}
                    page = metadata.get("page", "?")
                    content_type = metadata.get("content_type", "page_text")
                    url = metadata.get("url", "")

                    sources.append(
                        {
                            "page": page,
                            "content_type": content_type,
                            "text": document,
                            "url": url,
                        }
                    )
                    context_parts.append(
                        f"\n{document}"
                    )

                context = "\n\n".join(context_parts)
                if not context:
                    context = "No relevant source text was retrieved."

                answer = st.write_stream(stream_answer(context, question))

                with st.expander("Sources"):
                    render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                error_message = (
                    f"An error occurred while retrieving or generating an answer: {exc}"
                )
                st.error(error_message)
                st.caption(
                    f"Confirm that Ollama is running and the model "
                    f"'{LLM_MODEL}' is available (try `ollama list`)."
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )