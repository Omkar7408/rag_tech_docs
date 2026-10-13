import pymupdf as fitz
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb

file_path = "data/samsung.pdf"  # your real filename

def table_to_text(table):
    lines = []
    for row in table:
        clean_row = [str(cell).replace("\n", " ") if cell else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)

def build_chunks(file_path):
    chunks = []
    fitz_doc = fitz.open(file_path)

    with pdfplumber.open(file_path) as pdf:
        for i in range(len(pdf.pages)):
            text = fitz_doc[i].get_text().strip()
            tables = pdf.pages[i].extract_tables()
            table_texts = [table_to_text(t) for t in tables]

            combined = text
            if table_texts:
                combined += "\n\n[TABLES ON THIS PAGE]\n" + "\n\n".join(table_texts)

            if combined.strip():
                chunks.append({"page": i + 1, "content": combined})

    fitz_doc.close()
    return chunks

print("Extracting and chunking document...")
chunks = build_chunks(file_path)
print(f"Total chunks: {len(chunks)}")

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
texts = [c["content"] for c in chunks]
embeddings = model.encode(texts)

print("Storing in Chroma...")
client = chromadb.PersistentClient(path="./chroma_db_real")
collection = client.get_or_create_collection(name="microwave_manual")

# clear old data if rerunning
existing_ids = collection.get()["ids"]
if existing_ids:
    collection.delete(ids=existing_ids)

ids = [f"page_{c['page']}" for c in chunks]
metadatas = [{"page": c["page"]} for c in chunks]

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    ids=ids,
    metadatas=metadatas
)

print(f"Stored {collection.count()} chunks in Chroma (persisted at ./chroma_db_real)")
# ---- quick retrieval test ----
query = "what is the output power in watts?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

print(f"\n---- Query: '{query}' ----")
for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    print(f"\n[Page {meta['page']}]")
    print(doc[:300])