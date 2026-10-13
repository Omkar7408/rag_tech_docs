from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ---- Step 1: connect to existing Chroma collection ----
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="tech_docs")

model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- Step 2: retrieve relevant chunk(s) ----
query = "what is the latency for ap-south-1?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

retrieved_chunks = results['documents'][0]
context = "\n\n".join(retrieved_chunks)

print("---- Retrieved context ----")
print(context)

# ---- Step 3: send to local LLM (Ollama) for generation ----
prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

response = ollama.generate(model="llama3.2:1b", prompt=prompt)

print("\n---- Generated Answer ----")
print(response['response'])