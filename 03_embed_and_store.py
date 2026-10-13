from sentence_transformers import SentenceTransformer
import chromadb

# ---- Step 1: our structure-aware chunks from before ----
chunks = [
    """1. Introduction
This system uses a distributed cache to reduce latency across regions.""",

    """2. Configuration
The cache TTL is set via the CACHE_TTL environment variable, default 3600 seconds.

Table 1: Supported Regions
Region      | Latency (ms) | Status
us-east-1   | 12           | Active
eu-west-1   | 45           | Active
ap-south-1  | 78           | Beta""",

    """3. Error Handling
If a cache miss occurs, the system falls back to the primary database with a circuit breaker pattern."""
]

# ---- Step 2: load embedding model ----
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good for testing

# ---- Step 3: generate embeddings ----
print("Generating embeddings...")
embeddings = model.encode(chunks)
print(f"Generated {len(embeddings)} embeddings, each of dimension {len(embeddings[0])}")

# ---- Step 4: store in Chroma ----
print("Setting up Chroma...")
client = chromadb.PersistentClient(path="./chroma_db")  # saves to disk in project folder
collection = client.get_or_create_collection(name="tech_docs")

# add chunks with unique IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=ids
)

print(f"\nStored {collection.count()} chunks in Chroma.")

# ---- Step 5: quick retrieval test ----
query = "what is the latency for ap-south-1?"
query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

print(f"\n---- Query: '{query}' ----")
print("Top matching chunk(s):")
for doc in results['documents'][0]:
    print("\n" + doc)