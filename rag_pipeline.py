import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------------------------------------------------
# 1. Load Environment Variables (Groq API Key)
# ---------------------------------------------------------
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------
# 2. Setup Vector Database and Embedding Model
# ---------------------------------------------------------
print("[System] Initializing embedding model (this may download weights on the first run)...")
# We use a small, fast local embedding model from HuggingFace
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("[System] Initializing ChromaDB vector store...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Create or get a collection
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# ---------------------------------------------------------
# 3. Document Ingestion (Chunking & Embedding)
# ---------------------------------------------------------
def ingest_document(filepath: str):
    print(f"\n[System] Ingesting document: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Simple chunking: split by sentences (periods)
    chunks = [chunk.strip() for chunk in text.split(".") if len(chunk.strip()) > 10]
    
    print(f"[System] Document split into {len(chunks)} chunks. Embedding into vector space...")
    
    # Generate embeddings for each chunk
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Generate unique IDs for each chunk
    ids = [f"doc1_chunk_{i}" for i in range(len(chunks))]
    
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print("[System] Ingestion complete. Vector database is ready.")

# ---------------------------------------------------------
# 4. Retrieval-Augmented Generation (RAG) Query
# ---------------------------------------------------------
def rag_query(user_question: str):
    print(f"\nUser Question: {user_question}")
    print("[System] Searching vector database for relevant context...")
    
    # 1. Embed the user's question
    question_embedding = embedding_model.encode([user_question]).tolist()
    
    # 2. Query the vector database for the top 2 most similar chunks
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=2
    )
    
    retrieved_context = results['documents'][0]
    context_string = "\n".join(retrieved_context)
    
    print("\n[System] Retrieved Context:")
    print("--------------------------------------------------")
    print(context_string)
    print("--------------------------------------------------\n")
    
    # 3. Construct the prompt for the LLM
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context_string}

Question:
{user_question}
"""
    
    # 4. Generate the final answer using Groq
    print("[System] Generating answer using Groq LLM...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    print("\nAssistant:")
    print(answer)
    return answer

if __name__ == "__main__":
    # Ensure our sample document exists
    doc_path = "sample_document.txt"
    
    # Check if collection is empty, if so, ingest
    if collection.count() == 0:
        ingest_document(doc_path)
    else:
        print("[System] Vector database already contains documents. Skipping ingestion.")
        
    # Run a test query
    rag_query("What is the core mechanism of the Transformer architecture?")
