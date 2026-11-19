import json
import chromadb

# 1. Load your schema rules from Step 2
with open("data/schema_docs.json", "r") as f:
    schema_docs = json.load(f)

# 2. Initialize local ChromaDB vector store
client = chromadb.PersistentClient(path="vector_store")

# Delete old collection if re-running so there are no duplicates
try:
    client.delete_collection(name="schema_memory")
except Exception:
    pass

collection = client.create_collection(name="schema_memory")

# 3. Add table descriptions and business rules into ChromaDB
documents = [item["description"] for item in schema_docs]
ids = [item["table"] for item in schema_docs]

collection.add(
    documents=documents,
    ids=ids
)

# 4. Quick test query to verify RAG retrieval works
test_results = collection.query(
    query_texts=["What is the average house value and median income?"],
    n_results=2
)

print("Success! Schema RAG vector store created in 'vector_store/'")
print("Retrieved context:", test_results["documents"][0])
