import json
import pickle
import chromadb

from sentence_transformers import SentenceTransformer

JSON_PATH = "about_me.json"
PKL_PATH = "siri_knowledge.pkl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "siri_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def get_all_chunks(data):
    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict) and "id" in item
        ]

    chunks = []

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "id" in item:
                        chunks.append(item)

            elif isinstance(value, dict) and "id" in value:
                chunks.append(value)

    return chunks

all_chunks = get_all_chunks(data)

documents = []

for item in all_chunks:
    text = f"""
Topic: {item.get('topic', '')}

Questions:
{chr(10).join(item.get('question_patterns', []))}

Keywords:
{', '.join(item.get('keywords', []))}

Answer:
{item.get('answer', '')}
""".strip()

    documents.append(text)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

knowledge = {
    "chunks": all_chunks,
    "documents": documents,
    "embeddings": embeddings.tolist()
}

with open(PKL_PATH, "wb") as f:
    pickle.dump(knowledge, f)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

ids = [item["id"] for item in all_chunks]

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist()
)

print(f"Created {PKL_PATH}")
print(f"Processed {len(all_chunks)} knowledge chunks")
print(f"Created ChromaDB collection: {COLLECTION_NAME}")