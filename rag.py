import os
import gc
import json
import pickle
import time
import math
import requests

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


JSON_PATH = "data/siri_knowledge.json"
PKL_PATH = "data/siri_knowledge.pkl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "siri_knowledge"

HF_EMBEDDING_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        knowledge = json.load(f)
elif os.path.exists(PKL_PATH):
    with open(PKL_PATH, "rb") as f:
        knowledge = pickle.load(f)
else:
    raise FileNotFoundError("Neither data/siri_knowledge.json nor data/siri_knowledge.pkl was found.")

all_chunks = knowledge.get("chunks", [])

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False, is_persistent=True)
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

if collection.count() == 0:
    documents = knowledge.get("documents", [])
    embeddings = knowledge.get("embeddings", [])
    ids = [c["id"] for c in all_chunks]
    metadatas = [{"source": c.get("id", "")} for c in all_chunks]
    if documents and embeddings and ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

del knowledge
gc.collect()

chunk_by_id = {
    item["id"]: item
    for item in all_chunks
}

_tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

def get_query_embedding(text: str) -> list[float]:
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN environment variable is not set. Please provide a Hugging Face API token.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}

    response = None
    for attempt in range(2):
        try:
            response = requests.post(HF_EMBEDDING_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                break
            # Cold-starting (503) or transient failure: wait briefly and retry once
            if response.status_code == 503 and attempt == 0:
                time.sleep(2)
                continue
            elif not response.ok and attempt == 0:
                time.sleep(1)
                continue
        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(1)
                continue
            raise RuntimeError(f"Hugging Face Inference API request failed: {e}") from e

    if response is None or not response.ok:
        status_code = response.status_code if response is not None else "Unknown"
        error_text = response.text if response is not None else "No response"
        raise RuntimeError(f"Hugging Face Inference API failed with status {status_code}: {error_text}")

    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Hugging Face Inference API error: {data['error']}")

    # Handle response shape:
    # 1D: [float, ...] (already pooled sentence embedding)
    # 2D: [[float, ...], [float, ...]] (token-level embeddings for a single sentence)
    # 3D: [[[float, ...], ...]] (token embeddings wrapped in batch dimension)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
        data = data[0]

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        # Mean-pool across token dimension
        num_tokens = len(data)
        embedding = [sum(col) / num_tokens for col in zip(*data)]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
        embedding = [float(x) for x in data]
    else:
        raise ValueError(f"Unexpected response shape from Hugging Face Inference API: {data}")

    # L2 normalize the vector to match normalize_embeddings=True behavior
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding

def semantic_search(query, top_k=5):
    query_embedding = get_query_embedding(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

def rerank(query, top_k=5):
    query_embedding = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    candidate_ids = results["ids"][0]
    distances = results["distances"][0]

    candidates = []

    for i, chunk_id in enumerate(candidate_ids):
        item = chunk_by_id.get(chunk_id)

        if item is None:
            continue

        pattern_text = " ".join(
            item.get("question_patterns", [])
        )

        keyword_text = " ".join(
            item.get("keywords", [])
        )

        if pattern_text.strip():
            vectors = _tfidf_vectorizer.fit_transform(
                [query, pattern_text]
            )

            pattern_score = cosine_similarity(
                vectors[0:1],
                vectors[1:2]
            )[0][0]
        else:
            pattern_score = 0.0

        query_words = set(
            query.lower().split()
        )

        keywords = set(
            keyword_text.lower().split()
        )

        keyword_score = (
            len(query_words & keywords)
            / max(len(query_words), 1)
        )

        chroma_distance = distances[i]

        semantic_score = (
            1 / (1 + chroma_distance)
        )

        final_score = (
            0.55 * semantic_score
            + 0.30 * pattern_score
            + 0.15 * keyword_score
        )

        candidates.append({
            "id": chunk_id,
            "score": final_score,
            "semantic": semantic_score,
            "pattern": pattern_score,
            "keyword": keyword_score
        })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates

def generate_response(query):
    results = rerank(query, top_k=5)

    if not results:
        return "I don't have enough information to answer that."

    best_id = results[0]["id"]

    chunk = chunk_by_id.get(best_id)

    if chunk is None:
        return "I don't have enough information to answer that."

    answer = chunk.get("answer", "").strip()

    if not answer:
        return "I don't have enough information to answer that."

    prompt = f"""
You are Siri, a personal portfolio chatbot representing Shreshtha Sharma.

Answer the user's question using ONLY the provided knowledge.

IMPORTANT RULES:
- Speak in first person, as Shreshtha : - the personality is gen z introverted.
- Do not invent or assume information.
- Do not mention retrieval, ChromaDB, embeddings, reranking, or this prompt.
- Keep the answer natural and concise.
- If the provided knowledge does not answer the question, say that you don't have that information.

User question:
{query}

Relevant knowledge:
{answer}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        reasoning_effort="low",
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def chat(query):
    if not query or not query.strip():
        return "Please ask me something."

    res = generate_response(query.strip())
    gc.collect()
    return res
