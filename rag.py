import os
import gc
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Initialize PyTorch in single-thread inference configuration first to avoid runtime library duplication
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
torch.set_grad_enabled(False)

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


PKL_PATH = "data/siri_knowledge.pkl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "siri_knowledge"

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
if os.path.isdir(LOCAL_MODEL_DIR):
    os.environ["HF_HUB_OFFLINE"] = "1"
EMBEDDING_MODEL_NAME = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open(PKL_PATH, "rb") as f:
    knowledge = pickle.load(f)

all_chunks = knowledge.get("chunks", [])
del knowledge
gc.collect()

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False, is_persistent=True)
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

if collection.count() == 0:
    raise RuntimeError(
        f"Chroma collection '{COLLECTION_NAME}' is empty at {CHROMA_PATH}. "
        "Populate it first by running data/build_knowledge.py before starting the API."
    )

chunk_by_id = {
    item["id"]: item
    for item in all_chunks
}

_embedding_model = None
_tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": True}
        )
        _embedding_model.eval()
        gc.collect()

    return _embedding_model

def semantic_search(query, top_k=5):
    with torch.inference_mode():
        query_embedding = get_embedding_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1
        )[0]

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

def rerank(query, top_k=5):
    with torch.inference_mode():
        query_embedding = get_embedding_model().encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1
        )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
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
