
import os
import pickle
import chromadb
from dotenv import load_dotenv

load_dotenv()
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

PKL_PATH = "data/siri_knowledge.pkl"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "siri_knowledge"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

with open(PKL_PATH, "rb") as f:
    knowledge = pickle.load(f)

all_chunks = knowledge["chunks"]
documents = knowledge["documents"]
embeddings = knowledge["embeddings"]

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

ids = [item["id"] for item in all_chunks]

if collection.count() == 0:
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

def semantic_search(query, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

def rerank(query, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    candidate_ids = results["ids"][0]
    distances = results["distances"][0]

    data_map = {
        item["id"]: item
        for item in all_chunks
    }

    candidates = []

    for i, chunk_id in enumerate(candidate_ids):
        item = data_map.get(chunk_id)

        if item is None:
            continue

        pattern_text = " ".join(
            item.get("question_patterns", [])
        )

        keyword_text = " ".join(
            item.get("keywords", [])
        )

        if pattern_text.strip():
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )

            vectors = vectorizer.fit_transform(
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

    chunk = next(
        (
            item
            for item in all_chunks
            if item["id"] == best_id
        ),
        None
    )

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

    return generate_response(query.strip())
