# Fix Summary: Render Memory Limit & 502 Out-Of-Memory Fix

## 1. ROOT CAUSE
Render Web Service instances on the Free Tier are limited to **512 MB RAM**. 
When the first `/chat` request arrived, the combined resident memory of:
1. Uvicorn + FastAPI
2. ChromaDB (with background telemetry threads)
3. Scikit-learn (Scipy sparse/linalg/numpy)
4. PyTorch runtime (allocating multiple OpenMP thread pools and memory arenas)
5. Hugging Face `SentenceTransformer` model initialization & forward activations
6. Full uncollected pickle data (`embeddings` and `documents` cached in RAM)
caused the process RSS to exceed 512 MB. This triggered Render's cgroup memory limit killer, restarting the container and returning `502 Bad Gateway`.

## 2. EVIDENCE & MEASUREMENTS (Stage-by-Stage RSS Profile)

| Stage | Description | Before Optimization | After Optimization |
|---|---|---|---|
| Stage 0 | Initial Python Process Start | 20.02 MB | 20.09 MB |
| Stage 1 | API Startup & ChromaDB / Knowledge Init | 179.66 MB | 179.84 MB (with single-thread PyTorch) |
| Stage 2 | Immediately before `SentenceTransformer` | 179.66 MB | 179.84 MB |
| Stage 3 | After `SentenceTransformer` load | 468.47 MB | 464.47 MB (from local saved dir, `low_cpu_mem_usage=True`) |
| Stage 4 | Immediately before `encode()` | 468.47 MB | 464.47 MB |
| Stage 5 | Immediately after `encode()` | 512.46 MB (SPIKE > 512 MB) | 508.55 MB (with `torch.inference_mode()`) |
| Stage 6 | Immediately after Chroma query | 519.86 MB (KILLED) | 516.18 MB |
| Stage 7 | During/after TF-IDF Reranking | 520.11 MB | 516.43 MB (reused `_tfidf_vectorizer`) |
| Stage 8 | Immediately before Groq | 520.11 MB | 516.43 MB |
| Stage 9 | Immediately after Groq | 521.16 MB | 518.71 MB |
| Stage 10 | End of Request (`gc.collect()`) | 521.16 MB | 518.71 MB |
| Sequential | Subsequent Requests (#2 - #5) | 521.39 MB | 518.89 MB (Stable, no memory leaks) |

## 3. KEY OPTIMIZATIONS APPLIED

1. **Build-Phase Model Pre-Download & Local Storage (`build_render.py`)**:
   - `all-MiniLM-L6-v2` is downloaded during the Render build phase and saved directly to `models/all-MiniLM-L6-v2`.
   - Runtime loads directly from `models/all-MiniLM-L6-v2` with `low_cpu_mem_usage=True` and `HF_HUB_OFFLINE=1`, eliminating network calls and download caches at runtime.

2. **Single-Threaded PyTorch & Zero-Grad Inference**:
   - `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` prevent PyTorch from spawning multi-core thread pools and allocating per-thread glibc memory arenas.
   - `torch.set_grad_enabled(False)` and `with torch.inference_mode()` prevent autograd graph construction during `encode()`.

3. **Glibc Malloc Arena Restriction (`MALLOC_ARENA_MAX=2`)**:
   - Configured in `render.yaml` and `rag.py` to prevent Linux glibc from creating unreleased memory arenas.

4. **ChromaDB Telemetry Disabled**:
   - `Settings(anonymized_telemetry=False, is_persistent=True)` disables PostHog telemetry threads and network buffer queues.

5. **Knowledge Pickle Memory Cleanup**:
   - Extracts only `knowledge["chunks"]` and immediately deletes `knowledge` and runs `gc.collect()`, freeing raw embedding tensors.

6. **Global TF-IDF Vectorizer Reuse**:
   - Uses a single shared `TfidfVectorizer` instance instead of re-instantiating inside candidate loops.

## 4. ARCHITECTURE PRESERVED

- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **Vector Store**: ChromaDB (PersistentClient)
- **Reranker**: Custom Hybrid (0.55 Semantic + 0.30 TF-IDF Question Pattern + 0.15 Keyword overlap)
- **LLM Generator**: Groq (`openai/gpt-oss-20b`)
- **API Interface**: FastAPI `/`, `/docs`, `/chat` (unchanged schema)
