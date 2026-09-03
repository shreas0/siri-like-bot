#!/usr/bin/env python
"""
Pre-download the SentenceTransformer model for offline use on Render.
This runs during the Render build phase (not at runtime).
"""
import os
import sys
from pathlib import Path

print("=== Render Build: Pre-downloading embedding model ===", flush=True)

# Set environment variables to minimize resource usage
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Model directory (will be included in Render deployment)
model_dir = Path("models") / "all-MiniLM-L6-v2"
model_dir.parent.mkdir(exist_ok=True)

try:
    print("Loading and saving SentenceTransformer model 'all-MiniLM-L6-v2'...", flush=True)
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.save(str(model_dir))
    print(f"[OK] Model downloaded and saved to {model_dir}", flush=True)
    
    # Quick sanity check loading from saved local path
    local_model = SentenceTransformer(str(model_dir))
    test_embedding = local_model.encode(["test"], normalize_embeddings=True)
    print(f"[OK] Local model verified and can generate embeddings (shape: {test_embedding.shape})", flush=True)
    
    print("=== Build phase complete ===", flush=True)
    sys.exit(0)
    
except Exception as e:
    print(f"[ERROR] FAILED to download model: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
