"""Stage 1b: embeddings + a 'vector store' that is just a numpy array."""
import json, pathlib
import numpy as np
from chunk import chunk_corpus, ROOT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts):
    """Returns L2-normalised float32 vectors, so cosine similarity == dot product."""
    v = get_model().encode(list(texts), convert_to_numpy=True, show_progress_bar=False).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def build_index(max_chars=800, overlap=100, name="default"):
    chunks = chunk_corpus(max_chars, overlap)
    vecs = embed([c["text"] for c in chunks])
    d = ROOT / "data" / "index"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", vecs)
    (d / f"{name}.json").write_text(json.dumps(chunks))
    return vecs, chunks

def load_index(max_chars=800, overlap=100, name="default"):
    d = ROOT / "data" / "index"
    if not (d / f"{name}.npy").exists():
        return build_index(max_chars, overlap, name)
    return np.load(d / f"{name}.npy"), json.loads((d / f"{name}.json").read_text())
