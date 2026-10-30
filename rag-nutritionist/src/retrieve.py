"""Stage 1c: hand-written cosine-similarity top-k retrieval. No vector DB."""
import numpy as np
from embed import embed, load_index

class Retriever:
    def __init__(self, max_chars=800, overlap=100, name="default"):
        self.vecs, self.chunks = load_index(max_chars, overlap, name)

    def search(self, query, k=4):
        q = embed([query])[0]
        scores = self.vecs @ q                      # cosine similarity (vectors are unit length)
        top = np.argsort(-scores)[:k]               # indices of the k highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

if __name__ == "__main__":
    r = Retriever()
    print(len(r.chunks), "chunks;", r.vecs.shape)
    for q in ["How many calories should I cut to lose weight safely?", "Why is fiber good for me?"]:
        print("\nQ:", q)
        for h in r.search(q, 3):
            print(f"  {h['score']:.3f}  {h['id']}  [{h['section']}]")
