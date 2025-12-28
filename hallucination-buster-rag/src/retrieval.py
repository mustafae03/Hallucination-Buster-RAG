import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_PATH = "index/faiss.index"
META_PATH  = "index/chunks.json"
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def search(query: str, k=3):
    index, chunks = load_index()
    model = SentenceTransformer(EMB_MODEL)

    qvec = model.encode([query], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")

    scores, ids = index.search(qvec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "chunk_id": chunks[idx]["chunk_id"],
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"]
        })
    return results

if __name__ == "__main__":
    q = "RAG nedir ve halüsinasyonu nasıl azaltır?"
    res = search(q, k=3)
    print("QUERY:", q)
    print("-" * 80)
    for i, r in enumerate(res, 1):
        print(f"[{i}] score={r['score']:.3f} source={r['source']} chunk_id={r['chunk_id']}")
        print(r["text"])
        print("-" * 80)
