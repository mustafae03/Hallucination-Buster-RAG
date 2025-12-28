import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_PATH = "chunks/chunks.jsonl"
OUT_DIR = "index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks(path=CHUNKS_PATH):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    if not chunks:
        raise ValueError("chunks.jsonl boş görünüyor.")
    return chunks

def build_faiss_index(texts, model_name=EMB_MODEL):
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )
    vecs = np.asarray(vecs, dtype="float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine benzerlik için (normalize edildi)
    index.add(vecs)
    return index

def main():
    Path(OUT_DIR).mkdir(exist_ok=True)

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    index = build_faiss_index(texts)

    faiss.write_index(index, f"{OUT_DIR}/faiss.index")
    with open(f"{OUT_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"OK: {len(chunks)} chunk indexlendi -> {OUT_DIR}/faiss.index")
    print(f"OK: metadata yazıldı -> {OUT_DIR}/chunks.json")

if __name__ == "__main__":
    main()
