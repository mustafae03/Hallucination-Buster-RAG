import json
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss


INDEX_PATH = "index/faiss.index"
META_PATH  = "index/chunks.json"
EMB_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"


_embed_model = None
_faiss_index = None
_chunks = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMB_MODEL)
    return _embed_model

def load_index():
    global _faiss_index, _chunks
    if _faiss_index is None or _chunks is None:
        _faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
    return _faiss_index, _chunks


def retrieve(query: str, k: int = 3):
    index, chunks = load_index()
    model = get_embed_model()

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


def build_prompt(question: str, contexts):
    ctx_block = "\n\n".join(
        [f"[KAYNAK {i+1} | {c['source']} | score={c['score']:.3f}]\n{c['text']}"
         for i, c in enumerate(contexts)]
    )

    prompt = f"""
Sen bir ders projesi asistanısın. Görev: SADECE verilen kaynaklardan cevap üretmek.

Kurallar:
- Kaynakta olmayan bilgi ekleme.
- Emin değilsen açıkça "Kaynaklarda geçmiyor" de.
- Cevabın sonunda 2-3 madde ile hangi kaynakları kullandığını belirt (KAYNAK 1, KAYNAK 2 gibi).

Soru:
{question}

Kaynaklar:
{ctx_block}

Cevap:
"""
    return prompt.strip()


def ollama_generate(prompt: str, retries: int = 2, timeout: int = 300):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.2,
            # İstersen cevapları kısa tut:
            # "num_predict": 256
        }
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Ollama failed after retries: {last_err}")


if __name__ == "__main__":
    question = "RAG nedir ve halüsinasyonu nasıl azaltır?"
    contexts = retrieve(question, k=3)

    prompt = build_prompt(question, contexts)
    answer = ollama_generate(prompt)

    print("QUESTION:", question)
    print("=" * 80)
    print("ANSWER:\n", answer)
    print("=" * 80)
    print("RETRIEVED CONTEXTS:")
    for i, c in enumerate(contexts, 1):
        print(f"[{i}] {c['source']} score={c['score']:.3f} chunk_id={c['chunk_id']}")
