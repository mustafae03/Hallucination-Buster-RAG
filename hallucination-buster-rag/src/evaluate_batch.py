

import re
import time
import sys

from rag_answer import ollama_generate, build_prompt
from retrieval import search

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


QUESTIONS = [
    "RAG nedir ve halüsinasyonu nasıl azaltır?",
    "Embedding nedir? Neden vektöre çeviriyoruz?",
    "FAISS ne işe yarar?",
    "Chunking neden gerekli? Chunk size yanlış olursa ne olur?",
    "RAG ile arama (retrieval) adımı neden kritik?",
    "Cosine similarity neyi ölçer? Normalize embeddings niye var?",
    "Top-k neden 3 seçilir? 10 yapsak ne değişir?",
    "RAG olmadan LLM neden halüsinasyon yapar?",
    "Vector database ile klasik keyword search farkı nedir?",
    "RAG sistemlerinde prompt neden kritiktir?",
]


def safe(s: str) -> str:
    
    if s is None:
        return ""
    return str(s).encode("utf-8", "replace").decode("utf-8")


def tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))


def context_coverage(answer: str, contexts: list[str]) -> float:
    """
    Answer'daki kelimelerin ne kadarı gerçekten getirilen context'lerde geçiyor?
    (Basit bir metrik: kelime kesişimi / answer kelime sayısı)
    """
    answer_tokens = tokenize(answer or "")
    context_tokens = tokenize(" ".join(contexts) if contexts else "")
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def run_once(question: str, k: int = 3):

    answer_plain = ollama_generate(question)

    
    retrieved = search(question, k=k)  # list[{"text","source","chunk_id","score"}]
    contexts = [r["text"] for r in retrieved]
    prompt = build_prompt(question, retrieved)
    answer_rag = ollama_generate(prompt)

    
    cov_plain = context_coverage(answer_plain, contexts)
    cov_rag = context_coverage(answer_rag, contexts)

    top_sources = [r.get("source", "?") for r in retrieved]

    return answer_plain, answer_rag, cov_plain, cov_rag, top_sources


def main():
    print("\nRunning mini benchmark...\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {safe(q)}")

        start_time = time.time()
        answer_plain, answer_rag, cov_plain, cov_rag, top_sources = run_once(q, k=3)
        elapsed = time.time() - start_time

        print(f"  RAG OFF Answer: {safe(answer_plain).strip()}")
        print(f"  RAG OFF Context Coverage: {cov_plain:.2%}")

        print(f"  RAG ON Answer: {safe(answer_rag).strip()}")
        print(f"  RAG ON Context Coverage: {cov_rag:.2%}")

        print(f"  Top Sources: {top_sources}")
        print(f"  Time taken: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
