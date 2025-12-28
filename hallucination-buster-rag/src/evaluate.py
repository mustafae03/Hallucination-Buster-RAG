import re
from retrieval import search
from rag_answer import build_prompt, ollama_generate


def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


def context_coverage(answer, contexts):
    """
    Answer'daki kelimelerin ne kadarı
    gerçekten getirilen context'lerde geçiyor?
    """
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(" ".join(contexts))

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


def run_eval(question: str, k=3):
   
    answer_plain = ollama_generate(question)

  
    retrieved = search(question, k=k)

    contexts = [r["text"] for r in retrieved]
    prompt = build_prompt(question, retrieved)
    answer_rag = ollama_generate(prompt)

   
    cov_plain = context_coverage(answer_plain, contexts)
    cov_rag = context_coverage(answer_rag, contexts)

 
    print("\nQUESTION:")
    print(question)

    print("\n--- RAG OFF ANSWER ---")
    print(answer_plain)
    print(f"Context Coverage (RAG OFF): {cov_plain:.3f}")

    print("\n--- RAG ON ANSWER ---")
    print(answer_rag)
    print(f"Context Coverage (RAG ON):  {cov_rag:.3f}")

    print("\n--- RETRIEVED CONTEXTS ---")
    for i, r in enumerate(retrieved, 1):
        print(f"[{i}] score={r['score']:.3f} source={r['source']} chunk_id={r['chunk_id']}")


if __name__ == "__main__":
    q = "RAG nedir ve halüsinasyonu nasıl azaltır?"
    run_eval(q, k=3)
