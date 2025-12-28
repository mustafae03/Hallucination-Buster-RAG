import re, json
from pathlib import Path

def normalize(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(text: str, chunk_size=180, overlap=40):
   
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks

def ingest_folder(data_dir="data", out_path="chunks/chunks.jsonl"):
    Path("chunks").mkdir(exist_ok=True)

    files = [p for p in Path(data_dir).glob("*") if p.suffix.lower() in [".txt", ".md"]]
    if not files:
        raise FileNotFoundError("data/ klasöründe .txt veya .md dosyası bulunamadı.")

    cid = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for fp in files:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            text = normalize(raw)

            for ch in chunk_text(text):
                row = {"chunk_id": cid, "source": fp.name, "text": ch}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                cid += 1

    print(f"OK: {len(files)} dosyadan toplam {cid} chunk yazıldı -> {out_path}")

if __name__ == "__main__":
    ingest_folder()
