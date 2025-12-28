# Hallucination-Buster-RAG

Bu proje, Büyük Dil Modellerinin (LLM) sık karşılaşılan problemlerinden biri olan **halüsinasyon (uydurma bilgi üretimi)** problemini azaltmak amacıyla geliştirilmiş **Retrieval-Augmented Generation (RAG)** tabanlı bir mini soru–cevap sistemidir.

Projede, bir dil modelinin **harici bilgi olmadan (RAG OFF)** ve **dokümana dayalı olarak (RAG ON)** verdiği cevaplar karşılaştırmalı olarak incelenmiştir.

---

## Proje Amacı

Bu projenin temel amacı:

- LLM’lerin bağlam olmadan çalıştığında neden halüsinasyon ürettiğini göstermek  
- RAG yaklaşımının bu problemi nasıl azalttığını deneysel olarak ortaya koymak  
- RAG ON ve RAG OFF senaryolarını ölçülebilir bir metrik ile karşılaştırmaktır  

---

## Problem Tanımı

Büyük Dil Modelleri, eğitim sırasında gördükleri verilerden genelleme yaparak cevap üretir.  
Ancak **gerçek zamanlı veya özel doküman bilgisi** gerektiren sorularda, model güvenilir olmayan veya tamamen uydurma cevaplar üretebilir.  
Bu durum literatürde **hallucination** problemi olarak adlandırılır.

---

## Proje Çözümü 
Bu problem, **Retrieval-Augmented Generation (RAG)** yaklaşımı kullanılarak ele alınmıştır.

RAG yaklaşımında dil modeli:

- Tek başına cevap üretmez  
- Önce harici bir doküman koleksiyonundan ilgili bilgileri çeker  
- Ardından bu bağlamı kullanarak cevap üretir  

Bu sayede model, yalnızca kendi parametrelerine değil, **gerçek doküman içeriğine** dayanır.

---

## Sistem Mimarisi

1. Dokümanlar küçük parçalara (chunk) bölünür  
2. Her chunk embedding modeli ile sayısal vektöre dönüştürülür  
3. Vektörler FAISS tabanlı bir vektör veritabanında saklanır  
4. Kullanıcı sorusu embedding’e çevrilir  
5. En alakalı top-k doküman parçaları retrieve edilir  
6. Bu bağlam LLM’e verilerek cevap üretilir  

---

## RAG OFF ve RAG ON Karşılaştırması

- **RAG OFF:**  
  Model yalnızca kendi parametrelerine dayanarak cevap üretir.  
  Halüsinasyon riski yüksektir.

- **RAG ON:**  
  Model, retrieve edilen doküman parçalarını kullanarak cevap üretir.  
  Daha güvenilir ve kaynağa dayalı çıktılar elde edilir.

---

## Değerlendirme Yöntemi

Bu projede klasik sınıflandırma metrikleri yerine, metin üretimine uygun bir ölçüm kullanılmıştır.

### Context Coverage Oranı

Üretilen cevaptaki kelimelerin, kullanılan bağlam (context) içinde bulunma oranını ifade eder.

- Düşük oran → Halüsinasyon ihtimali yüksek  
- Yüksek oran → Kaynağa dayalı cevap  

Bu oran, RAG ON ve RAG OFF senaryoları için karşılaştırılmıştır.

---

## Kullanılan Teknolojiler

- Python  
- Ollama (Local LLM çalıştırma)  
- Sentence-Transformers (Embedding)  
- FAISS (Vector Database)  
- Retrieval-Augmented Generation (RAG)  

---

## Proje Dosya Yapısı
hallucination-buster-rag/
├── src/
│ ├── retrieval.py
│ ├── rag_answer.py
│ ├── evaluate_batch.py
├── index/
│ ├── faiss.index
│ └── chunks.json
├── notes1.txt
├── results.txt
├── README.md
└── requirements.txt

