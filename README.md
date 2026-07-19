<p align="center">
  <h1 align="center">🔍 Rankify</h1>
  <p align="center">
    <strong>Mini Search Engine — Temu Kembali Informasi</strong>
  </p>
  <p align="center">
    <em>TF-IDF · Semantic Search · Hybrid Fusion · Transformer Models · Advanced IR Metrics</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white" alt="Flask">
    <img src="https://img.shields.io/badge/NLP-Sastrawi-green" alt="Sastrawi">
    <img src="https://img.shields.io/badge/Transformers-SBERT-orange" alt="Transformers">
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  </p>
</p>

---

## 📖 Deskripsi

**Rankify** adalah mini search engine berbasis web yang mengimplementasikan konsep-konsep utama dalam **Temu Kembali Informasi (Information Retrieval)**. Aplikasi ini dibangun sebagai tugas UTS mata kuliah Temu Kembali Informasi dan mendemonstrasikan secara interaktif bagaimana sebuah search engine bekerja di balik layar.

**UPDATE:** Sistem telah di-upgrade dengan **Semantic Search** menggunakan transformer models (SBERT) dan **Hybrid Search** dengan multiple fusion strategies untuk meningkatkan relevansi hasil pencarian.

Dataset yang digunakan berisi **50 dokumen** seputar topik **Green Economy** (Ekonomi Hijau), mencakup energi terbarukan, pembangunan berkelanjutan, inovasi ramah lingkungan, dan topik terkait.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🔎 **Pencarian Ranked** | Pencarian dokumen berdasarkan relevansi menggunakan Cosine Similarity |
| 📊 **TF-IDF Scoring** | Pembobotan term dengan Log Frequency Weighting dan Inverse Document Frequency |
| 🤖 **Semantic Search** | Pencarian berbasis makna menggunakan transformer models (SBERT multilingual) |
| 🔀 **Hybrid Search** | 3 fusion strategies: Linear Combination, Reciprocal Rank Fusion (RRF), Semantic Reranking |
| 📈 **Advanced Metrics** | Evaluasi dengan MAP, MRR, NDCG@k, Precision@k, Recall@k |
| 📋 **Inverted Index** | Visualisasi pemetaan setiap term ke dokumen yang mengandungnya |
| ⚙️ **Pipeline Pre-processing** | Demonstrasi langkah-langkah: Case Folding → Punctuation Removal → Tokenization → Stop-word Removal → Stemming |
| 🇮🇩 **Bahasa Indonesia** | Mendukung stemming bahasa Indonesia menggunakan library **Sastrawi** |
| 🌐 **Bilingual Stop-words** | Penghapusan stop-words untuk Bahasa Indonesia dan Bahasa Inggris |
| 📊 **Statistik Engine** | Dashboard statistik: total dokumen, vocabulary size, rata-rata token per dokumen |
| 🔬 **Method Comparison** | Perbandingan side-by-side Traditional vs Semantic vs Hybrid |
| 🎨 **UI Modern** | Antarmuka glassmorphism dengan animasi dinamis, dark mode, dan fully responsive |


---

## 🏗️ Arsitektur & Konsep IR

```
┌────────────────────────────────────────────────────────────┐
│                        USER QUERY                          │
│                    "ekonomi hijau"                          │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   PRE-PROCESSING PIPELINE                    │
│                                                              │
│  1. Case Folding      → "ekonomi hijau"                      │
│  2. Punctuation Removal → "ekonomi hijau"                    │
│  3. Tokenization      → ["ekonomi", "hijau"]                 │
│  4. Stop-word Removal → ["ekonomi", "hijau"]                 │
│  5. Stemming (Sastrawi) → ["ekonomi", "hijau"]               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   TF-IDF COMPUTATION                         │
│                                                              │
│  TF  = 1 + log₁₀(raw_tf)     (Log Frequency Weighting)      │
│  IDF = log₁₀(N / df)                                        │
│  Weight = TF × IDF                                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              COSINE SIMILARITY (VSM)                         │
│                                                              │
│  sim(Q, D) = Σ(wQ × wD) / (|Q| × |D|)                      │
│                                                              │
│  Dokumen diranking berdasarkan skor similarity (descending)  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Instalasi & Menjalankan

### Prasyarat

- Python 3.10 atau lebih baru
- pip (Python package manager)

### Langkah-langkah

1. **Clone repository**
   ```bash
   git clone https://github.com/Sponscar/Mini-Search-Engine-Rankify.git
   cd Mini-Search-Engine-Rankify
   ```

2. **Buat virtual environment (opsional tapi disarankan)**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

5. **Buka browser** dan akses:
   ```
   http://localhost:5000
   ```

---

## 📁 Struktur Proyek

```
Rankify/
├── app.py                  # Flask web application (routes & API endpoints)
├── search_engine.py        # Core search engine (preprocessing, TF-IDF, cosine similarity)
├── analysis.py             # Script analisis: IDF, normalisasi, evaluasi (Precision, Recall, F-Measure)
├── data green economy.csv  # Dataset 50 dokumen tentang Green Economy
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Halaman utama (Jinja2 template)
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet (glassmorphism, dark mode, animations)
│   └── js/
│       └── app.js          # Frontend JavaScript (search, rendering, interactions)
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Halaman utama search engine |
| `POST` | `/api/search` | Pencarian dokumen (Traditional TF-IDF). Body: `{"query": "kata kunci"}` |
| `POST` | `/api/search-semantic` | Pencarian semantic (Transformer). Body: `{"query": "kata kunci"}` |
| `POST` | `/api/search-hybrid` | Pencarian hybrid. Body: `{"query": "...", "method": "rrf", "alpha": 0.5}` |
| `POST` | `/api/compare-methods` | Bandingkan semua metode. Body: `{"query": "...", "top_k": 10}` |
| `GET` | `/api/stats` | Statistik engine (total dokumen, vocabulary size, dll.) |
| `GET` | `/api/model-info` | Info model transformer yang digunakan |
| `GET` | `/api/inverted-index` | Seluruh inverted index |
| `POST` | `/api/preprocessing-demo` | Demo pipeline pre-processing. Body: `{"text": "teks input"}` |


### Contoh Request

```bash
# Pencarian dokumen
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "ekonomi hijau"}'
```

### Contoh Response

```json
{
  "query_original": "ekonomi hijau",
  "query_tokens": ["ekonomi", "hijau"],
  "total_results": 15,
  "results": [
    {
      "rank": 1,
      "doc_id": "D5",
      "score": 0.452831,
      "document": "...",
      "tf_idf_details": {
        "ekonomi": { "tf": 1.3010, "idf": 0.3979, "tfidf": 0.517584 },
        "hijau":   { "tf": 1.0000, "idf": 0.5229, "tfidf": 0.522879 }
      }
    }
  ]
}
```

---

## 📊 Script Analisis (`analysis.py`)

Script `analysis.py` menghasilkan analisis lengkap untuk laporan UTS:

| Bagian | Isi |
|--------|-----|
| **A. Analisis Bobot IDF** | Perbandingan IDF untuk kata umum vs kata jarang, tabel IDF beberapa term |
| **B. Analisis Efek Normalisasi** | Perbandingan ranking Cosine Similarity vs Dot Product (tanpa normalisasi) |
| **C. Evaluasi Sistem** | Precision, Recall, dan F-Measure untuk 2 skenario query berbeda |
| **D. Evaluasi Semantic & Hybrid** | Perbandingan Traditional vs Semantic vs Hybrid dengan MAP, MRR, NDCG@k |

Jalankan script analisis:
```bash
python analysis.py
```

---

## 🛠️ Tech Stack

| Teknologi | Kegunaan |
|-----------|----------|
| **Python** | Bahasa pemrograman utama |
| **Flask** | Web framework untuk backend & API |
| **Pandas** | Membaca dan memproses dataset CSV |
| **PySastrawi** | Stemming & stop-word removal Bahasa Indonesia |
| **Sentence Transformers** | Pre-trained transformer models untuk semantic search (SBERT) |
| **PyTorch** | Deep learning framework (dependency untuk transformers) |
| **Scikit-learn** | Machine learning utilities (cosine similarity, metrics) |
| **NumPy** | Array operations dan numerical computing |
| **HTML/CSS/JS** | Frontend dengan desain modern (glassmorphism, dark mode) |

---

## 📚 Konsep Information Retrieval yang Diimplementasikan

1. **Pre-processing** — Case folding, punctuation removal, tokenization, stop-word removal, stemming
2. **Inverted Index** — Struktur data mapping term → dokumen
3. **TF-IDF** — Term Frequency (Log Weighting) × Inverse Document Frequency
4. **Vector Space Model (VSM)** — Representasi dokumen dan query sebagai vektor
5. **Cosine Similarity** — Pengukuran kemiripan antara vektor query dan dokumen
6. **Ranked Retrieval** — Pengurutan hasil berdasarkan skor relevansi

---

## 👤 Author

**Sponscar**

- GitHub: [@Sponscar](https://github.com/Sponscar)

---

## 📄 License

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<p align="center">
  <strong>Rankify</strong> — Dibuat dengan ❤️ untuk mata kuliah Temu Kembali Informasi
</p>
