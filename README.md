
````markdown
# RAG Evaluation Dataset Generator for Perda/Perwal

Skrip ini digunakan untuk **membangun dataset evaluasi** berupa kumpulan *query–chunk relevan* dari korpus dokumen Perda/Perwal yang sudah di-*chunk* ke format JSON.

Tujuan utamanya adalah menghasilkan sekitar **100 query** yang:

- Mencakup **semua dokumen** (misalnya 61 file PDF asal)  
- Terbagi menjadi:
  - **Query per dokumen** (~92)
  - **Query lintas dokumen** (8)
- Dilengkapi informasi **chunk mana yang seharusnya relevan** untuk setiap query  
- Menggunakan model embedding `intfloat/multilingual-e5-base` untuk membangun query lintas dokumen (cross-document) secara semantik

> Catatan penting:  
> Repository ini **tidak melakukan embedding ke Qdrant** dan **tidak memanggil database vektor**.  
> Ini murni *utility* untuk membuat **dataset evaluasi** yang nantinya dipakai untuk menguji kualitas retriever (mis. Qdrant) dalam sistem RAG.

---

##  Konteks Pipeline

Biasanya skrip ini dipakai di tahap ke-3 dari pipeline:

1. **Chunking PDF → JSON**  
   - Ekstraksi teks per halaman  
   - Chunking dengan TextSplitter  
   - Simpan ke `chunks_folder/*.json`

2. **Embedding → Vector DB (Qdrant atau lainnya)**  
   - Mengubah setiap chunk jadi embedding  
   - Menyimpannya ke koleksi vektor dengan metadata (chunk_id, filename, dll.)

3. **(Skrip ini) Generate Dataset Evaluasi**   
   - Membaca semua chunk JSON  
   - Mengidentifikasi topik per dokumen  
   - Menghasilkan query + label chunk relevan  
   - Output ke `query_set_100_docs.json`  
   - Siap dipakai untuk evaluasi RAG (Recall@k, MRR, dsb.)

---

##  Fitur Utama

-  **Topic-aware query generation**  
  Skrip menggunakan `TOPIC_KEYWORDS` untuk mendeteksi topik (pajak, retribusi, izin, sanksi, keuangan, dll.) dan menghasilkan query yang sesuai konteks dokumen.

-  **Cross-document semantic queries**  
  Menggunakan `SentenceTransformer("intfloat/multilingual-e5-base")` dan cosine similarity untuk memilih pasangan chunk dari dokumen berbeda yang **relevan tapi tidak terlalu mirip** (similarity antara 0.5–0.8), lalu membangun query lintas dokumen.

-  **Coverage multi-dokumen**  
  Secara desain, query yang dihasilkan mencakup seluruh dokumen yang ada di `chunks_folder` (misalnya 61 file PDF asal, yang direpresentasikan dalam JSON).

-  **Output JSON siap-evaluasi**  
  Output utama: `query_set_100_docs.json`, berisi 100 objek query dengan struktur:

  ```json
  {
    "id": 1,
    "text": "Apa ketentuan tentang pajak daerah dalam Perda_PajakReklame?",
    "relevant": ["chunk_00123"],
    "reference": "Potongan teks chunk sebagai konteks..."
  }
````

---

##  Struktur Kode (Ringkas)

Skrip ini terdiri dari beberapa bagian utama:

1. **Inisialisasi model embedding**

   ```python
   embedding_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
   ```

2. **Definisi `TOPIC_KEYWORDS`**

   Kamus topik → daftar kata kunci, misalnya:

   ```python
   TOPIC_KEYWORDS = {
       "retribusi": ["retribusi", "jasa umum", "jasa usaha", "perizinan", "tarif retribusi"],
       "pajak": ["pajak", "PBB-P2", "reklame", "penanggung pajak", "objek pajak", "pajak hiburan", "pajak parkir"],
       ...
   }
   ```

3. **Fungsi `load_all_chunks(chunks_folder)`**

   * Membaca semua file `*.json` di sebuah folder
   * Menggabungkan semua chunk ke dalam satu list

4. **Fungsi `identify_topic(text)`**

   * Mencari kata kunci yang cocok di dalam teks
   * Mengembalikan nama topik (`"pajak"`, `"izin"`, dll.) atau `"lainnya"`

5. **Fungsi `group_chunks(chunks)`**

   * Mengelompokkan chunk berdasarkan `filename`
   * Menentukan topik dominan per dokumen

6. **Fungsi `generate_query_for_chunk(chunk, topic)`**

   * Membuat kalimat query berbasis topik dan nama dokumen
   * Contoh:
     `Apa ketentuan tentang pajak daerah dalam Perda_PajakReklame?`

7. **Fungsi `generate_cross_doc_query(...)`**

   * Memilih candidate chunk dari topik prioritas (`retribusi`, `izin`, `pengelolaan`, `pajak`, `hukum`)
   * Menghitung embedding semua candidate chunk
   * Random pilih pasangan chunk, cek `cosine_similarity`
   * Jika 0.5 < similarity < 0.8 → buat query lintas dokumen

     * Contoh:
       `Apa hubungan antara pajak di Perda_PajakReklame dan retribusi di Perda_PasarTradisional?`

8. **Fungsi `generate_query_set(...)` (fungsi utama)**

   * Load semua chunk
   * Kelompokkan per dokumen dan tentukan topik dominan
   * Hitung `queries_per_doc = (num_queries - 8) // jumlah_dokumen`
   * Generate query per dokumen (topik → template)
   * Generate query lintas dokumen (8 query)
   * Potong agar total 100 query
   * Assign ID (1..100)
   * Simpan ke `query_set_100_docs.json`
   * Cetak informasi cakupan dokumen:

     ```text
     Generated 100 queries covering 61 documents
     ```

---

##  Struktur Input

### Folder chunk

Default: `chunks_folder/`

Di dalamnya terdapat file JSON, misalnya:

* `perda_pajak_1.json`
* `perda_retribusi_2.json`
* `perwal_perizinan_3.json`
* dst.

### Struktur tiap file JSON

Setiap file JSON berisi **list** objek chunk, dengan minimal field:

```json
[
  {
    "chunk_id": "perda_pajak_1_0001",
    "text": "Ketentuan pajak daerah ditetapkan...",
    "filename": "perda_pajak_1.pdf"
  },
  {
    "chunk_id": "perda_pajak_1_0002",
    "text": "Objek pajak reklame adalah...",
    "filename": "perda_pajak_1.pdf"
  }
]
```

> Penting:
>
> * `chunk_id` harus unik per chunk
> * `filename` digunakan untuk:
>
>   * pengelompokan per dokumen
>   * membuat nama dokumen di dalam query (mis. `Perda_PajakReklame` dari `perda_pajak_1.pdf`)

---

##  Struktur Output

Output utama: `query_set_100_docs.json`

Contoh isi:

```json
[
  {
    "id": 1,
    "text": "Apa ketentuan tentang pajak daerah dalam perda_pajak_1?",
    "relevant": ["perda_pajak_1_0001"],
    "reference": "Ketentuan pajak daerah ditetapkan..."
  },
  {
    "id": 95,
    "text": "Apa hubungan antara pajak di perda_pajak_1 dan retribusi di perda_retribusi_2?",
    "relevant": ["perda_pajak_1_0002", "perda_retribusi_2_0007"],
    "reference": "Dasar hukum di perda_pajak_1: ... Pelaksanaan di perda_retribusi_2: ..."
  }
]
```

Field:

* `id` → ID berurutan (1..num_queries)
* `text` → kalimat query (dalam Bahasa Indonesia)
* `relevant` → list `chunk_id` yang *seharusnya* relevan untuk query tersebut
* `reference` → potongan teks chunk (untuk manusia, tidak dipakai model)

---

## ⚙️ Instalasi & Dependensi

### 1. Clone repo

```bash
git clone https://github.com/USERNAME/NAMA_REPO.git
cd NAMA_REPO
```

### 2. Buat virtual environment (opsional tapi direkomendasikan)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# atau
.\.venv\Scripts\activate   # Windows
```

### 3. Install dependensi

Buat file `requirements.txt` seperti:

```txt
sentence-transformers
scikit-learn
numpy
```

Lalu install:

```bash
pip install -r requirements.txt
```

---

##  Cara Menjalankan

Pastikan struktur folder:

```bash
.
├── generate_queries.py        # skrip ini
├── chunks_folder/             # berisi file JSON chunk
│   ├── perda_pajak_1.json
│   ├── perda_retribusi_2.json
│   └── ...
```

Jalankan:

```bash
python generate_queries.py
```

Atau jika nama filenya berbeda, sesuaikan `if __name__ == "__main__":`:

```python
if __name__ == "__main__":
    chunks_folder = "chunks_folder"  # ganti jika perlu
    generate_query_set(chunks_folder)
```

Setelah selesai, akan muncul file:

```bash
query_set_100_docs.json
```

Dan di terminal:

```text
Generated 100 queries covering 61 documents
```

---

## 🔧 Penyesuaian & Konfigurasi

### Ubah jumlah query

Secara default:

```python
generate_query_set(chunks_folder, output_path="query_set_100_docs.json", num_queries=100)
```

Jika ingin mengubah:

```python
generate_query_set(chunks_folder, num_queries=200)
```

> Catatan: 8 di antaranya tetap dialokasikan untuk query lintas dokumen, kecuali kakak ubah `num_queries` dan parameter di `generate_cross_doc_query`.

### Ubah topik & kata kunci

Edit `TOPIC_KEYWORDS` sesuai domain:

```python
TOPIC_KEYWORDS = {
    "retribusi": [...],
    "pajak": [...],
    "izin": [...],
    ...
}
```

Query per chunk akan menyesuaikan otomatis.

---

##  Hubungan dengan Qdrant / RAG

Skrip ini **tidak memanggil Qdrant**.
Namun, output `query_set_100_docs.json` bisa digunakan untuk:

1. Mengambil `query["text"]`
2. Meng-encode dengan model embedding yang sama (atau berbeda)
3. Mengirim ke Qdrant sebagai **search query**
4. Mengambil hasil top-k chunk (berdasarkan `chunk_id` atau metadata)
5. Membandingkan hasil itu dengan `query["relevant"]`
6. Menghitung:

   * Recall@k
   * Hit@k
   * MRR
   * dsb.

Dengan begitu, kakak bisa mengukur:

* seberapa baik Qdrant mengembalikan chunk yang tepat
* seberapa bagus model embedding yang digunakan untuk domain regulasi daerah

---

##  Keterbatasan

* Deteksi topik berbasis **keyword sederhana**, belum menggunakan classifier ML.
* Query yang dihasilkan **berbentuk template** (bukan natural question hasil LLM).
* Cross-document query bergantung pada random sampling dan threshold cosine similarity 0.5–0.8.
* Dataset ini lebih cocok sebagai **baseline evaluasi** daripada gold standard yang dikurasi manual.

---

## Lisensi

 MIT License

---

##  Author

Dikembangkan sebagai bagian dari eksperimen membangun **Regulation Copilot** untuk dokumen Perda/Perwal dan evaluasi sistem **Retrieval-Augmented Generation (RAG)** pada domain pemerintahan daerah.


