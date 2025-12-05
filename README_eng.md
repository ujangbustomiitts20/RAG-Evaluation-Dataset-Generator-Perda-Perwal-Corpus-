
````markdown
# RAG Evaluation Dataset Generator for Indonesian Local Regulations (Perda/Perwal)

This repository provides a Python utility for generating an **evaluation dataset** for Retrieval-Augmented Generation (RAG) systems built on top of Indonesian local regulations (Perda/Perwal).  
The script produces approximately **100 synthetic queries** and labels the **relevant chunks** that should be returned by a retriever (e.g., Qdrant).

This dataset acts as **ground truth** for evaluating:

- Retrieval accuracy (Recall@k, Hit@k)
- Ranking quality (MRR)
- Embedding performance
- Chunking and semantic indexing quality

> Important:  
> This script **does not embed vectors** and **does not interact with Qdrant**.  
> Embedding and vector indexing are expected to be done beforehand.  
> The purpose here is solely to generate an **evaluation dataset** for benchmarking a RAG pipeline.

---

## 🧱 Pipeline Context

This script is typically used in **Stage 3** of a RAG workflow:

1. **PDF → Text → JSON chunks**  
   - Extract text (e.g., pdfplumber)  
   - Chunk with LangChain TextSplitter  
   - Produce `chunks_folder/*.json`

2. **Embedding → Vector DB (Qdrant / FAISS / Milvus)**  
   - Convert chunks to embeddings  
   - Upsert into vector DB with metadata  

3. **(This script) Generate Evaluation Dataset** ✔️  
   - Read chunk JSON files  
   - Detect document-level topics  
   - Generate queries + relevant chunk mapping  
   - Produce `query_set_100_docs.json`  
   - Ready for retriever evaluation

---

## ✨ Key Features

### ✔ Topic-Aware Query Generation  
Uses a customizable dictionary of topic → keywords to determine each document’s dominant topic and generate queries accordingly.

### ✔ Cross-Document Semantic Queries  
Uses `SentenceTransformer("intfloat/multilingual-e5-base")` and cosine similarity (0.5–0.8) to create queries that span **two different documents**, testing multi-document retrieval.

### ✔ Full Corpus Coverage  
Queries are distributed across all documents (e.g., 61 PDF sources from Indonesian Perda/Perwal).

### ✔ Clean, JSON-Based Output  
Outputs `query_set_100_docs.json` with structure:

```json
{
  "id": 1,
  "text": "What are the regional tax provisions regulated in perda_pajak_1?",
  "relevant": ["perda_pajak_1_0001"],
  "reference": "Snippet of the chunk text for human reference..."
}
````

---

## 🧩 Code Structure

### 1. Embedding Model Initialization

```python
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
```

Used only for generating cross-document semantic queries.

### 2. Topic Keyword Dictionary

```python
TOPIC_KEYWORDS = {
    "retribusi": ["retribusi", "jasa umum", "tarif retribusi"],
    "pajak": ["pajak", "reklame", "PBB-P2"],
    ...
}
```

You may freely modify or extend this mapping.

### 3. Chunk Loading

```python
load_all_chunks(chunks_folder)
```

Loads all `*.json` chunk files containing:

```json
{
  "chunk_id": "perda_pajak_1_0001",
  "text": "...",
  "filename": "perda_pajak_1.pdf"
}
```

### 4. Topic Detection

```python
identify_topic(text)
```

Returns the matching topic or `"lainnya"` (other).

### 5. Document Grouping & Topic Assignment

```python
group_chunks(chunks)
```

Groups chunks by filename and assigns a dominant topic per document.

### 6. Query Generation (Per Document)

```python
generate_query_for_chunk(chunk, topic)
```

Creates template-based queries such as:

```
"What are the regional tax provisions in Perda_PajakReklame?"
```

### 7. Cross-Document Query Generation

```python
generate_cross_doc_query(...)
```

* Selects candidate chunks from priority topics
* Computes embeddings
* Randomly picks two chunks with 0.5 < similarity < 0.8
* Creates relational questions like:

```
"What is the relationship between tax in perda_pajak_1 and retribusi in perda_pasar_2?"
```

### 8. Main Function: Generating 100 Queries

```python
generate_query_set(chunks_folder, output_path, num_queries=100)
```

* ~92 document-specific queries
* 8 semantic cross-document queries
* Assigns IDs
* Saves final dataset
* Prints corpus coverage:

```
Generated 100 queries covering 61 documents
```

---

## 📁 Input Format

Place all chunk files in:

```
chunks_folder/
    perda_pajak_1.json
    perda_retribusi_2.json
    perwal_perizinan_3.json
    ...
```

Each JSON file contains a list of chunks with:

```json
{
  "chunk_id": "unique_id",
  "text": "chunk content",
  "filename": "original_document.pdf"
}
```

---

## 📤 Output Format

File: `query_set_100_docs.json`

Example:

```json
[
  {
    "id": 1,
    "text": "What are the tax regulations in perda_pajak_1?",
    "relevant": ["perda_pajak_1_0001"],
    "reference": "Tax regulations in this section include..."
  },
  {
    "id": 95,
    "text": "What is the relationship between tax in perda_pajak_1 and retribusi in perda_retribusi_2?",
    "relevant": ["perda_pajak_1_0002", "perda_retribusi_2_0007"],
    "reference": "Basis in perda_pajak_1: ... Implementation in perda_retribusi_2: ..."
  }
]
```

---

## ⚙ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2. Install dependencies

Create a `requirements.txt`:

```
sentence-transformers
scikit-learn
numpy
```

Install via:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Script

Ensure this structure:

```
.
├── generate_queries.py
├── chunks_folder/
│   ├── perda_xxx.json
│   └── ...
```

Run:

```bash
python generate_queries.py
```

This will produce:

```
query_set_100_docs.json
```

---

## 🧪 How to Use This Dataset (Evaluation)

This dataset can be used to evaluate any vector retriever (Qdrant, FAISS, Milvus, Elasticsearch vector search).

Typical evaluation steps:

1. Encode `query["text"]`
2. Search in Qdrant
3. Compare top-k retrieved `chunk_id`s against `query["relevant"]`
4. Compute:

   * Recall@k
   * Hit@k
   * Mean Reciprocal Rank (MRR)
   * NDCG@k

This provides a **quantitative baseline** for RAG retrieval quality.

---

## ⚠ Limitations

* Topic detection is keyword-based (not ML-driven).
* Query forms follow templates, not natural LLM-generated questions.
* Cross-document sampling is probabilistic; results vary slightly per run.
* Best suited as a **baseline evaluation dataset**, not a human-curated gold standard.

---

## 📜 License

MIT License (or modify as needed).

---

## 👤 Author

Developed as part of an ongoing experiment to build a **Regulation Copilot** for Indonesian local government regulations using Retrieval-Augmented Generation (RAG).


