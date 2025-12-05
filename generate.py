import json
import os
import glob
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi model embedding
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")

# Kata kunci untuk topik (diperbarui)
TOPIC_KEYWORDS = {
    "retribusi": ["retribusi", "jasa umum", "jasa usaha", "perizinan", "tarif retribusi"],
    "pajak": ["pajak", "PBB-P2", "reklame", "penanggung pajak", "objek pajak", "pajak hiburan", "pajak parkir"],
    "izin": ["izin", "bangunan gedung", "TKA", "persetujuan", "IMB", "SIUP", "izin lingkungan"],
    "sanksi": ["sanksi", "denda", "keterlambatan", "pelanggaran", "penyitaan"],
    "pelaporan": ["pelaporan", "lapor", "batas waktu", "surat keputusan", "pengajuan"],
    "pengelolaan": ["pengelolaan", "pasar", "kebersihan", "pasar tradisional", "UMKM"],
    "keuangan": ["anggaran", "APBD", "pendapatan daerah", "belanja daerah", "dana perimbangan"],
    "pemerintahan": ["kepala daerah", "DPRD", "otonomi daerah", "pelayanan publik", "aparatur sipil"],
    "infrastruktur": ["tata ruang", "RTRW", "drainase", "persampahan", "lingkungan hidup"],
    "kesejahteraan": ["kesehatan masyarakat", "pendidikan daerah", "bantuan sosial", "disabilitas"],
    "hukum": ["dasar hukum", "pasal", "ayat", "undang-undang", "harmonisasi", "mengingat"]
}

# Load semua chunks dari folder
def load_all_chunks(chunks_folder):
    chunk_data = []
    json_files = glob.glob(os.path.join(chunks_folder, "*.json"))
    for json_path in json_files:
        with open(json_path, "r") as f:
            chunks = json.load(f)
        for chunk in chunks:
            chunk_data.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "filename": chunk["filename"]
            })
    return chunk_data

# Identifikasi topik berdasarkan kata kunci
def identify_topic(text):
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return topic
    return "lainnya"

# Kelompokkan chunk per dokumen dan topik
def group_chunks(chunks):
    doc_chunks = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk["filename"]].append(chunk)
    doc_topics = {}
    for filename, chunks in doc_chunks.items():
        topics = defaultdict(int)
        for chunk in chunks:
            topic = identify_topic(chunk["text"])
            topics[topic] += 1
        doc_topics[filename] = max(topics, key=topics.get, default="lainnya")
    return doc_chunks, doc_topics

# Generate query berdasarkan topik
def generate_query_for_chunk(chunk, topic):
    chunk_text = chunk["text"]
    chunk_id = chunk["chunk_id"]
    filename = chunk["filename"]
    
    if topic == "retribusi":
        query = f"Apa jenis {topic} yang diatur dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "pajak":
        query = f"Apa ketentuan tentang {topic} daerah dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "izin":
        query = f"Bagaimana prosedur pengajuan {topic} dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "sanksi":
        query = f"Apa {topic} untuk pelanggaran peraturan dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "pelaporan":
        query = f"Apa kewajiban {topic} dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "pengelolaan":
        query = f"Bagaimana {topic} aset daerah diatur dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "keuangan":
        query = f"Apa pengaturan {topic} daerah dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "pemerintahan":
        query = f"Apa tugas {topic} daerah dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "infrastruktur":
        query = f"Apa ketentuan tentang {topic} daerah dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "kesejahteraan":
        query = f"Bagaimana {topic} masyarakat diatur dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    elif topic == "hukum":
        query = f"Apa {topic} yang mendasari {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    else:
        query = f"Apa ketentuan umum dalam {filename.replace('.pdf', '')}?"
        reference = f"{chunk_text[:200]}..."
    
    return {
        "id": None,
        "text": query,
        "relevant": [chunk_id],
        "reference": reference
    }

# Generate query lintas dokumen
def generate_cross_doc_query(doc_chunks, doc_topics, embedding_model, num_queries=8):
    cross_queries = []
    candidate_chunks = []
    
    # Pilih chunk dengan topik relevan
    for filename, chunks in doc_chunks.items():
        for chunk in chunks:
            topic = identify_topic(chunk["text"])
            if topic in ["retribusi", "izin", "pengelolaan", "pajak", "hukum"]:
                candidate_chunks.append(chunk)
    
    # Hitung embedding
    chunk_texts = [chunk["text"] for chunk in candidate_chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts, batch_size=32, show_progress_bar=True)
    
    for _ in range(num_queries * 2):  # Coba lebih banyak untuk filter
        idx1, idx2 = np.random.choice(len(candidate_chunks), 2, replace=False)
        chunk1, chunk2 = candidate_chunks[idx1], candidate_chunks[idx2]
        
        # Cek kesamaan semantik
        sim = cosine_similarity([chunk_embeddings[idx1]], [chunk_embeddings[idx2]])[0][0]
        if 0.5 < sim < 0.8:  # Relevan tapi tidak terlalu mirip
            query = f"Apa hubungan antara {identify_topic(chunk1['text'])} di {chunk1['filename'].replace('.pdf', '')} dan {identify_topic(chunk2['text'])} di {chunk2['filename'].replace('.pdf', '')}?"
            reference = f"Dasar hukum di {chunk1['filename'].replace('.pdf', '')}: {chunk1['text'][:100]}... Pelaksanaan di {chunk2['filename'].replace('.pdf', '')}: {chunk2['text'][:100]}..."
            cross_queries.append({
                "id": None,
                "text": query,
                "relevant": [chunk1["chunk_id"], chunk2["chunk_id"]],
                "reference": reference
            })
    
    return cross_queries[:num_queries]

# Main
def generate_query_set(chunks_folder, output_path="query_set_100_docs.json", num_queries=100):
    chunks = load_all_chunks(chunks_folder)
    doc_chunks, doc_topics = group_chunks(chunks)
    
    # Pastikan cakupan 61 dokumen
    queries_per_doc = max(1, (num_queries - 8) // len(doc_topics))  # 8 untuk cross-doc
    queries = []
    
    # Generate query per dokumen
    for filename, chunks in doc_chunks.items():
        topic = doc_topics[filename]
        selected_chunks = np.random.choice(chunks, min(queries_per_doc, len(chunks)), replace=False)
        for chunk in selected_chunks:
            queries.append(generate_query_for_chunk(chunk, topic))
    
    # Tambahkan query lintas dokumen
    cross_queries = generate_cross_doc_query(doc_chunks, doc_topics, embedding_model)
    queries.extend(cross_queries)
    
    # Potong ke 100 query
    queries = queries[:num_queries]
    
    # Assign ID
    for i, query in enumerate(queries, 1):
        query["id"] = i
    
    # Simpan
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    # Validasi cakupan
    covered_docs = set()
    for query in queries:
        for chunk_id in query["relevant"]:
            filename = next(chunk["filename"] for chunk in chunks if chunk["chunk_id"] == chunk_id)
            covered_docs.add(filename)
    
    print(f"Generated {len(queries)} queries covering {len(covered_docs)} documents")
    return queries

# Main
if __name__ == "__main__":
    chunks_folder = "chunks_folder"  # Ganti dengan path folder Anda
    generate_query_set(chunks_folder)
