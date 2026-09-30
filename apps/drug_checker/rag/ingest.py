import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'faiss_index')
CORPUS_PATH = os.path.join(PROJECT_ROOT, 'data', 'guidelines_corpus.json')


def build_faiss_index():
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Indexing {len(documents)} document sections...")
    texts = [f"{doc['source_doc']} - {doc['section']}: {doc['content']}" for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, 'index.faiss')
    meta_path = os.path.join(INDEX_DIR, 'metadata.json')

    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)

    print(f"FAISS index successfully saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_faiss_index()
