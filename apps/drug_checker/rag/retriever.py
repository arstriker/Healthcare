import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'faiss_index')

class GuidelineRetriever:
    def __init__(self):
        self.index_path = os.path.join(INDEX_DIR, 'index.faiss')
        self.meta_path = os.path.join(INDEX_DIR, 'metadata.json')
        self.model = None
        self.index = None
        self.metadata = []
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            return
        
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.35):
        if self.index is None or self.model is None or not self.metadata:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            confidence = float(score)
            if confidence >= score_threshold:
                meta = self.metadata[idx].copy()
                meta['confidence'] = round(confidence, 4)
                results.append(meta)

        return results
