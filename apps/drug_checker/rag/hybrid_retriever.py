import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INDEX_DIR = os.path.join(PROJECT_ROOT, 'data', 'faiss_index')
CORPUS_PATH = os.path.join(PROJECT_ROOT, 'data', 'guidelines_corpus.json')
DENSE_MODEL_NAME = 'all-MiniLM-L6-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class HybridRetriever:
    def __init__(self):
        self.index_path = os.path.join(INDEX_DIR, 'index.faiss')
        self.meta_path = os.path.join(INDEX_DIR, 'metadata.json')
        self.dense_model = None
        self.reranker = None
        self.faiss_index = None
        self.bm25_index = None
        self.metadata = []
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            return

        # Load metadata corpus
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        # Load FAISS Index & Dense Embedding Model
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self.faiss_index = faiss.read_index(self.index_path)

        # Build BM25 Inverted Index over metadata text
        tokenized_corpus = [
            f"{doc['source_doc']} {doc['section']} {doc['content']}".lower().split()
            for doc in self.metadata
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # Load Cross-Encoder Reranker
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load CrossEncoder reranker ({e}). Falling back to RRF fusion.")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.35,
        filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.metadata or self.faiss_index is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        # -------------------------------------------------------------
        # Step 1: Dense FAISS Vector Search
        # -------------------------------------------------------------
        query_vector = self.dense_model.encode([clean_query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        faiss_scores, faiss_indices = self.faiss_index.search(query_vector, len(self.metadata))

        faiss_ranks = {}
        for rank, idx in enumerate(faiss_indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                faiss_ranks[idx] = rank + 1

        # -------------------------------------------------------------
        # Step 2: BM25 Keyword Search
        # -------------------------------------------------------------
        tokenized_query = clean_query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1]

        bm25_ranks = {}
        for rank, idx in enumerate(bm25_sorted_indices):
            bm25_ranks[idx] = rank + 1

        # -------------------------------------------------------------
        # Step 3: Reciprocal Rank Fusion (RRF)
        # -------------------------------------------------------------
        rrf_scores = {}
        K = 60
        for idx in range(len(self.metadata)):
            r_faiss = faiss_ranks.get(idx, 999)
            r_bm25 = bm25_ranks.get(idx, 999)
            rrf_score = (1.0 / (K + r_faiss)) + (1.0 / (K + r_bm25))
            rrf_scores[idx] = rrf_score

        # Filter candidates by optional metadata filter
        candidate_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        if filter_doc_id:
            candidate_indices = [
                i for i in candidate_indices
                if self.metadata[i].get('doc_id') == filter_doc_id
            ]

        # Top 5 candidates for Cross-Encoder Reranking
        top_candidates = candidate_indices[:5]
        if not top_candidates:
            return []

        # -------------------------------------------------------------
        # Step 4: Cross-Encoder Re-ranker Scoring
        # -------------------------------------------------------------
        results = []
        if self.reranker:
            pairs = [
                (clean_query, f"{self.metadata[idx]['source_doc']} - {self.metadata[idx]['section']}: {self.metadata[idx]['content']}")
                for idx in top_candidates
            ]
            reranker_scores = self.reranker.predict(pairs)
            
            # Normalize sigmoid scores
            sigmoid_scores = 1.0 / (1.0 + np.exp(-reranker_scores))
            
            reranked_pairs = sorted(
                zip(top_candidates, sigmoid_scores),
                key=lambda x: x[1],
                reverse=True
            )

            for idx, score in reranked_pairs[:top_k]:
                conf = float(score)
                if conf >= score_threshold:
                    meta = self.metadata[idx].copy()
                    meta['confidence'] = round(conf, 4)
                    meta['search_mode'] = 'hybrid_bm25_faiss_reranked'
                    results.append(meta)
        else:
            # RRF fallback if CrossEncoder is unavailable
            for idx in top_candidates[:top_k]:
                faiss_score = float(faiss_scores[0][list(faiss_indices[0]).index(idx)]) if idx in faiss_indices[0] else 0.0
                if faiss_score >= score_threshold:
                    meta = self.metadata[idx].copy()
                    meta['confidence'] = round(faiss_score, 4)
                    meta['search_mode'] = 'hybrid_rrf_fusion'
                    results.append(meta)

        return results
