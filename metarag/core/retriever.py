# metarag/core/retriever.py

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

def _chunk_text(chunk) -> str:
    """Extract text from a chunk — supports Chunk objects or raw strings."""
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "text", None) or getattr(chunk, "page_content", "") or str(chunk)


class RetrieverInterface(ABC):
    """
    Contract for retriever implementations.
    User can implement their own or use built-ins.
    """
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Retrieve top k chunks similar to query.
        
        Args:
            query: query string
            k: number of results
        
        Returns:
            list of (chunk_text, similarity_score) tuples
        """
        pass


# ─────────────────────────────────────────────────────────────
# Retriever 1: BM25 (sparse, keyword-based)
# ─────────────────────────────────────────────────────────────

class BM25Retriever(RetrieverInterface):
    def __init__(self, chunks: List):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 required for BM25Retriever. Install: pip install rank-bm25"
            )

        self.chunks = chunks  # original objects (Chunk or str) — kept for return
        corpus_texts = [_chunk_text(c) for c in chunks]
        self.corpus = [t.lower().split() for t in corpus_texts]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 4) -> List[Tuple]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]  # returns original Chunk objects


# ─────────────────────────────────────────────────────────────
# Retriever 2: Dense (embedding-based)
# ─────────────────────────────────────────────────────────────

# AFTER
class DenseRetriever(RetrieverInterface):
    """
    Dense embedding-based retriever.
    Expects vector_db to already be built — this class does NOT build it.
    """
    def __init__(self, chunks: List[str], embeddings, vector_db):
        """
        Args:
            chunks: list of text chunks (for reference only, not re-embedded)
            embeddings: EmbeddingInterface object (used only for query embedding)
            vector_db: an ALREADY-BUILT VectorDBInterface instance
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_db = vector_db
        # No build() call here — caller is responsible for building once

    def retrieve(self, query: str, k: int = 4):
        query_embedding = self.embeddings.embed(query)
        return self.vector_db.search(query_embedding, k=k)

# ─────────────────────────────────────────────────────────────
# Retriever 3: Hybrid (BM25 + Dense combined)
# ─────────────────────────────────────────────────────────────

class HybridRetriever(RetrieverInterface):
    def __init__(self, chunks: List[str], embeddings, vector_db, alpha: float = 0.5):
        """
        Args:
            vector_db: an ALREADY-BUILT VectorDBInterface instance,
                       shared with DenseRetriever — not built here.

        NOTE: merge logic below keys on chunk object identity. This is correct
        ONLY because BM25Retriever and DenseRetriever/vector_db are constructed
        from the exact same chunk list instance (no copying) — see metarag.py's
        _setup_retrievers(). If that ever changes, this merge breaks silently.

        """
        self.bm25 = BM25Retriever(chunks)                       # own index, no embeddings needed
        self.dense = DenseRetriever(chunks, embeddings, vector_db)  # no longer builds internally
        self.alpha = alpha
    
    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """Hybrid retrieval with interpolation. Guards against empty results from either retriever."""
        bm25_results = self.bm25.retrieve(query, k=k * 2)
        dense_results = self.dense.retrieve(query, k=k * 2)

        all_chunks = {}

        for chunk, score in bm25_results:
            all_chunks[chunk] = {"bm25": score, "dense": 0}

        for chunk, score in dense_results:
            if chunk not in all_chunks:
                all_chunks[chunk] = {"bm25": 0, "dense": 0}
            all_chunks[chunk]["dense"] = score

        # Guard: if neither retriever returned anything, there's nothing to rank —
        # return empty instead of crashing on max([]).
        if not all_chunks:
            print(f"[HybridRetriever] No results from either BM25 or Dense for query: '{query[:50]}...'")
            return []

        hybrid_scores = {}

        bm25_scores = [s["bm25"] for s in all_chunks.values()]
        dense_scores = [s["dense"] for s in all_chunks.values()]

        max_bm25 = max(bm25_scores) or 1   # safe now — bm25_scores is guaranteed non-empty
        max_dense = max(dense_scores) or 1

        for chunk, scores in all_chunks.items():
            bm25_norm = scores["bm25"] / max_bm25
            dense_norm = scores["dense"] / max_dense
            hybrid_scores[chunk] = (1 - self.alpha) * bm25_norm + self.alpha * dense_norm

        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_results


# ─────────────────────────────────────────────────────────────
# Retriever 4: MMR (Maximal Marginal Relevance)
# ─────────────────────────────────────────────────────────────

class MMRRetriever(RetrieverInterface):
    """
    Maximal Marginal Relevance retriever.
    Balances relevance + diversity.
    Hand-coded (no sklearn dependency).

    Computes similarity directly against its own local embeddings —
    does NOT go through vector_db.search(), so there's no text-based
    index lookup and no risk of duplicate-chunk-text bugs.
    """

    def __init__(self, chunks: List, embeddings, vector_db, lambda_param: float = 0.6):
        self.chunks = chunks
        self.embeddings = embeddings
        self.vector_db = vector_db
        self.lambda_param = lambda_param

        chunk_texts = [_chunk_text(c) for c in chunks]           # ← new: extract text first
        chunk_embeddings = embeddings.embed_documents(chunk_texts)  # ← embed text, not Chunk objects
        self.chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

        norms = np.linalg.norm(self.chunk_embeddings, axis=1)
        self._norms = np.where(norms == 0, 1e-8, norms)

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """MMR retrieval — pure local computation, index-safe by construction."""
        query_embedding = np.array(self.embeddings.embed(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        query_norm = query_norm if query_norm != 0 else 1e-8

        # Step 1: relevance of every chunk to the query (vectorized)
        relevance_scores = (self.chunk_embeddings @ query_embedding) / (self._norms * query_norm)

        # Step 2: candidate pool = top 2k by relevance, as indices directly — no text matching
        candidate_size = min(k * 2, len(self.chunks))
        candidate_indices = list(np.argsort(relevance_scores)[::-1][:candidate_size])

        # Step 3: MMR greedy selection over indices
        selected: List[int] = []
        remaining = set(candidate_indices)

        while len(selected) < k and remaining:
            best_idx, best_score = None, -float("inf")

            for idx in remaining:
                rel_score = float(relevance_scores[idx])

                if selected:
                    sim_to_selected = [
                        float(np.dot(self.chunk_embeddings[idx], self.chunk_embeddings[s]))
                        / (self._norms[idx] * self._norms[s])
                        for s in selected
                    ]
                    max_similarity = max(sim_to_selected)
                else:
                    max_similarity = 0.0

                mmr_score = self.lambda_param * rel_score - (1 - self.lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        # Step 4: return REAL relevance scores, not a flat 1.0 placeholder
        return [(self.chunks[idx], float(relevance_scores[idx])) for idx in selected]


