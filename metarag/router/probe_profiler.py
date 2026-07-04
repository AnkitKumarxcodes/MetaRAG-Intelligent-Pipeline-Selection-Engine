# router/probe_profiler.py
# Runs per query at ask() time. One cheap retrieval against the real vector_db.
# Answers: how difficult is retrieval for this query?

from __future__ import annotations
from typing import Any


def _chunk_text(chunk) -> str:
    """Extract text — supports Chunk objects or raw strings."""
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "text", None) or getattr(chunk, "page_content", "") or str(chunk)


class ProbeProfiler:
    """
    Does one cheap similarity search against the vector DB.
    Returns signals about how well the corpus matches the query.

    Uses the SAME VectorDBInterface.search() contract as every retriever —
    no LangChain assumptions, works against InMemoryVectorDB, ChromaVectorDB,
    and FAISSVectorDB identically.

    Signals produced:
        avg_similarity      →  how well corpus matches on average
        max_similarity      →  best matching chunk score
        similarity_variance →  spread of scores — high = mixed quality
        redundancy          →  are top chunks saying the same thing?
    """

    def __init__(self, vector_db, embeddings, k: int = 5):
        """
        Args:
            vector_db: an ALREADY-BUILT VectorDBInterface instance
            embeddings: EmbeddingInterface object — needed to embed the query
                        before calling vector_db.search()
            k: number of results to probe with
        """
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.k = k

    def probe(self, query: str) -> dict:
        """One similarity search. Returns retrieval difficulty signals."""
        try:
            query_embedding = self.embeddings.embed(query)
            results = self.vector_db.search(query_embedding, k=self.k)
        except Exception as e:
            print(f"[ProbeProfiler] Search failed: {e}")
            return self._empty()

        if not results:
            return self._empty()

        scores = [score for _, score in results]
        texts = [_chunk_text(chunk) for chunk, _ in results]

        avg = sum(scores) / len(scores)
        var = sum((s - avg) ** 2 for s in scores) / len(scores)

        return {
            "avg_similarity": round(avg, 4),
            "max_similarity": round(max(scores), 4),
            "similarity_variance": round(var, 4),
            "redundancy": round(self._redundancy(texts), 4),
        }

    def _redundancy(self, texts: list) -> float:
        """Word overlap between top chunks — high = repetitive retrieval."""
        if len(texts) < 2:
            return 0.0

        pairs, total = 0, 0.0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                a = set(texts[i].lower().split())
                b = set(texts[j].lower().split())
                if a and b:
                    total += len(a & b) / max(len(a), len(b))
                    pairs += 1

        return total / pairs if pairs else 0.0

    def _empty(self) -> dict:
        return {
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "similarity_variance": 0.0,
            "redundancy": 0.0,
        }