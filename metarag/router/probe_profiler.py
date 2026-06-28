# router/probe_profiler.py
# Runs per query at ask() time. One cheap BM25 retrieval.
# Answers: how difficult is retrieval for this query?

from __future__ import annotations
from typing import Any


class ProbeProfiler:
    """
    Does one cheap similarity search against the vector DB.
    Returns signals about how well the corpus matches the query.

    Signals produced:
        avg_similarity      →  how well corpus matches on average
        max_similarity      →  best matching chunk score
        similarity_variance →  spread of scores — high = mixed quality
        redundancy          →  are top chunks saying the same thing?

    Probe weights = 50% of routing decision.
    These signals are grounded in reality — not assumptions.
    """

    def __init__(self, vectordb, k: int = 5):
        self.vectordb = vectordb
        self.k        = k

    def probe(self, query: str) -> dict:
        """
        One similarity search. Returns retrieval difficulty signals.
        """
        try:
            results = self.vectordb.db.similarity_search_with_relevance_scores(
                query, k=self.k
            )
        except Exception as e:
            print(f"[ProbeProfiler] Search failed: {e}")
            return self._empty()

        if not results:
            return self._empty()

        scores = [score for _, score in results]
        texts  = [doc.page_content for doc, _ in results]

        avg = sum(scores) / len(scores)
        var = sum((s - avg) ** 2 for s in scores) / len(scores)

        return {
            "avg_similarity":      round(avg, 4),
            "max_similarity":      round(max(scores), 4),
            "similarity_variance": round(var, 4),
            "redundancy":          round(self._redundancy(texts), 4),
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
            "avg_similarity":      0.0,
            "max_similarity":      0.0,
            "similarity_variance": 0.0,
            "redundancy":          0.0,
        }
