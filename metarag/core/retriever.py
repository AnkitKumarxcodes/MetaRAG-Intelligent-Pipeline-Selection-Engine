# retriever.py

from __future__ import annotations
from typing import List, Any, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# Base Retriever
# ─────────────────────────────────────────────────────────────

class BaseRetriever:
    """
    Base class for all MetaRAG retrievers.
    Every retriever must implement:
        - retrieve(query)              → List of documents
        - retrieve_with_score(query)   → List of (document, score) pairs
    """
    name: str = "base"

    def retrieve(self, query: str) -> List[Any]:
        raise NotImplementedError

    def retrieve_with_score(self, query: str) -> List[Tuple[Any, float]]:
        raise NotImplementedError

    def _check_db(self, db):
        if db is None:
            raise ValueError(
                "Vector DB not initialized. "
                "Call VectorDB.build() or VectorDB.load() first."
            )


# ─────────────────────────────────────────────────────────────
# 1. Dense Retriever
# ─────────────────────────────────────────────────────────────

class DenseRetriever(BaseRetriever):
    """
    Semantic retrieval using vector similarity.
    Uses Chroma or FAISS under the hood via VectorDB.

    Best for:
        - Conceptual / semantic queries
        - "How does X work?" "Why does Y happen?"
        - Long, nuanced questions

    Usage:
        retriever = DenseRetriever(vectordb, k=5)
        docs      = retriever.retrieve("what is relativity?")
        scored    = retriever.retrieve_with_score("what is relativity?")
    """
    name = "dense"

    def __init__(self, vectordb, k: int = 5):
        self.vectordb = vectordb
        self.k        = k

    def retrieve(self, query: str) -> List[Any]:
        self._check_db(self.vectordb.db)
        return self.vectordb.db.similarity_search(query, k=self.k)

    def retrieve_with_score(self, query: str) -> List[Tuple[Any, float]]:
        """
        Returns (document, relevance_score) pairs.
        Score range: 0.0 (irrelevant) → 1.0 (perfect match)
        Used by meta-evaluator to compare pipeline confidence.
        """
        self._check_db(self.vectordb.db)
        return self.vectordb.db.similarity_search_with_relevance_scores(
            query, k=self.k
        )

    def as_langchain_retriever(self):
        """Return LangChain-compatible retriever interface."""
        self._check_db(self.vectordb.db)
        return self.vectordb.db.as_retriever(
            search_kwargs={"k": self.k}
        )


# ─────────────────────────────────────────────────────────────
# 2. BM25 Retriever
# ─────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """
    Keyword-based sparse retrieval using BM25 algorithm.
    No embeddings, no vector DB — pure text matching.

    Best for:
        - Exact keyword / term queries
        - Short, specific queries ("error code 404", "version 2.1")
        - Queries with proper nouns, IDs, codes

    Usage:
        retriever = BM25Retriever(chunks, k=5)
        docs      = retriever.retrieve("Einstein relativity 1905")
    """
    name = "bm25"

    def __init__(self, chunks: List[Any], k: int = 5):
        from langchain_community.retrievers import BM25Retriever as LangChainBM25

        texts     = [c.text for c in chunks]
        metadatas = [getattr(c, "metadata", {}) for c in chunks]

        self.retriever = LangChainBM25.from_texts(
            texts=texts,
            metadatas=metadatas,
        )
        self.retriever.k = k
        self.k           = k
        self._texts      = texts    # kept for score approximation

    def retrieve(self, query: str) -> List[Any]:
        return self.retriever.invoke(query)

    def retrieve_with_score(self, query: str) -> List[Tuple[Any, float]]:
        """
        BM25 does not natively return normalised scores.
        We approximate by computing BM25 scores and normalising to [0, 1].
        """
        from rank_bm25 import BM25Okapi

        tokenized   = [t.split() for t in self._texts]
        bm25        = BM25Okapi(tokenized)
        raw_scores  = bm25.get_scores(query.split())

        max_score   = max(raw_scores) if max(raw_scores) > 0 else 1.0
        norm_scores = [s / max_score for s in raw_scores]

        docs = self.retrieve(query)

        # match docs back to their scores by content
        score_map = {
            self._texts[i]: norm_scores[i]
            for i in range(len(self._texts))
        }

        return [
            (doc, score_map.get(doc.page_content, 0.0))
            for doc in docs
        ]

    def as_langchain_retriever(self):
        """Return LangChain-compatible retriever interface."""
        return self.retriever


# ─────────────────────────────────────────────────────────────
# 3. Hybrid Retriever (BM25 + Dense)
# ─────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Combines BM25 sparse retrieval + dense semantic retrieval
    using LangChain's EnsembleRetriever with weighted scoring.

    alpha controls the balance:
        alpha = 0.0  →  pure BM25  (keyword-heavy)
        alpha = 0.5  →  balanced   (default)
        alpha = 1.0  →  pure dense (semantic-heavy)

    Best for:
        - General purpose queries
        - Mixed keyword + conceptual questions
        - Safe default when query type is unclear

    Usage:
        retriever = HybridRetriever(vectordb, chunks, k=5, alpha=0.5)
        docs      = retriever.retrieve("how does relativity affect GPS?")
    """
    name = "hybrid"

    def __init__(
        self,
        vectordb,
        chunks: List[Any],
        k: int = 5,
        alpha: float = 0.5,
    ):
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever as LangChainBM25

        self.vectordb = vectordb
        self.k        = k
        self.alpha    = alpha
        self._chunks  = chunks

        texts     = [c.text for c in chunks]
        metadatas = [getattr(c, "metadata", {}) for c in chunks]

        # BM25 leg
        bm25_retriever   = LangChainBM25.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = k

        # Dense leg
        self._check_db(vectordb.db)
        dense_retriever = vectordb.db.as_retriever(search_kwargs={"k": k})

        # Ensemble — weights must sum to 1.0
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[round(1 - alpha, 2), round(alpha, 2)],
        )

    def retrieve(self, query: str) -> List[Any]:
        return self.retriever.invoke(query)

    def retrieve_with_score(self, query: str) -> List[Tuple[Any, float]]:
        """
        Hybrid score = weighted average of BM25 + dense scores.
        """
        from rank_bm25 import BM25Okapi

        texts = [c.text for c in self._chunks]

        # BM25 scores — normalised
        tokenized  = [t.split() for t in texts]
        bm25       = BM25Okapi(tokenized)
        bm25_raw   = bm25.get_scores(query.split())
        max_bm25   = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_norm  = [s / max_bm25 for s in bm25_raw]

        # Dense scores
        dense_results = self.vectordb.db.similarity_search_with_relevance_scores(
            query, k=len(texts)
        )
        dense_map = {doc.page_content: score for doc, score in dense_results}

        # Combine
        combined = []
        for i, text in enumerate(texts):
            dense_score  = dense_map.get(text, 0.0)
            hybrid_score = (1 - self.alpha) * bm25_norm[i] + self.alpha * dense_score
            combined.append((text, hybrid_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_texts = [t for t, _ in combined[: self.k]]
        score_map = {t: s for t, s in combined}

        docs = self.retrieve(query)
        return [
            (doc, score_map.get(doc.page_content, 0.0))
            for doc in docs
        ]

    def as_langchain_retriever(self):
        return self.retriever


# ─────────────────────────────────────────────────────────────
# 4. MMR Retriever
# ─────────────────────────────────────────────────────────────

class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance retrieval.
    Balances relevance with diversity — avoids returning
    redundant chunks that say the same thing.

    diversity controls the balance:
        diversity = 0.0  →  pure similarity (like dense)
        diversity = 0.5  →  balanced        (default)
        diversity = 1.0  →  maximum diversity

    Best for:
        - Long documents with repeated content
        - Summarisation tasks
        - When top-k chunks tend to be near-duplicates

    Usage:
        retriever = MMRRetriever(vectordb, k=5, diversity=0.5)
        docs      = retriever.retrieve("summarise the key points")
    """
    name = "mmr"

    def __init__(self, vectordb, k: int = 5, diversity: float = 0.5):
        self.vectordb  = vectordb
        self.k         = k
        self.diversity = diversity

    def retrieve(self, query: str) -> List[Any]:
        self._check_db(self.vectordb.db)
        return self.vectordb.db.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.k * 3,     # fetch more, then diversify
            lambda_mult=self.diversity,
        )

    def retrieve_with_score(self, query: str) -> List[Tuple[Any, float]]:
        """
        MMR does not natively return scores.
        We retrieve via MMR then score each result via similarity.
        """
        self._check_db(self.vectordb.db)
        docs = self.retrieve(query)

        # score each MMR result via similarity search
        all_scored = self.vectordb.db.similarity_search_with_relevance_scores(
            query, k=self.k * 3
        )
        score_map = {doc.page_content: score for doc, score in all_scored}

        return [
            (doc, score_map.get(doc.page_content, 0.0))
            for doc in docs
        ]

    def as_langchain_retriever(self):
        self._check_db(self.vectordb.db)
        return self.vectordb.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":            self.k,
                "fetch_k":      self.k * 3,
                "lambda_mult":  self.diversity,
            },
        )


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

RETRIEVER_REGISTRY = {
    "dense":  DenseRetriever,
    "bm25":   BM25Retriever,
    "hybrid": HybridRetriever,
    "mmr":    MMRRetriever,
}


def get_retriever(
    retriever_type: str,
    vectordb=None,
    chunks: Optional[List[Any]] = None,
    k: int = 5,
    **kwargs,
) -> BaseRetriever:
    """
    Factory function to get any retriever by name.

    Args:
        retriever_type : "dense" | "bm25" | "hybrid" | "mmr"
        vectordb       : VectorDB instance (required for dense, hybrid, mmr)
        chunks         : List of Chunk objects (required for bm25, hybrid)
        k              : Number of results to retrieve
        **kwargs       : Extra args passed to retriever (alpha, diversity, etc.)

    Returns:
        Instantiated retriever ready to use.

    Usage:
        r = get_retriever("hybrid", vectordb=db, chunks=chunks, k=5, alpha=0.6)
        docs = r.retrieve("what is relativity?")
    """
    retriever_type = retriever_type.lower()

    if retriever_type not in RETRIEVER_REGISTRY:
        raise ValueError(
            f"Unknown retriever '{retriever_type}'. "
            f"Choose from: {list(RETRIEVER_REGISTRY.keys())}"
        )

    # validate required args per type
    if retriever_type in ("dense", "mmr") and vectordb is None:
        raise ValueError(f"'{retriever_type}' retriever requires a vectordb.")

    if retriever_type == "bm25" and chunks is None:
        raise ValueError("'bm25' retriever requires chunks.")

    if retriever_type == "hybrid" and (vectordb is None or chunks is None):
        raise ValueError("'hybrid' retriever requires both vectordb and chunks.")

    # build args per retriever type
    if retriever_type == "dense":
        return DenseRetriever(vectordb, k=k)

    elif retriever_type == "bm25":
        return BM25Retriever(chunks, k=k)

    elif retriever_type == "hybrid":
        return HybridRetriever(vectordb, chunks, k=k, **kwargs)

    elif retriever_type == "mmr":
        return MMRRetriever(vectordb, k=k, **kwargs)
