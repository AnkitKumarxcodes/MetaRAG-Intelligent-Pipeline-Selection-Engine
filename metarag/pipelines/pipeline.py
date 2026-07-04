# metarag/pipeline/pipeline.py

from __future__ import annotations
from typing import List, Optional, Any, Tuple


def _chunk_text(chunk) -> str:
    """Extract text — supports (text, score) tuples, Chunk objects, or raw strings."""
    if isinstance(chunk, tuple):
        return chunk[0]
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "text", None) or getattr(chunk, "page_content", "")


# ─────────────────────────────────────────────────────────────
# MultiQuery — expands one query into multiple variations
# ─────────────────────────────────────────────────────────────

# AFTER
class MultiQuery:
    """
    Expands a query into N variations using an LLM (GeneratorInterface).
    Falls back to just the original query if the LLM call fails —
    this must never crash ask()/benchmark(), since query expansion
    is an enhancement, not a required step.
    """

    PROMPT = """Generate {n} different versions of the following question.
Each version should ask the same thing but with different wording.
Return one question per line, no numbering, no bullets.

Question: {query}
"""

    def __init__(self, generator, n: int = 3):
        self.generator = generator
        self.n = n

    def expand(self, query: str) -> List[str]:
        prompt = self.PROMPT.format(query=query, n=self.n)
        try:
            result = self.generator.generate(prompt)
            variants = [q.strip() for q in result.strip().splitlines() if q.strip()]
            print(f"[MultiQuery] Expanded into {len(variants)} variants")
            return [query] + variants[: self.n]  # cap in case LLM over-generates
        except Exception as e:
            print(f"[MultiQuery] Expansion failed ({e}) — falling back to original query only")
            return [query]


# ─────────────────────────────────────────────────────────────
# HyDE — Hypothetical Document Embedding
# ─────────────────────────────────────────────────────────────

# AFTER
class HyDE:
    """
    Generates a hypothetical answer, retrieves based on that instead of raw query.
    Falls back to the original query if generation fails — HyDE is an
    enhancement to retrieval, not something that should crash the pipeline.
    """

    PROMPT = """Write a short, factual paragraph that would directly answer
the following question. Be concise. Do not say you don't know.

Question: {query}
Hypothetical answer:"""

    def __init__(self, generator):
        self.generator = generator

    def generate_hypothesis(self, query: str) -> str:
        prompt = self.PROMPT.format(query=query)
        try:
            hypothesis = self.generator.generate(prompt).strip()
            print(f"[HyDE] Generated hypothesis ({len(hypothesis)} chars)")
            return hypothesis
        except Exception as e:
            print(f"[HyDE] Hypothesis generation failed ({e}) — falling back to raw query")
            return query


# ─────────────────────────────────────────────────────────────
# Reranker — reorders chunks by relevance (optional dep)
# ─────────────────────────────────────────────────────────────

# AFTER
class Reranker:
    """
    Reorders retrieved chunks using a cross-encoder model.
    Optional dependency: sentence-transformers.
    If not installed, falls back to no-op (returns chunks unchanged, still truncated to k).

    top_k is now a DEFAULT only — callers can override per-call via rerank(k=...),
    so pipelines respect whatever k the user actually asked for at run(k=...) time.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        self.default_top_k = top_k
        self.model = None

        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model)
            print(f"[Reranker] Loaded {model}")
        except ImportError:
            print(
                "[Reranker] ⚠️  sentence-transformers not installed. "
                "Reranking disabled (passthrough). Install: pip install sentence-transformers"
            )

    def rerank(self, query: str, chunks: List[Any], k: int = None) -> List[Any]:
        effective_k = k if k is not None else self.default_top_k

        if not chunks:
            return chunks

        if self.model is None:
            return chunks[:effective_k]

        texts = [_chunk_text(c) for c in chunks]
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)

        # Merge rerank scores back onto chunks — replaces stale retrieval scores
        # with real post-rerank relevance (this also folds in fix #9)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        result = [
            (_chunk_text(chunk), float(score)) if not isinstance(chunk, tuple) else (chunk[0], float(score))
            for chunk, score in ranked[:effective_k]
        ]

        print(f"[Reranker] Reranked {len(chunks)} → top {len(result)} chunks")
        return result


# ─────────────────────────────────────────────────────────────
# Deduplicator — removes near-identical chunks
# ─────────────────────────────────────────────────────────────

class Deduplicator:
    """
    Removes chunks too similar to each other (word-overlap based, no dependencies).
    threshold: 0.0 = remove nothing, 1.0 = remove everything similar
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def deduplicate(self, chunks: List[Any]) -> List[Any]:
        if not chunks:
            return chunks

        seen = []
        result = []

        for chunk in chunks:
            text = _chunk_text(chunk)
            if not self._is_duplicate(text, seen):
                seen.append(text)
                result.append(chunk)

        print(f"[Deduplicator] {len(chunks)} → {len(result)} chunks after dedup")
        return result

    def _is_duplicate(self, text: str, seen: List[str]) -> bool:
        for s in seen:
            if self._similarity(text, s) >= self.threshold:
                return True
        return False

    def _similarity(self, a: str, b: str) -> float:
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / max(len(set_a), len(set_b))


# ─────────────────────────────────────────────────────────────
# Base Pipeline
# ─────────────────────────────────────────────────────────────

class BasePipeline:
    name: str = "base"

    def run(self, query: str) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Composable Pipeline — router builds combinations dynamically
# ─────────────────────────────────────────────────────────────

class Pipeline(BasePipeline):
    """
    Composable pipeline. Combine any retriever with optional
    multiquery, reranking, and HyDE.

    retriever: object with .retrieve(query, k) → List[(text, score)]
    """

    name = "custom"

    def __init__(
        self,
        retriever,
        multiquery: Optional[MultiQuery] = None,
        reranker: Optional[Reranker] = None,
        hyde: Optional[HyDE] = None,
        name: str = None,
    ):
        if not hasattr(retriever, "retrieve"):
            raise TypeError("retriever must have a .retrieve(query, k) method")

        self.retriever = retriever
        self.multiquery = multiquery
        self.reranker = reranker
        self.hyde = hyde
        self.dedup = Deduplicator()
        if name:
            self.name = name

    # AFTER
    def run(self, query: str, k: int = 4) -> dict:
        print(f"[{self.name}] Running on: '{query}'")

        retrieve_query = query
        hypothesis = None
        if self.hyde:
            hypothesis = self.hyde.generate_hypothesis(query)
            retrieve_query = hypothesis

        queries = [retrieve_query]
        if self.multiquery:
            queries = self.multiquery.expand(retrieve_query)

        all_chunks = []
        for q in queries:
            all_chunks.extend(self.retriever.retrieve(q, k=k))

        if self.reranker:
            all_chunks = self.reranker.rerank(query, all_chunks, k=k)   # k passed at call time — see fix below
        else:
            # No reranker — sort by score descending before truncating, so the
            # best k survive, not just the first k encountered across queries
            all_chunks.sort(key=lambda c: c[1] if isinstance(c, tuple) else 0, reverse=True)

        chunks = self.dedup.deduplicate(all_chunks)
        chunks = chunks[:k]   # ← the actual fix: always truncate back to k regardless of path taken

        return {
            "query": query,
            "chunks": chunks,
            "pipeline": self.name,
            "hypothesis": hypothesis,
        }


# ─────────────────────────────────────────────────────────────
# Preset Pipelines — common configurations, ready to use
# ─────────────────────────────────────────────────────────────

class StraightPipeline(Pipeline):
    """Fastest: query → retrieve → return. No extras."""
    name = "straight"

    def __init__(self, retriever):
        super().__init__(retriever)


class MultiQueryPipeline(Pipeline):
    """query → expand → retrieve for each → dedup."""
    name = "multiquery"

    def __init__(self, retriever, generator, n_variants: int = 3):
        super().__init__(retriever, multiquery=MultiQuery(generator, n=n_variants))


class RerankedPipeline(Pipeline):
    """query → retrieve → rerank → return top chunks."""
    name = "reranked"

    def __init__(self, retriever, reranker: Reranker):
        super().__init__(retriever, reranker=reranker)


class HyDEPipeline(Pipeline):
    """query → hypothesis → retrieve using hypothesis."""
    name = "hyde"

    def __init__(self, retriever, generator):
        super().__init__(retriever, hyde=HyDE(generator))


class FullPipeline(Pipeline):
    """query → expand → retrieve → rerank → dedup. Most thorough."""
    name = "full"

    def __init__(self, retriever, generator, reranker: Reranker, n_variants: int = 3):
        super().__init__(
            retriever,
            multiquery=MultiQuery(generator, n=n_variants),
            reranker=reranker,
        )


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

PIPELINE_REGISTRY = {
    "straight": StraightPipeline,
    "multiquery": MultiQueryPipeline,
    "reranked": RerankedPipeline,
    "hyde": HyDEPipeline,
    "full": FullPipeline,
}


def available_pipelines() -> List[str]:
    return list(PIPELINE_REGISTRY.keys())