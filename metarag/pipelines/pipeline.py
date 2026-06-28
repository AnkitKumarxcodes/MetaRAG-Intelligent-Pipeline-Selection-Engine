# pipeline.py

from __future__ import annotations
from typing import List, Optional, Any


# ─────────────────────────────────────────────────────────────
# MultiQuery
# Expands one query into multiple variations for broader retrieval
# ─────────────────────────────────────────────────────────────

class MultiQuery:
    """
    Takes one query, generates N variations using an LLM.
    Retrieves for each variation, merges and deduplicates results.

    Why: A single query might miss relevant chunks due to wording.
    Multiple variations cast a wider net.
    """

    PROMPT = """Generate {n} different versions of the following question.
Each version should ask the same thing but with different wording.
Return one question per line, no numbering, no bullets.

Question: {query}
"""

    def __init__(self, llm, n: int = 3):
        self.llm = llm
        self.n   = n

    def expand(self, query: str) -> List[str]:
        prompt   = self.PROMPT.format(query=query, n=self.n)
        result   = self.llm.invoke(prompt).content
        variants = [q.strip() for q in result.strip().splitlines() if q.strip()]
        print(f"[MultiQuery] Expanded into {len(variants)} variants")
        return [query] + variants   # always include original


# ─────────────────────────────────────────────────────────────
# HyDE — Hypothetical Document Embedding
# Generates a fake answer first, retrieves based on that
# ─────────────────────────────────────────────────────────────

class HyDE:
    """
    Instead of retrieving with the raw query,
    generates a hypothetical answer and retrieves based on that.

    Why: For vague queries, a hypothetical answer is closer
    to the actual document content than the query itself.
    """

    PROMPT = """Write a short, factual paragraph that would directly answer
the following question. Be concise. Do not say you don't know.

Question: {query}
Hypothetical answer:"""

    def __init__(self, llm):
        self.llm = llm

    def generate_hypothesis(self, query: str) -> str:
        prompt     = self.PROMPT.format(query=query)
        hypothesis = self.llm.invoke(prompt).content.strip()
        print(f"[HyDE] Generated hypothesis ({len(hypothesis)} chars)")
        return hypothesis


# ─────────────────────────────────────────────────────────────
# Reranker
# Reorders retrieved chunks by relevance to the query
# ─────────────────────────────────────────────────────────────

class Reranker:
    """
    Takes retrieved chunks and reorders them by relevance.
    Uses a cross-encoder model — more accurate than embedding similarity.

    Install: pip install sentence-transformers
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.top_k = top_k
        print(f"[Reranker] Loaded {model}")

    def rerank(self, query: str, chunks: List[Any]) -> List[Any]:
        if not chunks:
            return chunks

        pairs  = [(query, c.page_content) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        result = [chunk for chunk, _ in ranked[: self.top_k]]

        print(f"[Reranker] Reranked {len(chunks)} → top {len(result)} chunks")
        return result


# ─────────────────────────────────────────────────────────────
# Deduplication
# Removes near-identical chunks before sending to LLM
# ─────────────────────────────────────────────────────────────

class Deduplicator:
    """
    Removes chunks that are too similar to each other.
    Prevents the LLM from seeing the same information repeated.

    threshold: 0.0 = remove nothing, 1.0 = remove everything similar
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def deduplicate(self, chunks: List[Any]) -> List[Any]:
        if not chunks:
            return chunks

        seen   = []
        result = []

        for chunk in chunks:
            text = chunk.page_content
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
        # simple character overlap — no dependencies needed
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
# Pipeline A — Straight retrieval, no extras
# Fast, simple, good baseline
# ─────────────────────────────────────────────────────────────

class StraightPipeline(BasePipeline):
    """
    query → retrieve → return chunks

    No multiquery, no reranking, no dedup.
    Fastest pipeline. Router uses this for simple, clear queries.
    """
    name = "straight"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> dict:
        print(f"[{self.name}] Running on: '{query}'")
        chunks = self.retriever.retrieve(query)
        return {
            "query":    query,
            "chunks":   chunks,
            "pipeline": self.name,
        }


# ─────────────────────────────────────────────────────────────
# Pipeline B — MultiQuery retrieval
# Better coverage for ambiguous queries
# ─────────────────────────────────────────────────────────────

class MultiQueryPipeline(BasePipeline):
    """
    query → expand to N variants → retrieve for each → merge → deduplicate

    Router uses this when query is short, vague, or ambiguous.
    """
    name = "multiquery"

    def __init__(self, retriever, llm, n_variants: int = 3):
        self.retriever   = retriever
        self.multi_query = MultiQuery(llm, n=n_variants)
        self.dedup       = Deduplicator()

    def run(self, query: str) -> dict:
        print(f"[{self.name}] Running on: '{query}'")

        variants = self.multi_query.expand(query)

        all_chunks = []
        for q in variants:
            all_chunks.extend(self.retriever.retrieve(q))

        chunks = self.dedup.deduplicate(all_chunks)

        return {
            "query":    query,
            "chunks":   chunks,
            "pipeline": self.name,
            "variants": variants,
        }


# ─────────────────────────────────────────────────────────────
# Pipeline C — Reranked retrieval
# More precise, slightly slower
# ─────────────────────────────────────────────────────────────

class RerankedPipeline(BasePipeline):
    """
    query → retrieve → rerank → return top chunks

    Router uses this when precision matters more than speed.
    Good for factual, specific queries.
    """
    name = "reranked"

    def __init__(self, retriever, reranker: Reranker):
        self.retriever = retriever
        self.reranker  = reranker

    def run(self, query: str) -> dict:
        print(f"[{self.name}] Running on: '{query}'")

        chunks  = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, chunks)

        return {
            "query":    query,
            "chunks":   reranked,
            "pipeline": self.name,
        }


# ─────────────────────────────────────────────────────────────
# Pipeline D — HyDE retrieval
# Best for vague, open-ended queries
# ─────────────────────────────────────────────────────────────

class HyDEPipeline(BasePipeline):
    """
    query → generate hypothesis → retrieve using hypothesis → return chunks

    Router uses this when query is vague or conceptual.
    "explain...", "what is the difference...", "how does... work"
    """
    name = "hyde"

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.hyde      = HyDE(llm)

    def run(self, query: str) -> dict:
        print(f"[{self.name}] Running on: '{query}'")

        hypothesis = self.hyde.generate_hypothesis(query)
        chunks     = self.retriever.retrieve(hypothesis)

        return {
            "query":      query,
            "chunks":     chunks,
            "pipeline":   self.name,
            "hypothesis": hypothesis,
        }


# ─────────────────────────────────────────────────────────────
# Pipeline E — Full pipeline
# MultiQuery + Rerank + Dedup
# Slowest but most thorough
# ─────────────────────────────────────────────────────────────

class FullPipeline(BasePipeline):
    """
    query → expand → retrieve → rerank → deduplicate

    Router uses this for complex, multi-hop, or high-stakes queries.
    """
    name = "full"

    def __init__(self, retriever, llm, reranker: Reranker, n_variants: int = 3):
        self.retriever   = retriever
        self.multi_query = MultiQuery(llm, n=n_variants)
        self.reranker    = reranker
        self.dedup       = Deduplicator()

    def run(self, query: str) -> dict:
        print(f"[{self.name}] Running on: '{query}'")

        # expand
        variants   = self.multi_query.expand(query)

        # retrieve for all variants
        all_chunks = []
        for q in variants:
            all_chunks.extend(self.retriever.retrieve(q))

        # rerank
        reranked = self.reranker.rerank(query, all_chunks)

        # dedup
        chunks = self.dedup.deduplicate(reranked)

        return {
            "query":    query,
            "chunks":   chunks,
            "pipeline": self.name,
            "variants": variants,
        }

# pipeline.py — add this

class Pipeline(BasePipeline):
    """
    Composable pipeline. Router builds it with whatever 
    combination of steps it needs.
    
    router says → Pipeline(retriever, multiquery=True, rerank=True)
    """
    
    def __init__(
        self,
        retriever,
        multiquery: Optional[MultiQuery] = None,
        reranker:   Optional[Reranker]   = None,
        hyde:       Optional[HyDE]       = None,
    ):
        self.retriever  = retriever
        self.multiquery = multiquery
        self.reranker   = reranker
        self.hyde       = hyde
        self.dedup      = Deduplicator()

    def run(self, query: str) -> dict:
        
        # step 1 — HyDE transforms query if enabled
        retrieve_query = query
        hypothesis     = None
        if self.hyde:
            hypothesis     = self.hyde.generate_hypothesis(query)
            retrieve_query = hypothesis

        # step 2 — expand query if multiquery enabled
        queries = [retrieve_query]
        if self.multiquery:
            queries = self.multiquery.expand(retrieve_query)

        # step 3 — retrieve for all queries
        all_chunks = []
        for q in queries:
            all_chunks.extend(self.retriever.retrieve(q))

        # step 4 — rerank if enabled
        if self.reranker:
            all_chunks = self.reranker.rerank(query, all_chunks)

        # step 5 — always dedup
        chunks = self.dedup.deduplicate(all_chunks)

        return {
            "query":      query,
            "chunks":     chunks,
            "pipeline":   self.name,
            "hypothesis": hypothesis,
        }


# ─────────────────────────────────────────────────────────────
# Registry — router uses this to pick pipeline by name
# ─────────────────────────────────────────────────────────────

PIPELINE_REGISTRY = {
    "straight":   StraightPipeline,
    "multiquery": MultiQueryPipeline,
    "reranked":   RerankedPipeline,
    "hyde":       HyDEPipeline,
    "full":       FullPipeline,
}


def available_pipelines() -> List[str]:
    return list(PIPELINE_REGISTRY.keys())
