# router/selector.py

from __future__ import annotations
from .corpus_profiler import CorpusProfiler
from .query_profiler  import QueryProfiler
from .probe_profiler  import ProbeProfiler
from .router_interface import RouterInterface

class Router(RouterInterface):
    """
    Routes every query to the best pipeline using three signal sources:

        CorpusProfile  (35%)  →  what kind of corpus is this?
        QueryProfile   (15%)  →  what kind of query is this?
        ProbeProfile   (50%)  →  how hard is retrieval for this query?
        embeddings: EmbeddingInterface object (needed by ProbeProfiler to embed queries

    Routing logic is rule-based for now.
    Later replaced by a trained ML classifier on logged features.
    """

    def __init__(self, vectordb, embeddings , corpus_profile: dict):
        self.corpus_profile  = corpus_profile
        self.query_profiler  = QueryProfiler()
        self.probe_profiler  = ProbeProfiler(vectordb , embeddings)

    def route(self, query: str) -> dict:
        qp = self.query_profiler.profile(query)
        pp = self.probe_profiler.probe(query)

        features = {
            **self.corpus_profile,
            **qp,
            **pp,
        }

        pipeline = self._select(features)
        print(f"[Router] → '{pipeline}'")

        return {
            "query":    query,
            "pipeline": pipeline,
            "features": features,       # logged → becomes training data
        }

    def _select(self, f: dict) -> str:
        """
        Rule-based selection using all three profiles.

        Probe profile drives the decision most heavily.
        Corpus and query profiles break ties and handle edge cases.
        """

        # ── probe signals (50%) ───────────────────────────────
        max_sim = f.get("max_similarity",      0.0)
        avg_sim = f.get("avg_similarity",      0.0)
        redund  = f.get("redundancy",          0.0)
        var     = f.get("similarity_variance", 0.0)

        # ── corpus signals (35%) ──────────────────────────────
        ocr_ratio    = f.get("ocr_ratio",       0.0)
        numeric      = f.get("numeric_ratio",   0.0)
        short_docs   = f.get("short_doc_ratio", 0.0)
        avg_len      = f.get("avg_chunk_length", 500)

        # ── query signals (15%) ───────────────────────────────
        is_short     = f.get("is_short",        False)
        is_long      = f.get("is_long",         False)
        has_number   = f.get("contains_number", False)
        has_operator = f.get("has_operator",    False)
        is_wh        = f.get("starts_with_wh",  False)

        # ── routing rules ─────────────────────────────────────

        # answer clearly in corpus — fast precise retrieval
        if max_sim > 0.85 and redund < 0.3:
            return "reranked"

        # corpus is log-like or numeric heavy
        if numeric > 0.4 or short_docs > 0.6:
            if has_number or not is_wh:
                return "straight"
            return "hybrid"

        # retrieval is hard — answer not obvious in corpus
        if max_sim < 0.5 or avg_sim < 0.35:
            if is_short:
                return "hyde"
            return "multiquery"

        # high redundancy — top chunks are repetitive
        if redund > 0.6:
            return "mmr"

        # noisy OCR corpus
        if ocr_ratio > 0.3:
            return "hybrid"

        # complex multi-part query
        if is_long or has_operator:
            return "multiquery"

        # default
        return "hybrid"