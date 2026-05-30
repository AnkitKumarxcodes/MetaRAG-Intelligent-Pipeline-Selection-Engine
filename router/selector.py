# selector.py

from __future__ import annotations
import re
from typing import List


# ─────────────────────────────────────────────────────────────
# Query Classifier
# Figures out what kind of query this is
# ─────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Classifies a query into one of four types.

    keyword   →  short, exact, specific terms
    semantic  →  conceptual, explanation based
    vague     →  too short or unclear to retrieve directly
    complex   →  multi part, comparison, needs broad retrieval
    """
    query_lower = query.lower().strip()
    words       = query_lower.split()
    n_words     = len(words)

    # vague — too short to be specific
    if n_words <= 3:
        return "vague"

    # complex — multi part questions
    complex_signals = ["compare", "difference", "explain", "how does",
                       "why does", "what are", "summarise", "summarize"]
    if any(s in query_lower for s in complex_signals):
        return "complex"

    # keyword — looks like a search term or specific lookup
    keyword_signals = ["error", "code", "version", "id", "name",
                       "date", "number", "list", "find", "show"]
    has_quotes      = '"' in query or "'" in query
    if has_quotes or any(s in query_lower for s in keyword_signals):
        return "keyword"

    # default — semantic
    return "semantic"


# ─────────────────────────────────────────────────────────────
# Router
# Maps query type → pipeline name
# ─────────────────────────────────────────────────────────────

class Router:
    """
    Hardcoded rule based router.
    Analyses the query and picks the best pipeline.

    vague    →  hyde        (generate hypothesis first)
    keyword  →  straight    (BM25 retriever, no expansion needed)
    semantic →  reranked    (dense retriever + rerank for precision)
    complex  →  multiquery  (expand query, broad retrieval)

    Later this gets replaced by a trained ML classifier.
    """

    ROUTING_TABLE = {
        "vague":    "hyde",
        "keyword":  "straight",
        "semantic": "reranked",
        "complex":  "multiquery",
    }

    def route(self, query: str) -> dict:
        """
        Returns routing decision with reasoning.
        """
        query_type = classify_query(query)
        pipeline   = self.ROUTING_TABLE[query_type]

        print(f"[Router] query_type='{query_type}' → pipeline='{pipeline}'")

        return {
            "query":      query,
            "query_type": query_type,
            "pipeline":   pipeline,
        }