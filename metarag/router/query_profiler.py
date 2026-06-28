# router/query_profiler.py
# Runs per query at ask() time. Instant — no model, no embeddings.
# Answers: what kind of query is this?

from __future__ import annotations
import re


class QueryProfiler:
    """
    Extracts lightweight features from a query string.
    No model calls. Runs in microseconds.

    Signals produced:
        query_length     →  number of words
        char_count       →  total characters
        contains_number  →  dates, IDs, error codes
        contains_date    →  time-sensitive query
        is_question      →  ends with ?
        starts_with_wh   →  what/why/how/when/where/who
        has_operator     →  compare, vs, and, or, not
        is_short         →  ≤ 3 words — likely vague
        is_long          →  > 10 words — likely complex
    """

    WH_WORDS   = {"what", "why", "how", "when", "where", "who", "which"}
    OPERATORS  = {"and", "or", "not", "vs", "versus", "compare",
                  "difference", "between"}

    def profile(self, query: str) -> dict:
        query   = query.strip()
        words   = query.split()
        n       = len(words)
        lower   = query.lower()

        return {
            "query_length":    n,
            "char_count":      len(query),
            "contains_number": bool(re.search(r"\d", query)),
            "contains_date":   bool(re.search(
                r"\b(\d{4}|\d{1,2}[\/\-]\d{1,2}|"
                r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
                lower
            )),
            "is_question":     query.endswith("?"),
            "starts_with_wh":  words[0].lower() in self.WH_WORDS if words else False,
            "has_operator":    any(w in lower.split() for w in self.OPERATORS),
            "is_short":        n <= 3,
            "is_long":         n > 10,
        }
