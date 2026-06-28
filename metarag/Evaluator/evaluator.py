# evaluation/evaluator.py

from __future__ import annotations
from typing import List, Any, Optional

from metrics import (
    faithfulness  as _faithfulness,
    relevancy     as _relevancy,
    precision     as _precision,
    coverage      as _coverage,
    redundancy    as _redundancy,
)
from scorer import Scorer, ScoreResult


class Evaluator:
    """
    Evaluates an Answer object using all five metrics.
    Zero LLM calls. Fast. Returns a ScoreResult.

    preset: "balanced" | "precision" | "recall"

    Usage:
        evaluator = Evaluator(embeddings, preset="balanced")
        result    = evaluator.evaluate(answer)
        print(result.composite)
    """

    def __init__(self, embedding_model, preset: str = "balanced"):
        self.embeddings = embedding_model
        self.scorer     = Scorer(preset=preset)

    def evaluate(self, answer) -> ScoreResult:
        """
        Score an Answer object from generator.py.
        answer must have: .query .text .chunks .latency_ms
        """
        query      = answer.query
        text       = answer.text
        chunks     = answer.chunks
        latency_ms = getattr(answer, "latency_ms", 0.0)

        if not chunks:
            print("[Evaluator] Warning — no chunks, all scores will be zero")

        f  = _faithfulness(text,  chunks, self.embeddings)
        r  = _relevancy(query, text,      self.embeddings)
        p  = _precision(query,    chunks, self.embeddings)
        c  = _coverage(query,     chunks)
        rd = _redundancy(chunks,          self.embeddings)

        result = self.scorer.score(
            faithfulness = f,
            relevancy    = r,
            precision    = p,
            coverage     = c,
            redundancy   = rd,
            latency_ms   = latency_ms,
        )

        print(
            f"[Evaluator] "
            f"faith={result.faithfulness:.2f} "
            f"relev={result.relevancy:.2f} "
            f"prec={result.precision_avg:.2f} "
            f"cov={result.coverage:.2f} "
            f"redund={result.redundancy:.2f} "
            f"→ composite={result.composite:.2f}"
        )

        return result

    def set_preset(self, preset: str):
        """Switch scoring preset without recreating evaluator."""
        self.scorer = Scorer(preset=preset)
