# evaluation/scorer.py
# Combines metrics into a single comparable score.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Any


# ─────────────────────────────────────────────────────────────
# Score Result
# ─────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    faithfulness:  float
    relevancy:     float
    precision_avg: float
    precision_max: float
    precision_std: float
    coverage:      float
    redundancy:    float
    latency_ms:    float
    composite:     float

    def __repr__(self):
        return (
            f"ScoreResult(\n"
            f"  faithfulness  = {self.faithfulness:.2f}\n"
            f"  relevancy     = {self.relevancy:.2f}\n"
            f"  precision_avg = {self.precision_avg:.2f}\n"
            f"  precision_max = {self.precision_max:.2f}\n"
            f"  coverage      = {self.coverage:.2f}\n"
            f"  redundancy    = {self.redundancy:.2f}  (lower is better)\n"
            f"  latency_ms    = {self.latency_ms:.0f}\n"
            f"  composite     = {self.composite:.2f}\n"
            f")"
        )

    def as_dict(self) -> dict:
        return {
            "faithfulness":  self.faithfulness,
            "relevancy":     self.relevancy,
            "precision_avg": self.precision_avg,
            "precision_max": self.precision_max,
            "precision_std": self.precision_std,
            "coverage":      self.coverage,
            "redundancy":    self.redundancy,
            "latency_ms":    self.latency_ms,
            "composite":     self.composite,
        }


# ─────────────────────────────────────────────────────────────
# Default weights — three presets
# ─────────────────────────────────────────────────────────────

WEIGHTS = {

    # balanced — general RAG, employee docs
    "balanced": {
        "faithfulness": 0.30,
        "relevancy":    0.25,
        "precision":    0.20,
        "coverage":     0.15,
        "redundancy":   0.10,   # penalty
        "latency":      0.00,   # ignored
    },

    # precision — security logs, anomaly detection
    # latency matters, redundancy penalised harder
    "precision": {
        "faithfulness": 0.20,
        "relevancy":    0.20,
        "precision":    0.30,
        "coverage":     0.20,
        "redundancy":   0.10,
        "latency":      0.05,   # small latency penalty
    },

    # recall — research, summarisation
    # coverage matters most, redundancy tolerated
    "recall": {
        "faithfulness": 0.25,
        "relevancy":    0.20,
        "precision":    0.15,
        "coverage":     0.30,
        "redundancy":   0.05,
        "latency":      0.00,
    },
}


# ─────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────

class Scorer:
    """
    Combines individual metrics into one composite score.

    preset options: "balanced", "precision", "recall"
    or pass your own weights dict.
    """

    MAX_LATENCY_MS = 10_000     # 10 seconds = max penalty reference

    def __init__(self, preset: str = "balanced", weights: dict = None):
        if weights:
            self.weights = weights
        elif preset in WEIGHTS:
            self.weights = WEIGHTS[preset]
        else:
            raise ValueError(f"Unknown preset '{preset}'. Choose: {list(WEIGHTS.keys())}")

        self.preset = preset

    def score(
        self,
        faithfulness:  float,
        relevancy:     float,
        precision:     dict,
        coverage:      float,
        redundancy:    float,
        latency_ms:    float,
    ) -> ScoreResult:

        w = self.weights

        # latency penalty — normalised 0→1 where 1 = MAX_LATENCY_MS
        latency_norm    = min(latency_ms / self.MAX_LATENCY_MS, 1.0)
        latency_penalty = w.get("latency", 0.0) * latency_norm

        composite = (
            w["faithfulness"] * faithfulness
            + w["relevancy"]  * relevancy
            + w["precision"]  * precision["avg"]
            + w["coverage"]   * coverage
            - w["redundancy"] * redundancy
            - latency_penalty
        )

        return ScoreResult(
            faithfulness  = round(faithfulness, 4),
            relevancy     = round(relevancy, 4),
            precision_avg = round(precision["avg"], 4),
            precision_max = round(precision["max"], 4),
            precision_std = round(precision["std"], 4),
            coverage      = round(coverage, 4),
            redundancy    = round(redundancy, 4),
            latency_ms    = round(latency_ms, 2),
            composite     = round(max(0.0, composite), 4),
        )
