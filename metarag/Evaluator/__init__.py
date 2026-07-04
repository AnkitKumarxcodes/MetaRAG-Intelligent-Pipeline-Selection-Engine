# metarag/evaluation/__init__.py

from .metrics import faithfulness, relevancy, precision, coverage, redundancy
from .scorer import Scorer, ScoreResult, WEIGHTS
from .evaluator import Evaluator

__all__ = [
    # metrics (pure functions — exposed mainly for Mode 1 users who want
    # to compute a single metric standalone, not typically called directly
    # if using Evaluator)
    "faithfulness",
    "relevancy",
    "precision",
    "coverage",
    "redundancy",
    # scoring
    "Scorer",
    "ScoreResult",
    "WEIGHTS",
    # orchestration
    "Evaluator",
]