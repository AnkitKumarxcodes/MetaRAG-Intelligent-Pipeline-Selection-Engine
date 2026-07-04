# metarag/evaluation/metrics.py
# Pure functions. One metric each. No LLM. No side effects. No dependencies.

from __future__ import annotations
import math
from typing import List, Tuple, Any


def _text(chunk) -> str:
    """Extract text from a chunk — supports (text, score) tuples, Chunk objects, or raw strings."""
    if isinstance(chunk, tuple):
        return chunk[0]
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "text", None) or getattr(chunk, "page_content", "")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return max(0.0, dot / (mag_a * mag_b))


# ─────────────────────────────────────────────────────────────
# 1. Faithfulness — cosine(answer, context)
# ─────────────────────────────────────────────────────────────

def faithfulness(answer_text: str, chunks: List[Any], embeddings) -> float:
    if not answer_text or not chunks:
        return 0.0
    context = " ".join(_text(c) for c in chunks)
    answer_emb = embeddings.embed(answer_text)
    context_emb = embeddings.embed(context)
    return _cosine(answer_emb, context_emb)


# ─────────────────────────────────────────────────────────────
# 2. Relevancy — cosine(query, answer)
# ─────────────────────────────────────────────────────────────

def relevancy(query: str, answer_text: str, embeddings) -> float:
    if not query or not answer_text:
        return 0.0
    q_emb = embeddings.embed(query)
    a_emb = embeddings.embed(answer_text)
    return _cosine(q_emb, a_emb)


# ─────────────────────────────────────────────────────────────
# 3. Precision — max/avg/std of (query ↔ each chunk)
# ─────────────────────────────────────────────────────────────

def precision(query: str, chunks: List[Any], embeddings) -> dict:
    if not query or not chunks:
        return {"max": 0.0, "avg": 0.0, "std": 0.0}

    q_emb = embeddings.embed(query)
    scores = [_cosine(q_emb, embeddings.embed(_text(c))) for c in chunks]

    avg = sum(scores) / len(scores)
    std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))

    return {
        "max": round(max(scores), 4),
        "avg": round(avg, 4),
        "std": round(std, 4),
    }


# ─────────────────────────────────────────────────────────────
# 4. Coverage — query term overlap in chunks
# ─────────────────────────────────────────────────────────────

def coverage(query: str, chunks: List[Any]) -> float:
    if not query or not chunks:
        return 0.0

    query_terms = set(query.lower().split())
    chunk_text = " ".join(_text(c) for c in chunks).lower()
    chunk_words = set(chunk_text.split())

    matched = query_terms & chunk_words
    return len(matched) / len(query_terms) if query_terms else 0.0


# ─────────────────────────────────────────────────────────────
# 5. Redundancy — avg pairwise similarity between chunks
# ─────────────────────────────────────────────────────────────

def redundancy(chunks: List[Any], embeddings) -> float:
    if len(chunks) < 2:
        return 0.0

    embeddings_list = [embeddings.embed(_text(c)) for c in chunks]
    pairs, total = 0, 0.0

    for i in range(len(embeddings_list)):
        for j in range(i + 1, len(embeddings_list)):
            total += _cosine(embeddings_list[i], embeddings_list[j])
            pairs += 1

    return round(total / pairs, 4) if pairs > 0 else 0.0