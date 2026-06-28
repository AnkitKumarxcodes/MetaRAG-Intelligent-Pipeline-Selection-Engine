# evaluation/metrics.py
# Pure functions. One metric each. No LLM. No side effects.

from __future__ import annotations
import math
from typing import List, Any


def _text(chunk) -> str:
    return getattr(chunk, "page_content", None) or getattr(chunk, "text", "")


def _cosine(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return max(0.0, dot / (mag_a * mag_b))


# ─────────────────────────────────────────────────────────────
# 1. Faithfulness
# cosine(answer_embedding, concat_chunks_embedding)
# handles paraphrases unlike word overlap
# ─────────────────────────────────────────────────────────────

def faithfulness(answer_text: str, chunks: List[Any], embeddings) -> float:
    if not answer_text or not chunks:
        return 0.0
    context      = " ".join(_text(c) for c in chunks)
    answer_emb   = embeddings.embed_query(answer_text)
    context_emb  = embeddings.embed_query(context)
    return _cosine(answer_emb, context_emb)


# ─────────────────────────────────────────────────────────────
# 2. Relevancy
# cosine(query_embedding, answer_embedding)
# ─────────────────────────────────────────────────────────────

def relevancy(query: str, answer_text: str, embeddings) -> float:
    if not query or not answer_text:
        return 0.0
    q_emb = embeddings.embed_query(query)
    a_emb = embeddings.embed_query(answer_text)
    return _cosine(q_emb, a_emb)


# ─────────────────────────────────────────────────────────────
# 3. Precision
# returns max, avg, std of (query ↔ each chunk) similarities
# ─────────────────────────────────────────────────────────────

def precision(query: str, chunks: List[Any], embeddings) -> dict:
    if not query or not chunks:
        return {"max": 0.0, "avg": 0.0, "std": 0.0}

    q_emb  = embeddings.embed_query(query)
    scores = [_cosine(q_emb, embeddings.embed_query(_text(c))) for c in chunks]

    avg = sum(scores) / len(scores)
    std = math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores))

    return {
        "max": round(max(scores), 4),
        "avg": round(avg, 4),
        "std": round(std, 4),
    }


# ─────────────────────────────────────────────────────────────
# 4. Coverage
# how many query terms appear in retrieved chunks
# ─────────────────────────────────────────────────────────────

def coverage(query: str, chunks: List[Any]) -> float:
    if not query or not chunks:
        return 0.0

    query_terms = set(query.lower().split())
    chunk_text  = " ".join(_text(c) for c in chunks).lower()
    chunk_words = set(chunk_text.split())

    matched = query_terms & chunk_words
    return len(matched) / len(query_terms) if query_terms else 0.0


# ─────────────────────────────────────────────────────────────
# 5. Redundancy
# avg pairwise similarity between chunks
# high = chunks are repetitive = bad retrieval
# ─────────────────────────────────────────────────────────────

def redundancy(chunks: List[Any], embeddings) -> float:
    if len(chunks) < 2:
        return 0.0

    embeddings_list = [embeddings.embed_query(_text(c)) for c in chunks]
    pairs, total    = 0, 0.0

    for i in range(len(embeddings_list)):
        for j in range(i + 1, len(embeddings_list)):
            total += _cosine(embeddings_list[i], embeddings_list[j])
            pairs += 1

    return round(total / pairs, 4) if pairs > 0 else 0.0
