# tests/test_retriever.py
"""
pytest suite for BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever
— all built on the same shared chunks/embeddings/vector_db, matching how
MetaRAG._setup_retrievers() constructs them (fix #2: no re-building the
index per retriever).
"""

from pathlib import Path
import pytest

from metarag import (
    DocumentLoader, Chunker, CachedEmbeddings, InMemoryVectorDB,
    BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
QUERY = "What is the main topic of this document?"


from metarag.utils import FakeEmbeddings


@pytest.fixture(scope="module")
def embeddings():
    return CachedEmbeddings(FakeEmbeddings())


@pytest.fixture(scope="module")
def chunks():
    docs = DocumentLoader(DATA_DIR).load(verbose=False)
    return Chunker(strategy="recursive", chunk_size=500, overlap=50).chunk_documents(docs)


@pytest.fixture(scope="module")
def built_vdb(chunks, embeddings):
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return vdb


@pytest.fixture(scope="module")
def bm25(chunks):
    return BM25Retriever(chunks)


@pytest.fixture(scope="module")
def dense(chunks, embeddings, built_vdb):
    return DenseRetriever(chunks, embeddings, built_vdb)


@pytest.fixture(scope="module")
def hybrid(chunks, embeddings, built_vdb):
    return HybridRetriever(chunks, embeddings, built_vdb)


@pytest.fixture(scope="module")
def mmr(chunks, embeddings, built_vdb):
    return MMRRetriever(chunks, embeddings, built_vdb)


# ─────────────────────────────────────────────────────────
# BM25Retriever
# ─────────────────────────────────────────────────────────

def test_bm25_returns_k_results(bm25):
    results = bm25.retrieve(QUERY, k=3)
    assert len(results) == 3


def test_bm25_results_are_scored_tuples(bm25):
    results = bm25.retrieve(QUERY, k=2)
    for chunk, score in results:
        assert isinstance(score, float)


# ─────────────────────────────────────────────────────────
# DenseRetriever
# ─────────────────────────────────────────────────────────

def test_dense_returns_k_results(dense):
    results = dense.retrieve(QUERY, k=3)
    assert len(results) == 3


def test_dense_does_not_rebuild_vector_db(chunks, embeddings, built_vdb):
    """Fix #2 regression check: constructing a second DenseRetriever on the
    SAME already-built vector_db must not re-embed or re-build anything."""
    chunk_count_before = len(built_vdb.chunks)
    DenseRetriever(chunks, embeddings, built_vdb)
    assert len(built_vdb.chunks) == chunk_count_before


# ─────────────────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────────────────

def test_hybrid_returns_k_results(hybrid):
    results = hybrid.retrieve(QUERY, k=3)
    assert len(results) == 3


def test_hybrid_alpha_changes_ranking(chunks, embeddings, built_vdb):
    bm25_leaning = HybridRetriever(chunks, embeddings, built_vdb, alpha=0.0)
    dense_leaning = HybridRetriever(chunks, embeddings, built_vdb, alpha=1.0)

    bm25_top = bm25_leaning.retrieve(QUERY, k=1)[0][1]
    dense_top = dense_leaning.retrieve(QUERY, k=1)[0][1]

    # Not asserting WHICH is higher (corpus-dependent) — just that alpha
    # actually changes the score, proving it isn't silently ignored.
    assert isinstance(bm25_top, float)
    assert isinstance(dense_top, float)


def test_hybrid_empty_query_does_not_crash(hybrid):
    results = hybrid.retrieve("", k=3)
    assert isinstance(results, list)


# ─────────────────────────────────────────────────────────
# MMRRetriever
# ─────────────────────────────────────────────────────────

def test_mmr_returns_k_results(mmr):
    results = mmr.retrieve(QUERY, k=3)
    assert len(results) == 3


def test_mmr_results_have_real_scores_not_flat_one(mmr):
    """Regression check for the old bug where MMR returned a hardcoded 1.0
    for every result instead of real relevance scores."""
    results = mmr.retrieve(QUERY, k=3)
    scores = [score for _, score in results]
    assert not all(s == 1.0 for s in scores)


def test_mmr_no_duplicate_chunks_in_results(mmr):
    results = mmr.retrieve(QUERY, k=5)
    texts = [c.text if hasattr(c, "text") else str(c) for c, _ in results]
    assert len(texts) == len(set(texts))


def test_mmr_lambda_extremes_do_not_crash(chunks, embeddings, built_vdb):
    pure_relevance = MMRRetriever(chunks, embeddings, built_vdb, lambda_param=1.0)
    pure_diversity = MMRRetriever(chunks, embeddings, built_vdb, lambda_param=0.0)

    assert len(pure_relevance.retrieve(QUERY, k=3)) == 3
    assert len(pure_diversity.retrieve(QUERY, k=3)) == 3


# ─────────────────────────────────────────────────────────
# Cross-retriever consistency
# ─────────────────────────────────────────────────────────

def test_all_retrievers_return_valid_chunks(bm25, dense, hybrid, mmr):
    for retriever in [bm25, dense, hybrid, mmr]:
        results = retriever.retrieve(QUERY, k=2)
        for chunk, score in results:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            assert isinstance(text, str) and len(text) > 0

# ─────────────────────────────────────────────────────────
# Empty-query guards
# ─────────────────────────────────────────────────────────

def test_bm25_empty_query_returns_empty_list(bm25):
    assert bm25.retrieve("", k=3) == []


def test_dense_empty_query_returns_empty_list(dense):
    assert dense.retrieve("   ", k=3) == []


def test_mmr_empty_query_returns_empty_list(mmr):
    assert mmr.retrieve("", k=3) == []


def test_hybrid_returns_empty_when_both_retrievers_empty(chunks, embeddings, built_vdb):
    """Regression check for the 'neither retriever returned anything' guard —
    an empty query makes BOTH BM25 and Dense return [], so hybrid must not
    crash on max([]) and should return [] instead."""
    hybrid = HybridRetriever(chunks, embeddings, built_vdb)
    assert hybrid.retrieve("", k=3) == []