# tests/test_profilers.py
"""
Tests for QueryProfiler, CorpusProfiler, ProbeProfiler — the three signal
sources merged by MetaRAG._extract_query_features() for routing decisions.
"""

import pytest

from metarag.router.query_profiler import QueryProfiler
from metarag.router.corpus_profiler import CorpusProfiler
from metarag.router.probe_profiler import ProbeProfiler
from metarag import Chunk, InMemoryVectorDB

from metarag.utils import FakeEmbeddings

# ─────────────────────────────────────────────────────────
# QueryProfiler
# ─────────────────────────────────────────────────────────

@pytest.fixture
def qp():
    return QueryProfiler()


def test_query_profiler_returns_all_keys(qp):
    result = qp.profile("What is machine learning?")
    expected = {
        "query_length", "char_count", "contains_number", "contains_date",
        "is_question", "starts_with_wh", "has_operator", "is_short", "is_long",
    }
    assert set(result.keys()) == expected


def test_query_profiler_detects_question(qp):
    assert qp.profile("What is AI?")["is_question"] is True
    assert qp.profile("AI is powerful")["is_question"] is False


def test_query_profiler_detects_wh_word(qp):
    assert qp.profile("What is AI?")["starts_with_wh"] is True
    assert qp.profile("AI is powerful")["starts_with_wh"] is False


def test_query_profiler_detects_number(qp):
    assert qp.profile("What was the 2024 revenue?")["contains_number"] is True
    assert qp.profile("What was the revenue?")["contains_number"] is False


def test_query_profiler_short_vs_long(qp):
    assert qp.profile("cost")["is_short"] is True
    long_query = "What is the comparative difference between the two primary approaches discussed here"
    assert qp.profile(long_query)["is_long"] is True


def test_query_profiler_operator_detection(qp):
    assert qp.profile("compare A and B")["has_operator"] is True
    assert qp.profile("describe A")["has_operator"] is False


def test_query_profiler_empty_query(qp):
    result = qp.profile("")
    assert result["query_length"] == 0
    assert result["starts_with_wh"] is False


# ─────────────────────────────────────────────────────────
# CorpusProfiler
# ─────────────────────────────────────────────────────────

@pytest.fixture
def cp():
    return CorpusProfiler()


def test_corpus_profiler_returns_expected_keys(cp):
    chunks = [Chunk(text="Some normal sentence with words.") for _ in range(5)]
    result = cp.profile(chunks)
    expected = {"num_docs", "avg_chunk_length", "ocr_ratio", "duplicate_ratio", "numeric_ratio", "short_doc_ratio"}
    assert set(result.keys()) == expected


def test_corpus_profiler_empty_chunks(cp):
    assert cp.profile([]) == {}


def test_corpus_profiler_detects_duplicates(cp):
    chunks = [Chunk(text="Exact same content here for testing purposes today") for _ in range(4)]
    result = cp.profile(chunks)
    assert result["duplicate_ratio"] > 0.5


def test_corpus_profiler_detects_numeric_heavy(cp):
    chunks = [Chunk(text="123 456 789 100 200 300 revenue 2024") for _ in range(3)]
    result = cp.profile(chunks)
    assert result["numeric_ratio"] > 0.0


def test_corpus_profiler_detects_short_docs(cp):
    chunks = [Chunk(text="short") for _ in range(5)]
    result = cp.profile(chunks)
    assert result["short_doc_ratio"] == 1.0


def test_corpus_profiler_save_and_load(cp, tmp_path):
    chunks = [Chunk(text="Some normal content for the profile test") for _ in range(3)]
    profile = cp.profile(chunks)
    path = str(tmp_path / "corpus_profile.json")
    cp.save(path, profile)
    loaded = cp.load(path)
    assert loaded == profile


def test_corpus_profiler_load_missing_raises(cp, tmp_path):
    with pytest.raises(FileNotFoundError):
        cp.load(str(tmp_path / "does_not_exist.json"))


# ─────────────────────────────────────────────────────────
# ProbeProfiler
# ─────────────────────────────────────────────────────────

@pytest.fixture
def built_vdb():
    embeddings = FakeEmbeddings()
    chunks = [Chunk(text=f"Sample chunk number {i} about machine learning") for i in range(10)]
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return vdb, embeddings


def test_probe_profiler_returns_expected_keys(built_vdb):
    vdb, embeddings = built_vdb
    probe = ProbeProfiler(vdb, embeddings, k=5)
    result = probe.probe("machine learning")
    expected = {"avg_similarity", "max_similarity", "similarity_variance", "redundancy"}
    assert set(result.keys()) == expected


def test_probe_profiler_max_gte_avg(built_vdb):
    vdb, embeddings = built_vdb
    probe = ProbeProfiler(vdb, embeddings, k=5)
    result = probe.probe("machine learning")
    assert result["max_similarity"] >= result["avg_similarity"]


def test_probe_profiler_empty_index_returns_empty_shape():
    empty_vdb = InMemoryVectorDB()
    embeddings = FakeEmbeddings()
    probe = ProbeProfiler(empty_vdb, embeddings, k=5)
    result = probe.probe("test query")
    assert result["avg_similarity"] == 0.0
    assert result["max_similarity"] == 0.0


def test_probe_profiler_search_exception_returns_empty_shape():
    class BrokenVectorDB:
        def search(self, query_embedding, k=5):
            raise RuntimeError("index corrupted")

    probe = ProbeProfiler(BrokenVectorDB(), FakeEmbeddings(), k=5)
    result = probe.probe("test query")
    assert result == {
        "avg_similarity": 0.0, "max_similarity": 0.0,
        "similarity_variance": 0.0, "redundancy": 0.0,
    }