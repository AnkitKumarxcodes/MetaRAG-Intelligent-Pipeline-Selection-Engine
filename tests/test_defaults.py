# tests/test_defaults.py
"""
Tests for defaults.py — DEFAULTS singleton behavior: as_single()/as_list()
normalization, and that a mutation actually propagates into newly-built
components (the entire point of a shared mutable config object).
"""

import pytest

from metarag.defaults import DEFAULTS, MetaRAGDefaults
from metarag import HybridRetriever, MMRRetriever, Chunker, Chunk, InMemoryVectorDB
from metarag.pipelines.pipeline import Deduplicator, MultiQuery
from metarag.utils import FakeEmbeddings, FakeGenerator



@pytest.fixture(autouse=True)
def restore_defaults():
    """Every test gets a clean DEFAULTS state — mutations in one test
    must not leak into the next."""
    original = MetaRAGDefaults()
    yield
    for field_name in original.__dataclass_fields__:
        setattr(DEFAULTS, field_name, getattr(original, field_name))


# ─────────────────────────────────────────────────────────
# as_single() / as_list()
# ─────────────────────────────────────────────────────────

def test_as_single_returns_scalar_unchanged():
    DEFAULTS.hybrid_alpha = 0.5
    assert DEFAULTS.as_single("hybrid_alpha") == 0.5


def test_as_single_returns_first_of_list():
    DEFAULTS.hybrid_alpha = [0.3, 0.5, 0.7]
    assert DEFAULTS.as_single("hybrid_alpha") == 0.3


def test_as_list_wraps_scalar():
    DEFAULTS.hybrid_alpha = 0.5
    assert DEFAULTS.as_list("hybrid_alpha") == [0.5]


def test_as_list_returns_list_unchanged():
    DEFAULTS.hybrid_alpha = [0.3, 0.5, 0.7]
    assert DEFAULTS.as_list("hybrid_alpha") == [0.3, 0.5, 0.7]


def test_default_factory_values_are_sane():
    fresh = MetaRAGDefaults()
    assert 200 <= fresh.chunk_size <= 1500
    assert 0.0 <= fresh.hybrid_alpha <= 1.0
    assert 0.0 <= fresh.mmr_lambda <= 1.0
    assert fresh.chunk_strategy in ["fixed", "recursive", "semantic", "sentence", "sliding_window", "markdown"]
    assert fresh.eval_preset in ["balanced", "precision", "recall"]


# ─────────────────────────────────────────────────────────
# Mutation propagation — the actual point of DEFAULTS existing
# ─────────────────────────────────────────────────────────

def test_mutation_propagates_to_new_hybrid_retriever():
    chunks = [Chunk(text=f"chunk {i} about testing") for i in range(5)]
    embeddings = FakeEmbeddings()
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))

    DEFAULTS.hybrid_alpha = 0.9
    retriever = HybridRetriever(chunks, embeddings, vdb)
    assert retriever.alpha == 0.9

    DEFAULTS.hybrid_alpha = 0.1
    retriever2 = HybridRetriever(chunks, embeddings, vdb)
    assert retriever2.alpha == 0.1


def test_explicit_alpha_overrides_defaults():
    chunks = [Chunk(text=f"chunk {i} about testing") for i in range(5)]
    embeddings = FakeEmbeddings()
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))

    DEFAULTS.hybrid_alpha = 0.9
    retriever = HybridRetriever(chunks, embeddings, vdb, alpha=0.2)
    assert retriever.alpha == 0.2   # explicit param wins over DEFAULTS


def test_existing_instances_do_not_retroactively_change():
    """Documented limitation: mutating DEFAULTS only affects NEW constructions,
    not objects already built before the mutation."""
    chunks = [Chunk(text=f"chunk {i} about testing") for i in range(5)]
    embeddings = FakeEmbeddings()
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))

    DEFAULTS.hybrid_alpha = 0.5
    retriever = HybridRetriever(chunks, embeddings, vdb)
    assert retriever.alpha == 0.5

    DEFAULTS.hybrid_alpha = 0.99
    assert retriever.alpha == 0.5  # unchanged — already built


# ─────────────────────────────────────────────────────────
# Propagation to other DEFAULTS-reading components
# ─────────────────────────────────────────────────────────

def test_chunk_size_overlap_propagate_to_chunker():
    DEFAULTS.chunk_size = 300
    DEFAULTS.chunk_overlap = 20
    chunker = Chunker(strategy="fixed")
    assert chunker.chunk_size == 300
    assert chunker.overlap == 20


def test_mmr_lambda_propagates_to_mmr_retriever():
    chunks = [Chunk(text=f"chunk {i} about testing") for i in range(5)]
    embeddings = FakeEmbeddings()
    vdb = InMemoryVectorDB()
    vdb.build(chunks, embeddings.embed_documents([c.text for c in chunks]))

    DEFAULTS.mmr_lambda = 0.2
    retriever = MMRRetriever(chunks, embeddings, vdb)
    assert retriever.lambda_param == 0.2

from metarag.defaults import DEFAULTS
from metarag.pipelines import pipeline

def test_dedup_threshold_propagates_to_deduplicator():
    old = DEFAULTS.dedup_threshold
    try:
        DEFAULTS.dedup_threshold = 0.7
        dedup = Deduplicator()
        assert dedup.threshold == 0.7
    finally:
        DEFAULTS.dedup_threshold = old

def test_multiquery_n_variants_propagates_when_n_is_none():
    """NOTE: MultiQuery.__init__ signature is `n: int = 3`, not `n: int = None` —
    so DEFAULTS.multiquery_n_variants only applies if n=None is passed
    explicitly. MultiQuery(generator) with no n argument always gets the
    hardcoded 3 regardless of DEFAULTS, which is worth fixing upstream."""
    DEFAULTS.multiquery_n_variants = 4
    mq = MultiQuery(FakeGenerator(), n=None)
    assert mq.n == 4


def test_multiquery_uses_defaults_when_n_not_provided():
    DEFAULTS.multiquery_n_variants = 4

    mq = MultiQuery(FakeGenerator())

    assert mq.n == 4

def test_multiquery_explicit_n_overrides_defaults():
    DEFAULTS.multiquery_n_variants = 4

    mq = MultiQuery(FakeGenerator(), n=2)

    assert mq.n == 2