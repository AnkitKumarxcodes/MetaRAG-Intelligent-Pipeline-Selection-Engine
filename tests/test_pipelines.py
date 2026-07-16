from metarag.pipelines.pipeline import (
    Pipeline,
    StraightPipeline,
    MultiQueryPipeline,
    HyDEPipeline,
    RerankedPipeline,
    FullPipeline,
    MultiQuery,
    HyDE,
    Deduplicator,
    Reranker,
    PIPELINE_REGISTRY,
    available_pipelines,
)
import pytest


# ============================================================
# Fake Components
# ============================================================
from metarag.utils import  FakeGenerator , AlwaysFailGenerator , FakeRetriever , FakeReranker,FakeSklearnModel


# ============================================================
# MultiQuery
# ============================================================

def test_multiquery_expands():

    mq = MultiQuery(
        FakeGenerator(),
        n=3,
    )

    queries = mq.expand("What is AI?")

    assert len(queries) == 4

    assert queries[0] == "What is AI?"


def test_multiquery_failure_returns_original():

    mq = MultiQuery(
        AlwaysFailGenerator(),
        n=3,
    )

    queries = mq.expand("hello")

    assert queries == ["hello"]


# ============================================================
# HyDE
# ============================================================

def test_hyde_generates_hypothesis():

    hyde = HyDE(
        FakeGenerator()
    )

    text = hyde.generate_hypothesis(
        "What is AI?"
    )

    assert isinstance(text, str)

    assert len(text) > 10


def test_hyde_failure_returns_query():

    hyde = HyDE(
        AlwaysFailGenerator()
    )

    text = hyde.generate_hypothesis(
        "original query"
    )

    assert text == "original query"


# ============================================================
# Deduplicator
# ============================================================

def test_deduplicator_removes_duplicates():

    dedup = Deduplicator(
        threshold=0.8
    )

    chunks = [

        ("Artificial Intelligence is amazing", 0.9),

        ("Artificial Intelligence is amazing", 0.8),

        ("Deep Learning uses neural networks", 0.7),

    ]

    result = dedup.deduplicate(chunks)

    assert len(result) == 2


def test_deduplicator_empty():

    dedup = Deduplicator()

    assert dedup.deduplicate([]) == []


def test_deduplicator_similarity():

    dedup = Deduplicator()

    score = dedup._similarity(

        "hello world",

        "hello world",

    )

    assert score == 1.0


def test_deduplicator_different():

    dedup = Deduplicator()

    score = dedup._similarity(

        "apple banana",

        "machine learning",

    )

    assert score == 0.0

# ============================================================
# Base Pipeline
# ============================================================

def test_pipeline_requires_retriever():

    try:

        Pipeline(object())

        assert False

    except TypeError:

        assert True


def test_pipeline_run_returns_dict():

    pipeline = Pipeline(

        FakeRetriever(),

        name="test",

    )

    result = pipeline.run(

        "What is AI?",

        k=3,

    )

    assert isinstance(result, dict)

    assert "query" in result

    assert "chunks" in result

    assert "pipeline" in result

    assert "hypothesis" in result


def test_pipeline_returns_k_chunks():

    pipeline = Pipeline(

        FakeRetriever(),

        name="test",

    )

    result = pipeline.run(

        "What is AI?",

        k=2,

    )

    assert len(result["chunks"]) == 2


def test_pipeline_name():

    pipeline = Pipeline(

        FakeRetriever(),

        name="custom",

    )

    result = pipeline.run("AI")

    assert result["pipeline"] == "custom"


def test_pipeline_without_optional_modules():

    pipeline = Pipeline(

        FakeRetriever(),

    )

    result = pipeline.run("AI")

    assert result["hypothesis"] is None


# ============================================================
# Straight Pipeline
# ============================================================

def test_straight_pipeline():

    pipe = StraightPipeline(

        FakeRetriever()

    )

    result = pipe.run(

        "AI",

        k=3,

    )

    assert result["pipeline"] == "straight"

    assert len(result["chunks"]) == 3


# ============================================================
# MultiQuery Pipeline
# ============================================================

def test_multiquery_pipeline():

    pipe = MultiQueryPipeline(

        FakeRetriever(),

        FakeGenerator(),

        n_variants=2,

    )

    result = pipe.run(

        "AI",

        k=4,

    )

    assert result["pipeline"] == "multiquery"

    assert len(result["chunks"]) <= 4


# ============================================================
# HyDE Pipeline
# ============================================================

def test_hyde_pipeline():

    pipe = HyDEPipeline(

        FakeRetriever(),

        FakeGenerator(),

    )

    result = pipe.run(

        "AI",

        k=3,

    )

    assert result["pipeline"] == "hyde"

    assert result["hypothesis"] is not None

    assert len(result["chunks"]) == 3


def test_hyde_pipeline_failure():

    pipe = HyDEPipeline(

        FakeRetriever(),

        AlwaysFailGenerator(),

    )

    result = pipe.run(

        "AI",

        k=2,

    )

    assert result["hypothesis"] == "AI"


# ============================================================
# Reranked Pipeline
# ============================================================

def test_reranked_pipeline():

    pipe = RerankedPipeline(

        FakeRetriever(),

        FakeReranker(),

    )

    result = pipe.run(

        "AI",

        k=3,

    )

    assert result["pipeline"] == "reranked"

    assert len(result["chunks"]) == 3


def test_reranker_changes_order():

    retriever = FakeRetriever()

    reranker = FakeReranker()

    pipe = RerankedPipeline(

        retriever,

        reranker,

    )

    result = pipe.run(

        "AI",

        k=3,

    )

    assert result["chunks"][0][0] == "Chunk C about Deep Learning"

# ============================================================
# Full Pipeline
# ============================================================

def test_full_pipeline():

    pipe = FullPipeline(

        FakeRetriever(),

        FakeGenerator(),

        FakeReranker(),

        n_variants=2,

    )

    result = pipe.run(

        "AI",

        k=4,

    )

    assert result["pipeline"] == "full"

    assert result["hypothesis"] is None

    assert len(result["chunks"]) <= 4


def test_full_pipeline_returns_query():

    pipe = FullPipeline(

        FakeRetriever(),

        FakeGenerator(),

        FakeReranker(),

    )

    result = pipe.run("What is AI?")

    assert result["query"] == "What is AI?"


# ============================================================
# Registry
# ============================================================

def test_pipeline_registry():

    assert "straight" in PIPELINE_REGISTRY

    assert "multiquery" in PIPELINE_REGISTRY

    assert "hyde" in PIPELINE_REGISTRY

    assert "reranked" in PIPELINE_REGISTRY

    assert "full" in PIPELINE_REGISTRY


def test_available_pipelines():

    names = available_pipelines()

    assert isinstance(names, list)

    assert "straight" in names

    assert "full" in names


# ============================================================
# Pipeline Behaviour
# ============================================================

def test_pipeline_respects_k():

    pipe = StraightPipeline(

        FakeRetriever()

    )

    result = pipe.run(

        "AI",

        k=1,

    )

    assert len(result["chunks"]) == 1


def test_pipeline_chunks_are_sorted():

    pipe = StraightPipeline(

        FakeRetriever()

    )

    result = pipe.run(

        "AI",

        k=4,

    )

    scores = [

        score

        for _, score in result["chunks"]

    ]

    assert scores == sorted(

        scores,

        reverse=True,

    )


def test_pipeline_deduplicates():

    class DuplicateRetriever:

        def retrieve(self, query, k=4):

            return [

                ("AI", 0.9),

                ("AI", 0.8),

                ("Machine Learning", 0.7),

            ]


    pipe = StraightPipeline(

        DuplicateRetriever()

    )

    result = pipe.run(

        "AI",

        k=4,

    )

    assert len(result["chunks"]) == 2


def test_pipeline_empty_results():

    class EmptyRetriever:

        def retrieve(self, query, k=4):

            return []


    pipe = StraightPipeline(

        EmptyRetriever()

    )

    result = pipe.run(

        "AI",

    )

    assert result["chunks"] == []


# ============================================================
# Reranker
# ============================================================

def test_reranker_passthrough():

    reranker = Reranker()

    chunks = [

        ("A", 0.8),

        ("B", 0.7),

    ]

    result = reranker.rerank(

        "query",

        chunks,

        k=2,

    )

    assert len(result) == 2


# ============================================================
# BasePipeline
# ============================================================

def test_base_pipeline():

    from metarag.pipelines.pipeline import BasePipeline

    base = BasePipeline()

    try:

        base.run("AI")

        assert False

    except NotImplementedError:

        assert True


# ============================================================
# _chunk_text Helper
# ============================================================

def test_chunk_text_string():

    from metarag.pipelines.pipeline import _chunk_text

    assert _chunk_text("hello") == "hello"


def test_chunk_text_tuple():

    from metarag.pipelines.pipeline import _chunk_text

    assert _chunk_text(("hello", 0.8)) == "hello"


def test_chunk_text_object():

    from metarag.pipelines.pipeline import _chunk_text


    class Chunk:

        text = "sample"


    assert _chunk_text(Chunk()) == "sample"


# ============================================================
# End
# ============================================================