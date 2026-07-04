# metarag/__init__.py

"""
MetaRAG — AutoML for RAG.

Two ways to use this library:

  1. High-level (recommended for most users):
        from metarag import MetaRAG
        rag = MetaRAG(docs, embeddings, generator)
        rag.fit()
        rag.ask("...")

  2. Toolkit mode (compose your own pipeline, sklearn/PyTorch-style):
        from metarag import HybridRetriever, Reranker, Evaluator
        retriever = HybridRetriever(chunks, embeddings, vector_db)
        ...

NOTE ON STABILITY: MetaRAG's top-level class and its 12 public methods
(fit, ask, benchmark, status, leaderboard, analyze_query, analyze_corpus,
explain, save, load, get_benchmark_data, get_router_thresholds) plus the
configuration methods (set_llm, set_embeddings, set_router, rebuild) are
the stable, documented API as of v0.3.

Everything re-exported below from core/pipeline/evaluation/router (toolkit
mode) is usable and tested, but should be considered pre-1.0 / subject to
signature changes as the toolkit surface gets battle-tested. Pin your
MetaRAG version if you depend directly on these for production use.
"""

from .metarag import MetaRAG, Answer

# Toolkit mode — Mode 1 composable primitives
from .core import (
    Document,
    DocumentLoader,
    LoaderInterface,
    Chunk,
    Chunker,
    ChunkerInterface,
    EmbeddingInterface,
    CachedEmbeddings,
    VectorDBInterface,
    InMemoryVectorDB,
    ChromaVectorDB,
    FAISSVectorDB,
    RetrieverInterface,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MMRRetriever,
)

from .pipelines import (
    GeneratorInterface,
    OllamaGenerator,
    GeneratorWrapper,
    build_prompt,
    MultiQuery,
    HyDE,
    Reranker,
    Deduplicator,
    Pipeline,
    BasePipeline,
    StraightPipeline,
    MultiQueryPipeline,
    RerankedPipeline,
    HyDEPipeline,
    FullPipeline,
    available_pipelines,
)

from .Evaluator import (
    faithfulness,
    relevancy,
    precision,
    coverage,
    redundancy,
    Scorer,
    ScoreResult,
    Evaluator,
)

from .router import (
    RouterInterface,
    QueryProfiler,
    CorpusProfiler,
    ProbeProfiler,
    Router,
    LearnedRuleRouter,
)

__version__ = "0.3.0"

__all__ = [
    # high-level API
    "MetaRAG",
    "Answer",
    # core
    "Document",
    "DocumentLoader",
    "LoaderInterface",
    "Chunk",
    "Chunker",
    "ChunkerInterface",
    "EmbeddingInterface",
    "CachedEmbeddings",
    "VectorDBInterface",
    "InMemoryVectorDB",
    "ChromaVectorDB",
    "FAISSVectorDB",
    "RetrieverInterface",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "MMRRetriever",
    # pipeline
    "GeneratorInterface",
    "OllamaGenerator",
    "GeneratorWrapper",
    "build_prompt",
    "MultiQuery",
    "HyDE",
    "Reranker",
    "Deduplicator",
    "Pipeline",
    "BasePipeline",
    "StraightPipeline",
    "MultiQueryPipeline",
    "RerankedPipeline",
    "HyDEPipeline",
    "FullPipeline",
    "available_pipelines",
    # evaluation
    "faithfulness",
    "relevancy",
    "precision",
    "coverage",
    "redundancy",
    "Scorer",
    "ScoreResult",
    "Evaluator",
    # router
    "RouterInterface",
    "QueryProfiler",
    "CorpusProfiler",
    "ProbeProfiler",
    "Router",
    "LearnedRuleRouter",
]