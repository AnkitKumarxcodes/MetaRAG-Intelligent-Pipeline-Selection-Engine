# metarag/pipeline/__init__.py

from .generator import GeneratorInterface, OllamaGenerator, GeneratorWrapper, build_prompt
from .pipeline import (
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

__all__ = [
    # generator
    "GeneratorInterface",
    "OllamaGenerator",
    "GeneratorWrapper",
    "build_prompt",
    # pipeline stages
    "MultiQuery",
    "HyDE",
    "Reranker",
    "Deduplicator",
    # pipeline composition
    "Pipeline",
    "BasePipeline",
    "StraightPipeline",
    "MultiQueryPipeline",
    "RerankedPipeline",
    "HyDEPipeline",
    "FullPipeline",
    "available_pipelines",
]