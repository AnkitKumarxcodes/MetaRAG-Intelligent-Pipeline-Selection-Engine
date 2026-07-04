# metarag/core/__init__.py

from .loader import Document, DocumentLoader, LoaderInterface
from .chunking import Chunk, Chunker, ChunkerInterface
from .embeddings import EmbeddingInterface, CachedEmbeddings
from .vector_db import VectorDBInterface, InMemoryVectorDB, ChromaVectorDB, FAISSVectorDB
from .retriever import (
    RetrieverInterface,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MMRRetriever,
)

__all__ = [
    # loader
    "Document",
    "DocumentLoader",
    "LoaderInterface",
    # chunking
    "Chunk",
    "Chunker",
    "ChunkerInterface",
    # embeddings
    "EmbeddingInterface",
    "CachedEmbeddings",
    # vector_db
    "VectorDBInterface",
    "InMemoryVectorDB",
    "ChromaVectorDB",
    "FAISSVectorDB",
    # retriever
    "RetrieverInterface",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "MMRRetriever",
]