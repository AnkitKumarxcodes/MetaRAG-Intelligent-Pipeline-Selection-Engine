# Contracts

MetaRAG is built around lightweight interfaces ("contracts") rather than fixed implementations.

Any object that satisfies the required methods can be used inside the framework.

---

> [!TIP]
> **Duck Typing**
>
> MetaRAG generally checks **behavior, not inheritance**.
>
> Your custom component does **not** have to inherit from a MetaRAG interface—it only needs to provide the expected methods.

---

# Overview

| Component | Required Contract | Returns |
|-----------|-------------------|---------|
| Document Loader | `LoaderInterface` | `DocumentList` |
| Chunker | `ChunkerInterface` | `List[Chunk]` |
| Embeddings | `EmbeddingInterface` | Embedding Vector(s) |
| Vector Database | `VectorDBInterface` | Search Results |
| Retriever | `RetrieverInterface` | Retrieved Chunks |
| Generator | `GeneratorInterface` | Generated Text |
| Router | `RouterInterface` | Pipeline Name |

---

# LoaderInterface

Loads one or more documents into MetaRAG.

### Required

```python
load(
    verbose: bool = True
) -> DocumentList
```

| Input | Output |
|-------|--------|
| Path | `DocumentList` |

Built-in implementation

```text
DocumentLoader
```

---

# ChunkerInterface

Splits documents into retrieval chunks.

### Required

```python
chunk(
    documents
) -> List[Chunk]
```

| Input | Output |
|-------|--------|
| `DocumentList` | `List[Chunk]` |

Built-in implementation

```text
Chunker
```

---

# EmbeddingInterface

Generates vector representations.

### Required

```python
embed_query(text)

embed_documents(texts)
```

| Method | Returns |
|---------|----------|
| `embed_query()` | `List[float]` |
| `embed_documents()` | `List[List[float]]` |

`embed(text)` is also available as a built-in alias for `embed_query(text)`.

Built-in implementation

```text
CachedEmbeddings
```

---

# VectorDBInterface

Stores embeddings and performs similarity search.

### Required

```python
build(chunks, embeddings)

search(
    query_embedding,
    k
)
```

| Method | Returns |
|---------|----------|
| `build()` | None |
| `search()` | `List[(Chunk, score)]` |

Built-in implementations

```text
InMemoryVectorDB

ChromaVectorDB

FAISSVectorDB
```

---

# RetrieverInterface

Retrieves relevant chunks for a query.

### Required

```python
retrieve(
    query,
    k
)
```

| Input | Output |
|-------|--------|
| Query | `List[(Chunk, score)]` |

Built-in implementations

```text
BM25Retriever

DenseRetriever

HybridRetriever

MMRRetriever
```

---

# GeneratorInterface

Generates text from retrieved context.

### Required

```python
generate(
    prompt
)
```

| Input | Output |
|-------|--------|
| Prompt | Generated Text |

Built-in implementation

```text
OllamaGenerator
```

---

# RouterInterface

Chooses which retrieval pipeline should handle a query.

### Required

```python
route(
    features
)
```

Returns

```python
pipeline_name
```

### Optional

```python
explain(
    pipeline,
    features=None
)
```

The framework calls `explain()` only when available.

Built-in implementation

```text
Router
```

---

# GeneratorWrapper

`GeneratorWrapper` adapts any compatible generator into MetaRAG.

Responsibilities

- Builds prompts
- Measures latency
- Retries failed requests
- Returns generated text

Return type

```python
(
    generated_text,
    latency_ms
)
```

---

# BasePipeline

Every pipeline follows the same execution model.

```python
run(
    query,
    k
)
```

Returns

```python
{
    "query": ...,
    "chunks": ...,
    "pipeline": ...,
    "hypothesis": ...
}
```

Built-in pipelines

```text
Straight

MultiQuery

HyDE

Reranked

Full
```

---

# Pipeline Components

Some pipeline features are reusable and can be combined.

| Component | Purpose |
|-----------|---------|
| `MultiQuery` | Expands a query into multiple variants |
| `HyDE` | Generates a hypothetical answer for retrieval |
| `Reranker` | Reorders retrieved chunks |
| `Deduplicator` | Removes near-duplicate chunks |

---

> [!IMPORTANT]
> Pipeline components are optional.
>
> A pipeline may use any combination of:
>
> - Retriever
> - MultiQuery
> - HyDE
> - Reranker
> - Deduplicator

---

# Configuration Contract

Many built-in components read values from the shared configuration object.

Example

```python
from metarag import DEFAULTS

DEFAULTS.chunk_size = 600

DEFAULTS.k = 5

DEFAULTS.hybrid_alpha = 0.6
```

Components use these defaults unless explicitly overridden.

---

# Compatibility Matrix

The table below summarizes how MetaRAG components interact.

| Component | Interface | Primary Input | Primary Output |
|-----------|-----------|---------------|----------------|
| Document Loader | `LoaderInterface` | `Path` | `DocumentList` |
| Chunker | `ChunkerInterface` | `DocumentList` | `List[Chunk]` |
| Embeddings | `EmbeddingInterface` | `str`, `List[str]` | `List[float]`, `List[List[float]]` |
| Vector Database | `VectorDBInterface` | `List[Chunk]`, Embeddings | Search Index |
| Retriever | `RetrieverInterface` | Query | `List[(Chunk, score)]` |
| Pipeline | `BasePipeline` | Query | `PipelineResult` |
| Generator | `GeneratorInterface` | Prompt | `str` |
| Evaluator | `Evaluator` | `Answer` | `ScoreResult` |
| Router | `RouterInterface` | Feature Dictionary | Pipeline Name |

---

# End-to-End Contract Flow

```text
DocumentLoader
      │
      ▼
DocumentList
      │
      ▼
Chunker
      │
      ▼
List[Chunk]
      │
      ├──────────────┐
      ▼              ▼
 Embeddings      VectorDB
      │              │
      └──────┬───────┘
             ▼
        Retriever
             │
             ▼
         Pipeline
             │
             ▼
     GeneratorWrapper
             │
             ▼
          Answer
             │
             ▼
        Evaluator
             │
             ▼
       ScoreResult
```

---

## Design Philosophy

MetaRAG interfaces are intentionally lightweight.

A component is considered compatible if it provides the required behavior, allowing existing libraries, research code, or custom implementations to integrate with minimal changes.