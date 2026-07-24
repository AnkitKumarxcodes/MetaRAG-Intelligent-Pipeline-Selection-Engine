# Data Types

This page describes the core data structures used throughout MetaRAG.

Rather than documenting every implementation, it focuses on the objects exchanged between components and returned by the public API.

---

> [!NOTE]
> **Type Flow**
>
> ```text
> Path
>   │
>   ▼
> DocumentLoader
>   │
>   ▼
> DocumentList
>   │
>   ▼
> Chunker
>   │
>   ▼
> List[Chunk]
>   │
>   ├─────────────► Embeddings
>   │                    │
>   │                    ▼
>   │               Vector Database
>   │                    │
>   ▼                    ▼
>              Retriever / Pipeline
>                       │
>                       ▼
>                    Answer
>                       │
>                       ▼
>                  ScoreResult
> ```

---

# Document

Represents a single loaded document before chunking.

| Field | Type | Description |
|--------|------|-------------|
| `text` | `str` | Document contents |
| `metadata` | `dict` | Optional metadata (filename, source, etc.) |

Used by

```text
DocumentLoader
        │
        ▼
DocumentList
```

---

# DocumentList

A container returned by `DocumentLoader.load()`.

Besides behaving like a normal list, it stores loading statistics.

| Attribute | Type | Description |
|-----------|------|-------------|
| `loaded` | `ExtensionCollection` | Successfully loaded files |
| `skipped` | `ExtensionCollection` | Skipped files |

Example

```python
docs = loader.load()

len(docs)

docs.loaded.count

docs.loaded.files
```

---

# ExtensionCollection

Stores statistics grouped by file extension.

Example

```python
docs.loaded["pdf"].count

docs.loaded["pdf"].files
```

| Attribute | Type |
|-----------|------|
| `count` | `int` |
| `files` | `List[str]` |

---

# ExtensionStats

Statistics for one file extension.

Example

```python
pdf = docs.loaded["pdf"]

pdf.count

pdf.files
```

| Field | Type |
|--------|------|
| `count` | `int` |
| `files` | `List[str]` |

---

# Chunk

Represents one chunk produced by the `Chunker`.

| Field | Type |
|--------|------|
| `text` | `str` |
| `metadata` | `dict` |

Produced by

```text
Chunker
      │
      ▼
List[Chunk]
```

---

# Answer

Returned by

```python
rag.ask(...)
```

| Field | Type | Description |
|--------|------|-------------|
| `text` | `str` | Generated answer |
| `query` | `str` | Original query |
| `pipeline` | `str` | Pipeline used |
| `chunks` | `List[str]` | Retrieved context |
| `score` | `float` | Evaluation score (if available) |
| `latency_ms` | `float` | End-to-end latency |
| `sources` | `List[str]` | Top-3 source identifiers behind the answer |

Example

```python
answer = rag.ask(...)

print(answer.text)

print(answer.pipeline)

print(answer.latency_ms)
```

---

# ScoreResult

Returned by

```python
Evaluator.evaluate(...)
```

or internally during benchmarking.

| Field | Type |
|--------|------|
| `faithfulness` | `float` |
| `relevancy` | `float` |
| `precision_avg` | `float` |
| `precision_max` | `float` |
| `precision_std` | `float` |
| `coverage` | `float` |
| `redundancy` | `float` |
| `latency_ms` | `float` |
| `composite` | `float` |

Example

```python
score = evaluator.evaluate(answer)

score.composite

score.faithfulness
```

---

# Common Input Types

| Type | Used By |
|------|---------|
| `Path` | DocumentLoader |
| `DocumentList` | Chunker |
| `List[Chunk]` | Embeddings, VectorDB, Retriever |
| `str` | Query |
| `List[str]` | Batch embedding |
| `List[float]` | Embedding vector |
| `dict` | Router features |

---

# Common Return Types

| Method | Returns |
|---------|---------|
| `DocumentLoader.load()` | `DocumentList` |
| `Chunker.chunk()` | `List[Chunk]` |
| `Retriever.retrieve()` | `List[(Chunk, score)]` |
| `Pipeline.run()` | `dict` |
| `GeneratorWrapper.generate_text()` | `(str, latency_ms)` |
| `MetaRAG.ask()` | `Answer` |
| `MetaRAG.benchmark()` | `pandas.DataFrame` |
| `Evaluator.evaluate()` | `ScoreResult` |
| `Router.route()` | `str` |

---

# Pipeline Result

Every pipeline returns a dictionary with a common structure.

| Key | Type |
|-----|------|
| `query` | `str` |
| `chunks` | `List[(Chunk, score)]` |
| `pipeline` | `str` |
| `hypothesis` | `str \| None` |

---

# Feature Dictionary

Routers receive a merged feature dictionary combining three sources.

```text
QueryProfiler
        │
CorpusProfiler
        │
ProbeProfiler
        ▼
Merged Feature Dictionary
```

This dictionary is passed unchanged to

```python
router.route(features)
```

---

# Summary

| Category | Main Type |
|-----------|-----------|
| Documents | `Document`, `DocumentList` |
| Retrieval | `Chunk` |
| Generation | `Answer` |
| Evaluation | `ScoreResult` |
| Routing | `dict` (merged features) |
| Benchmarking | `pandas.DataFrame` |