# MetaRAG

> A modular framework for building, benchmarking, and experimenting with Retrieval-Augmented Generation (RAG) pipelines.

MetaRAG is a student-built open-source project that separates a RAG system into interchangeable components—document loading, chunking, embeddings, retrieval, pipelines, and evaluation.

Instead of assuming a single retrieval strategy is best, MetaRAG makes it easy to compare multiple pipelines on the same corpus and inspect their behaviour.

---

## What can MetaRAG do?

✓ Build complete RAG pipelines

✓ Mix and match retrieval components

✓ Compare retrieval pipelines

✓ Benchmark retrieval quality

✓ Inspect intermediate retrieval results

✓ Save benchmark results for later analysis

---

## A Typical Workflow

```text
Documents
   │
   ├── Load
   ├── Chunk
   ├── Embed
   ├── Retrieve
   ├── Generate
   └── Benchmark
```

Benchmarking isn't a separate utility—it is part of the development workflow.

---

## Example

```python
from metarag import MetaRAG

rag = MetaRAG(
    docs="data",
    embeddings=...,
    generator=...
)

rag.fit()

answer = rag.ask(
    "What is the main topic of this document?"
)

print(answer.text)
```

---

## Example Output

```text
DocumentLoader Report
------------------------------

Files Loaded : 8
Files Skipped: 0

Documents Extracted : 101

Chunker
------------------------------

Chunks Generated : 333

Benchmark
------------------------------

Benchmark rows : 595

Router thresholds saved.

Benchmark CSV saved.
```

---

## Package Structure

```text
metarag
├── core/          # Loading, chunking, embeddings, vector DBs, retrieval
├── pipelines/     # Retrieval pipelines & generators
├── Evaluator/     # Evaluation & scoring
├── router/        # Corpus/query/probe profiling & routing
├── metarag.py     # High-level framework (MetaRAG class)
└── defaults.py    # Global configuration (DEFAULTS)
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| Installation | Install MetaRAG and optional dependencies |
| Quick Start | Build your first pipeline |
| Architecture | Understand the framework design |
| Examples | Runnable demonstrations |
| API Reference | Public classes and interfaces |

---

## Included Examples

```text
loader_demo.py
chunker_demo.py
embeddings_demo.py
vector_db_demo.py
retriever_demo.py
pipeline_demo.py
metarag_demo.py
```

Each example focuses on one component before combining everything in `metarag_demo.py`.

---

## Project Status

MetaRAG is an actively developed student project.

The framework is intended for experimentation and learning. APIs may evolve before a stable `1.0` release.

Feedback, bug reports, and contributions are always welcome.