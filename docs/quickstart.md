# Quick Start

This guide walks through building your first RAG system with MetaRAG.

By the end, you will:

- Load documents
- Build the retrieval system
- Ask questions
- Benchmark multiple pipelines

---

## 1. Import MetaRAG

```python
from metarag import MetaRAG
from metarag import CachedEmbeddings
from metarag import OllamaGenerator
```

---

## 2. Create the Framework

```python
from langchain_ollama import OllamaEmbeddings

# LangChain is NOT a MetaRAG dependency — this is just one example
# embedding provider. Any object with .embed_query()/.embed_documents()
# works (see docs/references/contracts.md for EmbeddingInterface).
embeddings = CachedEmbeddings(
    OllamaEmbeddings(model="nomic-embed-text"),
    cache_dir=".metarag/embeddings",
)

rag = MetaRAG(
    docs="tests/data",
    embeddings=embeddings,
    generator=OllamaGenerator(model="mistral"),
)
```

---

## 3. Build the Pipeline

```python
rag.fit()
```

Example output

```text
Files Loaded        : 8

Documents Extracted : 101

Chunks Generated    : 333

Vector Index Built

Pipelines Built     : 7
```

---

## 4. Ask Questions

```python
answer = rag.ask(
    "What is the main topic of this document?"
)

print(answer.text)
```

Example

```text
The document discusses ...
```

---

## 5. Benchmark Pipelines

```python
queries = [
    "Summarize the document.",
    "What are the key findings?",
    "List important numbers."
]

results = rag.benchmark(
    queries,
    retrieval_only=True
)
```

Example output

```text
Benchmark Rows      : 595

Benchmark CSV Saved

Router Thresholds Saved
```

The benchmark results are also returned as a pandas DataFrame.

```python
print(results.head())
```

---

## 6. Inspect Results

```python
rag.leaderboard()

rag.dashboard()

rag.report()
```

These utilities summarize benchmark performance and compare pipeline behaviour. `rag.status()`, `rag.pipeline_graph()`, `rag.inspect(query)`, and `rag.trace(query)` are also available — see **API Reference**.

---

## 7. Save the Project

```python
rag.save()
```

Example output

```text
Project saved.

config.json

benchmark.csv

router_thresholds.json
```

---

## Complete Workflow

```text
MetaRAG
   │
   ├── fit()
   ├── ask()
   ├── benchmark()
   ├── leaderboard()
   ├── report()
   └── save()
```

---

## Next Steps

Explore the individual components:

- **Architecture** — framework design
- **Examples** — runnable demonstrations
- **API Reference** — public classes and methods

Or start experimenting by changing:

- retrieval strategy
- chunk size
- embedding model
- generator
- benchmark queries