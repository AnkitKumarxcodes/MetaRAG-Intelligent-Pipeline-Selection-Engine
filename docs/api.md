# API Reference

MetaRAG's public API is locked at **~17 methods** on the `MetaRAG` class (v0.3.8). Everything else — retrievers, vector DBs, chunkers, pipelines, generators — is a **configurable component** you can swap in the constructor or use standalone in toolkit mode.

---

# High-Level API

```python
from metarag import MetaRAG
```

## Constructor

```python
MetaRAG(
    docs,               # path(s) to documents
    embeddings,          # EmbeddingInterface — .embed_query() / .embed_documents()
    generator,           # GeneratorInterface — .generate(prompt)
    project="default",   # storage namespace: .metarag/<project>/
    vector_db=None,      # VectorDBInterface (default: InMemoryVectorDB)
    chunk_size=None,     # overrides DEFAULTS.chunk_size for this run
    chunk_overlap=None,
    chunk_strategy=None, # fixed|recursive|semantic|sentence|sliding_window|markdown
    k=None,
    eval_preset=None,    # balanced|precision|recall
    verbose=True,
)
```

Any of `chunk_size` / `chunk_overlap` / `chunk_strategy` / `k` / `eval_preset` passed here also propagates into `DEFAULTS`, so every retriever/pipeline/router built afterward picks it up automatically.

---

# Core Lifecycle

## `fit(force=False) -> MetaRAG`
Load documents → chunk → embed → index → build pipelines.
```python
rag.fit()
```

## `ask(query: str) -> Answer`
Retrieve + generate using the active router (falls back to the first built pipeline if no router is set).
```python
answer = rag.ask("What is the main topic of this document?")
answer.text, answer.pipeline, answer.score, answer.latency_ms, answer.sources
```

## `benchmark(queries, retrieval_only=True, train_router=True, save_csv=True) -> pandas.DataFrame`
Run every built pipeline against every query, score each, and (by default) train the built-in router on the results.
```python
df = rag.benchmark(["Summarize the document.", "List the key findings."])
```

---

# Inspection

| Method | Returns | Purpose |
|---|---|---|
| `status()` | `dict` | Project snapshot: fitted state, chunk count, registered pipelines, avg score, active router |
| `leaderboard(source="benchmark")` | summary | Pipeline ranking from `"benchmark"` (last `benchmark()` run) or `"logs"` (real `ask()` usage) |
| `analyze_query(query)` | `dict` | Standalone query diagnostic (length, complexity, keywords) — no router involved |
| `analyze_corpus()` | `dict` | Chunk count, avg chunk size, corpus profile |
| `explain(query)` | `dict` | Which pipeline the router picked and why, plus the full feature vector |

```python
rag.status()
rag.leaderboard()
rag.explain("How do I configure the retriever?")
```

---

# Observability *(v0.3)*

| Method | Purpose |
|---|---|
| `pipeline_graph(pipeline_name=None)` | Structural diagram of a pipeline's stages (introspects what's actually attached — not hardcoded) |
| `dashboard()` | Bar-chart leaderboard from the last `benchmark()` run |
| `report()` | Corpus profile summary |
| `inspect(query, k=None)` | Run every built **retriever** independently on one query, side-by-side |
| `trace(query, pipeline_name=None)` | Step-by-step chunk counts through one pipeline's stages |

```python
rag.pipeline_graph()   # every built pipeline
rag.trace("your query", pipeline_name="full")
```

---

# Persistence

## `save() -> MetaRAG`
Persists config (settings only, not the index/router) to `.metarag/<project>/config.json`.

## `load(project, embeddings, generator, vector_db=None) -> MetaRAG` *(classmethod)*
Rebuilds a `MetaRAG` shell from a saved config. Call `fit()` afterward to rehydrate the index — fast, since it hits disk caches.
```python
rag = MetaRAG.load("default", embeddings=embeddings, generator=generator)
rag.fit()
```
> **Note:** `load()` is a classmethod, not an instance method — `MetaRAG.load(...)`, not `rag.load(...)`.

---

# Benchmark & Router Data Access

| Method | Returns |
|---|---|
| `get_benchmark_data()` | Raw `benchmark.csv` as a DataFrame — the training set for your own router |
| `get_router_thresholds()` | Whatever the active router reports about its own state |

---

# Configuration — Swapping Components

```python
rag.set_llm(generator)                       # swap the generator
rag.set_embeddings(embeddings)                 # swap the embedding model
rag.set_router(router)                          # any object with .route(features) -> str
rag.set_router_from_model(sklearn_model, feature_cols)  # wrap a sklearn-style .predict() model
rag.update_router_thresholds(path=None)           # hot-reload a saved router_thresholds.json
rag.rebuild(force=True)                             # force a full re-fit()
```

`set_router_from_model` example:
```python
df = rag.get_benchmark_data()
cols = ["max_similarity", "avg_similarity", "redundancy", "query_length"]
clf = RandomForestClassifier().fit(df[cols], df["winning_pipeline"])
rag.set_router_from_model(clf, feature_cols=cols)
```

---

# Which Component Should I Use?

The API above is locked, but everything it orchestrates is swappable. Quick picks:

### Vector DB
| Use | When |
|---|---|
| `InMemoryVectorDB()` | Default — zero dependencies, fine up to a few thousand chunks |
| `ChromaVectorDB(persist_directory=...)` | You want the index to persist across runs |
| `FAISSVectorDB()` | Larger corpora, need faster similarity search |

### Retriever
| Use | When |
|---|---|
| `BM25Retriever(chunks)` | Keyword-heavy queries, logs, exact terms |
| `DenseRetriever(chunks, embeddings, vector_db)` | Semantic/paraphrased queries |
| `HybridRetriever(chunks, embeddings, vector_db, alpha=0.5)` | Default general-purpose choice — `alpha` 0.0=BM25 → 1.0=dense |
| `MMRRetriever(chunks, embeddings, vector_db, lambda_param=0.6)` | Repetitive corpus, want diverse chunks — `lambda_param` 0.0=diverse → 1.0=relevant |

### Chunk Strategy (`DEFAULTS.chunk_strategy`)
| Strategy | Best for |
|---|---|
| `fixed` | Quick baseline |
| `recursive` | General purpose (default) |
| `sentence` | Conversational text |
| `semantic` | Loosely topic-grouped text |
| `sliding_window` | Overlap-heavy retrieval |
| `markdown` | Structured docs — splits on headers |

### Pipeline (`available_pipelines()` → 5 built-in)
| Pipeline | What it does |
|---|---|
| `straight` | Retrieve only — fastest |
| `multiquery` | Expand query into variants, retrieve on all, merge |
| `hyde` | Generate a hypothetical answer, retrieve using that instead of the raw query |
| `reranked` | Retrieve, then cross-encoder rerank (needs `sentence-transformers`) |
| `full` | MultiQuery + Reranking — most thorough |

### Generator
Bring anything with `.generate(prompt) -> str`. `OllamaGenerator(model="mistral")` is the built-in free/local convenience wrapper.

---

# Toolkit Mode

Every component above is independently importable:
```python
from metarag import DocumentLoader, Chunker, HybridRetriever, InMemoryVectorDB, Evaluator
```
Useful for building a custom pipeline without going through the `MetaRAG` class. Toolkit-mode signatures are tested but pre-1.0 — pin your version if you depend on them directly in production.

---

# Public Modules

| Module | Purpose |
|---|---|
| `core` | Loading, chunking, embeddings, vector DBs, retrievers |
| `pipelines` | Retrieval pipelines and generators |
| `Evaluator` | Evaluation and scoring |
| `router` | Query/corpus/probe profiling and routing |
| `defaults` | Global framework configuration (`DEFAULTS`) |

---

# Version

This documentation describes the public API in **MetaRAG v0.3.8**. Interfaces may evolve before `1.0`.
