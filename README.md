<div align="center">
<img src="assets/metarag_logo.svg" width="680" alt="MetaRAG — Intelligent Pipeline Selection Engine"/>


<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-red?style=for-the-badge)]()

<br/>

> **MetaRAG** is an open-source toolkit that takes the guesswork out of RAG pipeline design.
> Instead of manually tuning chunking strategies, retrieval backends, and rerankers —
> MetaRAG benchmarks them all and automatically routes every query to the configuration
> that performs best on your data.
>
> *Think of it as AutoML, but for RAG.*

<br/>

---

</div>

## 📌 Table of Contents

- [Why MetaRAG](#-why-metarag)
- [Architecture](#-architecture)
- [Components](#-components)
- [Quickstart](#-quickstart)
- [Pipeline Selection](#-pipeline-selection)
- [Evaluation](#-evaluation)
- [Supported Models](#-supported-models)
- [Roadmap](#-roadmap)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)

---

## 🤔 Why MetaRAG

Every team building a RAG system faces the same unsolved problem:

```
Which chunking strategy should I use?
Which retrieval method works best for my documents?
How do I know if my pipeline is actually good?
What happens when a different query type breaks everything?
```

Current tools make you answer these questions manually — every time, for every project.

| Tool | Build RAG | Evaluate | Compare | Auto-Select | Learn |
|------|-----------|----------|---------|-------------|-------|
| LangChain | ✅ | ❌ | ❌ | ❌ | ❌ |
| LlamaIndex | ✅ | ❌ | ❌ | ❌ | ❌ |
| RAGAS | ❌ | ✅ | ❌ | ❌ | ❌ |
| **MetaRAG** | ✅ | ✅ | ✅ | ✅ | ✅ |

MetaRAG owns the **entire RAG workflow** — from raw documents to evaluated, auto-selected, continuously improving answers.

---

## 🏗 Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              USER INTERFACE             │
                        │         MetaRAG(docs).fit().ask()       │
                        └─────────────────┬───────────────────────┘
                                          │
                 ┌────────────────────────▼────────────────────────┐
                 │                    METARAG CORE                 │
                 │                                                 │
                 │   ┌──────────┐    ┌──────────┐   ┌──────────┐   │
    Documents ──►│   │  Loader  │───►│ Chunker  │──►│VectorDB  │   │
                 │   └──────────┘    └──────────┘   └────┬─────┘   │
                 │                                        │        │
                 │                              ┌─────────▼──────┐ │
                 │                              │   Retrievers   │ │
                 │                              │ BM25 │ Dense   │ │
                 │                              │ Hybrid │ MMR   │ │
                 │                              └─────────┬──────┘ │
                 │                                        │        │
    Query ──────►│   ┌──────────┐               ┌────────▼──────┐  │
                 │   │  Router  │──────────────► │   Pipelines  │  │
                 │   │(Selector)│               │Straight│HyDE  │  │
                 │   └──────────┘               │MQuery │Full   │  │
                 │                              └────────┬──────┘  │
                 │                                        │        │
                 │   ┌──────────┐               ┌────────▼──────┐  │
                 │   │ Learning │◄──────────────│   Evaluator   │  │
                 │   │  Loop    │               │ Fast │ Deep   │  │
                 │   └──────────┘               └────────┬──────┘  │ 
                 │                                        │        │
                 └────────────────────────────────────────┼────────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │   Answer   │
                                                    │text│score  │
                                                    │pipeline│ms │
                                                    └────────────┘
```

---

## 🧩 Components

### 📂 Document Loader
Loads any document type automatically — no configuration needed.

```python
from metarag.loader import DocumentLoader

loader = DocumentLoader("./data")                          # folder
loader = DocumentLoader(["./docs", "./reports"])           # multiple folders
loader = DocumentLoader("./data/contract.pdf")             # single file
docs   = loader.load_all(urls=["https://example.com/faq"]) # + web pages
```

| Format | Support |
|--------|---------|
| PDF | ✅ |
| TXT | ✅ |
| DOCX | ✅ |
| HTML | ✅ |
| CSV | ✅ |
| JSON | ✅ |
| Web URLs | ✅ |
| Sitemaps | ✅ |
| Nested directories | ✅ |

---

### ✂️ Chunker
Six strategies. One interface.

```python
from metarag.chunking import Chunker

chunker = Chunker(strategy="recursive")      # best default
chunks  = chunker.chunk_documents(docs)
print(chunker.stats(chunks))
```

| Strategy | Best For | Cost |
|----------|----------|------|
| `fixed` | Quick baseline | Free |
| `sentence` | Conversational text | Free |
| `recursive` | General purpose ⭐ | Free |
| `parent_child` | Long documents | Free |
| `semantic` | Topic coherent chunks | Embeddings |
| `proposition` | Maximum precision | LLM call |

---

### 🗄 Vector Database
Chroma and FAISS. Build once, load forever.

```python
from metarag.vector_db import VectorDB

db = VectorDB(embeddings, db_type="chroma")
db.build(chunks)     # indexes everything
db.load()            # reload next session — no rebuild
db.add(new_chunks)   # incremental updates
```

---

### 🔍 Retrievers
Four retrieval strategies, all with confidence scores.

```python
from metarag.retriever import get_retriever

r = get_retriever("bm25",   chunks=chunks)           # keyword
r = get_retriever("dense",  vectordb=db)             # semantic
r = get_retriever("hybrid", vectordb=db, chunks=chunks, alpha=0.5)  # combined
r = get_retriever("mmr",    vectordb=db)             # diverse results

docs   = r.retrieve("your query")
scored = r.retrieve_with_score("your query")         # with confidence
```

---

### ⚙️ Pipelines
Dynamic pipelines. Router decides which steps to activate.

```python
from metarag.pipeline import Pipeline, MultiQuery, Reranker, HyDE

# straight — just retrieve
pipeline = Pipeline(retriever=r)

# multiquery — expand query, retrieve broadly
pipeline = Pipeline(retriever=r, multiquery=MultiQuery(llm, n=3))

# reranked — retrieve then reorder by precision
pipeline = Pipeline(retriever=r, reranker=Reranker())

# hyde — generate hypothesis, retrieve on that
pipeline = Pipeline(retriever=r, hyde=HyDE(llm))

# full — everything combined
pipeline = Pipeline(retriever=r, multiquery=MultiQuery(llm), reranker=Reranker())
```

---

### 🤖 Generator
One place. All LLMs. Returns structured Answer objects.

```python
from metarag.generator import get_generator

generator = get_generator("ollama", model="llama3")   # free, local
generator = get_generator("groq",   model="llama3-8b-8192")  # free tier
generator = get_generator("openai", model="gpt-4o-mini")     # paid

answer = generator.generate(query, chunks, pipeline="hybrid")
print(answer.text)          # the answer
print(answer.latency_ms)    # how long it took
print(answer.chunks)        # what was used
```

---

### 📊 Evaluator
Two modes. Zero API cost in default mode.

```python
from metarag.evaluator import Evaluator

evaluator = Evaluator(embedding_model=embeddings)

result = evaluator.evaluate(answer)              # fast, ~50ms, always on
result = evaluator.evaluate(answer, deep=True)   # LLM judge, on demand

print(result.faithfulness)   # grounded in chunks?
print(result.relevancy)      # addresses the question?
print(result.precision)      # were chunks useful?
print(result.composite)      # single score for learning loop
```

---

### 🔀 Router / Selector
Hardcoded rules now. Trained ML model later.

```python
from metarag.selector import Router

router   = Router()
decision = router.route("what is the refund policy?")
# → {"query_type": "keyword", "pipeline": "straight"}
```

| Query Type | Example | Pipeline Selected |
|------------|---------|-------------------|
| `vague` | "refund" | HyDE |
| `keyword` | "error code 404" | Straight (BM25) |
| `semantic` | "how does auth work" | Reranked (Dense) |
| `complex` | "compare X and Y" | MultiQuery |

---

## ⚡ Quickstart

### Installation

```bash
git clone https://github.com/yourusername/metarag
cd metarag
pip install -r requirements.txt
```

### Dependencies

```bash
pip install pypdf beautifulsoup4 python-docx pandas requests lxml \
            nltk langchain langchain-text-splitters langchain-community \
            langchain-huggingface langchain-experimental langchain-ollama \
            rank-bm25 chromadb sentence-transformers datasets
```

### Setup Ollama (free, local)

```bash
# install from ollama.com then pull models
ollama pull llama3              # generation + evaluation
ollama pull nomic-embed-text    # embeddings
```

### Run the test

```bash
# put your documents in ./data
python main.py
```

### Output

```
=========================================================================================
PIPELINE       FAITH  RELEV   PREC  SCORE    LATENCY  ROUTER
=========================================================================================
reranked        0.84   0.91   0.78   0.84      1240ms
multiquery      0.81   0.88   0.76   0.82       890ms  ✅
hyde            0.79   0.85   0.74   0.79      1100ms
hybrid          0.74   0.82   0.71   0.76       340ms
dense           0.71   0.79   0.68   0.73       290ms
straight        0.63   0.71   0.61   0.65       120ms
=========================================================================================

🏆 Best pipeline : reranked  (score=0.84)
🔀 Router picked : multiquery
```

---

## 🔁 Pipeline Selection

MetaRAG does not commit to one pipeline. Every query gets the right one.

```
User asks a question
        │
        ▼
Router analyses query
        │
  ┌─────┴──────┐
  │ query type │
  └─────┬──────┘
        │
   ┌────▼─────────────────────────────────┐
   │ vague    → HyDE pipeline             │
   │ keyword  → Straight (BM25)           │
   │ semantic → Reranked (Dense)          │
   │ complex  → MultiQuery (Hybrid)       │
   └──────────────────────────────────────┘
        │
        ▼
  Best chunks retrieved
        │
        ▼
  Generator produces Answer
        │
        ▼
  Evaluator scores it
        │
        ▼
  Learning loop logs outcome
```

---

## 📐 Evaluation

MetaRAG uses a two-tier evaluation system with zero API cost by default.

### Fast Evaluation *(default, ~50ms)*
```
Faithfulness   →  word overlap between answer and retrieved chunks
Relevancy      →  cosine similarity between query and answer embeddings
Precision      →  cosine similarity between query and chunk embeddings
Composite      →  mean of all three — drives the learning loop
```

### Deep Evaluation *(on demand, local LLM)*
```
Faithfulness   →  LLM judge: "is this answer grounded in the context?"
Relevancy      →  LLM judge: "does this answer address the question?"
Precision      →  embedding similarity (same as fast)
```

No OpenAI. No cloud API. Runs entirely on your machine.

---

## 🤖 Supported Models

### Embeddings

| Model | Size | Speed | Quality | Cost |
|-------|------|-------|---------|------|
| `nomic-embed-text` (Ollama) | 270MB | Moderate | ⭐⭐⭐⭐ | Free |
| `all-MiniLM-L6-v2` (HuggingFace) | 80MB | Fast | ⭐⭐⭐ | Free |
| `BAAI/bge-small-en` (HuggingFace) | 130MB | Fast | ⭐⭐⭐⭐ | Free |
| OpenAI `text-embedding-3-small` | API | Fast | ⭐⭐⭐⭐⭐ | Paid |

### Generation

| Model | Provider | Cost | Speed |
|-------|----------|------|-------|
| `llama3` | Ollama (local) | Free | Moderate |
| `mistral` | Ollama (local) | Free | Fast |
| `llama3-8b-8192` | Groq API | Free tier | Very fast |
| `gpt-4o-mini` | OpenAI | Paid | Fast |

---

## 🗺 Roadmap

### v0.1 — Foundation *(current)*
- [x] Document loader — PDF, HTML, DOCX, CSV, JSON, URLs
- [x] 6 chunking strategies with unified interface
- [x] Vector database — Chroma + FAISS
- [x] 4 retrieval strategies with confidence scores
- [x] Dynamic pipeline composition — MultiQuery, HyDE, Reranker
- [x] Generator — Ollama, Groq, OpenAI
- [x] Two-tier evaluator — fast + deep
- [x] Rule-based router

### v0.2 — Intelligence
- [ ] `MetaRAG` top-level class — `fit()`, `ask()`, `leaderboard()`
- [ ] Learning loop — logs outcomes, improves routing over time
- [ ] Benchmark data generator — auto QA pairs from documents
- [ ] Trained ML router — LogisticRegression on collected data
- [ ] Trained ML evaluator — replaces embedding heuristics
- [ ] `RAGComparison` — compare any two configs fairly

### v0.3 — Toolkit
- [ ] `RAGTuner` — auto hyperparameter search (chunk_size, k, alpha)
- [ ] Experiment tracking — save and compare runs
- [ ] `pip install metarag` — proper package release
- [ ] CLI — `metarag fit ./data` `metarag ask "question"`

---

## 🔭 Future Scope

### 🤖 Agentic Workflow (v1.0)
MetaRAG will support **LangGraph-based agentic execution** — where the system can loop, retry with a different pipeline if confidence is low, and handle multi-hop questions that require multiple retrieval steps.

```
query → retrieve → evaluate
                      │
               score < 0.6?
                      │
              retry with different pipeline
                      │
               score >= 0.6?
                      │
                return answer
```

This turns MetaRAG from a pipeline selector into a **self-correcting retrieval agent**.

### 🌐 REST API (v1.5)
A FastAPI layer that exposes MetaRAG over HTTP — enabling any platform or language to use it.

```bash
POST /upload    # index a document set
POST /ask       # get an answer
GET  /leaderboard  # pipeline scores
GET  /history   # query history
```

Designed for organisations that cannot install Python directly — they just call the API.

### 🏢 Platform Integrations (v2.0)
Native integrations with where organisations actually work:

```
MetaRAG for Notion       →  query your Notion workspace
MetaRAG for Confluence   →  search your team's knowledge base
MetaRAG for SharePoint   →  enterprise document intelligence
MetaRAG for Slack        →  answer questions from channel history
```

### 🧠 Continuous Learning (v2.5)
A full ML training pipeline that learns from real usage:

```
Every query + score → training data
Periodic retraining → smarter router
Domain adaptation   → legal, medical, code — tuned per org
Human feedback loop → thumbs up/down improves quality
```

The router goes from hardcoded rules → sklearn classifier → fine-tuned BERT — automatically, as data accumulates.

### ☁️ MetaRAG Cloud (v3.0)
A hosted SaaS layer for organisations that do not want to manage infrastructure:

```
Upload documents via web UI
Ask questions via chat interface
See pipeline leaderboard and scores
Team collaboration — shared knowledge base
Usage analytics — what your team asks most
```

No terminal. No Python. No setup. Just intelligence over your documents.

---

## 📁 Project Structure

```
MetaRAG/
├── core/
│   ├── loader.py
│   ├── retriever.py
│   └── vector_db.py
|   └── chunking.py
|   └── embeddings.py
│
├── Evaluator/
│   ├── evaluator.py
│   └── ragas_eval.py
│
├── pipelines/
│   ├── generator.py
│   └── pipeline.py
│
├── data/docs/
│
├── metarag_db/
│
├── selector.py
├── main.py
├── .env
├── .gitignore
├── README.md
└── Requirement.txt
```

---

## 🤝 Contributing

MetaRAG is in active development. Contributions welcome in any of these areas:

- New retrieval strategies
- New chunking strategies
- New evaluation metrics
- Integration connectors (Notion, Confluence, Slack)
- ML router training pipeline
- Documentation and examples

```bash
git clone https://github.com/yourusername/metarag
cd metarag
pip install -r requirements.txt
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with the belief that RAG quality should be automatic, measurable, and continuously improving.**

*⭐ Star this repo if MetaRAG saves you time*

</div>
