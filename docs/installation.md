# Installation

MetaRAG requires **Python 3.10+** and can be installed with only the components you need.

---

## Install

```bash
pip install metarag-sdk
```

The base installation includes the core framework:

- Document Loader
- Chunker
- Embedding Interface
- Vector Database Interface
- Retrievers
- Pipelines
- MetaRAG Framework

---

## Optional Components

Some features require additional packages.

| Feature | Install |
|---------|---------|
| PDF Support | `pip install metarag-sdk[pdf]` |
| ChromaDB | `pip install metarag-sdk[chroma]` |
| FAISS | `pip install metarag-sdk[faiss]` |
| NLTK Sentence Chunking | `pip install metarag-sdk[nltk]` |
| Cross-Encoder Reranker | `pip install metarag-sdk[rerank]` |
| Ollama Generator | `pip install metarag-sdk[ollama]` |

Or install everything:

```bash
pip install metarag-sdk[all]
```

---

## Verify Installation

```bash
python -c "import metarag; print(metarag.__version__)"
```

Example

```text
0.3.8
```

---

## Development Installation

Clone the repository.

```bash
git clone https://github.com/AnkitKumarxcodes/MetaRAG---Intelligent-Pipeline-Selection-Engine.git

cd MetaRAG
```

Create a virtual environment.

**Windows**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install in editable mode.

```bash
pip install -e .
```

Development dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Run Examples

MetaRAG includes small demos for each major component.

```text
examples/
├── loader_demo.py
├── chunker_demo.py
├── embeddings_demo.py
├── vector_db_demo.py
├── retriever_demo.py
├── pipeline_demo.py
└── metarag_demo.py
```

Example:

```bash
python examples/loader_demo.py
```

Example output

```text
DocumentLoader Report
------------------------------

Files Loaded : 8
Files Skipped: 0

Documents Extracted : 101
```

---

## Run Tests

Run the complete test suite.

```bash
pytest
```

Or test individual modules.

```bash
pytest tests/test_loader.py

pytest tests/test_chunker.py

pytest tests/test_retriever.py
```

---

## Common Issues

### PDF files are skipped

Install PDF support.

```bash
pip install metarag-sdk[pdf]
```

---

### Ollama connection failed

Start the Ollama server.

```bash
ollama serve
```

---

### Editable install fails

Upgrade packaging tools.

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Next

Continue to **Quick Start** to build your first RAG pipeline using MetaRAG.