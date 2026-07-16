# examples/retriever_demo.py
"""
Retriever API demo — builds all 4 retrievers (BM25, Dense, Hybrid, MMR)
on the same chunk set + vector index, runs one query through each, and
prints their top results side by side so you can directly compare them.
"""

from pathlib import Path
import requests

from metarag import (
    DocumentLoader, Chunker, CachedEmbeddings, InMemoryVectorDB,
    BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

from metarag.utils import FakeEmbeddings

print("=" * 60)
print("Retriever API Demo")
print("=" * 60)

# ------------------------------------------------------------------
# Shared setup — one chunk set, one built vector_db, reused by all 4
# ------------------------------------------------------------------

docs = DocumentLoader(DATA_DIR).load(verbose=False)
chunks = Chunker(strategy="recursive", chunk_size=500, overlap=50).chunk_documents(docs)
embeddings = CachedEmbeddings(FakeEmbeddings())

print(f"\nDocuments : {len(docs)}")
print(f"Chunks    : {len(chunks)}")

vdb = InMemoryVectorDB()
chunk_embeddings = embeddings.embed_documents([c.text for c in chunks])
vdb.build(chunks, chunk_embeddings)
print("Vector index built.")

# ------------------------------------------------------------------
# Build all 4 retrievers — sharing chunks/embeddings/vdb, never
# re-building the index themselves (fix #2 behaviour)
# ------------------------------------------------------------------

retrievers = {
    "bm25":   BM25Retriever(chunks),
    "dense":  DenseRetriever(chunks, embeddings, vdb),
    "hybrid": HybridRetriever(chunks, embeddings, vdb),
    "mmr":    MMRRetriever(chunks, embeddings, vdb),
}

QUERY = "What is the main topic of this document?"
K = 3

print(f"\nQuery : '{QUERY}'   (k={K})")

for name, retriever in retrievers.items():
    print(f"\n{'-' * 60}")
    print(f"{name.upper()}")
    print(f"{'-' * 60}")

    results = retriever.retrieve(QUERY, k=K)
    for i, (chunk, score) in enumerate(results, start=1):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        source = chunk.metadata.get("source", "?") if hasattr(chunk, "metadata") else "?"
        print(f"  #{i}  score={score:.4f}  source={source}")
        print(f"      {text[:90]}...")

# ------------------------------------------------------------------
# HybridRetriever alpha — quick before/after comparison
# ------------------------------------------------------------------

print(f"\n{'=' * 60}")
print("Hybrid alpha comparison (0.2 = BM25-leaning, 0.8 = dense-leaning)")
print(f"{'=' * 60}")

for alpha in [0.2, 0.5, 0.8]:
    hybrid = HybridRetriever(chunks, embeddings, vdb, alpha=alpha)
    top = hybrid.retrieve(QUERY, k=1)[0]
    text = top[0].text if hasattr(top[0], "text") else str(top[0])
    print(f"\nalpha={alpha}  score={top[1]:.4f}")
    print(f"  {text[:90]}...")