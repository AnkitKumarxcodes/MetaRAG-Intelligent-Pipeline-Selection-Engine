# examples/vector_db_demo.py
"""
VectorDB API demo — builds a real index from your test PDFs using Ollama
embeddings, then exercises InMemoryVectorDB's build/search/add.
"""

from pathlib import Path

from metarag import DocumentLoader, Chunker, CachedEmbeddings, InMemoryVectorDB

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


from metarag.utils import FakeEmbeddings


print("=" * 60)
print("VectorDB API Demo")
print("=" * 60)

# ------------------------------------------------------------------
# Build real chunks + embeddings from the test corpus
# ------------------------------------------------------------------

loader = DocumentLoader(DATA_DIR)
docs = loader.load(verbose=False)
print(f"\nDocuments loaded : {len(docs)}")

chunker = Chunker(strategy="recursive", chunk_size=500, overlap=50)
chunks = chunker.chunk_documents(docs)
print(f"Chunks produced  : {len(chunks)}")

embeddings = CachedEmbeddings(FakeEmbeddings())
chunk_texts = [c.text for c in chunks]
chunk_embeddings = embeddings.embed_documents(chunk_texts)
print(f"Embeddings ready : {len(chunk_embeddings)} vectors, dim={len(chunk_embeddings[0])}")

# ------------------------------------------------------------------
# build()
# ------------------------------------------------------------------

print("\n=== build() ===")

vdb = InMemoryVectorDB()
vdb.build(chunks, chunk_embeddings)
print(f"Chunks stored : {len(vdb.chunks)}")

# ------------------------------------------------------------------
# search()
# ------------------------------------------------------------------

print("\n=== search() ===")

query = "What is the main topic of this document?"
query_embedding = embeddings.embed_query(query)
results = vdb.search(query_embedding, k=3)

print(f"Query : '{query}'")
for i, (chunk, score) in enumerate(results, start=1):
    text = chunk.text if hasattr(chunk, "text") else str(chunk)
    print(f"\n  #{i}  score={score:.4f}")
    print(f"      {text[:100]}...")

# ------------------------------------------------------------------
# add() — append a synthetic chunk, confirm it's searchable
# ------------------------------------------------------------------

print("\n=== add() ===")

from metarag import Chunk
new_chunk = Chunk(text="MetaRAG unit test marker chunk.", metadata={"source": "synthetic"})
new_embedding = embeddings.embed_query(new_chunk.text)

before = len(vdb.chunks)
vdb.add([new_chunk], [new_embedding])
after = len(vdb.chunks)
print(f"Chunks before add : {before}")
print(f"Chunks after add  : {after}")

marker_results = vdb.search(embeddings.embed_query("unit test marker chunk"), k=1)
top_text = marker_results[0][0].text if hasattr(marker_results[0][0], "text") else str(marker_results[0][0])
print(f"Top result after add : {top_text[:60]}")

# ------------------------------------------------------------------
# save() — InMemoryVectorDB doesn't persist, just confirms the warning
# ------------------------------------------------------------------

print("\n=== save() ===")
vdb.save()