# tests/test_vector_db.py
"""
pytest suite for InMemoryVectorDB — build/search/add, using real chunks
and real Ollama embeddings from the test corpus (same setup as vector_db_demo.py).
"""

from pathlib import Path
import pytest

from metarag import DocumentLoader, Chunker, CachedEmbeddings, InMemoryVectorDB, Chunk

DATA_DIR = Path(__file__).resolve().parent / "data"


from metarag.utils import FakeEmbeddings
    
@pytest.fixture(scope="module")
def embeddings():
    return CachedEmbeddings(FakeEmbeddings())


@pytest.fixture(scope="module")
def chunks():
    docs = DocumentLoader(DATA_DIR).load(verbose=False)
    return Chunker(strategy="recursive", chunk_size=500, overlap=50).chunk_documents(docs)


@pytest.fixture(scope="module")
def built_vdb(chunks, embeddings):
    vdb = InMemoryVectorDB()
    chunk_embeddings = embeddings.embed_documents([c.text for c in chunks])
    vdb.build(chunks, chunk_embeddings)
    return vdb


def test_build_stores_all_chunks(chunks, built_vdb):
    assert len(built_vdb.chunks) == len(chunks)


def test_build_embeddings_shape(built_vdb):
    assert built_vdb.embeddings.shape[0] == len(built_vdb.chunks)
    assert built_vdb.embeddings.shape[1] > 0


def test_search_returns_k_results(built_vdb, embeddings):
    query_embedding = embeddings.embed_query("What is the main topic?")
    results = built_vdb.search(query_embedding, k=3)
    assert len(results) == 3


def test_search_results_are_tuples(built_vdb, embeddings):
    query_embedding = embeddings.embed_query("What is the main topic?")
    results = built_vdb.search(query_embedding, k=2)
    for chunk, score in results:
        assert hasattr(chunk, "text") or isinstance(chunk, str)
        assert isinstance(score, float)


def test_search_scores_are_sorted_descending(built_vdb, embeddings):
    query_embedding = embeddings.embed_query("What is the main topic?")
    results = built_vdb.search(query_embedding, k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_add_increases_chunk_count(built_vdb, embeddings):
    before = len(built_vdb.chunks)
    new_chunk = Chunk(text="MetaRAG unit test marker chunk.", metadata={"source": "synthetic"})
    built_vdb.add([new_chunk], [embeddings.embed_query(new_chunk.text)])
    assert len(built_vdb.chunks) == before + 1


def test_added_chunk_is_searchable(built_vdb, embeddings):
    results = built_vdb.search(embeddings.embed_query("unit test marker chunk"), k=1)
    top_text = results[0][0].text if hasattr(results[0][0], "text") else str(results[0][0])
    assert "marker" in top_text.lower()


def test_search_raises_before_build():
    empty_vdb = InMemoryVectorDB()
    with pytest.raises(RuntimeError):
        empty_vdb.search([0.1, 0.2, 0.3], k=3)


def test_zero_vector_query_does_not_crash(built_vdb):
    zero_query = [0.0] * built_vdb.embeddings.shape[1]
    results = built_vdb.search(zero_query, k=3)
    assert len(results) == 3

# ─────────────────────────────────────────────────────────
# Shape-mismatch guard
# ─────────────────────────────────────────────────────────

def test_search_shape_mismatch_raises(built_vdb):
    with pytest.raises(ValueError):
        built_vdb.search([0.1, 0.2], k=3)   # wrong dimensionality


# ─────────────────────────────────────────────────────────
# ChromaVectorDB (optional dep — skip if chromadb not installed)
# ─────────────────────────────────────────────────────────

def test_chroma_build_search_add(tmp_path, chunks, embeddings):
    pytest.importorskip("chromadb")
    from metarag import ChromaVectorDB

    db = ChromaVectorDB(persist_directory=str(tmp_path / "chroma_index"))
    chunk_embeddings = embeddings.embed_documents([c.text for c in chunks])
    db.build(chunks, chunk_embeddings)

    results = db.search(embeddings.embed_query("What is the main topic?"), k=3)
    assert len(results) == 3
    for chunk, score in results:
        assert isinstance(score, float)

    new_chunk = Chunk(text="Chroma marker chunk.", metadata={"source": "synthetic"})
    db.add([new_chunk], [embeddings.embed_query(new_chunk.text)])
    hits = db.search(embeddings.embed_query("marker chunk"), k=1)
    assert "marker" in hits[0][0].text.lower()


# ─────────────────────────────────────────────────────────
# FAISSVectorDB (optional dep — skip if faiss not installed)
# ─────────────────────────────────────────────────────────

def test_faiss_build_and_search(tmp_path, chunks, embeddings):
    pytest.importorskip("faiss")
    from metarag import FAISSVectorDB

    db = FAISSVectorDB()
    chunk_embeddings = embeddings.embed_documents([c.text for c in chunks])
    db.build(chunks, chunk_embeddings)

    results = db.search(embeddings.embed_query("What is the main topic?"), k=3)
    assert len(results) == 3

    save_path = str(tmp_path / "faiss_index.bin")
    db.save(save_path)
    reloaded = FAISSVectorDB()
    reloaded.chunks = chunks   # index file only stores vectors, not chunk objects
    reloaded.load(save_path)
    assert len(reloaded.search(embeddings.embed_query("main topic"), k=2)) == 2