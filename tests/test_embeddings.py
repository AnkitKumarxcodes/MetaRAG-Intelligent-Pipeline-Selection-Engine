from pathlib import Path
import pytest
from metarag.core.embeddings import CachedEmbeddings


# ----------------------------------------------------
# Fake Embedding Model
# ----------------------------------------------------

from metarag.utils import FakeEmbeddings


# ----------------------------------------------------
# Tests
# ----------------------------------------------------

def test_cached_embeddings_creation(tmp_path):

    model = FakeEmbeddings()

    embedder = CachedEmbeddings(
        model,
        cache_dir=tmp_path,
    )

    assert embedder is not None


def test_embed_query_returns_vector(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    vector = embedder.embed_query("MetaRAG")

    assert isinstance(vector, list)
    assert len(vector) == 3


def test_embed_alias(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    assert embedder.embed("hello") == embedder.embed_query("hello")


def test_embed_documents_returns_vectors(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    vectors = embedder.embed_documents(
        [
            "one",
            "two",
            "three",
        ]
    )

    assert len(vectors) == 3

    for vec in vectors:
        assert isinstance(vec, list)


def test_same_query_same_embedding(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    v1 = embedder.embed_query("hello")

    v2 = embedder.embed_query("hello")

    assert v1 == v2


def test_cache_file_created(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    embedder.embed_query("cached")

    files = list(Path(tmp_path).glob("*.npy"))

    assert len(files) == 1


def test_batch_cache(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    texts = [
        "A",
        "B",
        "C",
    ]

    embedder.embed_documents(texts)

    files = list(Path(tmp_path).glob("*.npy"))

    assert len(files) == 3


def test_empty_batch(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    vectors = embedder.embed_documents([])

    assert vectors == []


def test_invalid_model():

    class BadModel:
        pass

    try:

        CachedEmbeddings(BadModel())

        assert False

    except TypeError:

        assert True


def test_model_name_detection(tmp_path):

    model = FakeEmbeddings()

    embedder = CachedEmbeddings(
        model,
        cache_dir=tmp_path,
    )

    assert embedder.model_name == "fake-model"


def test_cache_persistence(tmp_path):

    model = FakeEmbeddings()

    embedder1 = CachedEmbeddings(
        model,
        cache_dir=tmp_path,
    )

    vector1 = embedder1.embed_query("persist")

    embedder2 = CachedEmbeddings(
        model,
        cache_dir=tmp_path,
    )

    vector2 = embedder2.embed_query("persist")

    assert vector1 == vector2


def test_document_order_preserved(tmp_path):

    embedder = CachedEmbeddings(
        FakeEmbeddings(),
        cache_dir=tmp_path,
    )

    texts = [
        "abc",
        "abcdefgh",
        "a",
    ]

    vectors = embedder.embed_documents(texts)

    assert vectors[0][0] == 3.0
    assert vectors[1][0] == 8.0
    assert vectors[2][0] == 1.0