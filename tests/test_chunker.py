from pathlib import Path
import pytest

from metarag import DocumentLoader, Chunker

DATA_DIR = Path(__file__).resolve().parent / "data"

loader = DocumentLoader(DATA_DIR)
documents = loader.load(verbose=False)


def test_chunker_creation():

    chunker = Chunker()

    assert chunker is not None


def test_invalid_strategy():

    try:
        Chunker(strategy="abc")
        assert False
    except ValueError:
        assert True


def test_fixed_strategy():

    chunker = Chunker(strategy="fixed")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_recursive_strategy():

    chunker = Chunker(strategy="recursive")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_semantic_strategy():

    chunker = Chunker(strategy="semantic")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_sentence_strategy():

    chunker = Chunker(strategy="sentence")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_sliding_window_strategy():

    chunker = Chunker(strategy="sliding_window")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_markdown_strategy():

    chunker = Chunker(strategy="markdown")

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0


def test_chunk_attributes():

    chunker = Chunker()

    chunk = chunker.chunk_documents(documents)[0]

    assert hasattr(chunk, "text")
    assert hasattr(chunk, "source")
    assert hasattr(chunk, "start_idx")
    assert hasattr(chunk, "end_idx")
    assert hasattr(chunk, "metadata")


def test_chunk_text():

    chunker = Chunker()

    chunk = chunker.chunk_documents(documents)[0]

    assert isinstance(chunk.text, str)
    assert len(chunk.text) > 0


def test_metadata_preserved():

    chunker = Chunker()

    chunk = chunker.chunk_documents(documents)[0]

    assert isinstance(chunk.metadata, dict)

    assert "source" in chunk.metadata


def test_source_matches_metadata():

    chunker = Chunker()

    chunk = chunker.chunk_documents(documents)[0]

    assert chunk.source == chunk.metadata["source"]


def test_start_end_indices():

    chunker = Chunker()

    chunk = chunker.chunk_documents(documents)[0]

    assert chunk.start_idx >= 0
    assert chunk.end_idx >= chunk.start_idx


def test_chunk_size_limit():

    chunk_size = 400

    chunker = Chunker(
        strategy="fixed",
        chunk_size=chunk_size,
        overlap=0,
    )

    chunks = chunker.chunk_documents(documents)

    for chunk in chunks[:-1]:
        assert len(chunk.text) <= chunk_size


def test_chunk_documents_returns_list():

    chunker = Chunker()

    chunks = chunker.chunk_documents(documents)

    assert isinstance(chunks, list)


def test_cache():

    cache_dir = Path(__file__).parent / ".cache"

    chunker = Chunker()

    chunks1 = chunker.chunk_documents(
        documents,
        cache_dir=cache_dir,
        force=True,
    )

    chunks2 = chunker.chunk_documents(
        documents,
        cache_dir=cache_dir,
    )

    assert len(chunks1) == len(chunks2)


def test_all_chunks_have_text():

    chunker = Chunker()

    chunks = chunker.chunk_documents(documents)

    for chunk in chunks:

        assert len(chunk.text.strip()) > 0


def test_all_chunks_have_metadata():

    chunker = Chunker()

    chunks = chunker.chunk_documents(documents)

    for chunk in chunks:

        assert isinstance(chunk.metadata, dict)


def test_all_chunks_have_source():

    chunker = Chunker()

    chunks = chunker.chunk_documents(documents)

    for chunk in chunks:

        assert chunk.source is not None

from metarag.core.chunking import Chunker


def test_markdown_heading_metadata():

    text = """
# Introduction

This is the introduction.

## Installation

Install MetaRAG using pip.

### Usage

Load your documents and start querying.
"""

    chunker = Chunker(
        strategy="markdown",
        chunk_size=1000,
    )

    chunks = chunker.chunk(text, source="README.md")

    assert len(chunks) == 3

    assert chunks[0].metadata["heading"] == "Introduction"
    assert chunks[0].metadata["heading_level"] == 1

    assert chunks[1].metadata["heading"] == "Installation"
    assert chunks[1].metadata["heading_level"] == 2

    assert chunks[2].metadata["heading"] == "Usage"
    assert chunks[2].metadata["heading_level"] == 3


def test_markdown_source_preserved():

    text = """
# Title

Some content.
"""

    chunker = Chunker(strategy="markdown")

    chunk = chunker.chunk(text, source="notes.md")[0]

    assert chunk.source == "notes.md"
    assert chunk.metadata["source"] == "notes.md"


def test_fixed_overlap():

    text = "A" * 1000

    chunker = Chunker(
        strategy="fixed",
        chunk_size=200,
        overlap=50,
    )

    chunks = chunker.chunk(text)

    assert len(chunks) > 1

    for chunk in chunks:

        assert len(chunk.text) <= 200


def test_sliding_window_overlap():

    text = "A" * 1000

    chunker = Chunker(
        strategy="sliding_window",
        chunk_size=200,
        overlap=50,
    )

    chunks = chunker.chunk(text)

    assert len(chunks) > 1

    for chunk in chunks:

        assert len(chunk.text) <= 200


def test_chunk_string_representation():

    text = "Hello MetaRAG"

    chunk = Chunker().chunk(text)[0]

    assert str(chunk) == text


def test_chunk_repr():

    chunk = Chunker().chunk("Hello World")[0]

    assert "Chunk(" in repr(chunk)