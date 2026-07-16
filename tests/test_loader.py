from pathlib import Path
from metarag import DocumentLoader
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"

loader = DocumentLoader(DATA_DIR)


def test_loader_returns_document_list():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    assert report is not None
    assert len(report) > 0


def test_loaded_and_skipped_exist():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    assert hasattr(report, "loaded")
    assert hasattr(report, "skipped")


def test_loaded_count_matches_documents():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    assert report.loaded.count == len(report.loaded.files)


def test_loaded_files_is_list():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    assert isinstance(report.loaded.files, list)


def test_skipped_files_is_list():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    assert isinstance(report.skipped.files, list)


def test_loaded_by_extension():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    for ext, stats in report.loaded.by_extension.items():

        assert isinstance(ext, str)

        assert stats.count >= 0

        assert isinstance(stats.files, list)


def test_skipped_by_extension():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    for ext, stats in report.skipped.by_extension.items():

        assert isinstance(ext, str)

        assert stats.count >= 0

        assert isinstance(stats.files, list)


def test_extension_lookup():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    for ext in report.loaded.by_extension:

        stats = report.loaded[ext]

        assert stats.count == len(stats.files)


def test_document_metadata():

    loader = DocumentLoader(DATA_DIR)

    report = loader.load(verbose=False)

    for doc in report:

        assert doc.text is not None

        assert isinstance(doc.metadata, dict)

        assert "source" in doc.metadata