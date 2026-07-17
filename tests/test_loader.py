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

# ─────────────────────────────────────────────────────────
# names() / load_pdfs() / load_texts() / load_jsons() / load_format()
# ─────────────────────────────────────────────────────────

def test_names_returns_loaded_and_skipped_dict():

    loader = DocumentLoader(DATA_DIR)
    loader.load(verbose=False)

    result = loader.names()

    assert set(result.keys()) == {"loaded", "skipped"}
    assert isinstance(result["loaded"], dict)


def test_names_filters_by_which():

    loader = DocumentLoader(DATA_DIR)
    loader.load(verbose=False)

    result = loader.names(which="loaded")

    assert set(result.keys()) == {"loaded"}


def test_names_filters_by_extension():

    loader = DocumentLoader(DATA_DIR)
    loader.load(verbose=False)

    if "txt" in loader.loaded:
        result = loader.names(which="loaded", ext="txt")
        assert list(result["loaded"].keys()) == ["txt"]


def test_load_texts_returns_only_txt_docs():

    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_texts()

    for doc in docs:
        assert doc.metadata.get("type") == "txt"


def test_load_jsons_returns_only_json_docs():

    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_jsons()

    for doc in docs:
        assert doc.metadata.get("type") == "json"


def test_load_pdfs_returns_only_pdf_docs():

    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_pdfs()

    for doc in docs:
        assert doc.metadata.get("type") == "pdf"


def test_load_format_filters_by_arbitrary_type():

    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_format("markdown")

    for doc in docs:
        assert doc.metadata.get("type") == "markdown"


def test_loader_missing_path_raises():

    with pytest.raises(FileNotFoundError):
        DocumentLoader("/definitely/does/not/exist")