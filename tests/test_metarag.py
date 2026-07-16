# tests/test_metarag.py
"""
Integration test suite for MetaRAG — all 23 public methods, exercised
against a real (small) corpus with deterministic, offline fake
embeddings/generator so the suite runs fast and without network calls.
"""

from pathlib import Path
import pandas as pd
import pytest

from metarag import MetaRAG
from metarag.core.embeddings import EmbeddingInterface

DATA_DIR = Path(__file__).resolve().parent / "data"


# ─────────────────────────────────────────────────────────
# Fakes — deterministic, offline, no network
# ─────────────────────────────────────────────────────────

from metarag.utils import FakeEmbeddings , FakeGenerator , FakeSklearnModel


QUERIES = [
    "What is the main topic of this document?",
    "Summarize the key points.",
    "What numbers are mentioned?",
]


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    project_dir = tmp_path_factory.mktemp("metarag_project")
    instance = MetaRAG(
        docs=str(DATA_DIR),
        embeddings=FakeEmbeddings(),
        generator=FakeGenerator(),
        project=f"test_{project_dir.name}",
        k=3,
        verbose=False,
    )
    instance.fit()
    return instance


@pytest.fixture(scope="module")
def benchmarked_rag(rag):
    rag.benchmark(QUERIES, retrieval_only=True, train_router=True, save_csv=True)
    return rag


# ─────────────────────────────────────────────────────────
# fit()
# ─────────────────────────────────────────────────────────

def test_fit_sets_fitted_flag(rag):
    assert rag._fitted is True


def test_fit_builds_chunks(rag):
    assert rag._chunks is not None
    assert len(rag._chunks) > 0


def test_fit_builds_retrievers(rag):
    assert set(rag._retrievers.keys()) == {"bm25", "dense", "hybrid", "mmr"}


def test_fit_builds_pipelines(rag):
    expected = {"bm25", "dense", "hybrid", "mmr", "reranked", "full", "multiquery"}
    assert expected.issubset(set(rag._pipelines.keys()))


def test_fit_builds_evaluator(rag):
    assert rag._evaluator is not None


def test_fit_builds_corpus_profile(rag):
    assert rag._corpus_profile is not None
    assert "num_docs" in rag._corpus_profile


def test_ask_before_fit_raises():
    fresh = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(), verbose=False)
    with pytest.raises(RuntimeError):
        fresh.ask("test query")


# ─────────────────────────────────────────────────────────
# ask()
# ─────────────────────────────────────────────────────────

def test_ask_returns_answer(rag):
    answer = rag.ask("What is this document about?")
    assert hasattr(answer, "text")
    assert hasattr(answer, "pipeline")
    assert hasattr(answer, "score")
    assert hasattr(answer, "sources")


def test_ask_pipeline_is_valid(rag):
    answer = rag.ask("What is this document about?")
    assert answer.pipeline in rag._pipelines


def test_ask_score_is_float(rag):
    answer = rag.ask("What is this document about?")
    assert isinstance(answer.score, float)


def test_ask_writes_log(rag):
    before = len(rag._read_logs())
    rag.ask("Another test query")
    after = len(rag._read_logs())
    assert after == before + 1


# ─────────────────────────────────────────────────────────
# benchmark()
# ─────────────────────────────────────────────────────────

def test_benchmark_returns_dataframe(benchmarked_rag):
    df = benchmarked_rag.benchmark(QUERIES[:1], retrieval_only=True, train_router=False, save_csv=False)
    assert isinstance(df, pd.DataFrame)


def test_benchmark_covers_all_pipelines(benchmarked_rag):
    df = benchmarked_rag.get_benchmark_data()
    assert set(df["pipeline"].unique()) == set(benchmarked_rag._pipelines.keys())


def test_benchmark_has_winning_pipeline_column(benchmarked_rag):
    df = benchmarked_rag.get_benchmark_data()
    assert "winning_pipeline" in df.columns


def test_benchmark_trains_router(benchmarked_rag):
    assert benchmarked_rag._router is not None
    assert benchmarked_rag._router.is_trained


# ─────────────────────────────────────────────────────────
# status() / leaderboard()
# ─────────────────────────────────────────────────────────

def test_status_returns_dict(rag):
    info = rag.status()
    assert isinstance(info, dict)
    assert info["fitted"] is True


def test_leaderboard_from_benchmark(benchmarked_rag):
    summary = benchmarked_rag.leaderboard(source="benchmark")
    assert summary is not None
    assert "composite" in summary.columns


def test_leaderboard_from_logs(rag):
    rag.ask("Log entry query")
    ranked = rag.leaderboard(source="logs")
    assert ranked is not None


def test_leaderboard_invalid_source(rag):
    with pytest.raises(ValueError):
        rag.leaderboard(source="bogus")


# ─────────────────────────────────────────────────────────
# analyze_query() / analyze_corpus()
# ─────────────────────────────────────────────────────────

def test_analyze_query_structure(rag):
    result = rag.analyze_query("What is machine learning?")
    assert "complexity" in result
    assert "keywords" in result


def test_analyze_corpus_structure(rag):
    result = rag.analyze_corpus()
    assert "num_chunks" in result
    assert result["num_chunks"] > 0


# ─────────────────────────────────────────────────────────
# explain()
# ─────────────────────────────────────────────────────────

def test_explain_structure(benchmarked_rag):
    result = benchmarked_rag.explain("What is this about?")
    assert "selected_pipeline" in result
    assert result["selected_pipeline"] in benchmarked_rag._pipelines


def test_explain_no_router_fallback():
    fresh = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(), verbose=False)
    fresh.fit()
    result = fresh.explain("test")
    assert result["confidence"] == "none"


# ─────────────────────────────────────────────────────────
# Observability: pipeline_graph / dashboard / report / inspect / trace
# ─────────────────────────────────────────────────────────

def test_pipeline_graph_single(rag):
    output = rag.pipeline_graph("hybrid")
    assert "hybrid" in output.lower() or "Retriever" in output


def test_pipeline_graph_all(rag):
    output = rag.pipeline_graph()
    for name in rag._pipelines:
        assert f"[{name}]" in output


def test_pipeline_graph_unknown():
    fresh_output_check = "[pipeline_graph] Unknown pipeline"


def test_dashboard_returns_summary(benchmarked_rag):
    summary = benchmarked_rag.dashboard()
    assert summary is not None
    assert len(summary) == len(benchmarked_rag._pipelines)


def test_report_returns_corpus_profile(rag):
    profile = rag.report()
    assert isinstance(profile, dict)
    assert "num_docs" in profile


def test_inspect_returns_per_retriever_results(rag):
    results = rag.inspect("What is this document about?", k=2)
    assert set(results.keys()) == set(rag._retrievers.keys())
    for texts in results.values():
        assert len(texts) <= 2


def test_trace_returns_steps(rag):
    steps = rag.trace("What is this document about?", pipeline_name="hybrid")
    stage_names = [s["stage"] for s in steps]
    assert "Retrieve" in stage_names
    assert "Deduplicate" in stage_names


def test_trace_full_pipeline_has_all_stages(rag):
    steps = rag.trace("What is this document about?", pipeline_name="full")
    stage_names = [s["stage"] for s in steps]
    assert "MultiQuery" in stage_names
    assert "Rerank" in stage_names


# ─────────────────────────────────────────────────────────
# save() / load()
# ─────────────────────────────────────────────────────────

def test_save_writes_config(rag):
    rag.save()
    import os
    assert os.path.exists(rag._config_path)


def test_load_restores_config(rag):
    rag.save()
    loaded = MetaRAG.load(rag.project, embeddings=FakeEmbeddings(), generator=FakeGenerator())
    assert loaded.chunk_size == rag.chunk_size
    assert loaded.k == rag.k


def test_load_missing_project_raises():
    with pytest.raises(FileNotFoundError):
        MetaRAG.load("definitely_does_not_exist_project", embeddings=FakeEmbeddings(), generator=FakeGenerator())


# ─────────────────────────────────────────────────────────
# get_benchmark_data() / get_router_thresholds()
# ─────────────────────────────────────────────────────────

def test_get_benchmark_data_returns_df(benchmarked_rag):
    df = benchmarked_rag.get_benchmark_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_benchmark_data_missing_raises():
    fresh = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(),
                     project="never_benchmarked_project", verbose=False)
    with pytest.raises(FileNotFoundError):
        fresh.get_benchmark_data()


def test_get_router_thresholds(benchmarked_rag):
    stats = benchmarked_rag.get_router_thresholds()
    assert stats.get("status") == "trained"


def test_get_router_thresholds_none():
    fresh = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(), verbose=False)
    assert fresh.get_router_thresholds() == {}


# ─────────────────────────────────────────────────────────
# set_llm() / set_embeddings() / set_router() / set_router_from_model()
# ─────────────────────────────────────────────────────────

def test_set_llm_updates_generator(rag):
    new_gen = FakeGenerator()
    rag.set_llm(new_gen)
    assert rag.generator is new_gen


def test_set_llm_rejects_invalid():
    class Bad: pass
    rag_instance = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(), verbose=False)
    with pytest.raises(TypeError):
        rag_instance.set_llm(Bad())


def test_set_embeddings_updates(rag):
    new_emb = FakeEmbeddings(dim=8)
    rag.set_embeddings(new_emb)
    assert rag.embeddings is new_emb
    rag.set_embeddings(FakeEmbeddings())  # restore for subsequent tests


def test_set_router_accepts_valid(rag):
    class DummyRouter:
        def route(self, features): return "hybrid"
    rag.set_router(DummyRouter())
    assert rag._router.route({}) == "hybrid"


def test_set_router_rejects_invalid(rag):
    with pytest.raises(TypeError):
        rag.set_router(object())


def test_set_router_from_model(rag):
    rag.set_router_from_model(FakeSklearnModel(), feature_cols=["max_similarity"])
    picked = rag._router.route({"max_similarity": 0.5})
    assert picked == "straight"


# ─────────────────────────────────────────────────────────
# update_router_thresholds() / rebuild()
# ─────────────────────────────────────────────────────────

def test_update_router_thresholds(benchmarked_rag):
    benchmarked_rag.update_router_thresholds()
    assert benchmarked_rag._router.is_trained


def test_update_router_thresholds_missing_raises():
    fresh = MetaRAG(docs=str(DATA_DIR), embeddings=FakeEmbeddings(), generator=FakeGenerator(),
                     project="no_thresholds_project", verbose=False)
    with pytest.raises(FileNotFoundError):
        fresh.update_router_thresholds()


def test_rebuild_refits(rag):
    chunk_count_before = len(rag._chunks)
    rag.rebuild()
    assert rag._fitted is True
    assert len(rag._chunks) == chunk_count_before


# ─────────────────────────────────────────────────────────
# __repr__
# ─────────────────────────────────────────────────────────

def test_repr_does_not_crash(rag):
    text = repr(rag)
    assert "MetaRAG" in text
