# tests/test_router.py
"""
Tests for router.py (Router — cold-start rule-based routing, win-rate-driven
learned routing, persistence, introspection) and router_interface.py
(RouterInterface contract).

No fakes from metarag.utils are needed here — Router operates purely on
plain dicts (features) and pandas DataFrames (benchmark data), with no
embeddings/generator/retriever dependency, so there's nothing to fake.
"""

import json
import pytest
import pandas as pd

from metarag.router.router import Router
from metarag.router.router_interface import RouterInterface
from metarag.defaults import DEFAULTS, MetaRAGDefaults


@pytest.fixture(autouse=True)
def restore_defaults():
    """Every test gets a clean DEFAULTS state — mutations in one test
    must not leak into the next (same pattern as test_defaults.py)."""
    original = MetaRAGDefaults()
    yield
    for field_name in original.__dataclass_fields__:
        setattr(DEFAULTS, field_name, getattr(original, field_name))


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def make_benchmark_df(rows):
    return pd.DataFrame(rows)


def three_pipeline_benchmark():
    """
    2 queries x 3 pipelines ("hybrid", "mmr", "reranked").
    hybrid wins both queries -> win_rate 1.0, avg_composite 0.9
    mmr / reranked win nothing -> win_rate 0.0, avg_composite 0.5
    """
    rows = []
    for q in ["query one", "query two"]:
        for pipeline in ["hybrid", "mmr", "reranked"]:
            rows.append({
                "query": q,
                "pipeline": pipeline,
                "winning_pipeline": "hybrid",
                "composite": 0.9 if pipeline == "hybrid" else 0.5,
                "max_similarity": 0.7,
                "avg_similarity": 0.6,
                "redundancy": 0.2,
                "query_length": 3,
                "num_docs": 100,
            })
    return make_benchmark_df(rows)


def trained_router_with_evidence(override_pipeline: str, win_rate: float):
    """
    Build a router trained on 'hybrid' (real winner) + one other pipeline
    (override_pipeline), then manually force override_pipeline's win_rate
    so evidence-threshold behavior can be tested unambiguously.
    """
    rows = []
    for q in ["q1", "q2"]:
        rows.append({
            "query": q, "pipeline": "hybrid", "winning_pipeline": "hybrid",
            "composite": 0.6, "max_similarity": 0.6, "avg_similarity": 0.5,
            "redundancy": 0.2, "query_length": 8, "num_docs": 50,
        })
        rows.append({
            "query": q, "pipeline": override_pipeline, "winning_pipeline": "hybrid",
            "composite": 0.4, "max_similarity": 0.4, "avg_similarity": 0.3,
            "redundancy": 0.7, "query_length": 2, "num_docs": 50,
        })
    router = Router()
    router.train(make_benchmark_df(rows))

    router.thresholds[override_pipeline]["win_rate"] = win_rate
    router.rules = sorted(router.thresholds.items(), key=lambda x: x[1]["win_rate"], reverse=True)
    return router


# ─────────────────────────────────────────────────────────
# Cold-start routing (untrained router)
# ─────────────────────────────────────────────────────────

def test_new_router_starts_untrained():
    router = Router()
    assert router.is_trained is False
    assert router.thresholds == {}
    assert router.rules == []


def test_empty_features_falls_through_to_default():
    router = Router()
    assert router.route({}) == "multiquery"


def test_high_similarity_low_redundancy_picks_reranked():
    router = Router()
    features = {"max_similarity": 0.9, "redundancy": 0.1}
    assert router.route(features) == "reranked"


def test_numeric_heavy_with_number_picks_straight():
    router = Router()
    features = {"numeric_ratio": 0.5, "contains_number": True, "starts_with_wh": False}
    assert router.route(features) == "straight"


def test_numeric_heavy_wh_question_no_number_picks_hybrid():
    router = Router()
    features = {"numeric_ratio": 0.5, "contains_number": False, "starts_with_wh": True}
    assert router.route(features) == "hybrid"


def test_short_doc_heavy_with_number_picks_straight():
    router = Router()
    features = {"short_doc_ratio": 0.8, "contains_number": True, "starts_with_wh": False}
    assert router.route(features) == "straight"


def test_weak_max_similarity_picks_multiquery():
    router = Router()
    features = {"max_similarity": 0.3, "avg_similarity": 0.5}
    assert router.route(features) == "multiquery"


def test_weak_avg_similarity_picks_multiquery():
    router = Router()
    features = {"max_similarity": 0.6, "avg_similarity": 0.2}
    assert router.route(features) == "multiquery"


def test_high_redundancy_picks_mmr():
    router = Router()
    features = {"max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.9}
    assert router.route(features) == "mmr"


def test_ocr_heavy_picks_hybrid():
    router = Router()
    features = {
        "max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2,
        "ocr_ratio": 0.5,
    }
    assert router.route(features) == "hybrid"


def test_long_query_picks_multiquery():
    router = Router()
    features = {
        "max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2,
        "ocr_ratio": 0.0, "is_long": True,
    }
    assert router.route(features) == "multiquery"


def test_operator_query_picks_multiquery():
    router = Router()
    features = {
        "max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2,
        "ocr_ratio": 0.0, "has_operator": True,
    }
    assert router.route(features) == "multiquery"


def test_no_signals_match_falls_back_to_hybrid():
    router = Router()
    features = {
        "max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2,
        "ocr_ratio": 0.0, "is_long": False, "has_operator": False,
    }
    assert router.route(features) == "hybrid"


def test_missing_keys_degrade_gracefully_instead_of_crashing():
    """A router given only a partial feature dict (e.g. probe-only, no
    corpus/query signals) must not crash — every signal falls back via
    .get(key, default)."""
    router = Router()
    result = router.route({"max_similarity": 0.95})
    # redundancy defaults to 0.0 < 0.3, and 0.95 > 0.85 -> deterministically "reranked"
    assert result == "reranked"


# ─────────────────────────────────────────────────────────
# train()
# ─────────────────────────────────────────────────────────

def test_train_on_empty_dataframe_skips_training():
    router = Router()
    router.train(pd.DataFrame())
    assert router.is_trained is False


def test_train_without_winning_pipeline_column_skips_training():
    router = Router()
    df = pd.DataFrame([{"query": "q", "pipeline": "hybrid", "composite": 0.5}])
    router.train(df)
    assert router.is_trained is False


def test_train_sets_is_trained():
    router = Router()
    router.train(three_pipeline_benchmark())
    assert router.is_trained is True


def test_train_computes_win_rate_per_pipeline():
    router = Router()
    router.train(three_pipeline_benchmark())
    assert router.thresholds["hybrid"]["win_rate"] == 1.0
    assert router.thresholds["mmr"]["win_rate"] == 0.0
    assert router.thresholds["reranked"]["win_rate"] == 0.0


def test_train_computes_avg_composite_per_pipeline():
    router = Router()
    router.train(three_pipeline_benchmark())
    assert router.thresholds["hybrid"]["avg_composite"] == pytest.approx(0.9)
    assert router.thresholds["mmr"]["avg_composite"] == pytest.approx(0.5)


def test_train_sorts_rules_by_win_rate_descending():
    router = Router()
    router.train(three_pipeline_benchmark())
    assert router.rules[0][0] == "hybrid"
    assert router.rules[0][1]["win_rate"] == 1.0


def test_feature_medians_computed_from_winning_queries_when_present():
    rows = [
        {"query": q, "pipeline": "hybrid", "winning_pipeline": "hybrid",
         "composite": 0.9, "max_similarity": 0.8, "avg_similarity": 0.7,
         "redundancy": 0.1, "query_length": 5, "num_docs": 50}
        for q in ["q1", "q2"]
    ]
    router = Router()
    router.train(make_benchmark_df(rows))
    assert router.thresholds["hybrid"]["max_similarity"] == pytest.approx(0.8)


def test_feature_medians_fall_back_to_all_rows_when_pipeline_never_wins():
    """`source = wins if len(wins) > 0 else pipeline_rows` — a pipeline
    with zero wins still gets feature medians, computed from ALL its rows."""
    rows = [
        {"query": "q1", "pipeline": "mmr", "winning_pipeline": "hybrid",
         "composite": 0.4, "max_similarity": 0.3, "avg_similarity": 0.2,
         "redundancy": 0.5, "query_length": 4, "num_docs": 20},
        {"query": "q1", "pipeline": "hybrid", "winning_pipeline": "hybrid",
         "composite": 0.9, "max_similarity": 0.8, "avg_similarity": 0.7,
         "redundancy": 0.1, "query_length": 4, "num_docs": 20},
    ]
    router = Router()
    router.train(make_benchmark_df(rows))
    assert router.thresholds["mmr"]["max_similarity"] == pytest.approx(0.3)


def test_train_auto_saves_when_base_path_is_set(tmp_path):
    router = Router(base_path=str(tmp_path))
    router.train(three_pipeline_benchmark())
    assert (tmp_path / "router_thresholds.json").exists()


def test_train_does_not_save_without_base_path(tmp_path):
    router = Router()  # no base_path
    router.train(three_pipeline_benchmark())
    assert not (tmp_path / "router_thresholds.json").exists()


# ─────────────────────────────────────────────────────────
# route() dispatch — cold-start vs learned
# ─────────────────────────────────────────────────────────

def test_untrained_router_dispatches_to_cold_start():
    router = Router()
    features = {"max_similarity": 0.9, "redundancy": 0.1}
    assert router.route(features) == router._route_cold_start(features)


def test_trained_router_with_neutral_features_returns_benchmark_winner():
    router = Router()
    router.train(three_pipeline_benchmark())
    neutral = {"max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2, "query_length": 10}
    assert router.route(neutral) == "hybrid"


# ─────────────────────────────────────────────────────────
# _route_learned — refinement-rule overrides
# ─────────────────────────────────────────────────────────

def test_mmr_override_fires_with_sufficient_evidence():
    router = trained_router_with_evidence("mmr", win_rate=0.5)
    features = {"redundancy": 0.9, "max_similarity": 0.6, "avg_similarity": 0.5, "query_length": 8}
    assert router.route(features) == "mmr"


def test_mmr_override_does_not_fire_without_evidence():
    router = trained_router_with_evidence("mmr", win_rate=0.0)
    features = {"redundancy": 0.9, "max_similarity": 0.6, "avg_similarity": 0.5, "query_length": 8}
    assert router.route(features) == "hybrid"  # falls through to benchmark winner


def test_multiquery_override_on_weak_similarity():
    router = trained_router_with_evidence("multiquery", win_rate=0.5)
    features = {"redundancy": 0.1, "max_similarity": 0.1, "avg_similarity": 0.1, "query_length": 8}
    assert router.route(features) == "multiquery"


def test_multiquery_override_on_short_query_length():
    router = trained_router_with_evidence("multiquery", win_rate=0.5)
    # similarity is strong (won't trigger the weak-retrieval branch) but
    # query_length is at/below the learned median (2) -> second multiquery rule
    features = {"redundancy": 0.1, "max_similarity": 0.9, "avg_similarity": 0.9, "query_length": 1}
    assert router.route(features) == "multiquery"


def test_reranked_override_fires_with_sufficient_evidence():
    router = trained_router_with_evidence("reranked", win_rate=0.5)
    features = {"redundancy": 0.1, "max_similarity": 0.95, "avg_similarity": 0.9, "query_length": 8}
    assert router.route(features) == "reranked"


def test_evidence_threshold_reads_live_from_defaults():
    """Raising DEFAULTS.min_win_rate_for_rule_override mid-test should turn
    a previously-sufficient win_rate into insufficient evidence."""
    router = trained_router_with_evidence("mmr", win_rate=0.05)
    features = {"redundancy": 0.9, "max_similarity": 0.6, "avg_similarity": 0.5, "query_length": 8}

    DEFAULTS.min_win_rate_for_rule_override = 0.01
    assert router.route(features) == "mmr"

    DEFAULTS.min_win_rate_for_rule_override = 0.10
    assert router.route(features) == "hybrid"


def test_learned_threshold_used_instead_of_hardcoded_class_constant():
    """route() must use the LEARNED redundancy threshold for mmr, not the
    COLD_START_HIGH_REDUNDANCY (0.6) class constant."""
    router = trained_router_with_evidence("mmr", win_rate=0.5)
    router.thresholds["mmr"]["redundancy"] = 0.2  # learned value, well below the 0.6 constant

    features = {"redundancy": 0.3, "max_similarity": 0.6, "avg_similarity": 0.5, "query_length": 8}
    # 0.3 > learned 0.2 -> mmr fires, even though 0.3 < the 0.6 hardcoded constant
    assert router.route(features) == "mmr"


# ─────────────────────────────────────────────────────────
# save() / load()
# ─────────────────────────────────────────────────────────

def test_save_writes_thresholds_json(tmp_path):
    router = Router()
    router.train(three_pipeline_benchmark())
    router.save(path=str(tmp_path))

    target = tmp_path / "router_thresholds.json"
    assert target.exists()
    with open(target) as f:
        data = json.load(f)
    assert data["hybrid"]["win_rate"] == 1.0


def test_save_without_path_or_base_path_raises():
    router = Router()
    router.train(three_pipeline_benchmark())
    with pytest.raises(ValueError):
        router.save()


def test_load_restores_thresholds_and_trained_state(tmp_path):
    original = Router()
    original.train(three_pipeline_benchmark())
    original.save(path=str(tmp_path))

    fresh = Router()
    result = fresh.load(path=str(tmp_path))

    assert result is True
    assert fresh.is_trained is True
    assert fresh.thresholds["hybrid"]["win_rate"] == 1.0
    assert fresh.rules[0][0] == "hybrid"


def test_load_missing_file_returns_false_without_raising(tmp_path):
    router = Router()
    result = router.load(path=str(tmp_path))
    assert result is False
    assert router.is_trained is False


def test_load_without_path_or_base_path_returns_false():
    router = Router()
    assert router.load() is False


def test_round_trip_save_load_produces_identical_routing(tmp_path):
    original = Router()
    original.train(three_pipeline_benchmark())
    original.save(path=str(tmp_path))

    restored = Router()
    restored.load(path=str(tmp_path))

    features = {"max_similarity": 0.6, "avg_similarity": 0.5, "redundancy": 0.2, "query_length": 10}
    assert original.route(features) == restored.route(features)


def test_base_path_is_default_target_for_save_and_load(tmp_path):
    router = Router(base_path=str(tmp_path))
    router.train(three_pipeline_benchmark())  # auto-saves via base_path

    fresh = Router(base_path=str(tmp_path))
    assert fresh.load() is True
    assert fresh.is_trained is True


# ─────────────────────────────────────────────────────────
# explain()
# ─────────────────────────────────────────────────────────

def test_explain_untrained_router():
    router = Router()
    result = router.explain("hybrid")
    assert "cold-start" in result


def test_explain_unknown_pipeline_after_training():
    router = Router()
    router.train(three_pipeline_benchmark())
    assert router.explain("does_not_exist") == "Unknown pipeline: does_not_exist"


def test_explain_marks_benchmark_winner():
    router = Router()
    router.train(three_pipeline_benchmark())
    result = router.explain("hybrid")
    assert "benchmark winner" in result


def test_explain_marks_non_default_as_refinement_rule():
    router = Router()
    router.train(three_pipeline_benchmark())
    result = router.explain("mmr")
    assert "refinement rule" in result


def test_explain_includes_formatted_win_rate_and_score():
    router = Router()
    router.train(three_pipeline_benchmark())
    result = router.explain("hybrid")
    assert "100.0%" in result
    assert "0.90" in result


# ─────────────────────────────────────────────────────────
# get_stats()
# ─────────────────────────────────────────────────────────

def test_get_stats_untrained():
    router = Router()
    stats = router.get_stats()
    assert stats["status"] == "cold_start"
    assert "mode" in stats


def test_get_stats_trained():
    router = Router()
    router.train(three_pipeline_benchmark())
    stats = router.get_stats()

    assert stats["status"] == "trained"
    assert stats["num_pipelines"] == 3
    assert stats["top_pipeline"] == "hybrid"
    assert stats["top_win_rate"] == 1.0
    assert "hybrid" in stats["thresholds"]


# ─────────────────────────────────────────────────────────
# RouterInterface contract
# ─────────────────────────────────────────────────────────

def test_router_satisfies_router_interface():
    assert isinstance(Router(), RouterInterface)


def test_router_interface_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        RouterInterface()


def test_minimal_subclass_needs_only_route():
    class MinimalRouter(RouterInterface):
        def route(self, features):
            return "hybrid"

    router = MinimalRouter()
    assert router.route({}) == "hybrid"


def test_router_interface_default_explain():
    class MinimalRouter(RouterInterface):
        def route(self, features):
            return "hybrid"

    router = MinimalRouter()
    assert router.explain("hybrid") == "hybrid: selected by MinimalRouter"


def test_subclass_missing_route_cannot_be_instantiated():
    class BrokenRouter(RouterInterface):
        pass

    with pytest.raises(TypeError):
        BrokenRouter()