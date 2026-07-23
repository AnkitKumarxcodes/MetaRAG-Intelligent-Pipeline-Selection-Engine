# metarag/metarag.py

"""
MetaRAG v0.3 — Benchmark-Driven Pipeline Selection for RAG

MetaRAG doesn't ship "the best router." It ships the instrumentation to
build one: profile your corpus, benchmark every candidate pipeline on
your own queries, and either use the built-in reference router or plug
in your own model trained on that data.

Public API:
  fit()                    — load docs, chunk, embed, index, build pipelines
  ask()                     — retrieve + generate with the active router
  benchmark()                — evaluate all pipelines on your queries
  status()                   — project state snapshot
  leaderboard()               — pipeline rankings (benchmark or production logs)
  analyze_query()              — diagnose a single query
  analyze_corpus()              — diagnose the loaded corpus
  explain()                     — why the router picked what it picked
  save() / load()                — persist / restore config
  get_benchmark_data()            — raw benchmark DataFrame, for training your own router
  get_router_thresholds()          — inspect the active router's learned state
  set_llm() / set_embeddings()      — swap components
  set_router()                       — plug in ANY object with .route(features) -> str
  set_router_from_model()             — one-line adapter for sklearn-style .predict() models
  update_router_thresholds()           — hot-reload a saved router_thresholds.json
  rebuild()                             — force full re-fit()

Observability (new in v0.3):
  pipeline_graph()   — structural diagram of a pipeline's stages
  dashboard()         — bar-chart leaderboard from the last benchmark() run
  report()             — corpus profile summary
  inspect()             — compare what each retriever returns for one query
  trace()                — step-by-step trace of one pipeline execution

Internal (private):
  _setup_* / _extract_* / _train_* / _write_log / _read_logs
"""

from __future__ import annotations
import os
import json
import time
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .defaults import DEFAULTS


# ─────────────────────────────────────────────────────────────
# Answer (public return type)
# ─────────────────────────────────────────────────────────────

@dataclass
class Answer:
    """Answer returned by ask()."""
    text: str
    query: str
    pipeline: str
    chunks: List[Tuple[str, float]]
    score: float
    latency_ms: float
    sources: List[str] = field(default_factory=list)

    def __repr__(self):
        return (
            f"\n{'='*60}\n"
            f"  Query    : {self.query}\n"
            f"  Pipeline : {self.pipeline}\n"
            f"  Score    : {self.score:.2f}\n"
            f"  Latency  : {self.latency_ms:.0f}ms\n"
            f"{'─'*60}\n"
            f"  {self.text[:200]}...\n"
            f"{'='*60}"
        )


# ─────────────────────────────────────────────────────────────
# Generic router adapter — so a user's sklearn-style model can
# plug in without them hand-writing a RouterInterface class.
# ─────────────────────────────────────────────────────────────

class SklearnRouterAdapter:
    """
    Wraps any object with a .predict() method (sklearn classifiers,
    or anything following that convention) into something satisfying
    RouterInterface's contract: .route(features: dict) -> str.

    Example:
        df = rag.get_benchmark_data()
        feature_cols = ["max_similarity", "avg_similarity", "redundancy", "query_length"]
        X, y = df[feature_cols], df["winning_pipeline"]
        clf = RandomForestClassifier().fit(X, y)

        rag.set_router_from_model(clf, feature_cols=feature_cols)
    """

    def __init__(self, model, feature_cols: List[str]):
        if not hasattr(model, "predict"):
            raise TypeError("model must have a .predict(X) method")
        self.model = model
        self.feature_cols = feature_cols

    def route(self, features: Dict[str, Any]) -> str:
        x = [[features.get(c, 0) for c in self.feature_cols]]
        return self.model.predict(x)[0]

    def explain(self, pipeline: str, features: Optional[Dict] = None) -> str:
        return f"{pipeline}: predicted by {self.model.__class__.__name__}"


# ─────────────────────────────────────────────────────────────
# MetaRAG
# ─────────────────────────────────────────────────────────────

class MetaRAG:
    """
    MetaRAG — orchestrates loading, chunking, retrieval, generation,
    evaluation, benchmarking, and routing. Every stage is a swappable
    component (embeddings, generator, vector_db, router) — MetaRAG
    wires them together, it doesn't own the logic of any one of them.
    """
    from . import __version__

    VERSION = __version__

    def __init__(
        self,
        docs: Union[str, List[str]],
        embeddings,
        generator,
        project: str = "default",
        vector_db=None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
        k: int = None,
        eval_preset: str = None,
        verbose: bool = True,
        
    ):
        """
        Args:
            docs: path(s) to documents
            embeddings: object with .embed_query()/.embed_documents() (EmbeddingInterface)
            generator: object with .generate(prompt) (GeneratorInterface)
            project: project name — determines the .metarag/<project>/ storage path
            vector_db: VectorDBInterface object (default: InMemoryVectorDB)
            chunk_size / chunk_overlap / chunk_strategy / k / eval_preset:
                leave as None to use metarag.defaults.DEFAULTS; pass a value
                here to override DEFAULTS for this run (the override also
                propagates to DEFAULTS, so every component built afterward
                — retrievers, pipelines, router — picks it up automatically)
            verbose: print progress logs
        """
        required_methods = ("embed_query", "embed_documents")
        for method in required_methods:
            if not hasattr(embeddings, method):
                raise TypeError(
                    f"embeddings must have {required_methods} methods "
                    f"(EmbeddingInterface contract). Missing: {method}. "
                    f"Tip: wrap with metarag.CachedEmbeddings(your_model) if unsure."
                )
        if not hasattr(generator, "generate"):
            raise TypeError("generator must have .generate(prompt) method")

        self.docs_path = docs
        self.embeddings = embeddings
        self.generator = generator
        self._pipeline_builders = {}
        self._pipeline_cache = {}
        self._reranker = None

        from .pipelines.generator import GeneratorWrapper
        self._generator_wrapper = GeneratorWrapper(generator, model_name=generator.__class__.__name__)

        self.project = project

        # Any explicit override here propagates into DEFAULTS, so it's
        # visible to every retriever/pipeline/router built downstream —
        # not just stored as a local attribute on this instance.
        if chunk_size is not None: DEFAULTS.chunk_size = chunk_size
        if chunk_overlap is not None: DEFAULTS.chunk_overlap = chunk_overlap
        if chunk_strategy is not None: DEFAULTS.chunk_strategy = chunk_strategy
        if k is not None: DEFAULTS.k = k
        if eval_preset is not None: DEFAULTS.eval_preset = eval_preset

        self.chunk_size = DEFAULTS.as_single("chunk_size")
        self.chunk_overlap = DEFAULTS.as_single("chunk_overlap")
        self.chunk_strategy = DEFAULTS.as_single("chunk_strategy")
        self.k = DEFAULTS.as_single("k")
        self.eval_preset = DEFAULTS.as_single("eval_preset")
        self.verbose = verbose

        self._base = f".metarag/{project}"
        self._index_dir = f"{self._base}/index"
        self._cache_dir = f"{self._base}/cache"
        self._logs_dir = f"{self._base}/logs"
        self._profile_path = f"{self._base}/corpus_profile.json"
        self._benchmark_path = f"{self._base}/benchmark.csv"
        self._router_path = f"{self._base}/router_thresholds.json"
        self._log_path = f"{self._logs_dir}/queries.jsonl"
        self._config_path = f"{self._base}/config.json"

        for d in [self._index_dir, self._cache_dir, self._logs_dir]:
            os.makedirs(d, exist_ok=True)

        if vector_db is None:
            from .core.vector_db import InMemoryVectorDB
            self.vector_db = InMemoryVectorDB()
        else:
            self.vector_db = vector_db

        self._chunks = None
        self._corpus_profile = None
        self._retrievers = {}
        self._evaluator = None
        self._router = None
        self._fitted = False
        self._query_logs = []

        self._log(f"MetaRAG v{self.VERSION} — project='{project}'")

    # ═════════════════════════════════════════════════════════
    # CORE LIFECYCLE
    # ═════════════════════════════════════════════════════════
    def _get_reranker(self):
        if self._reranker is None:
            from .pipelines.pipeline import Reranker
            self._reranker = Reranker()
        return self._reranker
    
    def fit(self, force: bool = False) -> "MetaRAG":
        self._pipeline_cache.clear()
        self._reranker = None
        """Load documents → chunk → embed → index → build pipelines."""
        t0 = time.time()
        self._log("fit() starting...")

        self._load_docs_and_chunk(force=force)
        self._build_vector_index(force=force)
        self._build_corpus_profile(force=force)
        self._setup_retrievers()
        self._setup_evaluator()
        self._setup_pipelines()

        self._fitted = True
        elapsed = round((time.time() - t0) * 1000)
        self._log(f"fit() done in {elapsed}ms — ready to ask()")
        return self

    def ask(self, query: str) -> Answer:
        """Ask a question. The active router selects a pipeline; falls back
        to the first available pipeline if no router is configured."""
        self._check_fitted()
        t0 = time.time()

        if self._router is not None:
            features = self._extract_query_features(query)
            pipeline_name = self._router.route(features)
        else:
            pipeline_name = next(iter(self._pipeline_builders))  # no hardcoded "hybrid" — just take whatever exists first

        try:
            pipeline = self._get_pipeline(pipeline_name)
        except ValueError:
            fallback = next(iter(self._pipeline_builders))
            pipeline = self._get_pipeline(fallback)
            pipeline_name = fallback

        pipeline_result = pipeline.run(query, k=self.k)
        chunks = pipeline_result["chunks"]
        chunk_texts = [c[0] if isinstance(c, tuple) else str(c) for c in chunks]

        answer_text, _ = self._generator_wrapper.generate_text(query, chunk_texts)

        sources = []
        for chunk in chunks:
            chunk_obj = chunk[0] if isinstance(chunk, tuple) else chunk
            sources.append(
                chunk_obj.metadata.get("source", "unknown")
                if hasattr(chunk_obj, "metadata") else str(chunk_obj)[:50]
            )

        elapsed = round((time.time() - t0) * 1000, 2)
        answer = Answer(
            text=answer_text, query=query, pipeline=pipeline_name,
            chunks=chunks, score=0.0, latency_ms=elapsed, sources=sources[:3],
        )
        score_result = self._evaluator.evaluate(answer)
        answer.score = score_result.composite

        self._write_log(answer)
        return answer

    def benchmark(
        self,
        queries: List[str],
        retrieval_only: bool = True,
        train_router: bool = True,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Run every built pipeline against every query, score each, and
        optionally train the built-in reference router on the results."""
        self._check_fitted()
        t0 = time.time()
        self._log(f"benchmark() starting on {len(queries)} queries "
                   f"({len(self._pipeline_builders)} pipelines, retrieval_only={retrieval_only})...")

        from .Evaluator.metrics import precision, coverage, redundancy as chunk_redundancy_fn
        from .router.query_profiler import QueryProfiler
        from .router.probe_profiler import ProbeProfiler

        query_profiler = QueryProfiler()
        probe_profiler = ProbeProfiler(self.vector_db, self.embeddings, k=DEFAULTS.as_single("k"))
        results = []

        for i, query in enumerate(queries):
            self._log(f"  [{i+1}/{len(queries)}] {query[:50]}...")

            qp_features = query_profiler.profile(query)
            probe_features = probe_profiler.probe(query)

            pipeline_scores = {}
            query_results = []

            for pipeline_name in self._pipeline_builders:
                pipeline = self._get_pipeline(pipeline_name)
                retrieval_t0 = time.time()
                pipeline_result = pipeline.run(query, k=self.k)
                retrieval_ms = round((time.time() - retrieval_t0) * 1000, 2)

                chunks = pipeline_result["chunks"]
                chunk_texts = [c[0] if isinstance(c, tuple) else str(c) for c in chunks]

                base_row = {
                    "query": query, "pipeline": pipeline_name,
                    "query_length": qp_features["query_length"],
                    "num_docs": len(self._chunks) if self._chunks else 0,
                    "avg_similarity": probe_features["avg_similarity"],
                    "max_similarity": probe_features["max_similarity"],
                    "redundancy": probe_features["redundancy"],
                }

                if retrieval_only:
                    prec = precision(query, chunk_texts, self.embeddings)
                    cov = coverage(query, chunk_texts)
                    redund = chunk_redundancy_fn(chunk_texts, self.embeddings)
                    composite = round(0.5 * prec["avg"] + 0.3 * cov + 0.2 * (1 - redund), 4)

                    pipeline_scores[pipeline_name] = composite
                    row = {**base_row,
                           "precision_avg": prec["avg"], "coverage": cov,
                           "chunk_redundancy": redund, "composite": composite,
                           "latency_ms": retrieval_ms}
                else:
                    answer_text, gen_latency_ms = self._generator_wrapper.generate_text(query, chunk_texts)
                    total_latency_ms = round(retrieval_ms + gen_latency_ms, 2)
                    temp_answer = Answer(text=answer_text, query=query, pipeline=pipeline_name,
                                          chunks=chunk_texts, score=0.0, latency_ms=total_latency_ms)
                    score_result = self._evaluator.evaluate(temp_answer)
                    pipeline_scores[pipeline_name] = score_result.composite
                    row = {**base_row,
                           "faithfulness": score_result.faithfulness,
                           "relevancy": score_result.relevancy,
                           "precision_avg": score_result.precision_avg,
                           "composite": score_result.composite,
                           "latency_ms": total_latency_ms}

                query_results.append(row)

            if pipeline_scores:
                winning_pipeline = max(pipeline_scores, key=pipeline_scores.get)
                for row in query_results:
                    row["winning_pipeline"] = winning_pipeline

            results.extend(query_results)

        benchmark_df = pd.DataFrame(results)
        if save_csv:
            benchmark_df.to_csv(self._benchmark_path, index=False)
            self._log(f"Benchmark CSV saved: {self._benchmark_path}")
        if train_router:
            self._train_learned_router(benchmark_df)

        self._log(f"benchmark() done in {round((time.time() - t0) * 1000)}ms")
        return benchmark_df

    # ═════════════════════════════════════════════════════════
    # INSPECTION
    # ═════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """Print + return a snapshot of current project state."""
        logs = self._read_logs()
        avg_score = round(sum(r.get("score", 0) for r in logs) / len(logs), 3) if logs else 0
        router_status = self._router.__class__.__name__ if self._router is not None else "none (default fallback)"

        info = {
            "fitted": self._fitted,
            "chunks": len(self._chunks) if self._chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "k": self.k,
            "eval_preset": self.eval_preset,
            "registered_pipelines": list(self._pipeline_builders.keys()),
            "loaded_pipelines": list(self._pipeline_cache.keys()),
            "queries_logged": len(logs),
            "avg_score": avg_score,
            "router": router_status,
            "storage_path": self._base,
        }

        print(f"\n{'='*60}\n  MetaRAG v{self.VERSION} — '{self.project}'\n{'='*60}")
        for key, val in info.items():
            label = key.replace("_", " ").title()
            print(f"  {label:<16}: {val}")
        print(f"{'='*60}\n")

        return info

    def leaderboard(self, source: str = "benchmark"):
        """Rank pipelines. source='benchmark' reads the last benchmark() run,
        source='logs' reads real ask() usage over time."""
        if source == "benchmark":
            return self._leaderboard_from_benchmark()
        elif source == "logs":
            return self._leaderboard_from_logs()
        raise ValueError("source must be 'benchmark' or 'logs'")

    def _leaderboard_from_benchmark(self):
        if not os.path.exists(self._benchmark_path):
            print("[MetaRAG] No benchmark data yet. Run benchmark() first.")
            return None

        df = pd.read_csv(self._benchmark_path)
        has_llm_metrics = "faithfulness" in df.columns

        agg_cols = {"composite": "mean", "latency_ms": "mean"}
        agg_cols.update(
            {"faithfulness": "mean", "relevancy": "mean", "precision_avg": "mean"}
            if has_llm_metrics else
            {"precision_avg": "mean", "coverage": "mean", "chunk_redundancy": "mean"}
        )
        summary = df.groupby("pipeline").agg(agg_cols).round(2).sort_values("composite", ascending=False)

        header = ("FAITH", "RELEV", "PREC") if has_llm_metrics else ("PREC", "COVER", "REDUND")
        cols = ("faithfulness", "relevancy", "precision_avg") if has_llm_metrics \
            else ("precision_avg", "coverage", "chunk_redundancy")

        print("=" * 70)
        print(f"{'PIPELINE':<15}{header[0]:>8}{header[1]:>8}{header[2]:>8}{'SCORE':>8}{'LATENCY':>10}")
        print("=" * 70)
        for name, row in summary.iterrows():
            print(f"{name:<15}{row[cols[0]]:>8.2f}{row[cols[1]]:>8.2f}{row[cols[2]]:>8.2f}"
                  f"{row['composite']:>8.2f}{row['latency_ms']:>9.0f}ms")
        print("=" * 70)

        best = summary.index[0]
        print(f"\n🏆 Best pipeline: {best} (score={summary.loc[best, 'composite']:.2f})")

        if self._router is not None and len(df):
            sample_query = df["query"].iloc[0]
            router_pick = self._router.route(self._extract_query_features(sample_query))
            print(f"🔀 Router would pick: {router_pick}")

        return summary

    def _leaderboard_from_logs(self):
        logs = self._read_logs()
        if not logs:
            print("[MetaRAG] No queries logged yet. Run ask() first.")
            return None

        from collections import defaultdict
        pipe_scores = defaultdict(list)
        for row in logs:
            pipe_scores[row.get("pipeline", "unknown")].append(row.get("score", 0))

        ranked = sorted(pipe_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

        print(f"\n{'='*60}\n  Leaderboard (production usage) — {len(logs)} queries\n{'='*60}")
        print(f"{'Pipeline':<15} {'Queries':>8} {'Avg':>7} {'Max':>7}")
        print(f"{'─'*60}")
        for i, (name, scores) in enumerate(ranked):
            avg, max_score = round(sum(scores) / len(scores), 3), round(max(scores), 3)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"{medal} {name:<13} {len(scores):>8} {avg:>7.3f} {max_score:>7.3f}")
        print(f"{'='*60}\n")

        return ranked

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Lightweight standalone query diagnostic (no router involved)."""
        words = query.lower().split()
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        complexity = "high" if avg_word_len > 6 else "medium" if avg_word_len > 4 else "low"
        stopwords = {"the", "a", "an", "and", "or", "is", "was", "what", "how", "why"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return {
            "query": query, "length": len(query), "words": len(words),
            "avg_word_length": round(avg_word_len, 2), "complexity": complexity,
            "keywords": keywords[:5], "num_keywords": len(keywords),
        }

    def analyze_corpus(self) -> Dict[str, Any]:
        """Diagnose the loaded corpus."""
        if not self._chunks:
            return {"status": "not fitted"}

        texts = [c.text if hasattr(c, "text") else str(c) for c in self._chunks]
        total_chars = sum(len(t) for t in texts)

        return {
            "num_chunks": len(self._chunks),
            "total_characters": total_chars,
            "avg_chunk_size": round(total_chars / len(texts), 0) if texts else 0,
            "corpus_profile": self._corpus_profile,
        }

    def explain(self, query: str) -> Dict[str, Any]:
        """Why did (or would) the router pick this pipeline for this query."""
        self._check_fitted()
        features = self._extract_query_features(query)

        if self._router is not None:
            pipeline_name = self._router.route(features)
            explanation = (
                self._router.explain(pipeline_name)
                if hasattr(self._router, "explain")
                else f"selected by {self._router.__class__.__name__}"
            )
            confidence = self._router.__class__.__name__
        else:
            pipeline_name = next(iter(self._pipeline_builders))
            explanation = "no router configured — using first available pipeline"
            confidence = "none"

        return {
            "query": query,
            "selected_pipeline": pipeline_name,
            "confidence": confidence,
            "explanation": explanation,
            "features": features,
            "available_pipelines": list(self._pipeline_builders.keys()),
        }

    # ═════════════════════════════════════════════════════════
    # OBSERVABILITY (v0.3)
    # ═════════════════════════════════════════════════════════

    def pipeline_graph(self, pipeline_name: str = None) -> str:
        """
        Structural diagram of a pipeline's stages, built by introspecting
        whatever is actually attached to it — not a hardcoded description.
        If pipeline_name is None, shows every built pipeline.
        """
        self._check_fitted()
        names = [pipeline_name] if pipeline_name else list(self._pipeline_builders.keys())
        output_lines = []

        for name in names:
            try:
                pipeline = self._get_pipeline(name)
            except ValueError:
                output_lines.append(f"[pipeline_graph] Unknown pipeline: '{name}'")
                continue

            stages = ["Query"]
            if getattr(pipeline, "hyde", None):
                stages.append("HyDE (hypothesis generation)")
            if getattr(pipeline, "multiquery", None):
                stages.append(f"MultiQuery (expand ×{pipeline.multiquery.n})")

            retriever_name = pipeline.retriever.__class__.__name__
            stages.append(f"Retriever: {retriever_name}")

            if getattr(pipeline, "reranker", None) and pipeline.reranker.model is not None:
                stages.append("Reranker")

            stages.append("Deduplicator")
            stages.append("Chunks")

            diagram = f"\n[{name}]\n" + "\n  │\n  ▼\n".join(stages)
            output_lines.append(diagram)

        result = "\n".join(output_lines)
        print(result)
        return result

    def dashboard(self):
        """Bar-chart leaderboard from the last benchmark() run — same data
        as leaderboard(), rendered as relative-length bars instead of numbers."""
        if not os.path.exists(self._benchmark_path):
            print("[MetaRAG] No benchmark data yet. Run benchmark() first.")
            return None

        df = pd.read_csv(self._benchmark_path)
        summary = df.groupby("pipeline")["composite"].mean().sort_values(ascending=False)
        latency = df.groupby("pipeline")["latency_ms"].mean()

        max_score = summary.max() or 1
        bar_width = 30

        print(f"\n{'='*70}\n  BENCHMARK DASHBOARD\n{'='*70}")
        for pipeline_name, score in summary.items():
            filled = int((score / max_score) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {pipeline_name:<14}{bar}  {score:.2f}  ({latency[pipeline_name]:.0f}ms)")
        print(f"{'='*70}\n")

        return summary

    def report(self) -> Dict[str, Any]:
        """Corpus profile summary — reads whatever CorpusProfiler computed at fit()."""
        self._check_fitted()
        profile = self._corpus_profile or {}

        print(f"\n{'='*60}\n  CORPUS REPORT\n{'='*60}")
        for key, val in profile.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                print(f"  {label:<20}: {val:.2%}" if "ratio" in key else f"  {label:<20}: {val:.1f}")
            else:
                print(f"  {label:<20}: {val}")
        print(f"{'='*60}\n")

        return profile

    def inspect(self, query: str, k: int = None) -> Dict[str, list]:
        """
        Run every built RETRIEVER (not full pipeline) independently on one
        query, so you can directly compare what each strategy surfaces.
        """
        self._check_fitted()
        k = k or self.k
        results = {}

        print(f"\n{'='*60}\n  RETRIEVAL INSPECTOR — '{query[:50]}'\n{'='*60}")
        for name, retriever in self._retrievers.items():
            hits = retriever.retrieve(query, k=k)
            texts = [h[0].text if hasattr(h[0], "text") else str(h[0]) for h in hits]
            results[name] = texts

            print(f"\n{name.upper()}")
            for i, text in enumerate(texts, 1):
                print(f"  {i}. {text[:70]}...")
        print(f"{'='*60}\n")

        return results

    def trace(self, query: str, pipeline_name: str = None) -> List[Dict[str, Any]]:
        """
        Step-by-step trace of one pipeline execution: how many chunks
        survived each stage. Requires the pipeline's stages to report
        their own counts — falls back to before/after chunk counts if a
        stage doesn't have anything more specific to report.
        """
        self._check_fitted()
        pipeline_name = pipeline_name or (
            self._router.route(self._extract_query_features(query))
            if self._router is not None else next(iter(self._pipeline_builders))
        )
        try:
            pipeline = self._get_pipeline(pipeline_name)
        except ValueError:
            print(f"[trace] Unknown pipeline: '{pipeline_name}'")
            return []

        steps = []
        print(f"\n{'='*60}\n  PIPELINE TRACE — '{pipeline_name}'\n{'='*60}")

        retrieve_query = query
        if getattr(pipeline, "hyde", None):
            hypothesis = pipeline.hyde.generate_hypothesis(query)
            retrieve_query = hypothesis
            steps.append({"stage": "HyDE", "detail": f"generated {len(hypothesis)}-char hypothesis"})
            print(f"  HyDE           → generated hypothesis ({len(hypothesis)} chars)")

        queries = [retrieve_query]
        if getattr(pipeline, "multiquery", None):
            queries = pipeline.multiquery.expand(retrieve_query)
            steps.append({"stage": "MultiQuery", "detail": f"expanded to {len(queries)} variants"})
            print(f"  MultiQuery     → expanded to {len(queries)} query variants")

        all_chunks = []
        for q in queries:
            all_chunks.extend(pipeline.retriever.retrieve(q, k=self.k))
        steps.append({"stage": "Retrieve", "detail": f"{len(all_chunks)} chunks"})
        print(f"  Retrieve       → {len(all_chunks)} chunks")

        if getattr(pipeline, "reranker", None):
            all_chunks = pipeline.reranker.rerank(query, all_chunks, k=self.k)
            steps.append({"stage": "Rerank", "detail": f"{len(all_chunks)} chunks"})
            print(f"  Rerank         → {len(all_chunks)} chunks")

        final_chunks = pipeline.dedup.deduplicate(all_chunks)
        steps.append({"stage": "Deduplicate", "detail": f"{len(final_chunks)} chunks"})
        print(f"  Deduplicate    → {len(final_chunks)} chunks")
        print(f"{'='*60}\n")

        return steps

    # ═════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═════════════════════════════════════════════════════════

    def save(self) -> "MetaRAG":
        """Persist current config to disk (settings only — not the index/router)."""
        config = {
            "version": self.VERSION, "project": self.project,
            "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy, "k": self.k,
            "eval_preset": self.eval_preset, "fitted": self._fitted,
            "num_chunks": len(self._chunks) if self._chunks else 0,
        }
        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=2)
        self._log(f"Config saved → {self._config_path}")
        return self

    @classmethod
    def load(cls, project: str, embeddings, generator, vector_db=None) -> "MetaRAG":
        """Rebuild a MetaRAG shell from a saved config. Call fit() afterward
        to actually rehydrate the index (fast — hits disk caches)."""
        config_path = f".metarag/{project}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        rag = cls(
            docs=[], embeddings=embeddings, generator=generator, project=project,
            vector_db=vector_db,
            chunk_size=config.get("chunk_size"), chunk_overlap=config.get("chunk_overlap"),
            chunk_strategy=config.get("chunk_strategy"), k=config.get("k"),
            eval_preset=config.get("eval_preset"),
        )
        rag._fitted = config.get("fitted", False)
        print(f"[MetaRAG] Loaded project: {project}")
        return rag

    def get_benchmark_data(self) -> pd.DataFrame:
        """Raw benchmark results — the training set for building your own router."""
        if not os.path.exists(self._benchmark_path):
            raise FileNotFoundError("Benchmark CSV not found. Run benchmark() first.")
        return pd.read_csv(self._benchmark_path)

    def get_router_thresholds(self) -> Dict[str, Any]:
        """Inspect whatever the active router reports about its own state."""
        if self._router is None:
            return {}
        if hasattr(self._router, "get_stats"):
            return self._router.get_stats()
        return {"router_class": self._router.__class__.__name__}

    # ═════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═════════════════════════════════════════════════════════

    def set_llm(self, generator) -> "MetaRAG":
        """Replace the generator."""
        if not hasattr(generator, "generate"):
            raise TypeError("generator must have .generate(prompt) method")
        self.generator = generator
        from .pipelines.generator import GeneratorWrapper
        self._generator_wrapper = GeneratorWrapper(generator, model_name=generator.__class__.__name__)
        self._log("Generator updated")
        return self

    def set_embeddings(self, embeddings) -> "MetaRAG":
        """Replace the embeddings model."""
        required_methods = ("embed_query", "embed_documents")
        for method in required_methods:
            if not hasattr(embeddings, method):
                raise TypeError(f"embeddings must have {required_methods} methods")
        self.embeddings = embeddings
        self._log("Embeddings updated")
        return self

    def set_router(self, router) -> "MetaRAG":
        """
        Plug in ANY object satisfying RouterInterface's contract:
        .route(features: dict) -> str

        This is the main extension point — bring your own router,
        trained however you like, on get_benchmark_data().
        """
        if not hasattr(router, "route"):
            raise TypeError("router must have a .route(features) method")
        self._router = router
        self._log(f"Router updated → {router.__class__.__name__}")
        return self

    def set_router_from_model(self, model, feature_cols: List[str]) -> "MetaRAG":
        """
        Convenience wrapper: plug in any sklearn-style model (anything with
        .predict()) without writing your own adapter class.

        Example:
            df = rag.get_benchmark_data()
            cols = ["max_similarity", "avg_similarity", "redundancy", "query_length"]
            clf = RandomForestClassifier().fit(df[cols], df["winning_pipeline"])
            rag.set_router_from_model(clf, feature_cols=cols)
        """
        return self.set_router(SklearnRouterAdapter(model, feature_cols))

    def update_router_thresholds(self, path: str = None) -> "MetaRAG":
        """
        Hot-reload a saved router_thresholds.json into the active router
        (or a fresh Router, if none is active) — without re-running
        benchmark()/train(). Useful for shipping a pre-trained router
        alongside your corpus, or restoring a prior run's weights.

        Args:
            path: directory containing router_thresholds.json.
                Defaults to this project's own storage path.
        """
        from .router.router import Router

        base_path = path or self._base
        router = self._router if isinstance(self._router, Router) else Router(base_path)
        router.base_path = __import__("pathlib").Path(base_path)

        if not router.load():
            raise FileNotFoundError(f"No router_thresholds.json found at {base_path}")

        self._router = router
        self._log(f"Router thresholds reloaded from {base_path}")
        return self

    def rebuild(self, force: bool = True) -> "MetaRAG":
        """Force a full re-fit()."""
        self._log("Rebuilding index and pipelines...")
        self._pipeline_cache.clear()
        self._reranker = None
        return self.fit(force=force)

    # ═════════════════════════════════════════════════════════
    # INTERNAL
    # ═════════════════════════════════════════════════════════

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[MetaRAG] {msg}")

    def _load_docs_and_chunk(self, force: bool = False):
        from .core.loader import DocumentLoader
        from .core.chunking import Chunker

        self._log("Loading documents...")
        docs = DocumentLoader(self.docs_path).load(verbose = False)
        docs = DocumentLoader(self.docs_path).load(verbose=False)

        if docs.loaded.count == 0:
            raise ValueError(f"No documents found at '{self.docs_path}'")
        
        self._log(
            f"Loaded {docs.loaded.count} file(s)"
                )

        if docs.skipped.count:
            self._log(
                f"Skipped {docs.skipped.count} file(s)"
                )

        self._log("Chunking...")
        chunker = Chunker(strategy=self.chunk_strategy, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        self._chunks = chunker.chunk_documents(docs, cache_dir=f"{self._cache_dir}/chunks", force=force)
        self._log(f"{len(self._chunks)} chunks ready")

    def _build_vector_index(self, force: bool = False):
        self._log("Embedding chunks...")
        chunk_texts = [c.text if hasattr(c, "text") else str(c) for c in self._chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)

        self._log("Building vector index...")
        self.vector_db.build(self._chunks, chunk_embeddings)

    def _build_corpus_profile(self, force: bool = False):
        """Delegates to CorpusProfiler — the real profiler, not a manual
        reimplementation of a subset of its metrics."""
        from .router.corpus_profiler import CorpusProfiler

        self._log("Profiling corpus...")
        self._corpus_profile = CorpusProfiler().profile(self._chunks)
        self._corpus_profile["chunk_strategy"] = self.chunk_strategy

        with open(self._profile_path, "w") as f:
            json.dump(self._corpus_profile, f, indent=2)

    def _setup_retrievers(self):
        self._retrievers.clear()
        from .core.retriever import BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever
        if not self._chunks:
            raise ValueError("No chunks available. Run fit() on a non-empty corpus.")
        
        self._retrievers["bm25"] = BM25Retriever(self._chunks)
        self._retrievers["dense"] = DenseRetriever(self._chunks, self.embeddings, self.vector_db)
        self._retrievers["hybrid"] = HybridRetriever(self._chunks, self.embeddings, self.vector_db)
        self._retrievers["mmr"] = MMRRetriever(self._chunks, self.embeddings, self.vector_db)

        self._log(f"{len(self._retrievers)} retrievers ready")

    def _setup_pipelines(self):
        self._pipeline_builders.clear()
        from .pipelines.pipeline import (
            StraightPipeline,
            MultiQueryPipeline,
            RerankedPipeline,
            FullPipeline,
        )

        # Straight pipelines
        for retriever_name, retriever in self._retrievers.items():
            self._pipeline_builders[retriever_name] = (
                lambda r=retriever, n=retriever_name:
                    StraightPipeline(r, name=n)
            )

        # MultiQuery
        self._pipeline_builders["multiquery"] = lambda: MultiQueryPipeline(
            self._retrievers["hybrid"],
            self.generator,
            n_variants=DEFAULTS.as_single("multiquery_n_variants"),
        )

        # Reranked
        self._pipeline_builders["reranked"] = lambda: RerankedPipeline(
            self._retrievers["hybrid"],
            self._get_reranker(),
        )

        #Full
        self._pipeline_builders["full"] = lambda: FullPipeline(
            self._retrievers["hybrid"],
            self.generator,
            self._get_reranker(),
            n_variants=DEFAULTS.as_single("multiquery_n_variants"),
        )

        self._log(
            f"{len(self._pipeline_builders)} pipelines registered: "
            f"{list(self._pipeline_builders.keys())}"
        )
    
    def _get_pipeline(self, pipeline_name: str):
        if pipeline_name not in self._pipeline_cache:
            builder = self._pipeline_builders.get(pipeline_name)

            if builder is None:
                raise ValueError(f"Unknown pipeline '{pipeline_name}'")

            self._pipeline_cache[pipeline_name] = builder()

        return self._pipeline_cache[pipeline_name]

    def _setup_evaluator(self):
        from .Evaluator.evaluator import Evaluator
        self._evaluator = Evaluator(self.embeddings, preset=self.eval_preset)
        self._log("Evaluator ready")

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Query + probe features, MERGED with corpus-level features — so a
        router can see the full picture, not just per-query signals."""
        from .router.query_profiler import QueryProfiler
        from .router.probe_profiler import ProbeProfiler

        qp = QueryProfiler().profile(query)
        probe = ProbeProfiler(self.vector_db, self.embeddings, k=DEFAULTS.as_single("k")).probe(query)
        corpus = self._corpus_profile or {}

        return {
            **corpus,
            **qp,   
            "query_length": qp["query_length"], "char_count": qp["char_count"],
            "is_short": qp["is_short"], "is_long": qp["is_long"],
            "num_chunks": len(self._chunks) if self._chunks else 0,
            **probe,
        }

    def _train_learned_router(self, benchmark_df: pd.DataFrame):
        from .router.router import Router
        router = Router(self._base)
        router.train(benchmark_df)
        self._router = router

    def _write_log(self, answer: Answer):
        row = {
            "query": answer.query, "pipeline": answer.pipeline,
            "score": answer.score, "latency_ms": answer.latency_ms,
            "timestamp": time.time(),
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._query_logs.append(row)

    def _read_logs(self) -> List[Dict]:
        if not os.path.exists(self._log_path):
            return []
        rows = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return rows

    def __repr__(self):
        router_status = self._router.__class__.__name__ if self._router is not None else "none"
        return (
            f"MetaRAG(project='{self.project}', "
            f"fitted={self._fitted}, "
            f"chunks={len(self._chunks) if self._chunks else 0}, "
            f"registered={len(self._pipeline_builders)}, "
            f"loaded={len(self._pipeline_cache)}, "
            f"router={router_status})"
        )