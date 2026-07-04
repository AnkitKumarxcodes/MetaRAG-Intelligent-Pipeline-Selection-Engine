# metarag/metarag.py

"""
MetaRAG v0.2 — Intelligent Pipeline Selection Engine for RAG

Public API (12 methods):
  fit() — load docs, chunk, embed, index, build pipelines
  ask() — retrieve + generate with learned router
  benchmark() — evaluate all pipelines, train router
  save() / load() — persist state
  status() — show project status
  leaderboard() — pipeline rankings
  analyze_query() — diagnose query
  analyze_corpus() — diagnose corpus
  explain() — transparent routing decision
  get_benchmark_data() — access benchmark CSV
  get_router_thresholds() — inspect router
  set_llm() / set_embeddings() / rebuild() — reconfigure

Internal (private, not exposed):
  _setup_* methods
  _extract_* methods
  _train_* methods
  _write_log / _read_logs
"""

from __future__ import annotations
import os
import json
import time
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Answer (public return type)
# ─────────────────────────────────────────────────────────────

@dataclass
class Answer:
    """Answer returned by ask()."""
    text: str
    query: str
    pipeline: str
    chunks: List[Tuple[str, float]]  # (text, similarity_score)
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
# MetaRAG
# ─────────────────────────────────────────────────────────────

class MetaRAG:
    """
    MetaRAG v0.2 — Intelligent Pipeline Selection Engine
    
    Interface-based framework: user brings embeddings/LLM, 
    MetaRAG orchestrates retrieval, generation, evaluation, routing.
    """
    
    VERSION = "0.2.0"
    
    def __init__(
        self,
        docs: Union[str, List[str]],
        embeddings,
        generator,
        project: str = "default",
        vector_db = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "recursive",
        k: int = 4,
        eval_preset: str = "balanced",
        verbose: bool = True,
    ):
        """
        Initialize MetaRAG.
        
        Args:
            docs: path(s) to documents
            embeddings: object with .embed(text) method (EmbeddingInterface)
            generator: object with .generate(prompt) method (GeneratorInterface)
            project: project name (for storage)
            vector_db: VectorDBInterface object (default: InMemoryVectorDB)
            chunk_size: chunk size in chars
            chunk_overlap: overlap in chars
            chunk_strategy: "fixed", "recursive", "semantic", "sentence", etc.
            k: retrieve k chunks
            eval_preset: "balanced", "precision", "recall"
            verbose: print logs
        """
        # Validate user inputs
        if not hasattr(embeddings, "embed"):
            raise TypeError("embeddings must have .embed(text) method")
        if not hasattr(generator, "generate"):
            raise TypeError("generator must have .generate(prompt) method")
        
        # Config
        self.docs_path = docs
        self.embeddings = embeddings
        self.generator = generator
        self.project = project
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.k = k
        self.eval_preset = eval_preset
        self.verbose = verbose
        
        # Storage paths
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
        
        # Vector DB (default: InMemory, user can override)
        if vector_db is None:
            from .core.vector_db import InMemoryVectorDB
            self.vector_db = InMemoryVectorDB()
        else:
            self.vector_db = vector_db
        
        # Internal state
        self._chunks = None
        self._corpus_profile = None
        self._retrievers = {}  # registry of built retrievers
        self._pipelines = {}
        self._evaluator = None
        self._router = None
        self._fitted = False
        self._query_logs = []
        
        self._log(f"MetaRAG v{self.VERSION} — project='{project}'")
    
    # ═════════════════════════════════════════════════════════
    # PUBLIC API (12 methods)
    # ═════════════════════════════════════════════════════════
    
    # ─────────────────────────────────────────────────────────
    # 1. fit()
    # ─────────────────────────────────────────────────────────
    
    def fit(self, force: bool = False) -> "MetaRAG":
        """
        Load documents → chunk → embed → index → build pipelines.
        
        Args:
            force: if True, rebuild everything from scratch
        
        Returns:
            self (for chaining)
        """
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
    
    # ─────────────────────────────────────────────────────────
    # 2. ask()
    # ─────────────────────────────────────────────────────────
    
    def ask(self, query: str) -> Answer:
        """
        Ask a question. Router selects best pipeline. Returns Answer.
        
        Args:
            query: question string
        
        Returns:
            Answer with .text .pipeline .score .chunks .sources
        """
        self._check_fitted()
        t0 = time.time()
        
        # Route: use learned router if trained, else default to hybrid
        # AFTER
        if self._router is not None:
            features = self._extract_query_features(query)
            pipeline_name = self._router.route(features)
            explanation = (
                self._router.explain(pipeline_name)
                if hasattr(self._router, "explain")
                else f"selected by {self._router.__class__.__name__}"
            )
            confidence = self._router.__class__.__name__
        else:
            pipeline_name = "hybrid"
            explanation = "default (no router configured)"
            confidence = "default"
        
        # Run pipeline
        pipeline = self._pipelines.get(pipeline_name, self._pipelines["hybrid"])
        pipeline_result = pipeline.run(query, k=self.k)
        chunks = pipeline_result["chunks"]  # List[(text, score)] tuples
        
        # Extract chunk texts for generation
        chunk_texts = [chunk[0] if isinstance(chunk, tuple) else str(chunk) for chunk in chunks]
        
        # Generate answer
        answer_text, gen_latency_ms = self._generator_wrapper.generate_text(query, chunk_texts)
        
        # Extract sources (metadata from chunks if available)
        sources = []
        for chunk in chunks:  # chunks are (Chunk_or_str, score) tuples
            chunk_obj = chunk[0] if isinstance(chunk, tuple) else chunk
            if hasattr(chunk_obj, "metadata"):
                sources.append(chunk_obj.metadata.get("source", "unknown"))
            else:
                sources.append(str(chunk_obj)[:50])
        
        elapsed = round((time.time() - t0) * 1000, 2)

        answer = Answer(
            text=answer_text,
            query=query,
            pipeline=pipeline_name,
            chunks=chunks,
            score=0.0,          # placeholder, set below
            latency_ms=elapsed,
            sources=sources[:3],
        )

        score_result = self._evaluator.evaluate(answer)   # Evaluator reads answer.query/.text/.chunks/.latency_ms
        answer.score = score_result.composite             # now fill in the real score

        self._write_log(answer)
        score = answer
        return answer
    # ─────────────────────────────────────────────────────────
    # 3. benchmark()
    # ─────────────────────────────────────────────────────────
    
    def benchmark(
        self,
        queries: List[str],
        retrieval_only: bool = False,
        train_router: bool = True,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Run all pipelines on queries. Generate labels. Train router.
        
        Args:
            queries: list of test queries
            retrieval_only: if True, skip generation/evaluation
            train_router: if True, train LearnedRuleRouter after
            save_csv: if True, save to benchmark.csv
        
        Returns:
            benchmark_df (DataFrame with all results)
        """
        self._check_fitted()
        t0 = time.time()
        
        self._log(f"benchmark() starting on {len(queries)} queries...")
        
        results = []
        
        for i, query in enumerate(queries):
            self._log(f"  [{i+1}/{len(queries)}] {query[:50]}...")
            
            pipeline_scores = {}
            query_results = []
            
            # Run all pipelines
            for pipeline_name, pipeline in self._pipelines.items():
                pipeline_result = pipeline.run(query, k=self.k)
                chunks = pipeline_result["chunks"]
                chunk_texts = [c[0] if isinstance(c, tuple) else str(c) for c in chunks]
                
                if retrieval_only:
                    row = {
                        "query": query,
                        "pipeline": pipeline_name,
                        "chunks_retrieved": len(chunks),
                    }
                    query_results.append(row)
                else:
                    # Generate + evaluate
                    answer_text, _ = self._generator_wrapper.generate_text(query, chunk_texts)
                    temp_answer = Answer(
                        text=answer_text, query=query, pipeline=pipeline_name,
                        chunks=chunk_texts, score=0.0, latency_ms=0.0,
                    )
                    score_result = self._evaluator.evaluate(temp_answer)
                    pipeline_scores[pipeline_name] = score_result.composite
                    
                    row = {
                        "query": query,
                        "pipeline": pipeline_name,
                        "composite" : score_result.composite,
                    }
                    query_results.append(row)
            
            # Add winning pipeline
            if not retrieval_only and pipeline_scores:
                winning_pipeline = max(pipeline_scores, key=pipeline_scores.get)
                for row in query_results:
                    row["winning_pipeline"] = winning_pipeline
            
            results.extend(query_results)
        
        # Save CSV
        benchmark_df = pd.DataFrame(results)
        if save_csv:
            benchmark_df.to_csv(self._benchmark_path, index=False)
            self._log(f"Benchmark CSV saved: {self._benchmark_path}")
        
        # Train router
        if not retrieval_only and train_router:
            self._train_learned_router(benchmark_df)
        
        elapsed = round((time.time() - t0) * 1000)
        self._log(f"benchmark() done in {elapsed}ms")
        
        return benchmark_df
    
    # ─────────────────────────────────────────────────────────
    # 4. status()
    # ─────────────────────────────────────────────────────────
    
    def status(self):
        """Print current state of this MetaRAG project."""
        logs = self._read_logs()
        n_logs = len(logs)
        avg_score = (
            round(sum(r.get("score", 0) for r in logs) / n_logs, 3)
            if logs
            else 0
        )
        
        router_status = "learned" if (
            self._learned_router and self._learned_router.is_trained
        ) else "default"
        
        print(f"\n{'='*60}")
        print(f"  MetaRAG v{self.VERSION} — '{self.project}'")
        print(f"{'='*60}")
        print(f"  Fitted        : {'✅' if self._fitted else '❌'}")
        print(f"  Chunks        : {len(self._chunks) if self._chunks else 0}")
        print(f"  Chunk size    : {self.chunk_size}")
        print(f"  Chunk overlap : {self.chunk_overlap}")
        print(f"  k (retrieve)  : {self.k}")
        print(f"  Eval preset   : {self.eval_preset}")
        print(f"  Pipelines     : {len(self._pipelines)}")
        print(f"  Queries asked : {len(logs)}")
        print(f"  Avg score     : {avg_score}")
        print(f"  Router        : {router_status}")
        print(f"  Storage       : {self._base}/")
        print(f"{'='*60}\n")
    
    # ─────────────────────────────────────────────────────────
    # 5. leaderboard()
    # ─────────────────────────────────────────────────────────
    
    def leaderboard(self):
        """Show pipeline performance from logged queries."""
        logs = self._read_logs()
        
        if not logs:
            print("[MetaRAG] No queries logged yet. Run ask() first.")
            return
        
        from collections import defaultdict
        
        pipe_scores = defaultdict(list)
        for row in logs:
            pipeline = row.get("pipeline", "unknown")
            score = row.get("score", 0)
            pipe_scores[pipeline].append(score)
        
        ranked = sorted(
            pipe_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )
        
        print(f"\n{'='*60}")
        print(f"  Leaderboard — {len(logs)} queries")
        print(f"{'='*60}")
        print(f"{'Pipeline':<15} {'Queries':>8} {'Avg':>7} {'Max':>7}")
        print(f"{'─'*60}")
        
        for i, (name, scores) in enumerate(ranked):
            avg = round(sum(scores) / len(scores), 3)
            max_score = round(max(scores), 3)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"{medal} {name:<13} {len(scores):>8} {avg:>7.3f} {max_score:>7.3f}")
        
        print(f"{'='*60}\n")
    
    # ─────────────────────────────────────────────────────────
    # 6. analyze_query()
    # ─────────────────────────────────────────────────────────
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Diagnose query: length, complexity, keywords, expected pipeline.
        
        Args:
            query: query string
        
        Returns:
            dict with analysis
        """
        words = query.lower().split()
        
        # Simple complexity heuristic
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        complexity = "high" if avg_word_len > 6 else "medium" if avg_word_len > 4 else "low"
        
        # Extract keywords (no stopwords)
        stopwords = {"the", "a", "an", "and", "or", "is", "was", "what", "how", "why"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Suggest pipeline based on complexity
        if len(keywords) > 5:
            expected = "multiquery"
        elif avg_word_len > 6:
            expected = "hybrid"
        else:
            expected = "dense"
        
        return {
            "query": query,
            "length": len(query),
            "words": len(words),
            "avg_word_length": round(avg_word_len, 2),
            "complexity": complexity,
            "keywords": keywords[:5],
            "num_keywords": len(keywords),
            "expected_pipeline": expected,
        }
    
    # ─────────────────────────────────────────────────────────
    # 7. analyze_corpus()
    # ─────────────────────────────────────────────────────────
    
    def analyze_corpus(self) -> Dict[str, Any]:
        """
        Diagnose corpus: size, duplicates, stats.
        
        Returns:
            dict with analysis
        """
        if not self._chunks:
            return {"status": "not fitted"}
        
        total_chars = sum(len(chunk.text if hasattr(chunk, "text") else str(chunk)) for chunk in self._chunks)
        avg_chunk = total_chars / len(self._chunks) if self._chunks else 0
        
        # Simple duplicate detection (word overlap)
        texts = [
            chunk.text if hasattr(chunk, "text") else str(chunk) 
            for chunk in self._chunks
        ]
        
        duplicates = 0
        for i, t1 in enumerate(texts):
            for t2 in texts[i+1:]:
                overlap = len(set(t1.split()) & set(t2.split()))
                if overlap > len(set(t1.split())) * 0.8:
                    duplicates += 1
                    break
        
        return {
            "num_documents": len(self._chunks),
            "total_characters": total_chars,
            "avg_chunk_size": round(avg_chunk, 0),
            "min_chunk_size": min(len(texts), default=0),
            "max_chunk_size": max(len(texts), default=0),
            "estimated_duplicates": duplicates,
            "corpus_profile": self._corpus_profile,
        }
    
    # ─────────────────────────────────────────────────────────
    # 8. explain()
    # ─────────────────────────────────────────────────────────
    
    def explain(self, query: str) -> Dict[str, Any]:
        """
        Transparent explanation of routing decision for a query.
        
        Args:
            query: query string
        
        Returns:
            dict with routing explanation
        """
        self._check_fitted()
        
        # Get routing decision
        if self._learned_router and self._learned_router.is_trained:
            features = self._extract_query_features(query)
            pipeline_name = self._learned_router.route(features)
            explanation = self._learned_router.explain(pipeline_name)
            confidence = "learned"
        else:
            pipeline_name = "hybrid"
            explanation = "default (router not yet trained)"
            confidence = "default"
        
        # Get query analysis
        query_analysis = self.analyze_query(query)
        
        return {
            "query": query,
            "selected_pipeline": pipeline_name,
            "confidence": confidence,
            "explanation": explanation,
            "query_analysis": query_analysis,
            "available_pipelines": list(self._pipelines.keys()),
        }
    
    # ─────────────────────────────────────────────────────────
    # 9. save()
    # ─────────────────────────────────────────────────────────
    
    def save(self) -> "MetaRAG":
        """Save config, router state, benchmark data."""
        config = {
            "version": self.VERSION,
            "project": self.project,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_strategy": self.chunk_strategy,
            "k": self.k,
            "eval_preset": self.eval_preset,
            "fitted": self._fitted,
            "num_chunks": len(self._chunks) if self._chunks else 0,
        }
        
        with open(self._config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        self._log(f"Config saved → {self._config_path}")
        return self
    
    # ─────────────────────────────────────────────────────────
    # 10. load()
    # ─────────────────────────────────────────────────────────
    
    @classmethod
    def load(cls, project: str, embeddings, generator, vector_db=None) -> "MetaRAG":
        """
        Load MetaRAG from saved state.
        
        Args:
            project: project name
            embeddings: embedding model
            generator: LLM generator
            vector_db: optional vector DB (defaults to InMemory)
        
        Returns:
            loaded MetaRAG instance
        """
        config_path = f".metarag/{project}/config.json"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        rag = cls(
            docs=[],  # dummy, will load from cache
            embeddings=embeddings,
            generator=generator,
            project=project,
            vector_db=vector_db,
            chunk_size=config.get("chunk_size", 500),
            chunk_overlap=config.get("chunk_overlap", 50),
            chunk_strategy=config.get("chunk_strategy", "recursive"),
            k=config.get("k", 4),
            eval_preset=config.get("eval_preset", "balanced"),
        )
        
        # Mark as fitted (will load chunks, profile from cache in fit())
        rag._fitted = config.get("fitted", False)
        
        print(f"[MetaRAG] Loaded project: {project}")
        return rag
    
    # ─────────────────────────────────────────────────────────
    # 11. get_benchmark_data()
    # ─────────────────────────────────────────────────────────
    
    def get_benchmark_data(self) -> pd.DataFrame:
        """
        Return benchmark CSV as DataFrame.
        Users can train their own models on this data.
        
        Returns:
            DataFrame with all benchmark results
        """
        if not os.path.exists(self._benchmark_path):
            raise FileNotFoundError(f"Benchmark CSV not found. Run benchmark() first.")
        
        return pd.read_csv(self._benchmark_path)
    
    # ─────────────────────────────────────────────────────────
    # 12. get_router_thresholds()
    # ─────────────────────────────────────────────────────────
    
    def get_router_thresholds(self) -> Dict[str, Any]:
        """
        Return learned router thresholds (for inspection).
        
        Returns:
            dict of thresholds per pipeline
        """
        if self._learned_router is None:
            return {}
        
        return self._learned_router.get_thresholds()
    
    # ─────────────────────────────────────────────────────────
    # Configuration Methods (optional)
    # ─────────────────────────────────────────────────────────
    
    def set_llm(self, generator) -> "MetaRAG":
        """Replace LLM generator."""
        if not hasattr(generator, "generate"):
            raise TypeError("generator must have .generate(prompt) method")
        self.generator = generator
        self._log("LLM generator updated")
        return self
    
    def set_embeddings(self, embeddings) -> "MetaRAG":
        """Replace embeddings model."""
        if not hasattr(embeddings, "embed"):
            raise TypeError("embeddings must have .embed(text) method")
        self.embeddings = embeddings
        self._log("Embeddings model updated")
        return self
    
    def rebuild(self, force: bool = True) -> "MetaRAG":
        """Force rebuild of index and pipelines."""
        self._log("Rebuilding index and pipelines...")
        return self.fit(force=force)
    
    def set_router(self, router) -> "MetaRAG":
        """
        Replace the routing strategy. Accepts anything satisfying
        RouterInterface's contract: .route(features: dict) -> str

        Args:
            router: a Router, LearnedRuleRouter, or any user-supplied object
                    with a .route(features) method — e.g. a wrapper around
                    a model trained on rag.get_benchmark_data()

        Example:
            df = rag.get_benchmark_data()
            clf = RandomForestClassifier().fit(X, y)
            rag.set_router(MyRouterAdapter(clf, pipeline_names))
        """
        if not hasattr(router, "route"):
            raise TypeError("router must have a .route(features) method")

        self._router = router
        self._log(f"Router updated → {router.__class__.__name__}")
        return self
    
    # ═════════════════════════════════════════════════════════
    # INTERNAL (private, not exposed)
    # ═════════════════════════════════════════════════════════
    
    def _check_fitted(self):
        """Ensure fit() was called."""
        if not self._fitted:
            raise RuntimeError("Call fit() before ask()")
    
    def _log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[MetaRAG] {msg}")
    
    # ─────────────────────────────────────────────────────────
    # Internal Setup
    # ─────────────────────────────────────────────────────────
    
    def _load_docs_and_chunk(self, force: bool = False):
        """Load documents and chunk them."""
        from .core.loader import DocumentLoader
        from .core.chunking import Chunker
        
        self._log("Loading documents...")
        docs = DocumentLoader(self.docs_path).load()
        
        if not docs:
            raise ValueError(f"No documents found at '{self.docs_path}'")
        
        self._log(f"{len(docs)} documents loaded")
        
        self._log("Chunking...")
        chunker = Chunker(
            strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        
        self._chunks = chunker.chunk_documents(
            docs,
            cache_dir=f"{self._cache_dir}/chunks",
            force=force,
        )
        
        self._log(f"{len(self._chunks)} chunks ready")
    
    def _build_vector_index(self, force: bool = False):
        self._log("Embedding chunks...")
        chunk_texts = [c.text if hasattr(c, "text") else str(c) for c in self._chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)  # embed text

        self._log("Building vector index...")
        self.vector_db.build(self._chunks, chunk_embeddings)  # ← store Chunk objects, not chunk_texts
    
    def _build_corpus_profile(self, force: bool = False):
        """Build corpus statistics profile."""
        self._log("Profiling corpus...")
        
        chunk_texts = [
            chunk.text if hasattr(chunk, "text") else str(chunk)
            for chunk in self._chunks
        ]
        
        total_chars = sum(len(t) for t in chunk_texts)
        avg_chunk = total_chars / len(chunk_texts) if chunk_texts else 0
        
        self._corpus_profile = {
            "num_chunks": len(self._chunks),
            "total_characters": total_chars,
            "avg_chunk_size": avg_chunk,
            "strategy": self.chunk_strategy,
        }
        
        with open(self._profile_path, "w") as f:
            json.dump(self._corpus_profile, f, indent=2)
    

    def _setup_retrievers(self):
        from .core.retriever import BM25Retriever, DenseRetriever, HybridRetriever, MMRRetriever

        # Pass Chunk objects directly — retrievers extract text internally as needed,
        # and now return real Chunk objects (with metadata) instead of bare strings.
        self._retrievers["bm25"] = BM25Retriever(self._chunks)
        self._retrievers["dense"] = DenseRetriever(self._chunks, self.embeddings, self.vector_db)
        self._retrievers["hybrid"] = HybridRetriever(self._chunks, self.embeddings, self.vector_db)
        self._retrievers["mmr"] = MMRRetriever(self._chunks, self.embeddings, self.vector_db)

        self._log(f"{len(self._retrievers)} retrievers ready (chunk metadata preserved end-to-end)")
    
    def _setup_pipelines(self):
        """Initialize pipeline combinations."""
        from .pipelines.pipeline import (
            StraightPipeline,
            MultiQueryPipeline,
            RerankedPipeline,
            FullPipeline,
        )
        from .pipelines.pipeline import Reranker
        
        # Basic pipelines (always available)
        self._pipelines["straight"] = StraightPipeline(self._retrievers["bm25"])
        self._pipelines["dense"] = StraightPipeline(self._retrievers["dense"])
        self._pipelines["hybrid"] = StraightPipeline(self._retrievers["hybrid"])
        self._pipelines["mmr"] = StraightPipeline(self._retrievers["mmr"])
        
        # Advanced pipelines (with optional reranker)
        try:
            reranker = Reranker()
            self._pipelines["reranked"] = RerankedPipeline(
                self._retrievers["hybrid"],
                reranker
            )
            self._pipelines["full"] = FullPipeline(
                self._retrievers["hybrid"],
                self.generator,
                reranker,
                n_variants=2
            )
        except ImportError:
            self._log("⚠️  sentence-transformers not available (reranked pipelines disabled)")
        
        # MultiQuery (uses generator)
        self._pipelines["multiquery"] = MultiQueryPipeline(
            self._retrievers["hybrid"],
            self.generator,
            n_variants=2
        )
    
    def _setup_evaluator(self):
        """Initialize evaluator with 5 metrics."""
        from .Evaluator.evaluator import Evaluator
        
        self._evaluator = Evaluator(self.embeddings, preset=self.eval_preset)
        self._log("Evaluator ready")

    
    # ─────────────────────────────────────────────────────────
    # Internal Features
    # ─────────────────────────────────────────────────────────
    
    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features for router decision."""
        # Simple features (can be expanded with profilers)
        words = query.lower().split()
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        
        return {
            "query_length": len(query),
            "num_words": len(words),
            "avg_word_length": avg_word_len,
            "num_chunks": len(self._chunks) if self._chunks else 0,
        }
    
    # ─────────────────────────────────────────────────────────
    # Internal Router Training
    # ─────────────────────────────────────────────────────────
    
    def _train_learned_router(self, benchmark_df: pd.DataFrame):
        """
        Trains the default LearnedRuleRouter on benchmark data.
        This is the automatic Mode 2 path — a user can always override
        the result afterward via set_router() with their own implementation.
        """
        from .router.learned_rule_router import LearnedRuleRouter
        router = LearnedRuleRouter(self._base)
        router.train(benchmark_df)
        self._router = router
    
    # ─────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────
    
    def _write_log(self, answer: Answer):
        """Log query result to JSONL."""
        row = {
            "query": answer.query,
            "pipeline": answer.pipeline,
            "score": answer.score,
            "latency_ms": answer.latency_ms,
            "timestamp": time.time(),
        }
        
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        
        self._query_logs.append(row)
    
    def _read_logs(self) -> List[Dict]:
        """Read all logged queries."""
        if not os.path.exists(self._log_path):
            return []
        
        rows = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except:
                        continue
        
        return rows
    
    def __repr__(self):
        return (
            f"MetaRAG(project='{self.project}', "
            f"fitted={self._fitted}, "
            f"chunks={len(self._chunks) if self._chunks else 0}, "
            f"pipelines={len(self._pipelines)}, "
            f"router={'learned' if (self._learned_router and self._learned_router.is_trained) else 'default'})"
        )