# metarag.py
# MetaRAG — Intelligent Pipeline Selection Engine
#
# Usage:
#   from langchain_ollama import OllamaEmbeddings, ChatOllama
#   from metarag import MetaRAG
#
#   embeddings = OllamaEmbeddings(model="nomic-embed-text")
#   llm        = ChatOllama(model="mistral", temperature=0)
#
#   rag = MetaRAG(docs="./docs", embeddings=embeddings, llm=llm)
#   rag.fit()
#   answer = rag.ask("what is the refund policy?")
#   print(answer.text)

from __future__ import annotations
import os
import sys
import json
import time
from typing import Union, List, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Answer
# ─────────────────────────────────────────────────────────────

@dataclass
class Answer:
    text:       str
    query:      str
    pipeline:   str
    chunks:     list
    score:      float
    latency_ms: float
    features:   dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"\n{'='*55}\n"
            f"  Query    : {self.query}\n"
            f"  Pipeline : {self.pipeline}\n"
            f"  Score    : {self.score:.2f}\n"
            f"  Latency  : {self.latency_ms:.0f}ms\n"
            f"{'─'*55}\n"
            f"  {self.text}\n"
            f"{'='*55}"
        )


# ─────────────────────────────────────────────────────────────
# MetaRAG
# ─────────────────────────────────────────────────────────────

class MetaRAG:
    """
    MetaRAG — Intelligent Pipeline Selection Engine

    User is responsible for setting up embedding model and LLM.

    Usage:
        from langchain_ollama import OllamaEmbeddings, ChatOllama
        from metarag import MetaRAG

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm        = ChatOllama(model="mistral", temperature=0)

        rag = MetaRAG(
            docs       = "./docs",
            embeddings = embeddings,
            llm        = llm,
        )
        rag.fit()
        answer = rag.ask("what is the refund policy?")
        print(answer.text)
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        docs:        Union[str, List[str]],
        embeddings,                           # user provides this
        llm,                                  # user provides this
        project:     str  = "default",
        chunk_size:  int  = 500,
        k:           int  = 4,
        eval_preset: str  = "balanced",
        verbose:     bool = True,
    ):
        if embeddings is None:
            raise ValueError(
                "embeddings is required.\n"
                "Example:\n"
                "  from langchain_ollama import OllamaEmbeddings\n"
                "  embeddings = OllamaEmbeddings(model='nomic-embed-text')"
            )
        if llm is None:
            raise ValueError(
                "llm is required.\n"
                "Example:\n"
                "  from langchain_ollama import ChatOllama\n"
                "  llm = ChatOllama(model='mistral', temperature=0)"
            )

        self.docs_path   = docs
        self.embeddings  = embeddings
        self.llm         = llm
        self.project     = project
        self.chunk_size  = chunk_size
        self.k           = k
        self.eval_preset = eval_preset
        self.verbose     = verbose

        # ── storage ──────────────────────────────────────────
        self._base         = os.path.join(".metarag", project)
        self._index_dir    = os.path.join(self._base, "index")
        self._cache_dir    = os.path.join(self._base, "cache")
        self._logs_dir     = os.path.join(self._base, "logs")
        self._profile_path = os.path.join(self._base, "corpus_profile.json")
        self._log_path     = os.path.join(self._logs_dir, "queries.jsonl")

        for d in [self._index_dir, self._cache_dir, self._logs_dir]:
            os.makedirs(d, exist_ok=True)

        # ── internal state ────────────────────────────────────
        self._db             = None
        self._chunks         = None
        self._corpus_profile = None
        self._pipelines      = None
        self._router         = None
        self._generator      = None
        self._evaluator      = None
        self._fitted         = False

        self._log(f"MetaRAG v{self.VERSION} — project='{project}'")

    # ─────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────

    def fit(self, force: bool = False) -> "MetaRAG":
        """
        Load documents → chunk → index → profile → ready.

        Args:
            force : rebuild everything from scratch
        """
        t0 = time.time()
        self._log("fit() starting...")

        self._load_docs_and_index(force=force)
        self._load_corpus_profile(force=force)
        self._setup_pipelines()
        self._setup_router()
        self._setup_generator()
        self._setup_evaluator()

        self._fitted = True
        self._log(f"fit() done in {round((time.time()-t0)*1000)}ms — ready.")
        return self

    # ─────────────────────────────────────────────────────────
    # ASK
    # ─────────────────────────────────────────────────────────

    def ask(self, query: str) -> Answer:
        """
        Ask a question. Returns an Answer object.

        Args:
            query : your question string

        Returns:
            Answer with .text .pipeline .score .chunks .latency_ms
        """
        self._check_fitted()
        t0 = time.time()

        decision = self._router.route(query)
        pipeline  = decision["pipeline"]
        features  = decision["features"]

        result = self._pipelines[pipeline].run(query)
        chunks = result["chunks"]

        gen_answer = self._generator.generate(
            query    = query,
            chunks   = chunks,
            pipeline = pipeline,
        )

        score = self._evaluator.evaluate(gen_answer)

        answer = Answer(
            text       = gen_answer.text,
            query      = query,
            pipeline   = pipeline,
            chunks     = chunks,
            score      = score.composite,
            latency_ms = round((time.time() - t0) * 1000, 2),
            features   = features,
        )

        self._write_log(answer, score)
        return answer

    # ─────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────

    def status(self):
        """Print current state of this MetaRAG project."""
        logs      = self._read_logs()
        n_logs    = len(logs)
        avg_score = round(
            sum(r["score"] for r in logs) / n_logs, 3
        ) if logs else 0

        print(f"\n{'='*50}")
        print(f"  MetaRAG v{self.VERSION} — '{self.project}'")
        print(f"{'='*50}")
        print(f"  Fitted        : {'✅' if self._fitted else '❌ run fit()'}")
        print(f"  Chunks        : {len(self._chunks) if self._chunks else 0}")
        print(f"  Chunk size    : {self.chunk_size}")
        print(f"  k             : {self.k}")
        print(f"  Eval preset   : {self.eval_preset}")
        print(f"  Queries logged: {n_logs}")
        print(f"  Avg score     : {avg_score}")
        print(f"  Router        : rule-based")
        print(f"  Storage       : {self._base}/")
        print(f"{'='*50}\n")

    # ─────────────────────────────────────────────────────────
    # LEADERBOARD
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
            pipe_scores[row["pipeline"]].append(row["score"])

        ranked = sorted(
            pipe_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )

        print(f"\n{'='*50}")
        print(f"  Leaderboard — {len(logs)} queries logged")
        print(f"{'='*50}")
        print(f"{'Pipeline':<14} {'Queries':>8} {'Avg':>7} {'Best':>7}")
        print(f"{'─'*50}")

        for i, (name, scores) in enumerate(ranked):
            avg   = round(sum(scores) / len(scores), 3)
            best  = round(max(scores), 3)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"{medal} {name:<12} {len(scores):>8} {avg:>7.3f} {best:>7.3f}")

        print(f"{'='*50}\n")

    # ─────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────

    def save(self) -> "MetaRAG":
        """Save config to .metarag/<project>/config.json"""
        path = os.path.join(self._base, "config.json")
        config = {
            "version":    self.VERSION,
            "project":    self.project,
            "chunk_size": self.chunk_size,
            "k":          self.k,
            "eval_preset":self.eval_preset,
            "docs_path":  self.docs_path
                          if isinstance(self.docs_path, str)
                          else list(self.docs_path),
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        self._log(f"config saved → {path}")
        return self

    @classmethod
    def load(cls, path: str, embeddings, llm) -> "MetaRAG":
        """
        Restore MetaRAG from saved config.
        User must pass the same embeddings and llm objects.

        Args:
            path       : path to .metarag/<project>/ folder
            embeddings : your embedding model
            llm        : your LLM
        """
        config_path = os.path.join(path, "config.json") \
                      if os.path.isdir(path) else path

        with open(config_path) as f:
            config = json.load(f)

        rag = cls(
            docs        = config["docs_path"],
            embeddings  = embeddings,
            llm         = llm,
            project     = config["project"],
            chunk_size  = config["chunk_size"],
            k           = config["k"],
            eval_preset = config["eval_preset"],
        )
        rag.fit()
        return rag

    # ─────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────

    def _load_docs_and_index(self, force: bool = False):
        sys.path.extend(["./core", "./pipelines", "./Evaluator", "./router"])

        from core.loader    import DocumentLoader
        from core.chunking  import Chunker
        from core.vector_db import VectorDB

        self._log("loading documents...")
        docs = DocumentLoader(self.docs_path).load_all()
        if not docs:
            raise ValueError(f"No documents found at '{self.docs_path}'")
        self._log(f"{len(docs)} documents loaded")

        self._log("chunking...")
        self._chunks = Chunker(
            strategy   = "recursive",
            chunk_size = self.chunk_size,
            overlap    = 50,
        ).chunk_documents(
            docs,
            cache_dir = os.path.join(self._cache_dir, "chunks"),
            force     = force,
        )
        self._log(f"{len(self._chunks)} chunks ready")

        self._log("building vector index...")
        self._db = VectorDB(
            self.embeddings,
            db_type           = "chroma",
            persist_directory = self._index_dir,
        )
        self._db.build(self._chunks, force=force)

    def _load_corpus_profile(self, force: bool = False):
        from router.corpus_profiler import CorpusProfiler

        if not force and os.path.exists(self._profile_path):
            with open(self._profile_path) as f:
                self._corpus_profile = json.load(f)
            self._log("corpus profile loaded from cache")
        else:
            self._corpus_profile = CorpusProfiler().profile(self._chunks)
            with open(self._profile_path, "w") as f:
                json.dump(self._corpus_profile, f, indent=2)
            self._log("corpus profile saved")

    def _setup_pipelines(self):
        from core.retriever import get_retriever
        from pipelines.pipeline  import Pipeline, Reranker, MultiQuery, HyDE

        bm25_ret   = get_retriever("bm25",   chunks=self._chunks,  k=self.k)
        dense_ret  = get_retriever("dense",  vectordb=self._db,    k=self.k)
        hybrid_ret = get_retriever("hybrid", vectordb=self._db,
                                   chunks=self._chunks,             k=self.k)
        mmr_ret    = get_retriever("mmr",    vectordb=self._db,    k=self.k)
        reranker   = Reranker()

        self._pipelines = {
            "straight":   Pipeline(retriever=bm25_ret),
            "dense":      Pipeline(retriever=dense_ret),
            "hybrid":     Pipeline(retriever=hybrid_ret),
            "reranked":   Pipeline(retriever=dense_ret, reranker=reranker),
            "mmr":        Pipeline(retriever=mmr_ret),
            "multiquery": Pipeline(retriever=hybrid_ret,
                                   multiquery=MultiQuery(self.llm, n=2)),
            "hyde":       Pipeline(retriever=dense_ret,
                                   hyde=HyDE(self.llm)),
        }
        self._log(f"{len(self._pipelines)} pipelines ready")

    def _setup_router(self):
        from router.selector import Router
        self._router = Router(self._db, self._corpus_profile)
        self._log("router ready")

    def _setup_generator(self):
        from pipelines.generator import GeneratorWrapper
        self._generator = GeneratorWrapper(self.llm)
        self._log("generator ready")

    def _setup_evaluator(self):
        from Evaluator.evaluator import Evaluator
        self._evaluator = Evaluator(self.embeddings, preset=self.eval_preset)
        self._log("evaluator ready")

    def _write_log(self, answer: Answer, score):
        row = {
            "query":        answer.query,
            "pipeline":     answer.pipeline,
            "score":        answer.score,
            "faithfulness": score.faithfulness,
            "relevancy":    score.relevancy,
            "precision":    score.precision_avg,
            "coverage":     score.coverage,
            "redundancy":   score.redundancy,
            "latency_ms":   answer.latency_ms,
            "features":     answer.features,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _read_logs(self) -> list:
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

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Call fit() before ask().\n"
                "  rag.fit()"
            )

    def _log(self, msg: str):
        if self.verbose:
            print(f"[MetaRAG] {msg}")

    def __repr__(self):
        return (
            f"MetaRAG(project='{self.project}', "
            f"fitted={self._fitted}, "
            f"chunks={len(self._chunks) if self._chunks else 0})"
        )