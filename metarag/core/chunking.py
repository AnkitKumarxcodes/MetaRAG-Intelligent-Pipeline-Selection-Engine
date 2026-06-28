# chunking.py

from __future__ import annotations
from typing import Any, List, Optional
import re
from typing import List, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Data model for a chunk
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)   # source, strategy, index, etc.

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(len={len(self.text)}, preview='{preview}...')"


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────

class BaseChunker:
    name: str = "base"

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        raise NotImplementedError

    def chunk_many(self, docs: List[str], sources: Optional[List[str]] = None) -> List[Chunk]:
        """Chunk a list of documents, attaching source metadata."""
        all_chunks = []
        for i, doc in enumerate(docs):
            source = (sources[i] if sources and i < len(sources) else f"doc_{i}")
            all_chunks.extend(self.chunk(doc, source=source))
        return all_chunks

    def _make_chunk(self, text: str, source: str, index: int, **extra) -> Chunk:
        return Chunk(
            text=text.strip(),
            metadata={"source": source, "strategy": self.name, "index": index, **extra},
        )


# ─────────────────────────────────────────────────────────────
# 1. Fixed-Size Chunker
# ─────────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Splits text every N characters with optional overlap.
    Fast and simple — good baseline.
    """
    name = "fixed_size"

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        assert overlap < chunk_size, "overlap must be smaller than chunk_size"
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, source, idx,
                                               start=start, end=end))
                idx += 1
            start += self.chunk_size - self.overlap
        return chunks


# ─────────────────────────────────────────────────────────────
# 2. Sentence Chunker
# ─────────────────────────────────────────────────────────────

class SentenceChunker(BaseChunker):
    """
    Splits on sentence boundaries using NLTK.
    Groups N sentences per chunk.
    Install: pip install nltk
    """
    name = "sentence"

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            sentences = nltk.sent_tokenize(text)
        except ImportError:
            # Fallback: naive split on ". "
            sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        step = self.sentences_per_chunk - self.overlap_sentences
        for i in range(0, len(sentences), max(step, 1)):
            group = sentences[i: i + self.sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, source, len(chunks),
                                               sentence_start=i,
                                               sentence_end=i + len(group)))
        return chunks


# ─────────────────────────────────────────────────────────────
# 3. Recursive Chunker  (default recommendation)
# ─────────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Tries to split on paragraph → newline → sentence → word boundaries,
    only going smaller if the chunk still exceeds chunk_size.
    Best general-purpose default.
    Install: pip install langchain langchain-text-splitters
    """
    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=self.separators,
            )
            parts = splitter.split_text(text)
        except ImportError:
            # Fallback: mimic recursive splitting without langchain
            parts = self._fallback_split(text)

        return [self._make_chunk(p, source, i) for i, p in enumerate(parts) if p.strip()]

    def _fallback_split(self, text: str) -> List[str]:
        """Pure-Python fallback that mimics recursive splitting."""
        for sep in self.separators:
            if sep and sep in text:
                segments = text.split(sep)
                results = []
                current = ""
                for seg in segments:
                    if len(current) + len(seg) <= self.chunk_size:
                        current += sep + seg if current else seg
                    else:
                        if current:
                            results.append(current)
                        current = seg
                if current:
                    results.append(current)
                return results
        # Last resort: fixed-size
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]


# ─────────────────────────────────────────────────────────────
# 4. Semantic Chunker
# ─────────────────────────────────────────────────────────────

class SemanticChunker(BaseChunker):
    """
    Groups sentences by embedding similarity.
    A new chunk starts where cosine similarity drops sharply (topic shift).
    Install: pip install langchain-experimental langchain-openai
    """
    name = "semantic"

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold_type: str = "percentile",  # or "standard_deviation"
        breakpoint_threshold_amount: float = 90,
    ):
        self.embedding_model = embedding_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def _get_embeddings(self):
        if self.embedding_model:
            return self.embedding_model
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except ImportError:
            raise ImportError("Run: pip install langchain-openai  (or pass your own embedding_model)")

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
        except ImportError:
            raise ImportError("Run: pip install langchain-experimental")

        splitter = LCSemanticChunker(
            embeddings=self._get_embeddings(),
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
        )
        parts = splitter.split_text(text)
        return [self._make_chunk(p, source, i) for i, p in enumerate(parts) if p.strip()]


# ─────────────────────────────────────────────────────────────
# 5. Parent-Child Chunker
# ─────────────────────────────────────────────────────────────

class ParentChildChunker(BaseChunker):
    """
    Creates two levels of chunks:
      - Small child chunks → stored in vectorstore for precise retrieval
      - Large parent chunks → returned as context to the LLM

    Returns child chunks with 'parent_text' embedded in metadata.
    """
    name = "parent_child"

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 200,
        overlap: int = 20,
    ):
        self.parent_chunker = RecursiveChunker(chunk_size=parent_chunk_size, overlap=overlap)
        self.child_chunker  = RecursiveChunker(chunk_size=child_chunk_size,  overlap=overlap)

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        """Returns child chunks; each carries its parent text in metadata."""
        parents = self.parent_chunker.chunk(text, source=source)
        all_children = []
        for p_idx, parent in enumerate(parents):
            children = self.child_chunker.chunk(parent.text, source=source)
            for child in children:
                child.metadata.update({
                    "strategy":    self.name,
                    "parent_text": parent.text,
                    "parent_index": p_idx,
                })
                all_children.append(child)
        return all_children

    def chunk_with_parents(self, text: str, source: str = "") -> dict:
        """Returns both levels separately — useful for building retrievers."""
        parents  = self.parent_chunker.chunk(text, source=source)
        children = self.chunk(text, source=source)
        return {"parents": parents, "children": children}


# ─────────────────────────────────────────────────────────────
# 6. Proposition Chunker
# ─────────────────────────────────────────────────────────────

class PropositionChunker(BaseChunker):
    """
    Uses an LLM to decompose text into atomic, self-contained facts.
    Highest retrieval precision — but slow and costly.
    Install: pip install langchain-openai
    """
    name = "proposition"

    PROMPT = """Decompose the following text into a list of short, atomic, \
self-contained propositions. Each proposition must:
- Express exactly ONE fact or idea
- Be independently understandable without surrounding context
- Be a complete sentence

Return ONLY the propositions, one per line, with no numbering or bullets.

Text:
{text}
"""

    def __init__(self, llm=None, model: str = "gpt-4o-mini"):
        self.llm = llm
        self.model = model

    def _get_llm(self):
        if self.llm:
            return self.llm
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=self.model, temperature=0)
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        # Split into paragraphs first to keep LLM prompts manageable
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        llm = self._get_llm()
        all_props = []

        for para in paragraphs:
            if len(para) < 40:          # too short to decompose
                all_props.append(para)
                continue
            prompt = self.PROMPT.format(text=para)
            result = llm.invoke(prompt).content
            props = [line.strip() for line in result.splitlines() if line.strip()]
            all_props.extend(props)

        return [
            self._make_chunk(prop, source, i)
            for i, prop in enumerate(all_props)
            if prop
        ]


# ─────────────────────────────────────────────────────────────
# Unified Chunker — single interface for all strategies
# ─────────────────────────────────────────────────────────────

CHUNKER_REGISTRY = {
    "fixed":       FixedSizeChunker,
    "sentence":    SentenceChunker,
    "recursive":   RecursiveChunker,
    "semantic":    SemanticChunker,
    "parent_child": ParentChildChunker,
    "proposition": PropositionChunker,
}


class Chunker:
    """
    Unified interface for all chunking strategies.

    Usage:
        chunker = Chunker(strategy="recursive", chunk_size=500, overlap=50)
        chunks  = chunker.chunk(text)
        chunks  = chunker.chunk_many(docs)

    Supported strategies:
        "fixed"        — FixedSizeChunker
        "sentence"     — SentenceChunker
        "recursive"    — RecursiveChunker  (default)
        "semantic"     — SemanticChunker
        "parent_child" — ParentChildChunker
        "proposition"  — PropositionChunker
    """

    def __init__(self, strategy: str = "recursive", **kwargs):
        if strategy not in CHUNKER_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(CHUNKER_REGISTRY.keys())}"
            )
        self.strategy = strategy
        self._chunker: BaseChunker = CHUNKER_REGISTRY[strategy](**kwargs)

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        """Chunk a single document."""
        return self._chunker.chunk(text, source=source)

    def chunk_many(self, docs: List[str], sources: Optional[List[str]] = None) -> List[Chunk]:
        """Chunk multiple documents at once."""
        return self._chunker.chunk_many(docs, sources=sources)
    
    def chunk_documents(self, docs: List[Any], cache_dir: str = "./metarag_cache/chunks", force: bool = False) -> List[Chunk]:
        """
        Chunk Document objects while preserving metadata.
        Caches results to disk — same docs + same strategy = instant reload.

        Args:
            docs      : list of Document or str objects
            cache_dir : where to store cached chunks
            force     : if True, ignore cache and rechunk
        """
        import hashlib, pickle, os

        # cache key = strategy + hash of all doc text combined
        content_hash = hashlib.md5(
            "".join(
                getattr(d, "text", d) if not isinstance(d, str) else d
                for d in docs
            ).encode()
        ).hexdigest()
        cache_key  = f"{self.strategy}_{content_hash[:12]}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

        # return from cache if available and not forced
        if not force and os.path.exists(cache_path):
            print(f"[Chunker] Cache hit — loading chunks ({cache_key})...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # not cached — chunk fresh
        all_chunks = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                text, metadata = doc, {}
            else:
                text     = getattr(doc, "text", "")
                metadata = getattr(doc, "metadata", {})

            source     = metadata.get("source", f"doc_{i}")
            doc_chunks = self._chunker.chunk(text, source=source)

            for c in doc_chunks:
                for k, v in metadata.items():
                    if k not in ["strategy", "index"]:
                        c.metadata[k] = v

            all_chunks.extend(doc_chunks)

        # save to cache
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_chunks, f)
        print(f"[Chunker] {len(all_chunks)} chunks cached → {cache_path}")

        return all_chunks

    def stats(self, chunks: List[Chunk]) -> dict:
        """Return basic statistics about a set of chunks."""
        if not chunks:
            return {}
        lengths = [len(c.text) for c in chunks]
        return {
            "strategy":      self.strategy,
            "total_chunks":  len(chunks),
            "avg_length":    round(sum(lengths) / len(lengths)),
            "min_length":    min(lengths),
            "max_length":    max(lengths),
            "total_chars":   sum(lengths),
        }

    @staticmethod
    def available_strategies() -> List[str]:
        return list(CHUNKER_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge.
    It works by retrieving relevant documents from a knowledge base and passing them as context.

    The quality of RAG depends heavily on chunking strategy. Poor chunking leads to missing context
    or noisy retrieval. Different strategies suit different document types and query patterns.

    Semantic chunking groups sentences by meaning. It produces topically coherent chunks but
    requires embedding every sentence, making it slower and more expensive than simpler methods.
    """

    strategies = ["fixed", "sentence", "recursive", "parent_child"]

    for strategy in strategies:
        chunker = Chunker(strategy=strategy)
        chunks  = chunker.chunk(sample, source="demo.txt")
        stats   = chunker.stats(chunks)
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.upper()}")
        print(f"Stats:    {stats}")
        for c in chunks[:2]:
            print(f"  → {c}")