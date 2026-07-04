# metarag/core/chunking.py

import hashlib
import os
import pickle
import re
from typing import List, Literal
from abc import ABC, abstractmethod


# AFTER
class Chunk:
    """Represents a chunk of text, carrying forward the parent Document's metadata."""

    def __init__(
        self,
        text: str,
        source: str = None,
        start_idx: int = 0,
        end_idx: int = None,
        metadata: dict = None,
    ):
        self.text = text
        self.source = source
        self.start_idx = start_idx
        self.end_idx = end_idx or len(text)
        # Full metadata dict from the parent Document (page, type, row, custom fields, etc.)
        # 'source' is duplicated into metadata too, so callers can rely on
        # chunk.metadata.get("source") uniformly instead of chunk.source directly.
        self.metadata = dict(metadata) if metadata else {}
        if source and "source" not in self.metadata:
            self.metadata["source"] = source

    def __repr__(self):
        return f"Chunk({len(self.text)} chars, source={self.source})"

    def __str__(self):
        return self.text


class ChunkerInterface(ABC):
    """Contract for chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, source: str = None) -> List[Chunk]:
        """Chunk a single text."""
        pass


class Chunker(ChunkerInterface):
    """
    Split documents into chunks using 6 strategies.
    No LangChain, no external deps (except optional nltk).
    """
    
    STRATEGIES = ["fixed", "recursive", "semantic", "sentence", "sliding_window", "markdown"]
    
    def __init__(
        self,
        strategy: Literal["fixed", "recursive", "semantic", "sentence", "sliding_window", "markdown"] = "recursive",
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: {self.STRATEGIES}")
        
        self.strategy = strategy.lower()
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(
        self,
        documents: List,
        cache_dir: str = None,
        force: bool = False
    ) -> List[Chunk]:
        """
        Chunk documents using configured strategy.
        
        Args:
            documents: list of Document objects
            cache_dir: optional cache directory
            force: if True, ignore cache
        
        Returns:
            list of Chunk objects
        """
        # Check cache
        if cache_dir and not force:
            cached = self._load_cache(documents, cache_dir)
            if cached is not None:
                return cached
        
        # Chunk
        chunks = []

        for doc in documents:

            doc_chunks = self.chunk(
                doc.text,
                source=doc.metadata.get("source"),
                metadata=doc.metadata,   # pass the FULL dict, not just source
            )

            chunks.extend(doc_chunks)
        
        # Save cache
        if cache_dir:
            self._save_cache(documents, chunks, cache_dir)
        
        return chunks
    
    # AFTER
    def chunk(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        if self.strategy == "fixed":
            return self._chunk_fixed(text, source, metadata)
        elif self.strategy == "recursive":
            return self._chunk_recursive(text, source, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, source, metadata)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text, source, metadata)
        elif self.strategy == "sliding_window":
            return self._chunk_sliding_window(text, source, metadata)
        elif self.strategy == "markdown":
            return self._chunk_markdown(text, source, metadata)
    
    # ─────────────────────────────────────────────────────────
    # Strategy 1: Fixed Size
    # ─────────────────────────────────────────────────────────
    
    def _chunk_fixed(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text.strip()) > 10:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    start_idx=i,
                    end_idx=i + len(chunk_text),
                    metadata=metadata,
                ))
        return chunks
    
    # ─────────────────────────────────────────────────────────
    # Strategy 2: Recursive (splits on separators hierarchically)
    # ─────────────────────────────────────────────────────────
    
    def _chunk_recursive(self, text: str, source: str = None , metadata: dict = None) -> List[Chunk]:
        """Recursive chunking: split on para → sentence → chars."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, 0, source , metadata)
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        depth: int,
        source: str,
        metadata:None
    ) -> List[Chunk]:
        """Recursively split using separators."""
        if depth >= len(separators):
            return self._chunk_fixed(text, source , metadata)
        
        sep = separators[depth]
        chunks = []
        good_chunks = []
        
        for sub_text in text.split(sep):
            if len(sub_text) > self.chunk_size:
                # Too long, recurse
                sub_chunks = self._recursive_split(sub_text, separators, depth + 1, source , metadata)
                good_chunks.extend(sub_chunks)
            else:
                if sub_text.strip():
                    good_chunks.append(sub_text)
        
        # Merge small chunks
        current = ""
        for chunk_text in good_chunks:
            if len(current) + len(chunk_text) + len(sep) < self.chunk_size:
                current += sep + chunk_text if current else chunk_text
            else:
                if current.strip():
                    chunks.append(Chunk(text=current, source=source , metadata=metadata))
                current = chunk_text
        
        if current.strip():
            chunks.append(Chunk(text=current, source=source , metadata=metadata))
        
        return chunks
    
    # ─────────────────────────────────────────────────────────
    # Strategy 3: Semantic (split on sentence, group by similarity)
    # ─────────────────────────────────────────────────────────
    
    def _chunk_semantic(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) < self.chunk_size:
                current += " " + sent if current else sent
            else:
                if current.strip():
                    chunks.append(Chunk(text=current, source=source, metadata=metadata))
                current = sent

        if current.strip():
            chunks.append(Chunk(text=current, source=source, metadata=metadata))

        return chunks
    
    # ─────────────────────────────────────────────────────────
    # Strategy 4: Sentence (split on sentence boundaries)
    # ─────────────────────────────────────────────────────────
    
    def _chunk_sentence(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 < self.chunk_size:
                current += " " + sent if current else sent
            else:
                if current.strip():
                    chunks.append(Chunk(text=current, source=source, metadata=metadata))
                current = sent

        if current.strip():
            chunks.append(Chunk(text=current, source=source, metadata=metadata))

        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (tries nltk, falls back to regex)."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            return nltk.sent_tokenize(text)
        except ImportError:
            # Fallback: regex sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    # ─────────────────────────────────────────────────────────
    # Strategy 5: Sliding Window (fixed overlap, no gaps)
    # ─────────────────────────────────────────────────────────
    
    def _chunk_sliding_window(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]

            if len(chunk_text.strip()) > 10:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=source,
                    start_idx=i,
                    end_idx=i + len(chunk_text),
                    metadata=metadata,
                ))

            if i + self.chunk_size >= len(text):
                break

        return chunks
    
    # ─────────────────────────────────────────────────────────
    # Strategy 6: Markdown (split on headers)
    # ─────────────────────────────────────────────────────────
    
    def _chunk_markdown(self, text: str, source: str = None, metadata: dict = None) -> List[Chunk]:
        """
        Split Markdown on headers. Now also tracks the current heading level
        and text in each chunk's metadata (fixes the earlier "structural benefit"
        gap flagged in review — headers were previously discarded entirely).
        """
        chunks = []
        header_pattern = r'^(#{1,6})\s+(.*)$'

        current = ""
        current_heading = None
        current_heading_level = None

        for line in text.splitlines():
            header_match = re.match(header_pattern, line)

            if header_match:
                # Flush what we've accumulated under the previous heading
                if current.strip():
                    chunk_metadata = dict(metadata) if metadata else {}
                    if current_heading:
                        chunk_metadata["heading"] = current_heading
                        chunk_metadata["heading_level"] = current_heading_level
                    chunks.append(Chunk(text=current, source=source, metadata=chunk_metadata))

                current_heading_level = len(header_match.group(1))
                current_heading = header_match.group(2).strip()
                current = ""
            else:
                if len(current) + len(line) < self.chunk_size:
                    current += ("\n" + line if current else line)
                else:
                    chunk_metadata = dict(metadata) if metadata else {}
                    if current_heading:
                        chunk_metadata["heading"] = current_heading
                        chunk_metadata["heading_level"] = current_heading_level
                    chunks.append(Chunk(text=current, source=source, metadata=chunk_metadata))
                    current = line

        if current.strip():
            chunk_metadata = dict(metadata) if metadata else {}
            if current_heading:
                chunk_metadata["heading"] = current_heading
                chunk_metadata["heading_level"] = current_heading_level
            chunks.append(Chunk(text=current, source=source, metadata=chunk_metadata))

        return chunks
    
    # ─────────────────────────────────────────────────────────
    # Caching
    # ─────────────────────────────────────────────────────────
    
    def _cache_key(self, documents: List) -> str:
        """Generate cache key from documents."""
        doc_str = "".join(doc.text[:100] for doc in documents)
        config_str = f"{self.strategy}_{self.chunk_size}_{self.overlap}"
        combined = doc_str + config_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _save_cache(self, documents: List, chunks: List[Chunk], cache_dir: str):
        """Save chunks to cache."""
        os.makedirs(cache_dir, exist_ok=True)
        key = self._cache_key(documents)
        path = os.path.join(cache_dir, f"{key}.pkl")
        
        try:
            with open(path, "wb") as f:
                pickle.dump(chunks, f)
            print(f"[Chunker] {len(chunks)} chunks cached → {key[:8]}...")
        except Exception as e:
            print(f"[Chunker] Cache save error: {e}")
    
    def _load_cache(self, documents: List, cache_dir: str):
        """Load chunks from cache if exists."""
        if not os.path.exists(cache_dir):
            return None
        
        key = self._cache_key(documents)
        path = os.path.join(cache_dir, f"{key}.pkl")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "rb") as f:
                chunks = pickle.load(f)
            print(f"[Chunker] Cache hit — {len(chunks)} chunks loaded")
            return chunks
        except Exception as e:
            print(f"[Chunker] Cache load error: {e}")
            return None