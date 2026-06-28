# vector_db.py

from __future__ import annotations
import os
from typing import List, Any, Dict


class VectorDB:
    """
    Minimal vector database handler for MetaRAG.

    Responsibilities:
        - Build vector DB from chunks
        - Load existing vector DB from disk
        - Add new chunks incrementally
        - Save / reset the DB

    Supports:
        - Chroma  (persistent, auto-saves to disk)
        - FAISS   (in-memory, manual save/load)

    Does NOT handle:
        - retrieval logic
        - search modes
        - pipeline selection
        - evaluation

    Usage:
        from langchain_ollama import OllamaEmbeddings
        from vector_db import VectorDB

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        db = VectorDB(embeddings, db_type="chroma", persist_directory="./metarag_db")
        db.build(chunks)
        db.info()
    """

    SUPPORTED_DB = ("chroma", "faiss")

    def __init__(
        self,
        embedding_model,
        db_type: str = "chroma",
        persist_directory: str = "./metarag_db",
    ):
        if db_type.lower() not in self.SUPPORTED_DB:
            raise ValueError(
                f"Unsupported db_type '{db_type}'. Choose from: {self.SUPPORTED_DB}"
            )

        self.embedding_model   = embedding_model
        self.db_type           = db_type.lower()
        self.persist_directory = persist_directory
        self.db                = None
        self._chunk_count      = 0

    # ─────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────

    def build(self, chunks: List[Any], force: bool = False) -> "VectorDB":
        """
        Build a new vector DB from a list of Chunk objects.

        If the DB already exists on disk and force=False,
        loads from disk instead of rebuilding — saves minutes.

        Args:
            chunks : List of Chunk objects with .text and .metadata
            force  : if True, always rebuild even if DB exists

        Returns:
            self — for method chaining.
        """
        if not chunks:
            raise ValueError("Cannot build VectorDB from empty chunk list.")

        # ── skip rebuild if already exists ───────────────────
        if not force and self.db_type == "chroma":
            if os.path.exists(self.persist_directory):
                print(f"[VectorDB] Found existing Chroma DB — loading instead of rebuilding.")
                print(f"[VectorDB] Pass force=True to rebuild from scratch.")
                return self.load()

        if not force and self.db_type == "faiss":
            if os.path.exists(self.persist_directory):
                print(f"[VectorDB] Found existing FAISS index — loading instead of rebuilding.")
                return self.load()

        texts     = [c.text for c in chunks]
        metadatas = [getattr(c, "metadata", {}) for c in chunks]

        if self.db_type == "chroma":
            from langchain_community.vectorstores import Chroma

            self.db = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                persist_directory=self.persist_directory,
            )

        elif self.db_type == "faiss":
            from langchain_community.vectorstores import FAISS

            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
            )

        self._chunk_count = len(texts)
        print(f"[VectorDB] Built {self.db_type.upper()} — {self._chunk_count} chunks indexed.")
        return self

    # ─────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────

    def load(self) -> "VectorDB":
        """
        Load an existing vector DB from disk.
        - Chroma: loads from persist_directory automatically.
        - FAISS:  loads from persist_directory using load_local().

        Returns:
            self — for method chaining.
        """
        if self.db_type == "chroma":
            from langchain_community.vectorstores import Chroma

            if not os.path.exists(self.persist_directory):
                raise FileNotFoundError(
                    f"No Chroma DB found at '{self.persist_directory}'. "
                    f"Call build() first."
                )

            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )
            print(f"[VectorDB] Loaded Chroma DB from '{self.persist_directory}'.")

        elif self.db_type == "faiss":
            from langchain_community.vectorstores import FAISS

            if not os.path.exists(self.persist_directory):
                raise FileNotFoundError(
                    f"No FAISS index found at '{self.persist_directory}'. "
                    f"Call build() and save_faiss() first."
                )

            self.db = FAISS.load_local(
                self.persist_directory,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            print(f"[VectorDB] Loaded FAISS index from '{self.persist_directory}'.")

        return self

    # ─────────────────────────────────────────
    # SAVE (FAISS only)
    # ─────────────────────────────────────────

    def save_faiss(self) -> "VectorDB":
        """
        Save FAISS index to disk.
        Chroma auto-persists — no manual save needed.

        Returns:
            self — for method chaining.
        """
        self._check_initialized()

        if self.db_type != "faiss":
            print("[VectorDB] Chroma auto-persists to disk — no manual save needed.")
            return self

        os.makedirs(self.persist_directory, exist_ok=True)
        self.db.save_local(self.persist_directory)
        print(f"[VectorDB] FAISS index saved to '{self.persist_directory}'.")
        return self

    # ─────────────────────────────────────────
    # ADD (incremental)
    # ─────────────────────────────────────────

    def add(self, chunks: List[Any]) -> "VectorDB":
        """
        Add new chunks to an already-built or loaded DB.
        Useful for incremental ingestion without rebuilding.

        Args:
            chunks: New Chunk objects to add.

        Returns:
            self — for method chaining.
        """
        self._check_initialized()

        if not chunks:
            print("[VectorDB] No chunks provided to add.")
            return self

        texts     = [c.text for c in chunks]
        metadatas = [getattr(c, "metadata", {}) for c in chunks]

        self.db.add_texts(texts=texts, metadatas=metadatas)
        self._chunk_count += len(texts)
        print(f"[VectorDB] Added {len(texts)} chunks. Total: {self._chunk_count}.")
        return self

    # ─────────────────────────────────────────
    # GET RAW DB (for retriever.py)
    # ─────────────────────────────────────────

    def get_db(self):
        """
        Return the raw vector store object.
        Used by retriever.py to run searches against.
        """
        self._check_initialized()
        return self.db

    # ─────────────────────────────────────────
    # INFO
    # ─────────────────────────────────────────

    def info(self) -> Dict[str, Any]:
        """Return a dict of basic DB metadata."""
        return {
            "db_type":           self.db_type,
            "persist_directory": self.persist_directory,
            "initialized":       self.db is not None,
            "chunk_count":       self._chunk_count,
        }

    def stats(self):
        """Print a readable summary of the current DB state."""
        i = self.info()
        print(f"\n[VectorDB] ─────────────────────────────")
        print(f"  Type        : {i['db_type'].upper()}")
        print(f"  Initialized : {i['initialized']}")
        print(f"  Chunks      : {i['chunk_count']}")
        print(f"  Directory   : {i['persist_directory']}")
        print(f"────────────────────────────────────────\n")

    # ─────────────────────────────────────────
    # DELETE / RESET
    # ─────────────────────────────────────────

    def delete_collection(self):
        """
        Permanently delete the Chroma collection from disk.
        WARNING: This cannot be undone.
        """
        if self.db_type != "chroma":
            raise ValueError("delete_collection() is only supported for Chroma.")

        self._check_initialized()
        self.db.delete_collection()
        self.db = None
        self._chunk_count = 0
        print(f"[VectorDB] Chroma collection permanently deleted.")

    def reset(self):
        """
        Clear the in-memory DB reference.
        Does NOT delete anything from disk.
        Call load() to restore from disk after reset.
        """
        self.db = None
        self._chunk_count = 0
        print("[VectorDB] In-memory reference cleared. Disk data preserved.")

    # ─────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────

    def _check_initialized(self):
        if self.db is None:
            raise ValueError(
                "VectorDB is not initialized. Call build() or load() first."
            )

    def __repr__(self):
        return (
            f"VectorDB(type={self.db_type}, "
            f"initialized={self.db is not None}, "
            f"chunks={self._chunk_count})"
        )


# ─────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass, field

    # Minimal Chunk mock for demo
    @dataclass
    class Chunk:
        text: str
        metadata: dict = field(default_factory=dict)

    sample_chunks = [
        Chunk("Einstein developed the theory of relativity in 1905.",
              {"source": "physics.txt", "strategy": "recursive"}),
        Chunk("Special relativity deals with objects moving at constant speed.",
              {"source": "physics.txt", "strategy": "recursive"}),
        Chunk("General relativity extends this to include gravity.",
              {"source": "physics.txt", "strategy": "recursive"}),
    ]

    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    except Exception:
        print("[Demo] Ollama not available — skipping live demo.")
        embeddings = None

    if embeddings:
        # Chroma
        db = VectorDB(embeddings, db_type="chroma", persist_directory="./demo_db")
        db.build(sample_chunks)
        db.stats()

        # Add more
        extra = [Chunk("GPS satellites must account for relativistic effects.",
                       {"source": "physics.txt", "strategy": "recursive"})]
        db.add(extra)
        db.stats()

        # Reset and reload
        db.reset()
        db.load()
        db.stats()
        print(db)