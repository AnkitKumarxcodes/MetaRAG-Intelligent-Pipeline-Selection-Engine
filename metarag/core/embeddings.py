# embeddings.py

import os
import hashlib
import numpy as np


# ─────────────────────────────────────────────────────────────
# Embedding Cache
# Wraps any embedding model — same text = instant disk lookup
# ─────────────────────────────────────────────────────────────

class CachedEmbeddings:
    """
    Wraps any LangChain embedding model with a local disk cache.
    First call → embeds + saves. Every call after → loads from disk.

    Works transparently as a drop-in for any LangChain component
    that expects an embedding model.
    """

    def __init__(self, model, cache_dir: str = "./metarag_cache/embeddings"):
        self.model     = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, text: str) -> str:
        key = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")

    def embed_query(self, text: str):
        path = self._path(text)
        if os.path.exists(path):
            return np.load(path).tolist()
        embedding = self.model.embed_query(text)
        np.save(path, np.array(embedding))
        return embedding

    def embed_documents(self, texts: list) -> list:
        return [self.embed_query(t) for t in texts]

    # ── LangChain compatibility ──────────────────────────────
    # These make CachedEmbeddings work anywhere the raw model works

    def __getattr__(self, name):
        """Fall through to underlying model for anything not defined here."""
        return getattr(self.model, name)


# ─────────────────────────────────────────────────────────────
# Factory — always returns a cached embedding model
# ─────────────────────────────────────────────────────────────

def get_embedding(name: str, cache_dir: str = "./metarag_cache/embeddings"):
    """
    Returns a cached embedding model.
    First query to each unique text is embedded and saved.
    Subsequent queries for the same text load from disk instantly.

    Options:
        "nomic"  — Ollama nomic-embed-text (free, local)
        "bge"    — HuggingFace BAAI/bge-small-en (free, local)
        "openai" — OpenAI text-embedding-3-small (paid)
    """
    name = name.lower()

    if name == "nomic":
        from langchain_ollama import OllamaEmbeddings
        model = OllamaEmbeddings(model="nomic-embed-text")

    elif name == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    elif name == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = OpenAIEmbeddings()

    else:
        raise ValueError(f"Unsupported embedding: '{name}'. Choose from: nomic, bge, openai")

    return CachedEmbeddings(model, cache_dir=cache_dir)