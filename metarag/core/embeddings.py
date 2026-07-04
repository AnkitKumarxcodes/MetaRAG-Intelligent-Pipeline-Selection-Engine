# metarag/core/embeddings.py

import os
import hashlib
import numpy as np
from typing import List
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────
# Embedding Interface Contract
# ─────────────────────────────────────────────────────────────

class EmbeddingInterface(ABC):
    """
    Contract that any embedding model must follow.
    
    Implement these methods and it works with MetaRAG.
    """
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: string to embed
        
        Returns:
            list of floats (embedding vector)
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple document strings (optimized for batch).
        
        Args:
            texts: list of strings to embed
        
        Returns:
            list of embedding vectors
        """
        pass
    
    def embed(self, text: str) -> List[float]:
        """Alias for embed_query()."""
        return self.embed_query(text)


# ─────────────────────────────────────────────────────────────
# Cached Embeddings Wrapper
# ─────────────────────────────────────────────────────────────

class CachedEmbeddings(EmbeddingInterface):
    """
    Wraps any embedding model with a local disk cache.
    
    First call → embeds + saves to disk.
    Subsequent calls for same text → loads from disk instantly.
    
    Works with ANY embedding object that has:
        - embed_query(text: str) → List[float]
        - embed_documents(texts: List[str]) → List[List[float]]
    
    Args:
        model: embedding object (duck-typed, no inheritance required)
        cache_dir: directory to store cached embeddings
    
    Example:
        class MyEmbedding:
            def embed_query(self, text): ...
            def embed_documents(self, texts): ...
        
        cached = CachedEmbeddings(MyEmbedding())
        vec = cached.embed("hello world")  # cached on disk
    """
    
    def __init__(self, model, cache_dir: str = ".metarag/embeddings"):
        # Duck typing: check for required methods, not inheritance
        required_methods = ("embed_query", "embed_documents")
        for method in required_methods:
            if not hasattr(model, method):
                raise TypeError(
                    f"Embedding model must have {required_methods} methods. "
                    f"Missing: {method}"
                )
        
        self.model = model
        self.cache_dir = cache_dir
        self.model_name = self._get_model_name()
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_model_name(self) -> str:
        """Get unique model identifier for cache key."""
        # Try common naming attributes
        if hasattr(self.model, "model_name"):
            return self.model.model_name
        if hasattr(self.model, "model"):
            return str(self.model.model)
        # Fall back to class name
        return self.model.__class__.__name__
    
    def _path(self, text: str) -> str:
        """Get cache file path for text (includes model name)."""
        # Hash both model name and text to avoid collisions
        cache_key = f"{self.model_name}:{text}"
        key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.npy")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text (with caching)."""
        path = self._path(text)
        
        # Check cache first
        if os.path.exists(path):
            return np.load(path).tolist()
        
        # Embed + cache
        embedding = self.model.embed_query(text)
        np.save(path, np.array(embedding))
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.
        
        Strategy:
        1. Check cache for each text
        2. Batch embed uncached texts (optimized inference)
        3. Save batch to cache
        4. Return all results in original order
        """
        cached = []
        uncached_indices = []
        uncached_texts = []
        
        # Separate cached from uncached
        for i, text in enumerate(texts):
            path = self._path(text)
            if os.path.exists(path):
                cached.append((i, np.load(path).tolist()))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Batch embed uncached texts (much faster)
        if uncached_texts:
            embeddings = self.model.embed_documents(uncached_texts)
            
            # Save to cache
            for text, embedding in zip(uncached_texts, embeddings):
                path = self._path(text)
                np.save(path, np.array(embedding))
            
            # Merge with cached results
            for idx, emb in zip(uncached_indices, embeddings):
                cached.append((idx, emb))
        
        # Return in original order
        cached.sort(key=lambda x: x[0])
        return [emb for _, emb in cached]
    
    def embed(self, text: str) -> List[float]:
        """Alias for embed_query()."""
        return self.embed_query(text)