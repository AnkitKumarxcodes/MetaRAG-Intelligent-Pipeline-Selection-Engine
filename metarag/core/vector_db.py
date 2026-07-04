# metarag/core/vector_db.py

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from retriever import _chunk_text


class VectorDBInterface(ABC):
    """
    Contract for vector database implementations.
    User can implement their own (Chroma, FAISS, Pinecone, Weaviate, etc).
    """
    
    @abstractmethod
    def build(self, chunks: List[str], embeddings: List[List[float]]) -> "VectorDBInterface":
        """
        Build index from chunks + embeddings.
        
        Args:
            chunks: list of text chunks
            embeddings: list of embedding vectors
        
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float]]:
        """
        Search for top k similar chunks.
        
        Args:
            query_embedding: embedding vector of query
            k: number of results
        
        Returns:
            list of (chunk_text, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def add(self, chunks: List[str], embeddings: List[List[float]]) -> "VectorDBInterface":
        """Add chunks to existing index."""
        pass
    
    @abstractmethod
    def save(self, path: str = None) -> "VectorDBInterface":
        """Persist index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str = None) -> "VectorDBInterface":
        """Load index from disk."""
        pass


# ─────────────────────────────────────────────────────────────
# Native Implementation: In-Memory Vector DB (no deps)
# ─────────────────────────────────────────────────────────────

class InMemoryVectorDB(VectorDBInterface):
    """
    Simple in-memory vector database.
    No external dependencies.
    Good for prototyping, testing, small corpora (<100K vectors).
    """
    
    def __init__(self):
        self.chunks = []
        self.embeddings = None  # numpy array
        self.ids = []
    
    def build(self, chunks: List[str], embeddings: List[List[float]]) -> "InMemoryVectorDB":
        """Build in-memory index."""
        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.ids = [f"doc_{i}" for i in range(len(chunks))]
        
        print(f"[InMemoryVectorDB] Built index with {len(chunks)} chunks")
        return self
    
    def search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float]]:
        """Search using cosine similarity. Zero-vector embeddings are guarded against."""
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build() first.")

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Guard against zero-vector chunk embeddings (empty/broken embeddings)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1e-8, norms)
        normalized = self.embeddings / safe_norms

        # Guard against a zero-vector query embedding too
        query_norm = np.linalg.norm(query_vec)
        query_norm = query_norm if query_norm != 0 else 1e-8
        query_normalized = query_vec / query_norm

        similarities = np.dot(normalized, query_normalized)

        # Any chunk that had a genuinely zero embedding gets forced to the bottom
        # of the ranking (similarity -inf) rather than participating in nan-driven
        # undefined ordering.
        zero_embedding_mask = (norms.flatten() == 0)
        similarities = np.where(zero_embedding_mask, -np.inf, similarities)

        top_indices = np.argsort(similarities)[::-1][:k]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]
    
    def add(self, chunks: List[str], embeddings: List[List[float]]) -> "InMemoryVectorDB":
        """Add chunks to index."""
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        start_id = len(self.ids)
        self.ids.extend([f"doc_{start_id + i}" for i in range(len(chunks))])
        
        return self
    
    def save(self, path: str = None) -> "InMemoryVectorDB":
        """In-memory DB doesn't persist (data lost on restart)."""
        print("[InMemoryVectorDB] Warning: In-memory DB doesn't persist. Data will be lost on restart.")
        return self
    
    def load(self, path: str = None) -> "InMemoryVectorDB":
        """Nothing to load from disk."""
        return self


# ─────────────────────────────────────────────────────────────
# Optional Implementation: Chroma Vector DB
# ─────────────────────────────────────────────────────────────

class ChromaVectorDB(VectorDBInterface):
    def __init__(self, persist_directory: str = ".metarag/index"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required for ChromaVectorDB. Install: pip install chromadb")

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        self._chunks_by_id = {}  # id -> original chunk object (Chunk or str)

    def build(self, chunks: List, embeddings: List[List[float]]) -> "ChromaVectorDB":
        batch_size = 41666

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_chunks))]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=[_chunk_text(c) for c in batch_chunks],   # Chroma needs plain strings
            )

            # Keep original objects so search() can reconstruct them, metadata intact
            for doc_id, chunk in zip(batch_ids, batch_chunks):
                self._chunks_by_id[doc_id] = chunk

        print(f"[ChromaVectorDB] Built index with {len(chunks)} chunks → {self.persist_directory}")
        return self

    def search(self, query_embedding: List[float], k: int = 4) -> List[Tuple]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["distances"],   # documents not needed — we reconstruct from _chunks_by_id
        )

        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        similarities = [1 / (1 + d) for d in distances]

        # Return ORIGINAL chunk objects (with metadata intact), not raw Chroma strings
        return [(self._chunks_by_id.get(doc_id, doc_id), sim) for doc_id, sim in zip(ids, similarities)]

    def add(self, chunks: List, embeddings: List[List[float]]) -> "ChromaVectorDB":
        start_id = self.collection.count()
        batch_ids = [f"doc_{start_id + i}" for i in range(len(chunks))]

        self.collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=[_chunk_text(c) for c in chunks],
        )
        for doc_id, chunk in zip(batch_ids, chunks):
            self._chunks_by_id[doc_id] = chunk

        return self
    
    def save(self, path: str = None) -> "ChromaVectorDB":
        """Chroma persists automatically."""
        return self
    
    def load(self, path: str = None) -> "ChromaVectorDB":
        """Chroma loads automatically."""
        return self


# ─────────────────────────────────────────────────────────────
# Optional Implementation: FAISS Vector DB
# ─────────────────────────────────────────────────────────────

class FAISSVectorDB(VectorDBInterface):
    """
    Fast vector database using FAISS.
    Optional dependency: faiss-cpu or faiss-gpu
    """
    
    def __init__(self):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss required for FAISSVectorDB. "
                "Install: pip install faiss-cpu (or faiss-gpu)"
            )
        
        self.faiss = faiss.faiss
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def build(self, chunks: List[str], embeddings: List[List[float]]) -> "FAISSVectorDB":
        """Build FAISS index."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dim = embeddings_array.shape[1]
        
        # Create flat index with L2 distance
        self.index = self.faiss.IndexFlatL2(dim)
        self.index.add(embeddings_array)
        
        self.chunks = chunks
        self.embeddings = embeddings_array
        
        print(f"[FAISSVectorDB] Built index with {len(chunks)} chunks")
        return self
    
    def search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float]]:
        """Search using FAISS."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        
        # Convert L2 distance to similarity
        similarities = [1 / (1 + d) for d in distances[0]]
        
        return [(self.chunks[int(i)], float(sim)) for i, sim in zip(indices[0], similarities)]
    
    def add(self, chunks: List[str], embeddings: List[List[float]]) -> "FAISSVectorDB":
        """Add chunks to FAISS."""
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        self.index.add(new_embeddings)
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        return self
    
    def save(self, path: str = ".metarag/faiss_index.bin") -> "FAISSVectorDB":
        """Save FAISS index to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.faiss.write_index(self.index, path)
        print(f"[FAISSVectorDB] Index saved to {path}")
        return self
    
    def load(self, path: str = ".metarag/faiss_index.bin") -> "FAISSVectorDB":
        """Load FAISS index from disk."""
        self.index = self.faiss.read_index(path)
        print(f"[FAISSVectorDB] Index loaded from {path}")
        return self