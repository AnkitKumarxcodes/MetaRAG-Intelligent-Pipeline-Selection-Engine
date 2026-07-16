import hashlib
from metarag.core.embeddings import EmbeddingInterface


class FakeEmbeddings(EmbeddingInterface):
    """
    Deterministic hash-based embedding model.

    Offline.
    No dependencies.
    Stable across machines.
    Produces meaningful cosine similarities.
    """

    model_name = "fake-embeddings"

    def __init__(self, dim=32):
        self.dim = dim

    def _embed(self, text: str):
        vec = [0.0] * self.dim

        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0

        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed(self, text):
        return self._embed(text)

    def embed_query(self, text):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]
    

from metarag.pipelines.generator import GeneratorInterface


# ============================================================
# Generator Fakes
# ============================================================

class FakeGenerator(GeneratorInterface):
    """
    Deterministic offline generator.

    Handles:
    - Normal QA
    - MultiQuery
    - HyDE
    """

    model_name = "fake-generator"

    def generate(self, prompt: str) -> str:

        prompt = prompt.lower()

        if "generate" in prompt:
            return (
                "Variant 1\n"
                "Variant 2\n"
                "Variant 3"
            )

        if "hypothetical" in prompt:
            return (
                "This is a hypothetical answer "
                "used for retrieval."
            )

        return (
            "This is a generated answer "
            "based on the retrieved context."
        )


class RetryGenerator(GeneratorInterface):
    """
    Fails the first N-1 calls,
    succeeds afterwards.
    Useful for retry testing.
    """

    def __init__(self, failures=2):

        self.calls = 0
        self.failures = failures

    def generate(self, prompt):

        self.calls += 1

        if self.calls <= self.failures:
            raise RuntimeError("Temporary failure")

        return "Recovered"


class AlwaysFailGenerator(GeneratorInterface):
    """
    Always raises an exception.
    """

    def generate(self, prompt):

        raise RuntimeError("Fatal Error")

# ============================================================
# Retriever Fake
# ============================================================

from metarag import Chunk


class FakeRetriever:
    """
    Deterministic retriever returning
    fixed ranked chunks.
    """

    def retrieve(self, query, k=4):

        chunks = [

            (Chunk(text="Artificial Intelligence overview"), 0.98),

            (Chunk(text="Machine Learning basics"), 0.93),

            (Chunk(text="Deep Learning introduction"), 0.87),

            (Chunk(text="Natural Language Processing"), 0.81),

            (Chunk(text="Computer Vision"), 0.74),

        ]

        return chunks[:k]

# ============================================================
# Reranker Fake
# ============================================================

class FakeReranker:

    def rerank(self, query, chunks, k=None):

        if k is None:
            k = len(chunks)

        reranked = sorted(
            chunks[:k],
            key=lambda x: len(
                x[0].text if hasattr(x[0], "text") else str(x[0])
            ),
            reverse=True,
        )

        return reranked

# ============================================================
# Router Fake
# ============================================================

class FakeRouter:
    """
    Always routes to the same pipeline.
    """

    is_trained = True

    def __init__(self, pipeline="hybrid"):

        self.pipeline = pipeline

    def route(self, features):

        return self.pipeline

# ============================================================
# Sklearn Fake
# ============================================================

class FakeSklearnModel:
    """
    Minimal sklearn-like model.

    Only implements predict().
    """

    def __init__(self, prediction="straight"):

        self.prediction = prediction

    def predict(self, X):

        return [self.prediction] * len(X)

# ============================================================
# Empty Retriever
# ============================================================

class EmptyRetriever:

    def retrieve(self, query, k=4):

        return []
