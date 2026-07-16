from pathlib import Path

from metarag import (
    DocumentLoader,
    Chunker,
    CachedEmbeddings,
    InMemoryVectorDB,
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    MMRRetriever,
)

from metarag.pipelines.pipeline import (
    StraightPipeline,
    MultiQueryPipeline,
    HyDEPipeline,
    RerankedPipeline,
    FullPipeline,
    Reranker,
)


# ============================================================
# Embedding Wrapper
# ============================================================

from metarag.utils import FakeEmbeddings


# ============================================================
# Load Documents
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"

loader = DocumentLoader(DATA_DIR)

documents = loader.load(verbose=False)

print("=" * 70)
print("Pipeline API Demo")
print("=" * 70)

print(f"\nDocuments Loaded : {len(documents)}")


# ============================================================
# Chunking
# ============================================================

chunker = Chunker(strategy="recursive")

chunks = chunker.chunk_documents(documents)

print(f"Chunks Generated : {len(chunks)}")


# ============================================================
# Embeddings
# ============================================================

embedder = CachedEmbeddings(
    FakeEmbeddings()
)

embeddings = embedder.embed_documents(
    [c.text for c in chunks]
)

print(f"Embeddings       : {len(embeddings)}")


# ============================================================
# Vector DB
# ============================================================

vectordb = InMemoryVectorDB()

vectordb.build(chunks, embeddings)

print("Vector Index     : Built")


# ============================================================
# Retriever
# ============================================================

retriever = DenseRetriever(
    chunks,
    embedder,
    vectordb,
)


# ============================================================
# Generator
# ============================================================
from metarag.utils import FakeGenerator

generator = FakeGenerator()


reranker = Reranker()

query = "What is the main topic of this document?"


# ============================================================
# Pipelines
# ============================================================

pipelines = [

    StraightPipeline(retriever),

    MultiQueryPipeline(
        retriever,
        generator,
    ),

    HyDEPipeline(
        retriever,
        generator,
    ),

    RerankedPipeline(
        retriever,
        reranker,
    ),

    FullPipeline(
        retriever,
        generator,
        reranker,
    ),

]


for pipe in pipelines:

    print("\n" + "=" * 70)

    print(f"Pipeline: {pipe.name}")

    print("=" * 70)

    result = pipe.run(

        query,

        k=3,

    )

    print(f"Pipeline   : {result['pipeline']}")

    print(f"Query      : {result['query']}")

    if result["hypothesis"]:

        print("\nHypothesis")

        print("-" * 40)

        print(result["hypothesis"])

    print("\nRetrieved Chunks")

    print("-" * 40)

    for i, (chunk, score) in enumerate(

        result["chunks"],

        start=1,

    ):

        if hasattr(chunk, "text"):

            text = chunk.text

        else:

            text = str(chunk)

        print(

            f"{i}. score={score:.4f}"

        )

        print(

            text[:120].replace("\n", " ")

        )

        print()


print("=" * 70)

print("Done.")

print("=" * 70)