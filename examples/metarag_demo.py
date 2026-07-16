# examples/metarag_demo.py
"""Full MetaRAG walkthrough — fit, ask, benchmark, and every observability output."""

from pathlib import Path

from metarag import MetaRAG, OllamaGenerator

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

from metarag import CachedEmbeddings
from metarag.utils import FakeEmbeddings , FakeGenerator

    
embeddings = CachedEmbeddings(FakeEmbeddings())

rag = MetaRAG(
    docs=str(DATA_DIR),
    embeddings=embeddings,
    generator=FakeGenerator(),
    project="metarag_demo",
    k=3,
)

print("\n### fit() ###")
rag.fit()

print("\n### ask() ###")
answer = rag.ask("What is the main topic of this document?")
print(answer)

QUERIES = [
    "What is the main topic of this document?",
    "Summarize the key points.",
    "What numbers or figures are mentioned?",
]

print("\n### benchmark() ###")
df = rag.benchmark(QUERIES, retrieval_only=True, train_router=True, save_csv=True)
print(f"Benchmark rows: {len(df)}  ->  {rag._benchmark_path}")

print("\n### status() ###")
rag.status()

print("\n### leaderboard() ###")
rag.leaderboard()

print("\n### dashboard() ###")
rag.dashboard()

print("\n### report() ###")
rag.report()

print("\n### pipeline_graph('hybrid') ###")
rag.pipeline_graph("hybrid")

print("\n### inspect() ###")
rag.inspect(QUERIES[0], k=2)

print("\n### trace() ###")
rag.trace(QUERIES[0], pipeline_name="full")

print("\n### explain() ###")
print(rag.explain(QUERIES[0]))

print("\n### analyze_query() / analyze_corpus() ###")
print(rag.analyze_query(QUERIES[0]))
print(rag.analyze_corpus())

print("\n### save() ###")
rag.save()

print(f"\nDone. Files at: {rag._base}")