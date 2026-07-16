# MetaRAG Sample Knowledge Base

## Overview

MetaRAG is a modular Retrieval-Augmented Generation framework for
experimenting with document loading, chunking, embeddings, retrieval,
routing, benchmarking, and evaluation.

## Artificial Intelligence

Artificial Intelligence (AI) focuses on building systems capable of
reasoning, learning, planning, and assisting humans. Machine Learning is
a subset of AI, while Deep Learning is a subset of Machine Learning.

## Retrieval-Augmented Generation

A RAG pipeline generally consists of: 1. Document Loading 2. Chunking 3.
Embedding Generation 4. Vector Indexing 5. Retrieval 6. Optional
Reranking 7. Prompt Construction 8. Answer Generation 9. Evaluation

## Evaluation Metrics

-   Precision
-   Coverage
-   Faithfulness
-   Relevancy
-   Latency
-   Redundancy

## Example Notes

Different chunking strategies can affect retrieval quality. Hybrid
retrieval combines lexical and semantic search. Benchmarking multiple
pipelines enables data-driven router training.

## Sample Table

  Pipeline     Strength                 Weakness
  ------------ ------------------------ ------------------
  Straight     Fast                     Simple
  MultiQuery   Better Recall            More LLM Calls
  HyDE         Strong Semantic Recall   Extra Generation
  Reranked     Higher Precision         Extra Compute
  Full         Highest Quality          Highest Cost

## Conclusion

This document is intentionally verbose so chunkers generate multiple
chunks and retrievers have sufficient semantic diversity during testing.
