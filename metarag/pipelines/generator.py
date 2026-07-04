# metarag/pipeline/generator.py

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Any, Optional, Tuple
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────
# Answer
# ─────────────────────────────────────────────────────────────


def _chunk_text(chunk) -> str:
    """Extract text — supports (text, score) tuples, Chunk objects, or raw strings."""
    if isinstance(chunk, tuple):
        return chunk[0]
    if isinstance(chunk, str):
        return chunk
    return getattr(chunk, "text", None) or getattr(chunk, "page_content", "")


def build_prompt(query: str, chunks: List[Any]) -> str:
    """Format query + retrieved chunks into an LLM prompt."""
    context_parts = [f"[{i+1}] {_chunk_text(c).strip()}" for i, c in enumerate(chunks)]
    context = "\n\n".join(context_parts)

    return f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
Be concise and direct.

Context:
{context}

Question: {query}

Answer:"""


# ─────────────────────────────────────────────────────────────
# Generator Interface — user implements this
# ─────────────────────────────────────────────────────────────

class GeneratorInterface(ABC):
    """
    Contract for LLM generators.
    Implement this with any LLM: OpenAI, Anthropic, Gemini, Ollama, local, etc.
    
    Example:
        class MyGenerator(GeneratorInterface):
            def generate(self, prompt: str) -> str:
                response = my_llm_client.chat(prompt)
                return response.text
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: the full prompt string
        
        Returns:
            generated text (string)
        """
        pass


# ─────────────────────────────────────────────────────────────
# Native Implementation: Ollama (via direct HTTP, no LangChain)
# ─────────────────────────────────────────────────────────────

class OllamaGenerator(GeneratorInterface):
    """
    Direct Ollama API generator (no LangChain).
    Requires: Ollama running locally.
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url
        self.requests = requests

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. Run: ollama serve"
            ) from e

    def generate(self, prompt: str) -> str:
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()


# ─────────────────────────────────────────────────────────────
# GeneratorWrapper — orchestrates prompt building + timing + retries
# Wraps ANY GeneratorInterface implementation
# ─────────────────────────────────────────────────────────────

# BEFORE — GeneratorWrapper.generate() built its own Answer object
# AFTER — it returns (text, latency_ms) and lets the caller build whatever Answer it needs

class GeneratorWrapper:
    """
    Wraps any GeneratorInterface implementation.
    Handles prompt building, timing, and retries.
    Does NOT construct an Answer — that's the caller's responsibility,
    since different callers (MetaRAG.ask(), benchmark(), a Mode 1 user)
    need different Answer shapes.

    Args:
        generator: object implementing GeneratorInterface (duck-typed — needs .generate(prompt))
        model_name: optional label for logging/tracking
    """

    def __init__(self, generator, model_name: str = None):
        if not hasattr(generator, "generate"):
            raise TypeError(
                "generator must have a .generate(prompt) method. "
                "Implement GeneratorInterface or provide a compatible object."
            )
        self.generator = generator
        self.model_name = model_name or generator.__class__.__name__

    def generate_text(self, query: str, chunks: List[Any]) -> Tuple[str, float]:
        """
        Build prompt, call the generator, retry on rate limits.

        Returns:
            (generated_text, latency_ms)
        """
        prompt = build_prompt(query, chunks)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                text = self.generator.generate(prompt).strip()
                ms = round((time.time() - t0) * 1000, 2)
                return text, ms
            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    wait = 10 * (attempt + 1)
                    print(f"[Generator] Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == max_retries - 1:
                    raise e
                else:
                    print(f"[Generator] Attempt {attempt+1} failed: {e}. Retrying...")

        raise RuntimeError(f"[Generator] Failed after {max_retries} attempts")