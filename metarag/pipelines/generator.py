# generator.py

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Any, Optional


# ─────────────────────────────────────────────────────────────
# Answer — what every pipeline ultimately produces
# ─────────────────────────────────────────────────────────────

@dataclass
class Answer:
    """
    The final output of MetaRAG.
    Not just a string — carries everything downstream needs.

    text        → the actual answer
    query       → original question
    chunks      → what the LLM used to answer (for evaluation)
    pipeline    → which pipeline produced this
    model       → which LLM generated this
    latency_ms  → how long generation took
    """
    text:       str
    query:      str
    chunks:     List[Any]
    pipeline:   str
    model:      str
    latency_ms: float
    metadata:   dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"Answer(\n"
            f"  query      = '{self.query[:60]}'\n"
            f"  pipeline   = '{self.pipeline}'\n"
            f"  model      = '{self.model}'\n"
            f"  latency_ms = {self.latency_ms:.0f}\n"
            f"  answer     = '{self.text[:100]}...'\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────
# Prompt Builder
# One place where all prompts are defined
# ─────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: List[Any]) -> str:
    """
    Formats query + retrieved chunks into a clean LLM prompt.
    Chunks can be LangChain Documents or MetaRAG Chunk objects.
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        # handle both LangChain Document and MetaRAG Chunk
        text = getattr(chunk, "page_content", None) or getattr(chunk, "text", "")
        context_parts.append(f"[{i+1}] {text.strip()}")

    context = "\n\n".join(context_parts)

    return f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
Be concise and direct.

Context:
{context}

Question: {query}

Answer:"""


# ─────────────────────────────────────────────────────────────
# Base Generator
# ─────────────────────────────────────────────────────────────

class BaseGenerator:
    model_name: str = "base"

    def generate(self, query: str, chunks: List[Any], pipeline: str = "") -> Answer:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Ollama Generator — default, free, local
# ─────────────────────────────────────────────────────────────

class OllamaGenerator(BaseGenerator):
    """
    Uses a local Ollama model to generate answers.
    Free, private, runs on your machine.

    Install: ollama pull llama3
    """

    def __init__(self, model: str = "llama3", temperature: float = 0.0):
        from langchain_ollama import ChatOllama
        self.llm        = ChatOllama(model=model, temperature=temperature)
        self.model_name = model

    def generate(self, query: str, chunks: List[Any], pipeline: str = "") -> Answer:
        if not chunks:
            print("[Generator] Warning — no chunks provided")

        prompt = build_prompt(query, chunks)

        print(f"[OllamaGenerator] Generating with {self.model_name}...")
        t0   = time.time()
        text = self.llm.invoke(prompt).content.strip()
        ms   = round((time.time() - t0) * 1000, 2)

        print(f"[OllamaGenerator] Done in {ms}ms")

        return Answer(
            text       = text,
            query      = query,
            chunks     = chunks,
            pipeline   = pipeline,
            model      = self.model_name,
            latency_ms = ms,
        )


# ─────────────────────────────────────────────────────────────
# Groq Generator — free tier, fast
# ─────────────────────────────────────────────────────────────

class GroqGenerator(BaseGenerator):
    """
    Uses Groq API for fast inference.
    Free tier available — very fast.

    Install: pip install langchain-groq
    """

    def __init__(self, model: str = "llama3-8b-8192", api_key: Optional[str] = None):
        from langchain_groq import ChatGroq
        self.llm        = ChatGroq(model=model, api_key=api_key)
        self.model_name = model

    def generate(self, query: str, chunks: List[Any], pipeline: str = "") -> Answer:
        prompt = build_prompt(query, chunks)

        print(f"[GroqGenerator] Generating with {self.model_name}...")
        t0   = time.time()
        text = self.llm.invoke(prompt).content.strip()
        ms   = round((time.time() - t0) * 1000, 2)

        print(f"[GroqGenerator] Done in {ms}ms")

        return Answer(
            text       = text,
            query      = query,
            chunks     = chunks,
            pipeline   = pipeline,
            model      = self.model_name,
            latency_ms = ms,
        )


# ─────────────────────────────────────────────────────────────
# OpenAI Generator — paid, optional
# ─────────────────────────────────────────────────────────────

class OpenAIGenerator(BaseGenerator):
    """
    Uses OpenAI GPT models.
    Paid — only use if needed.

    Install: pip install langchain-openai
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from langchain_openai import ChatOpenAI
        self.llm        = ChatOpenAI(model=model, api_key=api_key)
        self.model_name = model

    def generate(self, query: str, chunks: List[Any], pipeline: str = "") -> Answer:
        prompt = build_prompt(query, chunks)

        print(f"[OpenAIGenerator] Generating with {self.model_name}...")
        t0   = time.time()
        text = self.llm.invoke(prompt).content.strip()
        ms   = round((time.time() - t0) * 1000, 2)

        print(f"[OpenAIGenerator] Done in {ms}ms")

        return Answer(
            text       = text,
            query      = query,
            chunks     = chunks,
            pipeline   = pipeline,
            model      = self.model_name,
            latency_ms = ms,
        )


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def get_generator(name: str = "ollama", **kwargs) -> BaseGenerator:
    """
    Get a generator by name.

    Options:
        "ollama"  — local, free (default)
        "groq"    — API, free tier
        "openai"  — API, paid
    """
    name = name.lower()

    if name == "ollama":
        return OllamaGenerator(**kwargs)
    elif name == "groq":
        return GroqGenerator(**kwargs)
    elif name == "openai":
        return OpenAIGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator '{name}'. Choose from: ollama, groq, openai")


# ─────────────────────────────────────────────────────────────
# GeneratorWrapper — accepts any LangChain LLM directly
# used by metarag.py so user passes their own llm object
# ─────────────────────────────────────────────────────────────

class GeneratorWrapper:
    """
    Wraps any LangChain chat model (Ollama, Groq, OpenAI etc.)
    so MetaRAG can call .generate() without knowing the provider.
    """

    def __init__(self, llm):
        self.llm        = llm
        self.model_name = getattr(llm, "model", "custom")

    def generate(self, query: str, chunks: List[Any], pipeline: str = "") -> Answer:
        prompt = build_prompt(query, chunks)

        for attempt in range(3):
            try:
                t0   = time.time()
                text = self.llm.invoke(prompt).content.strip()
                ms   = round((time.time() - t0) * 1000, 2)
                return Answer(
                    text       = text,
                    query      = query,
                    chunks     = chunks,
                    pipeline   = pipeline,
                    model      = self.model_name,
                    latency_ms = ms,
                )
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f"[Generator] Rate limited — waiting 10s...")
                    time.sleep(10)
                else:
                    raise e

        raise RuntimeError("[Generator] Failed after 3 attempts")