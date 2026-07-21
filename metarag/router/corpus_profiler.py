# router/corpus_profiler.py
# Runs once at fit() time. Profiles the document corpus.
# Answers: what kind of corpus is this?

from __future__ import annotations
import os
import re
import json
from typing import List, Any


def _text(chunk) -> str:
    return getattr(chunk, "page_content", None) or getattr(chunk, "text", "")


class CorpusProfiler:
    """
    Profiles a corpus of chunks at fit() time.
    Result is saved to disk and loaded on every ask().

    Signals produced:
        num_docs         →  how large is the corpus
        avg_chunk_length →  how dense is the text
        ocr_ratio        →  is this scanned / noisy text
        duplicate_ratio  →  how much repetition exists
        numeric_ratio    →  is this log-like / data-heavy
        short_doc_ratio  →  are docs very short (logs, FAQs)
    """

    def profile(self, chunks: List[Any]) -> dict:
        if not chunks:
            return {}

        texts   = [_text(c) for c in chunks if _text(c).strip()]
        lengths = [len(t) for t in texts]
        ocr_texts = [
            _text(c)
            for c in chunks
            if _text(c).strip() and self._is_ocr_candidate(c)
        ]

        return {
            "num_docs":         len(texts),
            "avg_chunk_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "ocr_ratio":         (
                                    round(self._ocr_ratio(ocr_texts), 3)
                                    if ocr_texts else None
                                ),
            "duplicate_ratio":  round(self._duplicate_ratio(texts), 3),
            "numeric_ratio":    round(self._numeric_ratio(texts), 3),
            "short_doc_ratio":  round(self._short_doc_ratio(lengths), 3),
        }
    
    def _is_ocr_candidate(self, chunk) -> bool:
        doc_type = getattr(chunk, "metadata", {}).get("type", "").lower()
        return doc_type in {
            "pdf",
            "image",
            "scanned_pdf",
        }

    def _ocr_ratio(self, texts: List[str]) -> float:
        noisy = 0

        for t in texts:
            words = t.split()
            if not words:
                continue

            single_chars = sum(1 for w in words if len(w) == 1)

            broken_words = len(
                            re.findall(r"(?:\b[a-zA-Z]\b\s+){2,}\b[a-zA-Z]\b", t)
                        )

            replacement_chars = t.count("�")

            punctuation_runs = len(re.findall(r"[^\w\s]{4,}", t))

            score = 0

            if single_chars / len(words) > 0.35:
                score += 1

            if broken_words > 5:
                score += 1

            if replacement_chars > 0:
                score += 2

            if punctuation_runs > 3:
                score += 1

            if score >= 2:
                noisy += 1

        return noisy / len(texts) if texts else 0.0

    def _duplicate_ratio(self, texts: List[str]) -> float:
        """How many chunks share the same first 80 chars."""
        seen  = set()
        dupes = 0
        for t in texts:
            key = t[:80].strip()
            if key in seen:
                dupes += 1
            seen.add(key)
        return dupes / len(texts) if texts else 0.0

    def _numeric_ratio(self, texts: List[str]) -> float:
        """High numeric density = logs, financial, structured data."""
        numeric = 0
        for t in texts:
            words = t.split()
            if not words:
                continue
            nums = sum(1 for w in words if re.search(r"\d", w))
            if nums / len(words) > 0.3:
                numeric += 1
        return numeric / len(texts) if texts else 0.0

    def _short_doc_ratio(self, lengths: List[int]) -> float:
        """Many short chunks = FAQs, logs, structured records."""
        short = sum(1 for l in lengths if l < 200)
        return short / len(lengths) if lengths else 0.0

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str, profile: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"[CorpusProfiler] Profile saved → {path}")

    def load(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No corpus profile at '{path}'. Run fit() first."
            )
        with open(path) as f:
            return json.load(f)
