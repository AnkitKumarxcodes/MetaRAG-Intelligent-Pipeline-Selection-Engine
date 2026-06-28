# evaluator/ragas_eval.py

from dataclasses import dataclass
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness


# ── Simple Data Models ─────────────────────────────────────

@dataclass
class EvalResult:
    pipeline: str
    question: str
    scores: dict
    overall_score: float


# ── Evaluator ─────────────────────────────────────────────

class RAGASEvaluator:
    """
    Lightweight RAGAS evaluator (optimized for MetaRAG)

    Focus:
    - fast
    - stable
    - usable for pipeline selection
    """

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

        # 🔥 Only essential metrics
        self.metrics = [
            answer_relevancy,
            faithfulness
        ]

    # ── Build dataset ─────────────────────────────────────

    def _build_dataset(self, query, result, ground_truth):
        return Dataset.from_list([{
            "question": query,
            "answer": result["answer"],
            # 🔥 reduce context size (important for speed)
            "contexts": [result["context"][:300]],
            "ground_truth": ground_truth
        }])

    # ── Evaluate one pipeline ─────────────────────────────

    def evaluate(
        self,
        query: str,
        result: dict,
        ground_truth: str,
        pipeline_name: str
    ) -> Optional[EvalResult]:

        dataset = self._build_dataset(query, result, ground_truth)

        try:
            scores = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )

            score_dict = scores.to_pandas().iloc[0].to_dict()

            # 🔥 handle NaN safely
            relevancy = score_dict.get("answer_relevancy") or 0
            faith = score_dict.get("faithfulness") or 0

            # 🔥 weighted score (you will tune later)
            overall = 0.7 * relevancy + 0.3 * faith

            return EvalResult(
                pipeline=pipeline_name,
                question=query,
                scores={
                    "answer_relevancy": relevancy,
                    "faithfulness": faith
                },
                overall_score=overall
            )

        except Exception as e:
            print(f"[Eval Error] {pipeline_name}: {e}")
            return None

    # ── Evaluate multiple pipelines ───────────────────────

    def evaluate_all(self, query, results, ground_truth):
        """
        results = {
            "SimpleRAG": {...},
            "MultiQueryRAG": {...},
            ...
        }
        """
        eval_results = []

        for name, result in results.items():
            res = self.evaluate(query, result, ground_truth, name)
            if res:
                eval_results.append(res)

        return eval_results

    # ── Select best pipeline ─────────────────────────────

    @staticmethod
    def select_best(eval_results):
        if not eval_results:
            return None

        return max(eval_results, key=lambda x: x.overall_score)