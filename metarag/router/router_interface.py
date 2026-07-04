# router/router_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any


class RouterInterface(ABC):
    """
    Contract for any router pluggable into MetaRAG.

    Implement this with rule-based logic, learned thresholds, a trained
    ML classifier, or anything else — MetaRAG only ever calls .route(features).

    Example (wrapping your own sklearn model trained on rag.get_benchmark_data()):

        class MyRouter(RouterInterface):
            def __init__(self, model, pipeline_names):
                self.model = model
                self.pipeline_names = pipeline_names

            def route(self, features: dict) -> str:
                X = [[features.get(k, 0) for k in self.model.feature_names_in_]]
                pred = self.model.predict(X)[0]
                return self.pipeline_names[pred]
    """

    @abstractmethod
    def route(self, features: Dict[str, Any]) -> str:
        """
        Args:
            features: dict of corpus/query/probe signals
                      (see CorpusProfiler, QueryProfiler, ProbeProfiler)

        Returns:
            pipeline name (must match a key in MetaRAG._pipelines)
        """
        pass