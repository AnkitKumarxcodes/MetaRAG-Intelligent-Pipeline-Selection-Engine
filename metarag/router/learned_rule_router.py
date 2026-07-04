import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from .router_interface import RouterInterface

class LearnedRuleRouter(RouterInterface):
    """Learn routing thresholds from benchmark data, apply interpretable rules."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.thresholds = {}
        self.rules = []
        self.is_trained = False
    
    def train(self, benchmark_df: pd.DataFrame) -> None:
        """Learn thresholds from benchmark data."""
        if len(benchmark_df) == 0:
            print("[LearnedRuleRouter] Empty benchmark, skipping training")
            return
        
        # Group by pipeline and extract winning patterns
        for pipeline in benchmark_df['pipeline'].unique():
            winning = benchmark_df[benchmark_df['pipeline'] == pipeline]
            
            # Learn threshold for each key feature from winning cases
            self.thresholds[pipeline] = {
                'max_similarity': float(winning['max_similarity'].quantile(0.5)),
                'avg_similarity': float(winning['avg_similarity'].quantile(0.5)),
                'redundancy': float(winning['redundancy'].quantile(0.5)),
                'query_length': float(winning['query_length'].quantile(0.5)),
                'num_docs': int(winning['num_docs'].quantile(0.5)),
                'win_rate': len(winning) / len(benchmark_df),
                'avg_composite': float(winning['composite'].mean()),
            }
        
        # Build priority order by win rate
        self.rules = sorted(
            self.thresholds.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        self.is_trained = True
        self.save()
        
        # Log training results
        print(f"[LearnedRuleRouter] Trained on {len(benchmark_df)} rows")
        print(f"[LearnedRuleRouter] Learned {len(self.thresholds)} pipelines")
        for pipeline, stats in self.rules[:3]:
            print(f"  {pipeline}: {stats['win_rate']:.1%} win rate, avg score {stats['avg_composite']:.2f}")
    
    def route(self, features: Dict) -> str:
        """Route query using learned thresholds."""
        if not self.is_trained:
            print("[LearnedRuleRouter] Not trained, falling back to 'hybrid'")
            return 'hybrid'
        
        max_sim = features.get('max_similarity', 0)
        redundancy = features.get('redundancy', 0)
        query_len = features.get('query_length', 0)
        
        # Rule 1: High similarity + low redundancy → use retrieval-focused
        if max_sim > self.thresholds.get('hybrid', {}).get('max_similarity', 0.7):
            return 'hybrid'
        
        # Rule 2: High redundancy → use diversity-focused
        if redundancy > self.thresholds.get('mmr', {}).get('redundancy', 0.55):
            return 'mmr'
        
        # Rule 3: Low similarity → use expansion-focused
        if max_sim < self.thresholds.get('multiquery', {}).get('max_similarity', 0.45):
            return 'multiquery'
        
        # Rule 4: Short queries might benefit from expansion
        if query_len <= self.thresholds.get('multiquery', {}).get('query_length', 5):
            return 'multiquery'
        
        # Fallback: return highest win-rate pipeline
        return self.rules[0][0] if self.rules else 'hybrid'
    
    def save(self) -> None:
        """Save learned thresholds to disk."""
        path = self.base_path / "router_thresholds.json"
        with open(path, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        print(f"[LearnedRuleRouter] Thresholds saved to {path}")
    
    def load(self) -> bool:
        """Load thresholds from disk. Returns True if successful."""
        path = self.base_path / "router_thresholds.json"
        if not path.exists():
            print(f"[LearnedRuleRouter] No saved thresholds at {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                self.thresholds = json.load(f)
            self.rules = sorted(
                self.thresholds.items(),
                key=lambda x: x[1].get('win_rate', 0),
                reverse=True
            )
            self.is_trained = True
            print(f"[LearnedRuleRouter] Loaded thresholds from {path}")
            return True
        except Exception as e:
            print(f"[LearnedRuleRouter] Failed to load: {e}")
            return False
    
    def explain(self, pipeline: str, features: Optional[Dict] = None) -> str:
        """Explain why a pipeline was chosen."""
        if pipeline not in self.thresholds:
            return f"Unknown pipeline: {pipeline}"
        
        t = self.thresholds[pipeline]
        explanation = (
            f"{pipeline}: "
            f"sim>{t['max_similarity']:.2f}, "
            f"redun<{t['redundancy']:.2f}, "
            f"({t['win_rate']:.1%} win rate)"
        )
        return explanation
    
    def get_stats(self) -> Dict:
        """Return training statistics."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "num_pipelines": len(self.thresholds),
            "thresholds": self.thresholds,
            "top_pipeline": self.rules[0][0] if self.rules else None,
            "top_win_rate": self.rules[0][1]['win_rate'] if self.rules else 0,
        }