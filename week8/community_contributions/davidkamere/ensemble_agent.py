from pathlib import Path

import pandas as pd

from config import ENSEMBLE_MODEL_PATH

try:
    import joblib
except Exception:
    joblib = None


class EnsembleAgent:
    def __init__(self, model_path: Path = ENSEMBLE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.available = False
        self._load()

    def _load(self):
        if joblib is None:
            self.available = False
            return
        if not self.model_path.exists():
            self.available = False
            return
        try:
            self.model = joblib.load(self.model_path)
            self.available = True
        except Exception:
            self.available = False

    @staticmethod
    def _features(llm_price: float, rag_price: float, heuristic_price: float):
        mn = min(llm_price, rag_price, heuristic_price)
        mx = max(llm_price, rag_price, heuristic_price)
        return pd.DataFrame(
            [{
                "llm_price": float(llm_price),
                "rag_price": float(rag_price),
                "heuristic_price": float(heuristic_price),
                "min_price": float(mn),
                "max_price": float(mx),
            }]
        )

    def predict(self, llm_price: float, rag_price: float, heuristic_price: float):
        if not self.available or self.model is None:
            return None
        try:
            x = self._features(float(llm_price), float(rag_price), float(heuristic_price))
            y = float(self.model.predict(x)[0])
            return max(0.0, y)
        except Exception:
            return None
