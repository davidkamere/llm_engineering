import os
from pathlib import Path

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "deal_copilot.db"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_MODEL_PATH = BASE_DIR / "ensemble_model.pkl"
