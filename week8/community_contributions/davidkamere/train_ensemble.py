import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import ENSEMBLE_MODEL_PATH
from rag import reference_items

try:
    import joblib
except Exception:
    joblib = None


def build_training_frame(seed=42, per_item=40):
    random.seed(seed)
    rows = []

    for item in reference_items:
        fair = float(item["fair_price"])
        for _ in range(per_item):
            llm_price = fair * random.uniform(0.82, 1.18)
            rag_price = fair * random.uniform(0.88, 1.12)
            heuristic_price = fair * random.uniform(0.75, 1.25)

            rows.append(
                {
                    "llm_price": llm_price,
                    "rag_price": rag_price,
                    "heuristic_price": heuristic_price,
                    "min_price": min(llm_price, rag_price, heuristic_price),
                    "max_price": max(llm_price, rag_price, heuristic_price),
                    "target_fair_price": fair,
                }
            )

    return pd.DataFrame(rows)


def train_and_save():
    if joblib is None:
        raise RuntimeError("joblib is required. Install with `uv add joblib`.")

    df = build_training_frame()
    feature_cols = ["llm_price", "rag_price", "heuristic_price", "min_price", "max_price"]

    X = df[feature_cols]
    y = df["target_fair_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    joblib.dump(model, ENSEMBLE_MODEL_PATH)

    print("Saved:", ENSEMBLE_MODEL_PATH)
    print("RMSE:", round(float(rmse), 4))
    print("R2:", round(float(r2), 4))
    print("Coefficients:", dict(zip(feature_cols, np.round(model.coef_, 4))))


if __name__ == "__main__":
    train_and_save()
