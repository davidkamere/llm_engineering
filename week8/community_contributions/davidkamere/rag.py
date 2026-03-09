import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

reference_items = [
    {"name": "Apple AirPods Pro 2", "description": "wireless earbuds active noise cancellation", "fair_price": 220.0},
    {"name": "Sony WH-1000XM5", "description": "noise cancelling over-ear headphones", "fair_price": 320.0},
    {"name": "Nintendo Switch OLED", "description": "gaming console handheld oled", "fair_price": 350.0},
    {"name": "Instant Pot Duo 6QT", "description": "electric pressure cooker kitchen appliance", "fair_price": 95.0},
    {"name": "Logitech MX Master 3S", "description": "wireless productivity mouse", "fair_price": 99.0},
    {"name": "Kindle Paperwhite", "description": "ereader waterproof 16gb", "fair_price": 150.0},
    {"name": "Dyson V8 Vacuum", "description": "cordless vacuum cleaner", "fair_price": 350.0},
    {"name": "JBL Flip 6", "description": "portable bluetooth speaker waterproof", "fair_price": 110.0},
    {"name": "Samsung 55 inch 4K TV", "description": "smart tv 4k uhd", "fair_price": 500.0},
    {"name": "Anker 20W USB-C Charger", "description": "phone fast charger compact", "fair_price": 20.0},
]

_kb_df = pd.DataFrame(reference_items)
_kb_texts = (_kb_df["name"] + " " + _kb_df["description"]).tolist()
_vectorizer = TfidfVectorizer(stop_words="english")
_kb_matrix = _vectorizer.fit_transform(_kb_texts)


def lookup_comparables(title, description, k=3):
    query = f"{title} {description}".strip()
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _kb_matrix)[0]
    idxs = sims.argsort()[::-1][:k]

    comps = []
    for idx in idxs:
        row = _kb_df.iloc[idx]
        comps.append(
            {
                "name": row["name"],
                "fair_price": float(row["fair_price"]),
                "similarity": float(sims[idx]),
            }
        )
    return comps
