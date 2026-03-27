import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import gradio as gr
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

try:
    import joblib
except Exception:
    joblib = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "deal_copilot.db"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_MODEL_PATH = BASE_DIR / "ensemble_model.pkl"


def _client():
    if not OPENROUTER_API_KEY:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cur, table, col_name, col_type):
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deals_seen (
            deal_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            seen_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id TEXT,
            title TEXT,
            listed_price REAL,
            estimated_price REAL,
            discount_pct REAL,
            confidence REAL,
            rationale TEXT,
            url TEXT,
            created_at TEXT
        )
        """
    )
    _ensure_column(cur, "opportunities", "llm_price", "REAL")
    _ensure_column(cur, "opportunities", "rag_price", "REAL")
    _ensure_column(cur, "opportunities", "ensemble_price", "REAL")
    _ensure_column(cur, "opportunities", "planner_score", "REAL")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deal_id TEXT,
            message TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_trace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            agent_name TEXT,
            event TEXT,
            payload TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def trace(run_id, agent_name, event, payload):
    conn = get_conn()
    conn.execute(
        "INSERT INTO agent_trace(run_id, agent_name, event, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (run_id, agent_name, event, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
    )
    conn.commit()
    conn.close()


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
            [
                {
                    "llm_price": float(llm_price),
                    "rag_price": float(rag_price),
                    "heuristic_price": float(heuristic_price),
                    "min_price": float(mn),
                    "max_price": float(mx),
                }
            ]
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


def train_ensemble(per_item=40):
    if joblib is None:
        raise RuntimeError("joblib is required. Install with `uv add joblib`.")

    rows = []
    for item in reference_items:
        fair = float(item["fair_price"])
        for _ in range(per_item):
            llm_price = fair * (0.82 + 0.36 * os.urandom(1)[0] / 255.0)
            rag_price = fair * (0.88 + 0.24 * os.urandom(1)[0] / 255.0)
            heuristic_price = fair * (0.75 + 0.5 * os.urandom(1)[0] / 255.0)
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

    df = pd.DataFrame(rows)
    feature_cols = ["llm_price", "rag_price", "heuristic_price", "min_price", "max_price"]

    X = df[feature_cols]
    y = df["target_fair_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float((mean_squared_error(y_test, preds)) ** 0.5)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, ENSEMBLE_MODEL_PATH)
    return {"rmse": round(rmse, 4), "r2": round(float(r2), 4)}


class ScannerAgent:
    def __init__(self):
        self.rss_sources = [
            "https://www.techradar.com/rss",
            "https://www.theverge.com/rss/index.xml",
            "https://slickdeals.net/newsearch.php?mode=frontpage&searcharea=deals&searchin=first&rss=1",
            "https://www.engadget.com/rss.xml",
            "https://www.cnet.com/rss/deals/",
            "https://www.techradar.com/rss/tag/deals",
            "https://www.tomshardware.com/feeds/all",
            "https://feeds.arstechnica.com/arstechnica/index",
        ]

    def scan(self, max_items=12):
        deals = []
        for src in self.rss_sources:
            try:
                feed = feedparser.parse(src)
                for e in feed.entries[:max_items]:
                    title = getattr(e, "title", "").strip()
                    summary = re.sub("<[^<]+?>", " ", getattr(e, "summary", ""))
                    link = getattr(e, "link", "")
                    m = re.search(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)", f"{title} {summary}")
                    listed = float(m.group(1)) if m else None
                    if listed is None:
                        continue
                    deal_id = re.sub(r"[^a-zA-Z0-9]+", "-", f"{title}-{link}")[:100].lower()
                    deals.append(
                        {
                            "deal_id": deal_id,
                            "title": title,
                            "description": summary[:300],
                            "url": link,
                            "listed_price": listed,
                            "source": src,
                        }
                    )
            except Exception:
                continue

        if not deals:
            deals = [
                {
                    "deal_id": "sample-airpods",
                    "title": "AirPods Pro 2 sale for $169",
                    "description": "wireless earbuds anc discount",
                    "url": "https://example.com/airpods",
                    "listed_price": 169.0,
                    "source": "sample",
                },
                {
                    "deal_id": "sample-switch",
                    "title": "Nintendo Switch OLED listed at $249",
                    "description": "portable gaming console oled bundle",
                    "url": "https://example.com/switch",
                    "listed_price": 249.0,
                    "source": "sample",
                },
                {
                    "deal_id": "sample-logi",
                    "title": "MX Master 3S promo at $59",
                    "description": "wireless office mouse productivity",
                    "url": "https://example.com/mx3s",
                    "listed_price": 59.0,
                    "source": "sample",
                },
                {
                    "deal_id": "sample-kindle",
                    "title": "Kindle Paperwhite now $99",
                    "description": "ereader waterproof deal",
                    "url": "https://example.com/kindle",
                    "listed_price": 99.0,
                    "source": "sample",
                },
            ]

        return deals


class ValueAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def estimate(self, deal):
        comps = lookup_comparables(deal["title"], deal["description"], k=3)
        rag_price = sum(c["fair_price"] for c in comps) / len(comps)
        confidence = sum(c["similarity"] for c in comps) / len(comps)
        rationale = f"Comparable average from KB: {round(rag_price, 2)}"
        llm_price = rag_price

        if self.client:
            try:
                msg = (
                    "Estimate fair market price in USD using comparables. "
                    "Return JSON with keys fair_price and rationale.\n"
                    f"Deal title: {deal['title']}\n"
                    f"Description: {deal['description']}\n"
                    f"Listed price: {deal['listed_price']}\n"
                    f"Comparables: {json.dumps(comps)}"
                )
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a pricing analyst."},
                        {"role": "user", "content": msg},
                    ],
                    temperature=0,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/davidkamere/llm_engineering",
                        "X-Title": "week8-deal-triage-copilot",
                    },
                )
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"([0-9]+(?:\.[0-9]{1,2})?)", text)
                llm_price = float(m.group(1)) if m else rag_price
                rationale = text[:280] if text else rationale
                confidence = float(min(1.0, max(0.0, confidence + 0.15)))
            except Exception:
                pass

        heuristic_price = (deal["listed_price"] + rag_price) / 2.0
        estimated = 0.6 * llm_price + 0.4 * rag_price

        return {
            "llm_price": float(llm_price),
            "rag_price": float(rag_price),
            "heuristic_price": float(heuristic_price),
            "estimated_price": float(estimated),
            "confidence": float(confidence),
            "rationale": rationale,
            "comparables": comps,
        }


class PlannerAgent:
    def select(self, scored, min_discount_pct=20.0, min_confidence=0.35):
        picks = []
        for x in scored:
            est = x["estimated_price"]
            listed = x["listed_price"]
            if est <= 0:
                continue
            discount_pct = ((est - listed) / est) * 100
            x["discount_pct"] = discount_pct
            if discount_pct >= min_discount_pct and x["confidence"] >= min_confidence:
                x["planner_note"] = "Passed heuristic thresholds"
                x["planner_score"] = round(discount_pct * x["confidence"], 2)
                picks.append(x)

        picks.sort(key=lambda y: (y["discount_pct"], y["confidence"]), reverse=True)
        return picks


class PlannerLLMAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def refine(self, candidates, top_k=5):
        if not candidates:
            return []

        if self.client is None:
            return candidates[:top_k]

        packed = []
        for c in candidates[:12]:
            packed.append(
                {
                    "deal_id": c["deal_id"],
                    "title": c["title"],
                    "listed_price": c["listed_price"],
                    "estimated_price": c["estimated_price"],
                    "discount_pct": round(c["discount_pct"], 2),
                    "confidence": round(c["confidence"], 3),
                }
            )

        prompt = (
            "Select the best shopping opportunities. "
            "Return strict JSON list, max 5 items, each with keys: deal_id, priority_score, planner_note. "
            "Prioritize higher discount_pct and confidence.\n\n"
            f"Candidates: {json.dumps(packed)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a deal triage planner."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/davidkamere/llm_engineering",
                    "X-Title": "week8-deal-triage-copilot",
                },
            )
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            parsed = json.loads(m.group(0) if m else raw)
            picked_map = {x.get("deal_id"): x for x in parsed if isinstance(x, dict) and x.get("deal_id")}

            refined = []
            for c in candidates:
                item = picked_map.get(c["deal_id"])
                if not item:
                    continue
                c = dict(c)
                c["planner_score"] = float(item.get("priority_score", c.get("planner_score", 0.0)))
                c["planner_note"] = str(item.get("planner_note", "LLM-selected"))[:220]
                refined.append(c)

            if refined:
                refined.sort(key=lambda y: y.get("planner_score", 0.0), reverse=True)
                return refined[:top_k]
        except Exception:
            pass

        return candidates[:top_k]


class NotifierAgent:
    def notify(self, opp):
        msg = (
            f"Deal: {opp['title']} | listed=${opp['listed_price']:.2f} | "
            f"estimated=${opp['estimated_price']:.2f} | "
            f"discount={opp['discount_pct']:.1f}%"
        )
        conn = get_conn()
        conn.execute(
            "INSERT INTO alerts(deal_id, message, created_at) VALUES (?, ?, ?)",
            (opp["deal_id"], msg, utc_now_iso()),
        )
        conn.commit()
        conn.close()
        return msg


class DealAgentFramework:
    def __init__(self, client, model_name):
        self.scanner = ScannerAgent()
        self.valuer = ValueAgent(client, model_name)
        self.planner = PlannerAgent()
        self.llm_planner = PlannerLLMAgent(client, model_name)
        self.ensemble = EnsembleAgent()
        self.notifier = NotifierAgent()
        self.model_name = model_name

    def seen(self, deal_id):
        conn = get_conn()
        row = conn.execute("SELECT 1 FROM deals_seen WHERE deal_id=?", (deal_id,)).fetchone()
        conn.close()
        return row is not None

    def mark_seen(self, deal):
        conn = get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO deals_seen(deal_id, title, source, seen_at) VALUES (?, ?, ?, ?)",
            (deal["deal_id"], deal["title"], deal["source"], utc_now_iso()),
        )
        conn.commit()
        conn.close()

    def save_opportunity(self, opp):
        note = opp.get("planner_note", "")
        rationale = opp.get("rationale", "")
        merged_rationale = f"{rationale} | Planner: {note}" if note else rationale

        conn = get_conn()
        conn.execute(
            """
            INSERT INTO opportunities(
                deal_id, title, listed_price, estimated_price, discount_pct, confidence,
                rationale, url, created_at, llm_price, rag_price, ensemble_price, planner_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opp["deal_id"],
                opp["title"],
                opp["listed_price"],
                opp["estimated_price"],
                opp["discount_pct"],
                opp["confidence"],
                merged_rationale,
                opp["url"],
                utc_now_iso(),
                opp.get("llm_price"),
                opp.get("rag_price"),
                opp.get("ensemble_price"),
                opp.get("planner_score"),
            ),
        )
        conn.commit()
        conn.close()

    def run_once(self, min_discount_pct=20.0, min_confidence=0.35, max_items=10, model_name=None):
        run_id = f"run-{int(time.time())}"
        if model_name:
            self.valuer.model = model_name
            self.llm_planner.model = model_name

        trace(
            run_id,
            "Framework",
            "start",
            {
                "min_discount_pct": min_discount_pct,
                "min_confidence": min_confidence,
                "model": self.valuer.model,
                "ensemble_available": self.ensemble.available,
            },
        )

        scanned = self.scanner.scan(max_items=max_items)
        trace(run_id, "ScannerAgent", "scanned", {"count": len(scanned)})

        scored = []
        ensemble_used = 0
        for deal in scanned:
            if self.seen(deal["deal_id"]):
                continue
            self.mark_seen(deal)
            enriched = {**deal, **self.valuer.estimate(deal)}

            ensemble_price = self.ensemble.predict(
                llm_price=enriched["llm_price"],
                rag_price=enriched["rag_price"],
                heuristic_price=enriched["heuristic_price"],
            )
            enriched["ensemble_price"] = ensemble_price
            if ensemble_price is not None:
                enriched["estimated_price"] = float(ensemble_price)
                ensemble_used += 1

            scored.append(enriched)

        trace(run_id, "ValueAgent", "valued", {"count": len(scored)})
        trace(run_id, "EnsembleAgent", "predicted", {"count": ensemble_used, "model_available": self.ensemble.available})

        heuristic_picks = self.planner.select(scored, min_discount_pct=min_discount_pct, min_confidence=min_confidence)
        trace(run_id, "PlannerAgent", "heuristic_selected", {"count": len(heuristic_picks)})

        picks = self.llm_planner.refine(heuristic_picks, top_k=5)
        trace(run_id, "PlannerLLMAgent", "llm_refined", {"count": len(picks)})

        alerts = []
        for opp in picks:
            self.save_opportunity(opp)
            alerts.append(self.notifier.notify(opp))

        trace(run_id, "NotifierAgent", "alerted", {"count": len(alerts)})

        summary = {
            "run_id": run_id,
            "scanned": len(scanned),
            "scored_new": len(scored),
            "ensemble_used": ensemble_used,
            "heuristic_shortlisted": len(heuristic_picks),
            "final_shortlisted": len(picks),
            "top_discount_pct": round(max([p["discount_pct"] for p in picks], default=0.0), 2),
        }

        pd.DataFrame([summary]).to_csv(ARTIFACT_DIR / f"{run_id}_summary.csv", index=False)
        trace(run_id, "Framework", "end", summary)
        return summary


def latest_opportunities(limit=25):
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT deal_id, title, listed_price, estimated_price, discount_pct, confidence,
               llm_price, rag_price, ensemble_price, planner_score, rationale, url, created_at
        FROM opportunities ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(
            columns=[
                "deal_id",
                "title",
                "listed_price",
                "estimated_price",
                "discount_pct",
                "confidence",
                "llm_price",
                "rag_price",
                "ensemble_price",
                "planner_score",
                "rationale",
                "url",
                "created_at",
            ]
        )

    return pd.DataFrame([dict(r) for r in rows])


def latest_alerts(limit=20):
    conn = get_conn()
    rows = conn.execute(
        "SELECT deal_id, message, created_at FROM alerts ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=["deal_id", "message", "created_at"])

    return pd.DataFrame([dict(r) for r in rows])


def latest_trace(limit=80):
    conn = get_conn()
    rows = conn.execute(
        "SELECT run_id, agent_name, event, payload, created_at FROM agent_trace ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        return ""

    lines = []
    for r in rows[::-1]:
        lines.append(f"[{r['created_at']}] {r['run_id']} | {r['agent_name']} | {r['event']} | {r['payload']}")
    return "\n".join(lines)


def dashboard_snapshot(limit=30):
    opp_df = latest_opportunities(limit=limit)
    if len(opp_df):
        opp_df = opp_df.copy()
        opp_df["estimated_savings"] = (opp_df["estimated_price"] - opp_df["listed_price"]).round(2)
        opp_df["discount_pct"] = opp_df["discount_pct"].round(2)
        opp_df["confidence"] = opp_df["confidence"].round(3)
        opp_df = opp_df.sort_values(["discount_pct", "confidence"], ascending=False)

    alerts_df = latest_alerts(limit=limit)

    kpis = {
        "opportunities": int(len(opp_df)),
        "alerts": int(len(alerts_df)),
        "avg_discount_pct": round(float(opp_df["discount_pct"].mean()), 2) if len(opp_df) else 0.0,
        "max_discount_pct": round(float(opp_df["discount_pct"].max()), 2) if len(opp_df) else 0.0,
        "total_estimated_savings": round(float(opp_df["estimated_savings"].clip(lower=0).sum()), 2) if len(opp_df) else 0.0,
    }

    return kpis, opp_df, alerts_df


def _render_alerts(alerts_df):
    if alerts_df.empty:
        return "No alerts yet."
    lines = []
    for _, row in alerts_df.head(12).iterrows():
        lines.append(f"[{row['created_at']}] {row['message']}")
    return "\n".join(lines)


def _priority_band(discount_pct):
    if discount_pct >= 35:
        return "HIGH"
    if discount_pct >= 20:
        return "MEDIUM"
    return "LOW"


def _top_picks_markdown(opp_df):
    if opp_df.empty:
        return "No opportunities selected in this run."

    rows = opp_df.head(3)
    lines = ["### Top 3 Picks"]
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        lines.append(
            f"{i}. **{r['title']}** | Discount: `{r['discount_pct']:.1f}%` | "
            f"Savings: `${r['estimated_savings']:.2f}` | Confidence: `{r['confidence']:.2f}`"
        )
        rationale = str(r.get("rationale", "")).strip()
        if rationale:
            lines.append(f"   Reason: {rationale[:220]}")
    return "\n".join(lines)


def _details_markdown(opp_df):
    if opp_df.empty:
        return "No opportunity details available."

    rows = opp_df.head(3)
    lines = ["### Opportunity Links"]
    for _, r in rows.iterrows():
        url = r.get("url", "")
        title = r.get("title", "Deal")
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


def _kpi_cards(summary, kpis):
    card1 = f"**Final Shortlisted**\n\n## {summary.get('final_shortlisted', 0)}"
    card2 = f"**Avg Discount %**\n\n## {kpis.get('avg_discount_pct', 0)}"
    card3 = f"**Total Savings ($)**\n\n## {kpis.get('total_estimated_savings', 0)}"
    card4 = f"**Run ID**\n\n`{summary.get('run_id', '-')}`"
    return card1, card2, card3, card4


def run_pipeline(min_discount_pct, min_confidence, max_items, model_name):
    fw = DealAgentFramework(client=_client(), model_name=model_name)
    summary = fw.run_once(
        min_discount_pct=float(min_discount_pct),
        min_confidence=float(min_confidence),
        max_items=int(max_items),
        model_name=model_name,
    )

    kpis, opp_df, alerts_df = dashboard_snapshot(limit=40)
    trace_text = latest_trace(limit=160)

    cols = [
        "title",
        "listed_price",
        "llm_price",
        "rag_price",
        "ensemble_price",
        "estimated_price",
        "estimated_savings",
        "discount_pct",
        "confidence",
        "planner_score",
        "rationale",
        "url",
        "created_at",
    ]

    opp_view = opp_df[cols] if len(opp_df) else opp_df
    if len(opp_view):
        opp_view = opp_view.copy()
        opp_view["priority_band"] = opp_view["discount_pct"].apply(_priority_band)
        ordered_cols = [
            "priority_band",
            "title",
            "listed_price",
            "llm_price",
            "rag_price",
            "ensemble_price",
            "estimated_price",
            "estimated_savings",
            "discount_pct",
            "confidence",
            "planner_score",
            "rationale",
            "url",
            "created_at",
        ]
        opp_view = opp_view[ordered_cols]

    top_md = _top_picks_markdown(opp_view)
    details_md = _details_markdown(opp_view)
    alerts_text = _render_alerts(alerts_df)
    c1, c2, c3, c4 = _kpi_cards(summary, kpis)

    return c1, c2, c3, c4, opp_view, top_md, details_md, alerts_text, trace_text


def build_app():
    with gr.Blocks(title="Week8 Deal Triage Copilot") as demo:
        gr.Markdown("## Deal Triage Copilot")
        gr.Markdown("Run the pipeline and inspect ranked opportunities, estimated savings, alerts, and full agent trace.")

        with gr.Row():
            min_discount = gr.Slider(5, 60, value=20, step=1, label="Min Discount %")
            min_conf = gr.Slider(0.1, 0.95, value=0.35, step=0.05, label="Min Confidence")
            max_items = gr.Slider(3, 30, value=8, step=1, label="Max Items")
        model_name = gr.Textbox(value=OPENROUTER_MODEL, label="OpenRouter Model")
        run_btn = gr.Button("Run Once", variant="primary")

        with gr.Row():
            kpi1 = gr.Markdown("**Final Shortlisted**\n\n## -")
            kpi2 = gr.Markdown("**Avg Discount %**\n\n## -")
            kpi3 = gr.Markdown("**Total Savings ($)**\n\n## -")
            kpi4 = gr.Markdown("**Run ID**\n\n`-`")

        with gr.Row():
            opp_table = gr.Dataframe(label="Ranked Opportunities")
            top_picks = gr.Markdown(label="Top Picks")

        with gr.Row():
            details_md = gr.Markdown(label="Opportunity Links")
            alerts_box = gr.Textbox(label="Recent Alerts", lines=16)

        trace_box = gr.Textbox(label="Agent Trace", lines=18)

        run_btn.click(
            fn=run_pipeline,
            inputs=[min_discount, min_conf, max_items, model_name],
            outputs=[kpi1, kpi2, kpi3, kpi4, opp_table, top_picks, details_md, alerts_box, trace_box],
        )

    return demo


def run_cli():
    init_db()
    metrics = train_ensemble()
    print({"ensemble_training": metrics})
    client = _client()
    framework = DealAgentFramework(client=client, model_name=OPENROUTER_MODEL)
    summary = framework.run_once(min_discount_pct=20, min_confidence=0.35, max_items=8, model_name=OPENROUTER_MODEL)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
