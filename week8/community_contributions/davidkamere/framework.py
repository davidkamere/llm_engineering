import time

import pandas as pd

from agents import NotifierAgent, PlannerAgent, PlannerLLMAgent, ScannerAgent, ValueAgent
from config import ARTIFACT_DIR
from db import get_conn, init_db, trace, utc_now_iso
from ensemble_agent import EnsembleAgent


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


init_db()
