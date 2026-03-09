import json
import re

import feedparser

from db import get_conn, utc_now_iso
from rag import lookup_comparables


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
