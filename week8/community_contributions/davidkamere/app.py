import gradio as gr
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from framework import DealAgentFramework, dashboard_snapshot, latest_trace


def _client():
    if not OPENROUTER_API_KEY:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


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


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
