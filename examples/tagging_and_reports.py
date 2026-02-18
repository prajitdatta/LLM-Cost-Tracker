"""
examples/tagging_and_reports.py

Tag calls by feature/env/user and generate cost attribution reports.

Run:
    python examples/tagging_and_reports.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import CostTracker
from src.reporter import (
    model_breakdown, tag_breakdown, efficiency_report,
    top_expensive_calls, print_model_breakdown
)

def main():
    print("=" * 60)
    print("  LLM Cost Tracker — Tagging & Cost Attribution")
    print("=" * 60)

    tracker = CostTracker()

    # Simulate a day of production traffic across features
    calls = [
        # feature=chat (high volume, cheap model)
        *[("gpt-4o-mini", 800,  300, {"feature": "chat",       "env": "prod",    "tier": "free"}) for _ in range(20)],
        *[("gpt-4o",      1200, 500, {"feature": "chat",       "env": "prod",    "tier": "paid"}) for _ in range(5)],
        # feature=document-qa (medium volume, long context)
        *[("gpt-4o",      4000, 800, {"feature": "doc-qa",     "env": "prod",    "tier": "paid"}) for _ in range(8)],
        # feature=code-review (low volume, expensive model)
        *[("claude-3-5-sonnet-20241022", 2000, 1200, {"feature": "code-review", "env": "prod", "tier": "paid"}) for _ in range(3)],
        # feature=summarise (background jobs)
        *[("gpt-4o-mini", 3000, 400, {"feature": "summarise",  "env": "prod",    "tier": "free"}) for _ in range(10)],
        # staging traffic
        *[("gpt-4o-mini", 500,  200, {"feature": "chat",       "env": "staging", "tier": "free"}) for _ in range(5)],
    ]

    for model, inp, out, tags in calls:
        tracker.record(model, inp, out, tags=tags)

    print(f"\n  Total calls:  {tracker.call_count():,}")
    print(f"  Total cost:   ${tracker.total_cost():.4f}")

    # ── By model ──────────────────────────────────────────────
    print_model_breakdown(tracker)

    # ── By feature ────────────────────────────────────────────
    print("\n[Cost by feature]")
    rows = tag_breakdown(tracker, "feature")
    for r in rows:
        bar = "█" * max(1, int(r["pct_of_total"] / 4))
        print(f"  {r['tag_value']:<15} ${r['total_cost']:>8.4f}  {r['pct_of_total']:>5.1f}%  {bar}")

    # ── By tier ───────────────────────────────────────────────
    print("\n[Cost by tier (free vs paid)]")
    for r in tag_breakdown(tracker, "tier"):
        print(f"  {r['tag_value']:<8} ${r['total_cost']:.4f}  ({r['pct_of_total']:.0f}% of spend,  {r['call_count']} calls)")

    # ── By env ────────────────────────────────────────────────
    print("\n[Cost by environment]")
    for r in tag_breakdown(tracker, "env"):
        print(f"  {r['tag_value']:<10} ${r['total_cost']:.4f}  ({r['call_count']} calls)")

    # ── Efficiency ────────────────────────────────────────────
    print("\n[Model efficiency — cost per 1K output tokens]")
    for r in efficiency_report(tracker):
        print(f"  {r['model_id']:<42} ${r['cost_per_1k_output']:.4f}/1k output  ratio={r['output_input_ratio']:.2f}")

    # ── Top expensive calls ───────────────────────────────────
    print("\n[Top 3 most expensive individual calls]")
    for r in top_expensive_calls(tracker, n=3):
        print(f"  ${r.total_cost:.6f}  {r.model_id}  in={r.input_tokens} out={r.output_tokens}  {r.tags}")

    # ── JSON export preview ───────────────────────────────────
    import json
    data = json.loads(tracker.to_json())
    print(f"\n[JSON export — session_summary]")
    summary = data["session_summary"]
    for k, v in summary.items():
        if k != "cost_by_model":
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
