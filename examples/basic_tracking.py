"""
examples/basic_tracking.py

Hello world: track a single LLM API call, then a multi-call session.

Run:
    python examples/basic_tracking.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import CostTracker

def main():
    print("=" * 55)
    print("  LLM Cost Tracker — Basic Tracking Example")
    print("=" * 55)

    # ── 1. Track a single call ────────────────────────────────
    tracker = CostTracker()
    record = tracker.record(
        model_id="gpt-4o-mini",
        input_tokens=750,
        output_tokens=300,
        tags={"feature": "chat", "user": "demo"},
    )
    print(f"\n[Single call]")
    print(f"  {record}")
    print(f"  Cost: ${record.total_cost:.6f}")
    print(f"  Provider: {record.provider}")

    # ── 2. Multi-model session ────────────────────────────────
    tracker = CostTracker()
    calls = [
        ("gpt-4o-mini",               1200, 400, {"feature": "draft"}),
        ("gpt-4o",                    800,  600, {"feature": "review"}),
        ("claude-3-5-sonnet-20241022",1500, 800, {"feature": "final"}),
        ("gpt-4o-mini",               300,  150, {"feature": "summary"}),
    ]
    for model, inp, out, tags in calls:
        tracker.record(model, inp, out, tags=tags)

    print(f"\n[Session summary — {len(calls)} calls]")
    print(tracker.summary())

    # ── 3. Cost breakdown by feature ─────────────────────────
    print("\n[Cost by feature]")
    by_feature = tracker.cost_by_tag("feature")
    for feature, cost in by_feature.items():
        print(f"  {feature:<12} ${cost:.6f}")

if __name__ == "__main__":
    main()
