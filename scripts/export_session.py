"""
scripts/export_session.py

Demonstrate session export to JSON and CSV.

Run:
    python scripts/export_session.py
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import CostTracker

def main():
    tracker = CostTracker()
    calls = [
        ("gpt-4o-mini", 800,  300, {"feature": "chat",   "user": "u001"}),
        ("gpt-4o",      1200, 500, {"feature": "review",  "user": "u002"}),
        ("claude-3-5-sonnet-20241022", 1000, 700, {"feature": "draft", "user": "u001"}),
        ("gemini-1.5-flash", 600, 250, {"feature": "search","user": "u003"}),
        ("gpt-4o-mini", 300,  100, {"feature": "chat",   "user": "u002"}),
    ]
    for model, inp, out, tags in calls:
        tracker.record(model, inp, out, tags=tags)

    # ── JSON export ───────────────────────────────────────────
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = f.name
        tracker.save_json(json_path)

    with open(json_path) as f:
        data = json.load(f)

    print(f"\n[JSON export — {json_path}]")
    print(f"  Total calls: {data['session_summary']['total_calls']}")
    print(f"  Total cost:  ${data['session_summary']['total_cost_usd']:.6f}")
    print(f"  First call:  {data['calls'][0]['model_id']} — ${data['calls'][0]['total_cost']:.6f}")

    # ── CSV export ────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        tracker.save_csv(csv_path)

    with open(csv_path) as f:
        lines = f.read().strip().split("\n")

    print(f"\n[CSV export — {csv_path}]")
    print(f"  Header: {lines[0][:100]}")
    print(f"  Rows:   {len(lines) - 1}")

    # ── Clean up ──────────────────────────────────────────────
    os.unlink(json_path)
    os.unlink(csv_path)
    print(f"\n  ✅ Export/import round-trip complete")

if __name__ == "__main__":
    main()
