"""
examples/batch_cost_estimation.py

Estimate cost BEFORE running a batch job. Compare models. Set a budget.

Run:
    python examples/batch_cost_estimation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calculator import calculate, batch_estimate, monthly_projection, break_even_analysis
from src.models import compare_models

def main():
    print("=" * 65)
    print("  LLM Cost Tracker — Pre-flight Batch Cost Estimation")
    print("=" * 65)

    # Scenario: processing 50,000 customer support tickets
    # Each ticket: ~800 input tokens (ticket + context), ~300 output tokens (response draft)
    TICKET_COUNT   = 50_000
    AVG_INPUT      = 800
    AVG_OUTPUT     = 300

    print(f"\n[Scenario: {TICKET_COUNT:,} support tickets, {AVG_INPUT}/{AVG_OUTPUT} avg tokens]")

    candidates = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307",
                  "claude-3-5-sonnet-20241022", "gemini-1.5-flash", "deepseek-chat"]

    print(f"\n  {'Model':<42} {'Per ticket':>10} {'Total 50k':>12}")
    print(f"  {'─'*68}")

    for mid in candidates:
        try:
            est = calculate(mid, AVG_INPUT, AVG_OUTPUT)
            total = est.total_cost * TICKET_COUNT
            print(f"  {mid:<42} ${est.total_cost:>9.6f} ${total:>11.2f}")
        except KeyError:
            print(f"  {mid:<42} not found")

    # ── Batch estimate with variance ──────────────────────────
    print(f"\n[Batch estimate — variable ticket sizes on gpt-4o-mini]")
    calls = []
    import random
    random.seed(42)
    for _ in range(20):
        inp = random.randint(300, 2000)
        out = random.randint(100, 600)
        calls.append({"input_tokens": inp, "output_tokens": out})

    result = batch_estimate("gpt-4o-mini", calls)
    print(f"  Calls:        {result['call_count']}")
    print(f"  Total tokens: {result['total_tokens']:,}")
    print(f"  Total cost:   ${result['total_cost_usd']:.6f}")
    print(f"  Avg per call: ${result['avg_cost_per_call']:.6f}")
    print(f"  Min per call: ${result['min_cost_per_call']:.6f}")
    print(f"  Max per call: ${result['max_cost_per_call']:.6f}")

    # ── ROI comparison: mini vs sonnet ────────────────────────
    print(f"\n[Break-even: GPT-4o-mini vs Claude 3.5 Sonnet for 50k tickets]")
    be = break_even_analysis("claude-3-5-sonnet-20241022", "gpt-4o-mini", AVG_INPUT, AVG_OUTPUT)
    savings_total = be["savings_per_call"] * TICKET_COUNT
    print(f"  Switching to {be['cheaper_model']} saves:")
    print(f"    ${be['savings_per_call']:.6f} per ticket")
    print(f"    ${savings_total:.2f} for the full batch")
    print(f"    {be['pct_cheaper']:.1f}% cheaper")

    # ── Monthly scale ─────────────────────────────────────────
    print(f"\n[If this runs daily — monthly and annual cost projections]")
    daily_calls = TICKET_COUNT
    for mid in ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini-1.5-flash"]:
        try:
            proj = monthly_projection(mid, daily_calls, AVG_INPUT, AVG_OUTPUT)
            print(f"  {mid:<40} monthly=${proj['monthly_cost']:>8,.2f}  annual=${proj['annual_cost']:>10,.2f}")
        except KeyError:
            pass

if __name__ == "__main__":
    main()
