"""
examples/multi_model_comparison.py

Compare cost across GPT-4o, Claude 3.5, Gemini 1.5 Flash, and Mistral Large
for a realistic workload: 1000 input tokens, 500 output tokens per call.

Run:
    python examples/multi_model_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calculator import calculate, monthly_projection, break_even_analysis
from src.reporter import print_comparison_table, comparison_table

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "mistral-small-latest",
    "mistral-large-latest",
    "deepseek-chat",
]

def main():
    print("=" * 65)
    print("  LLM Cost Tracker — Multi-Model Comparison")
    print("=" * 65)

    INPUT_TOKENS  = 1000
    OUTPUT_TOKENS = 500

    # ── 1. Side-by-side cost table ───────────────────────────
    print_comparison_table(MODELS, INPUT_TOKENS, OUTPUT_TOKENS)

    # ── 2. Monthly projection at 10k calls/day ────────────────
    print(f"\n[Monthly projection — 10,000 calls/day, {INPUT_TOKENS}/{OUTPUT_TOKENS} avg tokens]")
    print(f"  {'Model':<42} {'Monthly':>10} {'Annual':>12}")
    print(f"  {'─'*68}")
    for mid in MODELS:
        try:
            proj = monthly_projection(mid, 10_000, INPUT_TOKENS, OUTPUT_TOKENS)
            print(f"  {mid:<42} ${proj['monthly_cost']:>9,.2f} ${proj['annual_cost']:>11,.2f}")
        except KeyError:
            print(f"  {mid:<42} not found")

    # ── 3. Break-even analysis ────────────────────────────────
    print(f"\n[Break-even: switching from GPT-4o → GPT-4o-mini]")
    be = break_even_analysis("gpt-4o", "gpt-4o-mini", INPUT_TOKENS, OUTPUT_TOKENS)
    print(f"  Cheaper model:         {be['cheaper_model']}")
    print(f"  Savings per call:      ${be['savings_per_call']:.6f}")
    print(f"  Savings per 1K calls:  ${be['savings_per_1k_calls']:.4f}")
    print(f"  Savings per 1M calls:  ${be['savings_per_1m_calls']:.2f}")
    print(f"  % cheaper:             {be['pct_cheaper']:.1f}%")

    print(f"\n[Break-even: switching from GPT-4o → DeepSeek Chat]")
    be2 = break_even_analysis("gpt-4o", "deepseek-chat", INPUT_TOKENS, OUTPUT_TOKENS)
    print(f"  Cheaper model:         {be2['cheaper_model']}")
    print(f"  % cheaper:             {be2['pct_cheaper']:.1f}%")
    print(f"  Savings per 1M calls:  ${be2['savings_per_1m_calls']:.2f}")

    # ── 4. Cheapest option per provider ──────────────────────
    from src.models import list_providers, cheapest_model
    print(f"\n[Cheapest model per provider (by output token price)]")
    for provider in list_providers():
        try:
            m = cheapest_model(provider=provider)
            print(f"  {provider:<15} {m.model_id:<42} ${m.output_per_1k:.6f}/1k out")
        except ValueError:
            pass

if __name__ == "__main__":
    main()
