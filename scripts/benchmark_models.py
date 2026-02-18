"""
scripts/benchmark_models.py

Print the full pricing table for all 40+ models in the registry.
Use this to quickly check prices and compare providers.

Run:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --provider openai
    python scripts/benchmark_models.py --sort input
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from src.models import list_models, list_providers, REGISTRY

def main():
    parser = argparse.ArgumentParser(description="Print LLM pricing table")
    parser.add_argument("--provider", default=None, help="Filter by provider")
    parser.add_argument("--sort", default="output", choices=["input","output","provider","context"],
                        help="Sort column")
    args = parser.parse_args()

    models = list_models(provider=args.provider)

    sort_key = {
        "input":    lambda m: m.input_per_1k,
        "output":   lambda m: m.output_per_1k,
        "provider": lambda m: (m.provider, m.output_per_1k),
        "context":  lambda m: m.context_window,
    }[args.sort]

    models = sorted(models, key=sort_key)

    title = f"LLM PRICING TABLE — {len(models)} models"
    if args.provider:
        title += f" ({args.provider})"
    title += f" — sorted by {args.sort} price"

    print(f"\n{'═'*95}")
    print(f"  {title}")
    print(f"{'═'*95}")
    print(f"  {'Model':<44} {'Provider':<12} {'$/1k in':>8} {'$/1k out':>9} {'$/1M in':>9} {'$/1M out':>10} {'ctx':>8}")
    print(f"  {'─'*91}")

    current_provider = None
    for m in models:
        if m.provider != current_provider:
            if current_provider is not None:
                print(f"  {'─'*91}")
            current_provider = m.provider

        print(
            f"  {m.model_id:<44} {m.provider:<12} "
            f"${m.input_per_1k:>7.4f} ${m.output_per_1k:>8.4f} "
            f"${m.input_per_1k*1000:>8.2f} ${m.output_per_1k*1000:>9.2f} "
            f"{m.context_window//1000:>6}k"
        )

    print(f"{'═'*95}")
    print(f"\n  Total models: {len(models)}")
    print(f"  Providers:    {', '.join(list_providers())}")

    # Cost comparison for a standard workload
    print(f"\n  Cost for 1000 input + 500 output tokens:")
    print(f"  {'─'*50}")
    for m in sorted(models, key=lambda m: m.cost_for_tokens(1000, 500)):
        cost = m.cost_for_tokens(1000, 500)
        bar  = "█" * max(1, int(cost * 500000))
        print(f"  {m.model_id:<44} ${cost:.6f}  {bar}")

if __name__ == "__main__":
    main()
