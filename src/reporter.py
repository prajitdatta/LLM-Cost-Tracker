"""
src/reporter.py

Reporter — generate structured cost reports from a CostTracker session.

Report types:
    model_breakdown()    — Cost per model with token and call stats
    provider_breakdown() — Cost per provider
    tag_breakdown()      — Cost per tag value (feature, user, env, etc.)
    time_series()        — Cost aggregated by hour/day/week
    top_expensive_calls()— The N most expensive individual calls
    efficiency_report()  — Output/input token ratio, cost per output token
    comparison_table()   — Side-by-side model comparison for a given workload
"""

from __future__ import annotations
import time
from typing import Optional
from src.tracker import CostTracker, CallRecord
from src.models import get_model


# ── Report data structures ────────────────────────────────────────────────────

def model_breakdown(tracker: CostTracker) -> list[dict]:
    """
    Detailed cost breakdown per model.

    Returns list of dicts sorted by total_cost descending, each with:
        model_id, provider, call_count, total_tokens, input_tokens,
        output_tokens, total_cost, avg_cost_per_call, pct_of_total
    """
    total_session_cost = tracker.total_cost() or 1e-10

    by_model: dict[str, list[CallRecord]] = {}
    for r in tracker.filter():
        by_model.setdefault(r.model_id, []).append(r)

    rows = []
    for model_id, records in by_model.items():
        total_cost = sum(r.total_cost     for r in records)
        rows.append({
            "model_id":        model_id,
            "provider":        records[0].provider,
            "call_count":      len(records),
            "input_tokens":    sum(r.input_tokens  for r in records),
            "output_tokens":   sum(r.output_tokens for r in records),
            "total_tokens":    sum(r.total_tokens  for r in records),
            "total_cost":      total_cost,
            "avg_cost_per_call": total_cost / len(records),
            "pct_of_total":    total_cost / total_session_cost * 100,
        })

    return sorted(rows, key=lambda r: r["total_cost"], reverse=True)


def provider_breakdown(tracker: CostTracker) -> list[dict]:
    """Cost breakdown per provider, sorted by total_cost descending."""
    total_session_cost = tracker.total_cost() or 1e-10

    by_provider: dict[str, list[CallRecord]] = {}
    for r in tracker.filter():
        by_provider.setdefault(r.provider, []).append(r)

    rows = []
    for provider, records in by_provider.items():
        total_cost = sum(r.total_cost for r in records)
        models_used = sorted(set(r.model_id for r in records))
        rows.append({
            "provider":       provider,
            "call_count":     len(records),
            "total_tokens":   sum(r.total_tokens for r in records),
            "total_cost":     total_cost,
            "pct_of_total":   total_cost / total_session_cost * 100,
            "models_used":    models_used,
            "model_count":    len(models_used),
        })

    return sorted(rows, key=lambda r: r["total_cost"], reverse=True)


def tag_breakdown(tracker: CostTracker, tag_key: str) -> list[dict]:
    """
    Cost breakdown by a specific tag key.

    Args:
        tracker:  CostTracker instance
        tag_key:  The tag key to group by (e.g. "feature", "user_id", "env")

    Returns:
        List of dicts with tag_value, call_count, total_cost, pct_of_total
    """
    total_session_cost = tracker.total_cost() or 1e-10

    by_tag: dict[str, list[CallRecord]] = {}
    for r in tracker.filter():
        key = r.tags.get(tag_key, "(untagged)")
        by_tag.setdefault(key, []).append(r)

    rows = []
    for tag_value, records in by_tag.items():
        total_cost = sum(r.total_cost for r in records)
        rows.append({
            "tag_key":        tag_key,
            "tag_value":      tag_value,
            "call_count":     len(records),
            "total_tokens":   sum(r.total_tokens for r in records),
            "total_cost":     total_cost,
            "avg_cost_per_call": total_cost / len(records),
            "pct_of_total":   total_cost / total_session_cost * 100,
        })

    return sorted(rows, key=lambda r: r["total_cost"], reverse=True)


def time_series(
    tracker: CostTracker,
    granularity: str = "hour",
    since: Optional[float] = None,
) -> list[dict]:
    """
    Cost aggregated into time buckets.

    Args:
        tracker:     CostTracker instance
        granularity: "minute" | "hour" | "day" | "week"
        since:       Unix timestamp — only include calls after this time

    Returns:
        List of {bucket, call_count, total_cost, total_tokens} dicts sorted by bucket
    """
    bucket_seconds = {
        "minute": 60,
        "hour":   3600,
        "day":    86400,
        "week":   604800,
    }.get(granularity, 3600)

    records = tracker.filter(since=since)
    buckets: dict[int, list[CallRecord]] = {}

    for r in records:
        bucket = int(r.timestamp // bucket_seconds) * bucket_seconds
        buckets.setdefault(bucket, []).append(r)

    rows = []
    for bucket_ts, bucket_records in sorted(buckets.items()):
        rows.append({
            "bucket":        bucket_ts,
            "bucket_label":  _format_timestamp(bucket_ts, granularity),
            "call_count":    len(bucket_records),
            "total_tokens":  sum(r.total_tokens for r in bucket_records),
            "total_cost":    sum(r.total_cost   for r in bucket_records),
            "models_used":   sorted(set(r.model_id for r in bucket_records)),
        })

    return rows


def top_expensive_calls(tracker: CostTracker, n: int = 10) -> list[CallRecord]:
    """
    Return the N most expensive individual calls.

    Useful for identifying runaway calls or unexpectedly large prompts.
    """
    return sorted(tracker.filter(), key=lambda r: r.total_cost, reverse=True)[:n]


def efficiency_report(tracker: CostTracker) -> list[dict]:
    """
    Efficiency metrics per model:
    - output/input ratio (higher = more verbose output per dollar of input)
    - cost per 1K output tokens
    - cache hit rate (if any calls used prompt caching)

    Helps identify which models give the most output value for their cost.
    """
    by_model: dict[str, list[CallRecord]] = {}
    for r in tracker.filter():
        by_model.setdefault(r.model_id, []).append(r)

    rows = []
    for model_id, records in by_model.items():
        total_input   = sum(r.input_tokens    for r in records)
        total_output  = sum(r.output_tokens   for r in records)
        total_cached  = sum(r.cached_tokens   for r in records)
        total_cost    = sum(r.total_cost      for r in records)

        output_ratio  = total_output / total_input if total_input else 0
        cost_per_1k_output = (total_cost / total_output * 1000) if total_output else 0
        cache_hit_rate = total_cached / total_input if total_input else 0

        rows.append({
            "model_id":          model_id,
            "output_input_ratio":round(output_ratio, 3),
            "cost_per_1k_output":round(cost_per_1k_output, 6),
            "cache_hit_rate":    round(cache_hit_rate, 3),
            "total_cached_tokens": total_cached,
            "call_count":        len(records),
        })

    return sorted(rows, key=lambda r: r["cost_per_1k_output"])


def comparison_table(
    model_ids: list[str],
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> list[dict]:
    """
    Side-by-side model comparison for a given hypothetical workload.

    Args:
        model_ids:     Models to compare
        input_tokens:  Hypothetical input tokens per call
        output_tokens: Hypothetical output tokens per call

    Returns:
        List of comparison rows sorted by cost ascending
    """
    rows = []
    for mid in model_ids:
        try:
            model = get_model(mid)
            cost = model.cost_for_tokens(input_tokens, output_tokens)
            rows.append({
                "model_id":     model.model_id,
                "provider":     model.provider,
                "display_name": model.display_name,
                "input_cost":   (input_tokens  / 1000) * model.input_per_1k,
                "output_cost":  (output_tokens / 1000) * model.output_per_1k,
                "total_cost":   cost,
                "cost_per_1k_calls": cost * 1000,
                "context_window":    model.context_window,
                "input_per_1k":      model.input_per_1k,
                "output_per_1k":     model.output_per_1k,
            })
        except KeyError:
            rows.append({
                "model_id": mid, "error": f"Not found in registry",
                "total_cost": float("inf"),
            })

    return sorted(rows, key=lambda r: r["total_cost"])


def print_model_breakdown(tracker: CostTracker):
    """Pretty-print model breakdown to stdout."""
    rows = model_breakdown(tracker)
    if not rows:
        print("No calls recorded.")
        return

    total_cost = tracker.total_cost()
    print(f"\n{'═'*80}")
    print(f"  LLM COST BREAKDOWN BY MODEL")
    print(f"{'═'*80}")
    print(f"  {'Model':<40} {'Calls':>6} {'Tokens':>10} {'Cost USD':>12} {'Share':>8}")
    print(f"  {'─'*76}")

    for r in rows:
        bar = "█" * max(1, int(r["pct_of_total"] / 5))
        print(
            f"  {r['model_id']:<40} {r['call_count']:>6,} "
            f"{r['total_tokens']:>10,} ${r['total_cost']:>11.6f} "
            f"{r['pct_of_total']:>6.1f}%  {bar}"
        )

    print(f"  {'─'*76}")
    print(f"  {'TOTAL':<40} {sum(r['call_count'] for r in rows):>6,} "
          f"{sum(r['total_tokens'] for r in rows):>10,} ${total_cost:>11.6f}")
    print(f"{'═'*80}\n")


def print_comparison_table(model_ids: list[str], input_tokens: int = 1000, output_tokens: int = 500):
    """Pretty-print model comparison table."""
    rows = comparison_table(model_ids, input_tokens, output_tokens)
    print(f"\n{'═'*85}")
    print(f"  MODEL COST COMPARISON  (input={input_tokens:,} tokens, output={output_tokens:,} tokens)")
    print(f"{'═'*85}")
    print(f"  {'Model':<38} {'Provider':<12} {'$/1k in':>8} {'$/1k out':>9} {'Total':>12} {'per 1K calls':>14}")
    print(f"  {'─'*81}")

    for r in rows:
        if "error" in r:
            print(f"  {r['model_id']:<38} {'ERROR':>12} — {r['error']}")
            continue
        print(
            f"  {r['model_id']:<38} {r['provider']:<12} "
            f"${r['input_per_1k']:>7.4f} ${r['output_per_1k']:>8.4f} "
            f"${r['total_cost']:>11.6f} ${r['cost_per_1k_calls']:>13.4f}"
        )
    print(f"{'═'*85}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_timestamp(ts: float, granularity: str) -> str:
    """Format a Unix timestamp into a human-readable bucket label."""
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts)
    formats = {
        "minute": "%Y-%m-%d %H:%M",
        "hour":   "%Y-%m-%d %H:00",
        "day":    "%Y-%m-%d",
        "week":   "%Y-W%W",
    }
    return dt.strftime(formats.get(granularity, "%Y-%m-%d %H:00"))
