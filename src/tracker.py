"""
src/tracker.py

CostTracker — the central tracking object.

Records every LLM API call with full metadata: model, tokens, cost, tags,
latency, and timestamp. Supports filtering, aggregation, and export.

Usage:
    tracker = CostTracker()
    tracker.record(model_id="gpt-4o-mini", input_tokens=500, output_tokens=200)

    # With tags for feature-level cost attribution
    tracker.record("gpt-4o", 1200, 400, tags={"feature": "summarisation", "env": "prod"})

    # With budget management
    tracker = CostTracker(session_budget=10.00)
    tracker.record("claude-3-5-sonnet-20241022", 800, 600)

    print(tracker.total_cost())
    print(tracker.summary())
"""

from __future__ import annotations
import time
import json
import csv
import io
from dataclasses import dataclass, field
from typing import Optional

from src.models import get_model, ModelPrice
from src.calculator import calculate, CostEstimate
from src.budget import BudgetManager, BudgetAlert, BudgetExceededError
from src.parser import TokenUsage, parse_response


# ── Call record ───────────────────────────────────────────────────────────────

@dataclass
class CallRecord:
    """
    Complete record of a single LLM API call.

    Attributes:
        call_id:        Auto-incrementing integer ID
        model_id:       Canonical model identifier
        provider:       Provider name
        input_tokens:   Prompt token count
        output_tokens:  Completion token count
        total_tokens:   input + output
        input_cost:     USD cost for input tokens
        output_cost:    USD cost for output tokens
        total_cost:     Total USD cost
        tags:           Arbitrary key-value metadata for filtering
        timestamp:      Unix timestamp of the call
        latency_ms:     Optional measured latency in milliseconds
        cached_tokens:  Tokens served from prompt cache
        reasoning_tokens: Chain-of-thought tokens (o1/R1 models)
        metadata:       Any extra data (request ID, user ID, etc.)
    """
    call_id:          int
    model_id:         str
    provider:         str
    input_tokens:     int
    output_tokens:    int
    total_tokens:     int
    input_cost:       float
    output_cost:      float
    total_cost:       float
    tags:             dict[str, str] = field(default_factory=dict)
    timestamp:        float = field(default_factory=time.time)
    latency_ms:       Optional[float] = None
    cached_tokens:    int = 0
    reasoning_tokens: int = 0
    metadata:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "call_id":          self.call_id,
            "model_id":         self.model_id,
            "provider":         self.provider,
            "input_tokens":     self.input_tokens,
            "output_tokens":    self.output_tokens,
            "total_tokens":     self.total_tokens,
            "input_cost":       self.input_cost,
            "output_cost":      self.output_cost,
            "total_cost":       self.total_cost,
            "tags":             self.tags,
            "timestamp":        self.timestamp,
            "latency_ms":       self.latency_ms,
            "cached_tokens":    self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "metadata":         self.metadata,
        }

    def __repr__(self) -> str:
        tag_str = " ".join(f"{k}={v}" for k, v in self.tags.items())
        return (
            f"CallRecord(#{self.call_id} {self.model_id} "
            f"in={self.input_tokens} out={self.output_tokens} "
            f"${self.total_cost:.6f}{' ' + tag_str if tag_str else ''})"
        )


# ── Main tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Track LLM API costs across a session.

    Args:
        session_budget:  Optional hard session budget in USD
        auto_cutoff:     If True, raises BudgetExceededError when session budget hit
        warn_at:         Fraction of budget that triggers WARNING alert (default 0.8)
    """

    def __init__(
        self,
        session_budget: Optional[float] = None,
        auto_cutoff: bool = False,
        warn_at: float = 0.80,
    ):
        self._calls:   list[CallRecord] = []
        self._call_id: int = 0
        self.budget    = BudgetManager()

        if session_budget is not None:
            self.budget.set_session_budget(
                session_budget,
                warning_threshold=warn_at,
                auto_cutoff=auto_cutoff,
            )

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        model_id:         str,
        input_tokens:     int,
        output_tokens:    int,
        tags:             Optional[dict[str, str]] = None,
        latency_ms:       Optional[float] = None,
        cached_tokens:    int = 0,
        reasoning_tokens: int = 0,
        metadata:         Optional[dict] = None,
    ) -> CallRecord:
        """
        Record a completed LLM API call.

        Args:
            model_id:         Model used (name or alias)
            input_tokens:     Number of input tokens
            output_tokens:    Number of output tokens
            tags:             Key-value metadata for filtering (e.g. {"feature": "chat"})
            latency_ms:       API call latency in milliseconds
            cached_tokens:    Tokens served from prompt cache (if any)
            reasoning_tokens: Internal CoT tokens (o1/R1 models)
            metadata:         Any extra data to attach to the record

        Returns:
            CallRecord with computed costs

        Raises:
            KeyError: If model_id is not in the pricing registry
            BudgetExceededError: If auto_cutoff=True and budget is exceeded
        """
        self.budget.check(model_id)

        estimate = calculate(model_id, input_tokens, output_tokens)
        self._call_id += 1

        record = CallRecord(
            call_id=self._call_id,
            model_id=estimate.model_id,
            provider=estimate.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=estimate.input_cost,
            output_cost=estimate.output_cost,
            total_cost=estimate.total_cost,
            tags=tags or {},
            latency_ms=latency_ms,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            metadata=metadata or {},
        )
        self._calls.append(record)
        self.budget.record_spend(estimate.total_cost, model_id=estimate.model_id)
        return record

    def record_response(
        self,
        model_id:   str,
        response:   object,
        provider:   Optional[str] = None,
        tags:       Optional[dict[str, str]] = None,
        latency_ms: Optional[float] = None,
        metadata:   Optional[dict] = None,
    ) -> CallRecord:
        """
        Record a call by parsing token counts directly from the provider response.

        Args:
            model_id:  Model used
            response:  Raw API response (dict or SDK response object)
            provider:  Provider hint for parsing ("openai", "anthropic", etc.)
            tags:      Metadata tags
            latency_ms: Call latency
            metadata:  Extra metadata

        Returns:
            CallRecord with costs computed from parsed token counts
        """
        usage = parse_response(response, provider=provider)
        return self.record(
            model_id=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tags=tags,
            latency_ms=latency_ms,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            metadata=metadata,
        )

    # ── Querying ──────────────────────────────────────────────────────────────

    def filter(
        self,
        model_id:  Optional[str] = None,
        provider:  Optional[str] = None,
        tags:      Optional[dict[str, str]] = None,
        since:     Optional[float] = None,
        until:     Optional[float] = None,
    ) -> list[CallRecord]:
        """
        Filter call records by criteria.

        Args:
            model_id:  Filter by exact model ID
            provider:  Filter by provider name
            tags:      Filter by tag key-value pairs (AND match)
            since:     Unix timestamp — only calls after this time
            until:     Unix timestamp — only calls before this time

        Returns:
            Filtered list of CallRecord objects
        """
        results = self._calls

        if model_id:
            results = [r for r in results if r.model_id == model_id]
        if provider:
            results = [r for r in results if r.provider.lower() == provider.lower()]
        if tags:
            results = [
                r for r in results
                if all(r.tags.get(k) == v for k, v in tags.items())
            ]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        if until is not None:
            results = [r for r in results if r.timestamp <= until]

        return results

    def total_cost(self, **filter_kwargs) -> float:
        """Total cost in USD, with optional filtering."""
        return sum(r.total_cost for r in self.filter(**filter_kwargs))

    def total_tokens(self, **filter_kwargs) -> dict[str, int]:
        """Total token counts (input, output, total), with optional filtering."""
        records = self.filter(**filter_kwargs)
        return {
            "input":  sum(r.input_tokens  for r in records),
            "output": sum(r.output_tokens for r in records),
            "total":  sum(r.total_tokens  for r in records),
        }

    def call_count(self, **filter_kwargs) -> int:
        """Number of recorded calls, with optional filtering."""
        return len(self.filter(**filter_kwargs))

    def cost_by_model(self) -> dict[str, float]:
        """Total cost per model, sorted by cost descending."""
        totals: dict[str, float] = {}
        for r in self._calls:
            totals[r.model_id] = totals.get(r.model_id, 0.0) + r.total_cost
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    def cost_by_provider(self) -> dict[str, float]:
        """Total cost per provider, sorted by cost descending."""
        totals: dict[str, float] = {}
        for r in self._calls:
            totals[r.provider] = totals.get(r.provider, 0.0) + r.total_cost
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    def cost_by_tag(self, tag_key: str) -> dict[str, float]:
        """
        Total cost broken down by a tag value.

        Args:
            tag_key: The tag key to group by (e.g. "feature" or "user_id")

        Returns:
            Dict of {tag_value → total_cost}

        Example:
            tracker.cost_by_tag("feature")
            # {"summarisation": 0.023, "chat": 0.089, "(untagged)": 0.001}
        """
        totals: dict[str, float] = {}
        for r in self._calls:
            key = r.tags.get(tag_key, "(untagged)")
            totals[key] = totals.get(key, 0.0) + r.total_cost
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))

    def avg_cost_per_call(self, **filter_kwargs) -> float:
        """Average cost per call, with optional filtering."""
        records = self.filter(**filter_kwargs)
        if not records:
            return 0.0
        return sum(r.total_cost for r in records) / len(records)

    def avg_latency_ms(self, **filter_kwargs) -> Optional[float]:
        """Average latency across calls that reported latency."""
        records = [r for r in self.filter(**filter_kwargs) if r.latency_ms is not None]
        if not records:
            return None
        return sum(r.latency_ms for r in records) / len(records)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable cost summary for the current session."""
        if not self._calls:
            return "No calls recorded yet."

        total = self.total_cost()
        tok   = self.total_tokens()
        lines = [
            f"{'─'*50}",
            f"  LLM Cost Tracker — Session Summary",
            f"{'─'*50}",
            f"  Total calls:    {len(self._calls):,}",
            f"  Total tokens:   {tok['total']:,} (in={tok['input']:,} out={tok['output']:,})",
            f"  Total cost:     ${total:.6f}",
            f"  Avg per call:   ${self.avg_cost_per_call():.6f}",
            f"",
            f"  Cost by model:",
        ]
        for model, cost in self.cost_by_model().items():
            pct = cost / total * 100 if total else 0
            bar = "█" * max(1, int(pct / 5))
            lines.append(f"    {model:<40} ${cost:.6f}  {pct:5.1f}%  {bar}")

        budget_status = self.budget.status()
        if budget_status["budgets"]:
            lines.append(f"")
            lines.append(f"  Budget status:")
            for scope, b in budget_status["budgets"].items():
                lines.append(
                    f"    {scope:<20} ${b['spent']:.4f} / ${b['limit']:.4f} "
                    f"({b['pct_used']:.0f}%)  [{b['status'].upper()}]"
                )

        lines.append(f"{'─'*50}")
        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Export all call records as JSON string."""
        return json.dumps(
            {
                "session_summary": {
                    "total_calls": len(self._calls),
                    "total_cost_usd": self.total_cost(),
                    "total_tokens": self.total_tokens(),
                    "cost_by_model": self.cost_by_model(),
                    "cost_by_provider": self.cost_by_provider(),
                },
                "calls": [r.to_dict() for r in self._calls],
            },
            indent=indent,
        )

    def to_csv(self) -> str:
        """Export all call records as CSV string."""
        if not self._calls:
            return ""
        output = io.StringIO()
        fieldnames = [
            "call_id", "model_id", "provider", "input_tokens", "output_tokens",
            "total_tokens", "input_cost", "output_cost", "total_cost",
            "timestamp", "latency_ms", "cached_tokens", "reasoning_tokens",
        ]
        # Add dynamic tag columns
        all_tag_keys = sorted(set(k for r in self._calls for k in r.tags.keys()))
        fieldnames.extend(f"tag_{k}" for k in all_tag_keys)

        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in self._calls:
            row = r.to_dict()
            row.pop("tags", None)
            row.pop("metadata", None)
            for k in all_tag_keys:
                row[f"tag_{k}"] = r.tags.get(k, "")
            writer.writerow(row)
        return output.getvalue()

    def save_json(self, path: str):
        """Save session to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"Session saved to {path}")

    def save_csv(self, path: str):
        """Save session to a CSV file."""
        with open(path, "w") as f:
            f.write(self.to_csv())
        print(f"Session saved to {path}")

    def __repr__(self) -> str:
        return (
            f"CostTracker(calls={len(self._calls)}, "
            f"total_cost=${self.total_cost():.6f})"
        )
