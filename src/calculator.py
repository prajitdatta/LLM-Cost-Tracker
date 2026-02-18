"""
src/calculator.py

Standalone cost calculator — pure functions, no tracking state.

Use this for:
    - Pre-flight cost estimation before running a batch
    - Comparing model costs for a given workload
    - Budget planning with token estimates

For live tracking of actual API calls, use src/tracker.py instead.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from src.models import get_model, ModelPrice, compare_models


@dataclass
class CostEstimate:
    """Result of a cost calculation."""
    model_id:       str
    provider:       str
    input_tokens:   int
    output_tokens:  int
    input_cost:     float   # USD
    output_cost:    float   # USD
    total_cost:     float   # USD
    cost_per_1k_calls: float  # USD if you ran 1,000 calls like this

    def __repr__(self) -> str:
        return (
            f"CostEstimate({self.model_id} | "
            f"in={self.input_tokens:,}tok out={self.output_tokens:,}tok | "
            f"${self.total_cost:.6f})"
        )

    def summary(self) -> str:
        return (
            f"Model:         {self.model_id} ({self.provider})\n"
            f"Input tokens:  {self.input_tokens:,}  → ${self.input_cost:.6f}\n"
            f"Output tokens: {self.output_tokens:,}  → ${self.output_cost:.6f}\n"
            f"Total cost:    ${self.total_cost:.6f}\n"
            f"Cost per 1K calls: ${self.cost_per_1k_calls:.4f}"
        )


def calculate(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> CostEstimate:
    """
    Calculate cost for a single LLM call.

    Args:
        model_id:      Model identifier (name or alias)
        input_tokens:  Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        CostEstimate with full breakdown

    Raises:
        KeyError: If model is not found in pricing registry
        ValueError: If token counts are negative
    """
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be >= 0, got {input_tokens}")
    if output_tokens < 0:
        raise ValueError(f"output_tokens must be >= 0, got {output_tokens}")

    model = get_model(model_id)
    input_cost  = (input_tokens  / 1000) * model.input_per_1k
    output_cost = (output_tokens / 1000) * model.output_per_1k
    total_cost  = input_cost + output_cost

    return CostEstimate(
        model_id=model.model_id,
        provider=model.provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        cost_per_1k_calls=total_cost * 1000,
    )


def estimate_from_text(
    model_id: str,
    input_text: str,
    estimated_output_tokens: int = 500,
    chars_per_token: float = 4.0,
) -> CostEstimate:
    """
    Estimate cost from raw text (no tokeniser required).

    Uses character-based approximation: 1 token ≈ 4 characters for English.

    Args:
        model_id:                Model identifier
        input_text:              The input text (prompt + context)
        estimated_output_tokens: Expected output length in tokens
        chars_per_token:         Characters per token ratio (default 4.0)

    Returns:
        CostEstimate (note: token counts are approximate)
    """
    estimated_input_tokens = max(1, math.ceil(len(input_text) / chars_per_token))
    return calculate(model_id, estimated_input_tokens, estimated_output_tokens)


def batch_estimate(
    model_id: str,
    calls: list[dict],
) -> dict:
    """
    Estimate total cost for a batch of calls.

    Args:
        model_id: Model to use for all calls
        calls:    List of dicts with "input_tokens" and "output_tokens" keys

    Returns:
        Dict with total_cost, call_count, breakdown per call, and statistics

    Example:
        calls = [
            {"input_tokens": 500, "output_tokens": 200},
            {"input_tokens": 1200, "output_tokens": 800},
        ]
        result = batch_estimate("gpt-4o-mini", calls)
    """
    model = get_model(model_id)
    estimates = []
    total_input  = 0
    total_output = 0
    total_cost   = 0.0

    for i, call in enumerate(calls):
        in_tok  = call.get("input_tokens", 0)
        out_tok = call.get("output_tokens", 0)
        est = calculate(model_id, in_tok, out_tok)
        estimates.append(est)
        total_input  += in_tok
        total_output += out_tok
        total_cost   += est.total_cost

    costs = [e.total_cost for e in estimates]

    return {
        "model_id":         model.model_id,
        "provider":         model.provider,
        "call_count":       len(calls),
        "total_input_tokens":  total_input,
        "total_output_tokens": total_output,
        "total_tokens":     total_input + total_output,
        "total_cost_usd":   total_cost,
        "avg_cost_per_call": total_cost / len(calls) if calls else 0.0,
        "min_cost_per_call": min(costs) if costs else 0.0,
        "max_cost_per_call": max(costs) if costs else 0.0,
        "estimates":        estimates,
    }


def monthly_projection(
    model_id: str,
    calls_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
) -> dict:
    """
    Project monthly and annual costs based on daily usage patterns.

    Args:
        model_id:          Model identifier
        calls_per_day:     Expected number of API calls per day
        avg_input_tokens:  Average input tokens per call
        avg_output_tokens: Average output tokens per call

    Returns:
        Dict with daily/weekly/monthly/annual cost projections
    """
    daily_est = calculate(model_id, avg_input_tokens, avg_output_tokens)
    daily_cost   = daily_est.total_cost * calls_per_day

    return {
        "model_id":          model_id,
        "calls_per_day":     calls_per_day,
        "avg_input_tokens":  avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "cost_per_call":     daily_est.total_cost,
        "daily_cost":        daily_cost,
        "weekly_cost":       daily_cost * 7,
        "monthly_cost":      daily_cost * 30,
        "annual_cost":       daily_cost * 365,
        "daily_tokens":      (avg_input_tokens + avg_output_tokens) * calls_per_day,
    }


def break_even_analysis(
    model_a: str,
    model_b: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """
    At what scale does switching from model_a to model_b pay off?

    Useful for deciding whether a cheaper model is worth the quality tradeoff.

    Returns:
        Dict comparing both models with savings per call and per 1M calls
    """
    est_a = calculate(model_a, input_tokens, output_tokens)
    est_b = calculate(model_b, input_tokens, output_tokens)

    savings_per_call = est_a.total_cost - est_b.total_cost
    cheaper = model_b if savings_per_call > 0 else model_a
    pct_saving = abs(savings_per_call) / max(est_a.total_cost, est_b.total_cost) * 100

    return {
        "model_a":         {"model_id": est_a.model_id, "cost_per_call": est_a.total_cost},
        "model_b":         {"model_id": est_b.model_id, "cost_per_call": est_b.total_cost},
        "cheaper_model":   cheaper,
        "savings_per_call": abs(savings_per_call),
        "savings_per_1k_calls": abs(savings_per_call) * 1000,
        "savings_per_1m_calls": abs(savings_per_call) * 1_000_000,
        "pct_cheaper":     round(pct_saving, 2),
    }
