<div align="center">

# 💸 LLM Cost Tracker

### Track, analyse, and control your LLM API spending across every major provider

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Providers](https://img.shields.io/badge/providers-11-orange.svg)]()
[![Models](https://img.shields.io/badge/models-41+-blue.svg)]()

**41+ models · 11 providers · Zero external dependencies**

*Know exactly what every LLM call costs — before it shows up on your invoice.*

</div>

---

## The Problem

You're running LLM-powered features in production. Costs are climbing. You don't know which model, feature, or user is responsible. Your monthly invoice from OpenAI arrives and you can't explain 60% of it.

`llm-cost-tracker` solves this. It is a pure-Python library — no external dependencies, no API keys, no database — that gives you:

- **Exact per-call costs** computed from the official pricing registry
- **Session aggregation** with filtering by model, provider, tag, or time window
- **Budget management** with configurable thresholds, alerts, and hard cutoffs
- **Feature-level attribution** via arbitrary tagging (`feature=chat`, `user=u001`, `env=prod`)
- **Pre-flight estimation** so you know what a batch will cost before you run it
- **Multi-provider parsing** — one function call extracts token counts from any provider response

---

## Quickstart

```bash
git clone https://github.com/prajitdatta/llm-cost-tracker
cd llm-cost-tracker
python examples/basic_tracking.py
```

**Zero pip installs for core functionality.** Only `pytest` needed to run the test suite.

---

## Core Usage

### Track a call

```python
from src.tracker import CostTracker

tracker = CostTracker()
record = tracker.record(
    model_id="gpt-4o-mini",
    input_tokens=750,
    output_tokens=300,
    tags={"feature": "chat", "user": "u001", "env": "prod"},
)

print(record)
# CallRecord(#1 gpt-4o-mini in=750 out=300 $0.000292 feature=chat user=u001 env=prod)

print(tracker.summary())
```

### Parse tokens directly from the provider response

```python
# Works with OpenAI, Anthropic, Google, Mistral, Cohere, Bedrock — auto-detected
response = openai_client.chat.completions.create(...)
record = tracker.record_response("gpt-4o-mini", response)
```

### Set a budget — get alerts before you overspend

```python
tracker = CostTracker(session_budget=10.00, auto_cutoff=False)

# Fine-grained control
tracker.budget.set_model_budget("gpt-4o", limit_usd=5.00, auto_cutoff=True)
tracker.budget.set_daily_budget(2.00, on_alert=lambda a: notify_slack(a.message))
```

### Cost attribution by feature, user, or environment

```python
# After recording 1000 calls with tags...
by_feature = tracker.cost_by_tag("feature")
# {"chat": 0.023, "doc-qa": 0.089, "summarise": 0.011}

by_env = tracker.cost_by_tag("env")
# {"prod": 0.098, "staging": 0.025}
```

### Calculate cost before running anything

```python
from src.calculator import calculate, batch_estimate, monthly_projection

# Single call
est = calculate("gpt-4o-mini", input_tokens=1000, output_tokens=500)
print(f"${est.total_cost:.6f}")   # $0.000450

# Batch of 50,000 support tickets
total = calculate("gpt-4o-mini", 800, 300).total_cost * 50_000
print(f"${total:.2f}")            # $6.75

# Monthly projection at 10k calls/day
proj = monthly_projection("gpt-4o-mini", calls_per_day=10_000, avg_input_tokens=800, avg_output_tokens=300)
print(f"${proj['monthly_cost']:.2f}/month")
```

---

## Supported Models & Providers

| Provider | Models Included |
|----------|----------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo, o1, o1-mini, o3-mini |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.5 Flash-8B, Gemini 1.0 Pro |
| **Mistral** | Mistral Large, Mistral Small, Mistral 7B, Mixtral 8x7B, Mixtral 8x22B, Codestral |
| **Meta / Llama** | Llama 3.1 405B, Llama 3.1 70B, Llama 3.1 8B, Llama 3 70B |
| **Cohere** | Command R+, Command R, Command |
| **DeepSeek** | DeepSeek Chat (V3), DeepSeek Reasoner (R1) |
| **AI21** | Jamba 1.5 Large, Jamba 1.5 Mini |
| **Perplexity** | Sonar Large Online, Sonar Small Online |
| **AWS Bedrock** | Amazon Titan Express/Lite, Claude on Bedrock |
| **Groq** | Llama 3.3 70B, Mixtral 8x7B |

All prices are USD per 1,000 tokens. Update prices by editing `src/models.py` — no code changes needed.

---

## Project Structure

```
llm-cost-tracker/
│
├── src/                         # Core library — zero external dependencies
│   ├── models.py                # Pricing registry: 41+ models, 11 providers
│   ├── calculator.py            # Cost calculation, batch estimation, projections
│   ├── parser.py                # Token extraction from any provider response format
│   ├── budget.py                # BudgetManager: limits, alerts, auto-cutoff
│   ├── tracker.py               # CostTracker: per-call recording, session aggregation
│   └── reporter.py              # Reports: model breakdown, tag attribution, time-series
│
├── tests/                       # Full test suite — 100+ assertions
│   ├── test_models.py           # Pricing accuracy, alias resolution, provider coverage
│   ├── test_calculator.py       # Cost math, batch estimates, monthly projections
│   ├── test_parser.py           # Response parsing for every provider format
│   ├── test_budget.py           # Budget limits, alert thresholds, auto-cutoff
│   ├── test_tracker.py          # Session tracking, aggregation, tag filtering, export
│   └── test_reporter.py         # Report generation, comparison tables, efficiency
│
├── examples/                    # Runnable examples — no API keys needed
│   ├── basic_tracking.py        # Hello world: track a single call
│   ├── budget_alerts.py         # Set budget, watch alerts fire, demo auto-cutoff
│   ├── multi_model_comparison.py# Compare GPT-4o vs Claude vs Gemini vs DeepSeek
│   ├── tagging_and_reports.py   # Tag by feature/env/user, generate attribution report
│   └── batch_cost_estimation.py # Estimate 50k-call batch cost before running it
│
└── scripts/
    ├── benchmark_models.py      # Print full pricing table, sort by any column
    └── export_session.py        # Export session to JSON and CSV
```

---

## API Reference

### `CostTracker`

```python
tracker = CostTracker(session_budget=None, auto_cutoff=False, warn_at=0.80)

# Record calls
tracker.record(model_id, input_tokens, output_tokens, tags={}, latency_ms=None)
tracker.record_response(model_id, api_response, provider=None, tags={})

# Query
tracker.total_cost()
tracker.total_tokens()
tracker.call_count()
tracker.cost_by_model()
tracker.cost_by_provider()
tracker.cost_by_tag(tag_key)
tracker.filter(model_id=None, provider=None, tags=None, since=None, until=None)
tracker.avg_cost_per_call()
tracker.avg_latency_ms()

# Export
tracker.to_json()
tracker.to_csv()
tracker.save_json(path)
tracker.save_csv(path)
tracker.summary()
```

### `BudgetManager`

```python
budget = BudgetManager()
budget.set_session_budget(limit_usd, warning_threshold=0.80, auto_cutoff=False, on_alert=None)
budget.set_daily_budget(limit_usd, auto_cutoff=False)
budget.set_model_budget(model_id, limit_usd, auto_cutoff=False)

budget.record_spend(amount_usd, model_id=None)   # Returns list[BudgetAlert]
budget.check(model_id=None)                       # Raises BudgetExceededError if over limit
budget.status()                                   # Full status dict
budget.get_alerts()                               # All triggered alerts
budget.reset("session" | "daily" | model_id)
```

### `calculator`

```python
from src.calculator import calculate, estimate_from_text, batch_estimate, monthly_projection, break_even_analysis

calculate("gpt-4o-mini", input_tokens=1000, output_tokens=500)  → CostEstimate
estimate_from_text("gpt-4o-mini", text, estimated_output_tokens=500)  → CostEstimate
batch_estimate("gpt-4o-mini", [{"input_tokens": 500, "output_tokens": 200}, ...])  → dict
monthly_projection("gpt-4o-mini", calls_per_day=1000, avg_input_tokens=500, avg_output_tokens=200)  → dict
break_even_analysis("gpt-4o", "gpt-4o-mini", input_tokens=1000, output_tokens=500)  → dict
```

### `parser`

```python
from src.parser import parse_response

# Auto-detects provider from response structure
usage = parse_response(api_response)  # → TokenUsage(input, output, total, cached, reasoning)

# Or specify explicitly
usage = parse_response(response, provider="openai")
usage = parse_response(response, provider="anthropic")
usage = parse_response(response, provider="google")
```

---

## Running Examples

```bash
# Hello world — single call tracking
python examples/basic_tracking.py

# Budget alerts and auto-cutoff
python examples/budget_alerts.py

# Multi-model cost comparison (GPT-4o vs Claude vs Gemini vs DeepSeek)
python examples/multi_model_comparison.py

# Tag-based cost attribution report
python examples/tagging_and_reports.py

# Pre-flight batch cost estimation
python examples/batch_cost_estimation.py

# Full pricing table for all 41+ models
python scripts/benchmark_models.py
python scripts/benchmark_models.py --provider openai --sort output
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover: pricing accuracy, alias resolution, cost calculation math, response parsing for all providers, budget alert logic, auto-cutoff behaviour, session aggregation, tag filtering, report generation, CSV/JSON export.

---

## Model Cost Comparison (1k input + 500 output tokens)

```
deepseek-chat              $0.000049
gemini-1.5-flash           $0.000225
gpt-4o-mini                $0.000450
claude-3-haiku-20240307    $0.000875
gemini-1.5-pro             $0.003750
mistral-large-latest       $0.007500
gpt-4o                     $0.007500
claude-3-5-sonnet          $0.010500
claude-3-opus-20240229     $0.052500
```

Run `python scripts/benchmark_models.py` for the full live table.

---

## Topics

`llm-cost-tracking` · `openai-cost` · `anthropic-cost` · `llm-pricing` · `gpt-4o-pricing` · `claude-pricing` · `gemini-pricing` · `token-cost` · `llm-budget` · `ai-cost-management` · `llm-observability` · `mlops` · `llm-monitoring` · `api-cost-tracking` · `openai-token-counter` · `llm-billing` · `cost-per-token` · `ai-spend-management` · `production-llm` · `llm-analytics`

---

## Who This Is For

**AI/ML engineers** who ship LLM-powered features to production and need to understand cost structure before it hits the monthly invoice.

**Engineering teams** evaluating model tradeoffs — not just on quality but on cost per output token, cost per feature, and monthly projection at scale.

**Startups** building on top of LLM APIs who need budget guardrails before their inference spend outpaces revenue.

**Researchers** running large-scale experiments who need pre-flight cost estimates and per-run attribution.

---

## Contributing

The pricing registry lives in `src/models.py` as a plain Python dict. To add a model or update a price:

1. Add or edit the `ModelPrice` entry in `REGISTRY`
2. Add any aliases to the `aliases` list
3. Run `pytest tests/test_models.py` to verify

No code changes needed for price updates — just the registry dict.

---

## Author

**Prajit Datta** — AI Research Scientist, AFRY | [prajitdatta.com](https://prajitdatta.com)

Built from the recurring experience of watching LLM API invoices grow faster than the value they deliver — and needing a precise, dependency-free way to track why.

---

<div align="center">

**⭐ Star if you've ever been surprised by an LLM API invoice.**

*The star tells the next engineer that a clean, dependency-free solution exists.*

</div>
