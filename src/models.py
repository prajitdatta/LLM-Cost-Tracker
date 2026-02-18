"""
src/models.py

Pricing registry for 40+ LLM models across all major providers.

Prices are in USD per 1,000 tokens (input and output separately).
Sources: official provider pricing pages as of early 2025.
Update prices by editing the REGISTRY dict — no code changes needed.

Providers covered:
    OpenAI      — GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo, o1, o3
    Anthropic   — Claude 3.5 Sonnet/Haiku, Claude 3 Opus/Sonnet/Haiku
    Google      — Gemini 1.5 Pro/Flash, Gemini 1.0 Pro
    Meta        — Llama 3.1 (via hosted APIs)
    Mistral     — Mistral Large/Medium/Small/7B
    Cohere      — Command R, Command R+
    AI21        — Jamba 1.5
    Perplexity  — Sonar models
    Together AI — Community-hosted open models
    AWS Bedrock — Titan, Claude on Bedrock pricing
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPrice:
    """
    Pricing for a single model.

    Attributes:
        model_id:       Canonical model identifier (e.g. "gpt-4o")
        provider:       Provider name (e.g. "openai")
        display_name:   Human-readable name
        input_per_1k:   USD cost per 1,000 input tokens
        output_per_1k:  USD cost per 1,000 output tokens
        context_window: Max context window in tokens
        aliases:        Alternative model IDs that map to this pricing
        notes:          Any special notes (e.g. "cached input at 50% discount")
    """
    model_id:       str
    provider:       str
    display_name:   str
    input_per_1k:   float
    output_per_1k:  float
    context_window: int
    aliases:        list[str] = field(default_factory=list)
    notes:          str = ""

    def cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost in USD for a given token count."""
        return (
            (input_tokens  / 1000) * self.input_per_1k +
            (output_tokens / 1000) * self.output_per_1k
        )

    def cost_per_1m_tokens(self) -> dict[str, float]:
        """Return cost per 1M tokens (common unit in provider marketing)."""
        return {
            "input":  self.input_per_1k  * 1000,
            "output": self.output_per_1k * 1000,
        }

    def __repr__(self) -> str:
        return (
            f"ModelPrice({self.model_id} | {self.provider} | "
            f"in=${self.input_per_1k:.4f}/1k out=${self.output_per_1k:.4f}/1k)"
        )


# ── Master pricing registry ───────────────────────────────────────────────────
# Format: model_id → ModelPrice
# All prices USD per 1,000 tokens. Last updated: early 2025.

REGISTRY: dict[str, ModelPrice] = {

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o": ModelPrice(
        model_id="gpt-4o", provider="openai",
        display_name="GPT-4o",
        input_per_1k=0.002500, output_per_1k=0.010000,
        context_window=128_000,
        aliases=["gpt-4o-2024-11-20", "gpt-4o-2024-08-06"],
        notes="Flagship multimodal model. Cached input: $0.00125/1k",
    ),
    "gpt-4o-mini": ModelPrice(
        model_id="gpt-4o-mini", provider="openai",
        display_name="GPT-4o mini",
        input_per_1k=0.000150, output_per_1k=0.000600,
        context_window=128_000,
        aliases=["gpt-4o-mini-2024-07-18"],
        notes="Best value for high-volume tasks",
    ),
    "gpt-4-turbo": ModelPrice(
        model_id="gpt-4-turbo", provider="openai",
        display_name="GPT-4 Turbo",
        input_per_1k=0.010000, output_per_1k=0.030000,
        context_window=128_000,
        aliases=["gpt-4-turbo-preview", "gpt-4-turbo-2024-04-09"],
    ),
    "gpt-4": ModelPrice(
        model_id="gpt-4", provider="openai",
        display_name="GPT-4",
        input_per_1k=0.030000, output_per_1k=0.060000,
        context_window=8_192,
        aliases=["gpt-4-0613"],
    ),
    "gpt-3.5-turbo": ModelPrice(
        model_id="gpt-3.5-turbo", provider="openai",
        display_name="GPT-3.5 Turbo",
        input_per_1k=0.000500, output_per_1k=0.001500,
        context_window=16_385,
        aliases=["gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"],
    ),
    "o1": ModelPrice(
        model_id="o1", provider="openai",
        display_name="OpenAI o1",
        input_per_1k=0.015000, output_per_1k=0.060000,
        context_window=200_000,
        aliases=["o1-2024-12-17"],
        notes="Reasoning model. Includes internal chain-of-thought tokens.",
    ),
    "o1-mini": ModelPrice(
        model_id="o1-mini", provider="openai",
        display_name="OpenAI o1-mini",
        input_per_1k=0.003000, output_per_1k=0.012000,
        context_window=128_000,
        aliases=["o1-mini-2024-09-12"],
    ),
    "o3-mini": ModelPrice(
        model_id="o3-mini", provider="openai",
        display_name="OpenAI o3-mini",
        input_per_1k=0.001100, output_per_1k=0.004400,
        context_window=200_000,
        notes="Efficient reasoning model",
    ),

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-3-5-sonnet-20241022": ModelPrice(
        model_id="claude-3-5-sonnet-20241022", provider="anthropic",
        display_name="Claude 3.5 Sonnet",
        input_per_1k=0.003000, output_per_1k=0.015000,
        context_window=200_000,
        aliases=["claude-3-5-sonnet", "claude-sonnet-4-5-20251001"],
        notes="Best intelligence. Cached input: $0.00030/1k",
    ),
    "claude-3-5-haiku-20241022": ModelPrice(
        model_id="claude-3-5-haiku-20241022", provider="anthropic",
        display_name="Claude 3.5 Haiku",
        input_per_1k=0.000800, output_per_1k=0.004000,
        context_window=200_000,
        aliases=["claude-3-5-haiku", "claude-haiku-4-5-20251001"],
    ),
    "claude-3-opus-20240229": ModelPrice(
        model_id="claude-3-opus-20240229", provider="anthropic",
        display_name="Claude 3 Opus",
        input_per_1k=0.015000, output_per_1k=0.075000,
        context_window=200_000,
        aliases=["claude-3-opus"],
        notes="Most powerful Claude 3 model",
    ),
    "claude-3-sonnet-20240229": ModelPrice(
        model_id="claude-3-sonnet-20240229", provider="anthropic",
        display_name="Claude 3 Sonnet",
        input_per_1k=0.003000, output_per_1k=0.015000,
        context_window=200_000,
        aliases=["claude-3-sonnet"],
    ),
    "claude-3-haiku-20240307": ModelPrice(
        model_id="claude-3-haiku-20240307", provider="anthropic",
        display_name="Claude 3 Haiku",
        input_per_1k=0.000250, output_per_1k=0.001250,
        context_window=200_000,
        aliases=["claude-3-haiku"],
        notes="Fastest Claude 3 model",
    ),

    # ── Google ────────────────────────────────────────────────────────────────
    "gemini-1.5-pro": ModelPrice(
        model_id="gemini-1.5-pro", provider="google",
        display_name="Gemini 1.5 Pro",
        input_per_1k=0.001250, output_per_1k=0.005000,
        context_window=2_000_000,
        aliases=["gemini-1.5-pro-latest", "gemini-1.5-pro-002"],
        notes="Up to 128k tokens: $0.00125/1k in. Over 128k: $0.0025/1k in",
    ),
    "gemini-1.5-flash": ModelPrice(
        model_id="gemini-1.5-flash", provider="google",
        display_name="Gemini 1.5 Flash",
        input_per_1k=0.000075, output_per_1k=0.000300,
        context_window=1_000_000,
        aliases=["gemini-1.5-flash-latest", "gemini-1.5-flash-002"],
        notes="Best price for high-volume Gemini tasks",
    ),
    "gemini-1.5-flash-8b": ModelPrice(
        model_id="gemini-1.5-flash-8b", provider="google",
        display_name="Gemini 1.5 Flash-8B",
        input_per_1k=0.0000375, output_per_1k=0.000150,
        context_window=1_000_000,
    ),
    "gemini-1.0-pro": ModelPrice(
        model_id="gemini-1.0-pro", provider="google",
        display_name="Gemini 1.0 Pro",
        input_per_1k=0.000500, output_per_1k=0.001500,
        context_window=32_760,
    ),

    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral-large-latest": ModelPrice(
        model_id="mistral-large-latest", provider="mistral",
        display_name="Mistral Large",
        input_per_1k=0.003000, output_per_1k=0.009000,
        context_window=128_000,
        aliases=["mistral-large-2411"],
    ),
    "mistral-small-latest": ModelPrice(
        model_id="mistral-small-latest", provider="mistral",
        display_name="Mistral Small",
        input_per_1k=0.000200, output_per_1k=0.000600,
        context_window=32_000,
        aliases=["mistral-small-2409"],
    ),
    "open-mistral-7b": ModelPrice(
        model_id="open-mistral-7b", provider="mistral",
        display_name="Mistral 7B",
        input_per_1k=0.000250, output_per_1k=0.000250,
        context_window=32_000,
        aliases=["mistral-tiny"],
    ),
    "open-mixtral-8x7b": ModelPrice(
        model_id="open-mixtral-8x7b", provider="mistral",
        display_name="Mixtral 8x7B",
        input_per_1k=0.000700, output_per_1k=0.000700,
        context_window=32_000,
        aliases=["mistral-small"],
    ),
    "open-mixtral-8x22b": ModelPrice(
        model_id="open-mixtral-8x22b", provider="mistral",
        display_name="Mixtral 8x22B",
        input_per_1k=0.002000, output_per_1k=0.006000,
        context_window=64_000,
    ),
    "codestral-latest": ModelPrice(
        model_id="codestral-latest", provider="mistral",
        display_name="Codestral",
        input_per_1k=0.001000, output_per_1k=0.003000,
        context_window=32_000,
        notes="Optimised for code generation",
    ),

    # ── Cohere ────────────────────────────────────────────────────────────────
    "command-r-plus": ModelPrice(
        model_id="command-r-plus", provider="cohere",
        display_name="Command R+",
        input_per_1k=0.002500, output_per_1k=0.010000,
        context_window=128_000,
        aliases=["command-r-plus-08-2024"],
    ),
    "command-r": ModelPrice(
        model_id="command-r", provider="cohere",
        display_name="Command R",
        input_per_1k=0.000150, output_per_1k=0.000600,
        context_window=128_000,
        aliases=["command-r-08-2024"],
    ),
    "command": ModelPrice(
        model_id="command", provider="cohere",
        display_name="Command",
        input_per_1k=0.001000, output_per_1k=0.002000,
        context_window=4_096,
    ),

    # ── Meta / Llama (via hosted APIs) ────────────────────────────────────────
    "llama-3.1-405b-instruct": ModelPrice(
        model_id="llama-3.1-405b-instruct", provider="meta",
        display_name="Llama 3.1 405B",
        input_per_1k=0.003000, output_per_1k=0.003000,
        context_window=128_000,
        aliases=["meta-llama/Meta-Llama-3.1-405B-Instruct"],
        notes="Pricing via Together AI / Fireworks AI",
    ),
    "llama-3.1-70b-instruct": ModelPrice(
        model_id="llama-3.1-70b-instruct", provider="meta",
        display_name="Llama 3.1 70B",
        input_per_1k=0.000900, output_per_1k=0.000900,
        context_window=128_000,
        aliases=["meta-llama/Meta-Llama-3.1-70B-Instruct"],
    ),
    "llama-3.1-8b-instruct": ModelPrice(
        model_id="llama-3.1-8b-instruct", provider="meta",
        display_name="Llama 3.1 8B",
        input_per_1k=0.000200, output_per_1k=0.000200,
        context_window=128_000,
        aliases=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
    ),
    "llama-3-70b-instruct": ModelPrice(
        model_id="llama-3-70b-instruct", provider="meta",
        display_name="Llama 3 70B",
        input_per_1k=0.000900, output_per_1k=0.000900,
        context_window=8_192,
    ),

    # ── AI21 ──────────────────────────────────────────────────────────────────
    "jamba-1.5-large": ModelPrice(
        model_id="jamba-1.5-large", provider="ai21",
        display_name="Jamba 1.5 Large",
        input_per_1k=0.002000, output_per_1k=0.008000,
        context_window=256_000,
    ),
    "jamba-1.5-mini": ModelPrice(
        model_id="jamba-1.5-mini", provider="ai21",
        display_name="Jamba 1.5 Mini",
        input_per_1k=0.000200, output_per_1k=0.000400,
        context_window=256_000,
    ),

    # ── Perplexity ────────────────────────────────────────────────────────────
    "llama-3.1-sonar-large-128k-online": ModelPrice(
        model_id="llama-3.1-sonar-large-128k-online", provider="perplexity",
        display_name="Sonar Large (Online)",
        input_per_1k=0.001000, output_per_1k=0.001000,
        context_window=127_072,
        notes="Includes web search. Additional $5/1000 search requests",
    ),
    "llama-3.1-sonar-small-128k-online": ModelPrice(
        model_id="llama-3.1-sonar-small-128k-online", provider="perplexity",
        display_name="Sonar Small (Online)",
        input_per_1k=0.000200, output_per_1k=0.000200,
        context_window=127_072,
    ),

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    "amazon.titan-text-express-v1": ModelPrice(
        model_id="amazon.titan-text-express-v1", provider="aws-bedrock",
        display_name="Amazon Titan Text Express",
        input_per_1k=0.000800, output_per_1k=0.000800,
        context_window=8_192,
    ),
    "amazon.titan-text-lite-v1": ModelPrice(
        model_id="amazon.titan-text-lite-v1", provider="aws-bedrock",
        display_name="Amazon Titan Text Lite",
        input_per_1k=0.000300, output_per_1k=0.000400,
        context_window=4_096,
    ),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPrice(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0", provider="aws-bedrock",
        display_name="Claude 3.5 Sonnet (Bedrock)",
        input_per_1k=0.003000, output_per_1k=0.015000,
        context_window=200_000,
    ),

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    "deepseek-chat": ModelPrice(
        model_id="deepseek-chat", provider="deepseek",
        display_name="DeepSeek Chat (V3)",
        input_per_1k=0.000140, output_per_1k=0.000280,
        context_window=64_000,
        notes="Cache hit: $0.000014/1k. Exceptional price/performance ratio.",
    ),
    "deepseek-reasoner": ModelPrice(
        model_id="deepseek-reasoner", provider="deepseek",
        display_name="DeepSeek Reasoner (R1)",
        input_per_1k=0.000550, output_per_1k=0.002190,
        context_window=64_000,
        notes="Chain-of-thought reasoning model",
    ),

    # ── Groq (ultra-fast inference) ───────────────────────────────────────────
    "llama-3.3-70b-versatile": ModelPrice(
        model_id="llama-3.3-70b-versatile", provider="groq",
        display_name="Llama 3.3 70B (Groq)",
        input_per_1k=0.000590, output_per_1k=0.000790,
        context_window=128_000,
        notes="Ultra-low latency via Groq LPU inference",
    ),
    "mixtral-8x7b-32768": ModelPrice(
        model_id="mixtral-8x7b-32768", provider="groq",
        display_name="Mixtral 8x7B (Groq)",
        input_per_1k=0.000240, output_per_1k=0.000240,
        context_window=32_768,
        notes="Groq-hosted Mixtral with sub-100ms latency",
    ),
}

# ── Alias index ───────────────────────────────────────────────────────────────
# Built at import time: alias → canonical model_id
_ALIAS_MAP: dict[str, str] = {}
for _model_id, _mp in REGISTRY.items():
    for _alias in _mp.aliases:
        _ALIAS_MAP[_alias.lower()] = _model_id


# ── Public API ────────────────────────────────────────────────────────────────

def get_model(model_id: str) -> ModelPrice:
    """
    Look up a model by ID or alias. Case-insensitive.

    Args:
        model_id: Model identifier or alias

    Returns:
        ModelPrice for the requested model

    Raises:
        KeyError: If model is not found in the registry
    """
    key = model_id.lower().strip()

    # Direct match
    if key in REGISTRY:
        return REGISTRY[key]

    # Alias match
    if key in _ALIAS_MAP:
        return REGISTRY[_ALIAS_MAP[key]]

    # Prefix match (e.g. "gpt-4o" matches "gpt-4o-2024-11-20")
    for reg_key in REGISTRY:
        if reg_key.startswith(key) or key.startswith(reg_key):
            return REGISTRY[reg_key]

    available = sorted(REGISTRY.keys())
    raise KeyError(
        f"Model '{model_id}' not found in pricing registry.\n"
        f"Available models: {available[:10]}... (and {len(available)-10} more)"
    )


def list_models(provider: Optional[str] = None) -> list[ModelPrice]:
    """
    List all models, optionally filtered by provider.

    Args:
        provider: Filter by provider name (e.g. "openai", "anthropic")

    Returns:
        Sorted list of ModelPrice objects
    """
    models = list(REGISTRY.values())
    if provider:
        models = [m for m in models if m.provider.lower() == provider.lower()]
    return sorted(models, key=lambda m: (m.provider, m.model_id))


def list_providers() -> list[str]:
    """Return sorted list of all providers in the registry."""
    return sorted(set(m.provider for m in REGISTRY.values()))


def cheapest_model(provider: Optional[str] = None, by: str = "output") -> ModelPrice:
    """
    Find the cheapest model by input or output token price.

    Args:
        provider: Optionally restrict to one provider
        by:       "input" or "output" (default "output")

    Returns:
        The cheapest ModelPrice
    """
    models = list_models(provider)
    if not models:
        raise ValueError(f"No models found for provider '{provider}'")
    key = "output_per_1k" if by == "output" else "input_per_1k"
    return min(models, key=lambda m: getattr(m, key))


def most_expensive_model(provider: Optional[str] = None) -> ModelPrice:
    """Find the most expensive model by output token price."""
    models = list_models(provider)
    if not models:
        raise ValueError(f"No models found for provider '{provider}'")
    return max(models, key=lambda m: m.output_per_1k)


def compare_models(*model_ids: str, input_tokens: int = 1000, output_tokens: int = 1000) -> list[dict]:
    """
    Compare costs across multiple models for a given token count.

    Args:
        *model_ids:     Model IDs to compare
        input_tokens:   Number of input tokens
        output_tokens:  Number of output tokens

    Returns:
        List of dicts sorted by total cost ascending
    """
    results = []
    for mid in model_ids:
        try:
            mp = get_model(mid)
            cost = mp.cost_for_tokens(input_tokens, output_tokens)
            results.append({
                "model_id":     mp.model_id,
                "provider":     mp.provider,
                "display_name": mp.display_name,
                "input_cost":   (input_tokens / 1000) * mp.input_per_1k,
                "output_cost":  (output_tokens / 1000) * mp.output_per_1k,
                "total_cost":   cost,
            })
        except KeyError:
            results.append({
                "model_id": mid, "provider": "unknown",
                "display_name": mid, "total_cost": float("inf"),
                "error": f"Model '{mid}' not in registry",
            })
    return sorted(results, key=lambda r: r["total_cost"])
