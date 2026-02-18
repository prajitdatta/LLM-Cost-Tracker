"""
src/parser.py

Response parsers for every major LLM provider API format.

Each parser extracts token counts from the provider's API response object
(or response dict) so CostTracker can record accurate costs.

Supported providers:
    OpenAI      — /v1/chat/completions response format
    Anthropic   — Messages API response format
    Google      — Gemini generateContent response format
    Mistral     — Mistral AI chat completion format
    Cohere      — Command R response format
    AWS Bedrock — Bedrock InvokeModel response format
    Generic     — Dict-based fallback for any provider

Usage:
    from src.parser import parse_response, TokenUsage

    # Auto-detect provider from response structure
    usage = parse_response(response_dict)
    print(usage.input_tokens, usage.output_tokens)

    # Or use a specific parser
    usage = parse_openai(response_dict)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """
    Normalised token usage extracted from any provider response.

    Attributes:
        input_tokens:   Number of tokens in the prompt/input
        output_tokens:  Number of tokens in the completion/output
        total_tokens:   input + output (may differ from sum if provider reports separately)
        cached_tokens:  Input tokens served from cache (lower cost)
        reasoning_tokens: Tokens used for internal chain-of-thought (o1/R1 models)
        raw:            The original usage dict from the provider
    """
    input_tokens:    int
    output_tokens:   int
    total_tokens:    int
    cached_tokens:   int = 0
    reasoning_tokens: int = 0
    raw:             dict = None

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}
        # Recompute total if not set correctly
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        parts = [f"in={self.input_tokens}", f"out={self.output_tokens}"]
        if self.cached_tokens:
            parts.append(f"cached={self.cached_tokens}")
        if self.reasoning_tokens:
            parts.append(f"reasoning={self.reasoning_tokens}")
        return f"TokenUsage({', '.join(parts)})"


# ── Provider-specific parsers ─────────────────────────────────────────────────

def parse_openai(response: Any) -> TokenUsage:
    """
    Parse OpenAI /v1/chat/completions response.

    Handles both object-style (response.usage) and dict-style responses.
    Extracts cached_tokens and reasoning_tokens for o1/o3 models.

    Example response usage block:
        {
            "prompt_tokens": 150,
            "completion_tokens": 300,
            "total_tokens": 450,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": 100}
        }
    """
    usage = _extract_usage(response)
    if usage is None:
        return TokenUsage(0, 0, 0, raw={"error": "no usage field in response"})

    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens  = usage.get("total_tokens", input_tokens + output_tokens)

    # Cached tokens (prompt caching discount)
    cached_tokens = 0
    prompt_details = usage.get("prompt_tokens_details", {})
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens", 0)

    # Reasoning tokens (o1/o3 chain-of-thought)
    reasoning_tokens = 0
    completion_details = usage.get("completion_tokens_details", {})
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens", 0)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        raw=usage,
    )


def parse_anthropic(response: Any) -> TokenUsage:
    """
    Parse Anthropic Messages API response.

    Example response usage block:
        {
            "input_tokens": 150,
            "output_tokens": 300,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 50
        }
    """
    usage = _extract_usage(response)
    if usage is None:
        return TokenUsage(0, 0, 0, raw={"error": "no usage field in response"})

    input_tokens  = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Anthropic prompt caching
    cached_tokens = usage.get("cache_read_input_tokens", 0)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached_tokens,
        raw=usage,
    )


def parse_google(response: Any) -> TokenUsage:
    """
    Parse Google Gemini generateContent response.

    Handles both the REST API dict format and the google-generativeai SDK
    response objects.

    Example usageMetadata:
        {
            "promptTokenCount": 150,
            "candidatesTokenCount": 300,
            "totalTokenCount": 450
        }
    """
    # Handle SDK response objects
    if hasattr(response, "usage_metadata"):
        meta = response.usage_metadata
        if hasattr(meta, "prompt_token_count"):
            input_t  = getattr(meta, "prompt_token_count", 0) or 0
            output_t = getattr(meta, "candidates_token_count", 0) or 0
            total_t  = getattr(meta, "total_token_count", input_t + output_t) or 0
            return TokenUsage(input_t, output_t, total_t, raw={"sdk_object": True})

    # Handle dict response
    if isinstance(response, dict):
        meta = response.get("usageMetadata", response.get("usage_metadata", {}))
        if isinstance(meta, dict):
            input_t  = meta.get("promptTokenCount", meta.get("prompt_token_count", 0))
            output_t = meta.get("candidatesTokenCount", meta.get("candidates_token_count", 0))
            total_t  = meta.get("totalTokenCount", input_t + output_t)
            return TokenUsage(input_t, output_t, total_t, raw=meta)

    return TokenUsage(0, 0, 0, raw={"error": "unrecognised google response format"})


def parse_mistral(response: Any) -> TokenUsage:
    """
    Parse Mistral AI chat completion response.
    Format mirrors OpenAI with prompt_tokens/completion_tokens.
    """
    usage = _extract_usage(response)
    if usage is None:
        return TokenUsage(0, 0, 0, raw={"error": "no usage field"})

    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens  = usage.get("total_tokens", input_tokens + output_tokens)

    return TokenUsage(input_tokens, output_tokens, total_tokens, raw=usage)


def parse_cohere(response: Any) -> TokenUsage:
    """
    Parse Cohere Command R response.

    Cohere uses meta.tokens or meta.billed_units:
        {
            "meta": {
                "tokens": {"input_tokens": 150, "output_tokens": 300},
                "billed_units": {"input_tokens": 150, "output_tokens": 300}
            }
        }
    """
    if isinstance(response, dict):
        meta = response.get("meta", {})
        if isinstance(meta, dict):
            tokens = meta.get("tokens", meta.get("billed_units", {}))
            if isinstance(tokens, dict):
                input_t  = tokens.get("input_tokens", 0)
                output_t = tokens.get("output_tokens", 0)
                return TokenUsage(input_t, output_t, input_t + output_t, raw=tokens)

    # Some Cohere response formats
    if hasattr(response, "meta"):
        try:
            tokens = response.meta.tokens
            return TokenUsage(
                tokens.input_tokens, tokens.output_tokens,
                tokens.input_tokens + tokens.output_tokens,
            )
        except AttributeError:
            pass

    return TokenUsage(0, 0, 0, raw={"error": "unrecognised cohere response"})


def parse_bedrock(response: Any) -> TokenUsage:
    """
    Parse AWS Bedrock InvokeModel response.

    Bedrock wraps provider-specific responses. Usage is in:
        response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-input-token-count"]
    or in the body for Claude on Bedrock (same as Anthropic format).
    """
    if isinstance(response, dict):
        # Claude on Bedrock
        if "usage" in response:
            usage = response["usage"]
            return TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                raw=usage,
            )
        # Header-based counting
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        input_t  = int(headers.get("x-amzn-bedrock-input-token-count", 0))
        output_t = int(headers.get("x-amzn-bedrock-output-token-count", 0))
        if input_t or output_t:
            return TokenUsage(input_t, output_t, input_t + output_t, raw=headers)

    return TokenUsage(0, 0, 0, raw={"error": "unrecognised bedrock response"})


def parse_generic(response: Any) -> TokenUsage:
    """
    Generic parser — tries common field names in order.

    Useful for providers not explicitly supported or for custom API wrappers.
    Tries: usage, meta, token_count, tokens
    """
    # Try to get a usage dict by any common name
    usage = None
    for field in ("usage", "meta", "token_usage", "tokens", "token_count"):
        candidate = (
            response.get(field) if isinstance(response, dict)
            else getattr(response, field, None)
        )
        if isinstance(candidate, dict) and candidate:
            usage = candidate
            break

    if usage is None:
        # Last resort: look for token count fields anywhere in top-level dict
        if isinstance(response, dict):
            usage = response

    if not usage:
        return TokenUsage(0, 0, 0)

    # Try every combination of field names
    input_tokens = (
        usage.get("input_tokens") or
        usage.get("prompt_tokens") or
        usage.get("promptTokenCount") or
        usage.get("input_token_count") or
        0
    )
    output_tokens = (
        usage.get("output_tokens") or
        usage.get("completion_tokens") or
        usage.get("candidatesTokenCount") or
        usage.get("generated_token_count") or
        0
    )
    total_tokens = (
        usage.get("total_tokens") or
        usage.get("totalTokenCount") or
        input_tokens + output_tokens
    )

    return TokenUsage(input_tokens, output_tokens, total_tokens, raw=usage)


# ── Auto-detect parser ────────────────────────────────────────────────────────

def parse_response(response: Any, provider: str | None = None) -> TokenUsage:
    """
    Parse a provider API response, auto-detecting the provider if not specified.

    Detection logic:
        - OpenAI:    has "prompt_tokens" in usage
        - Anthropic: has "input_tokens" in usage (no "promptTokenCount")
        - Google:    has "usageMetadata" or "usage_metadata"
        - Cohere:    has "meta" with nested "tokens"
        - Bedrock:   has "ResponseMetadata"
        - Mistral:   same as OpenAI (mirrors OpenAI format)
        - Generic:   fallback

    Args:
        response: The raw API response (dict or SDK response object)
        provider: Optional provider hint ("openai", "anthropic", etc.)

    Returns:
        Normalised TokenUsage
    """
    if provider:
        provider = provider.lower()
        parsers = {
            "openai":    parse_openai,
            "anthropic": parse_anthropic,
            "google":    parse_google,
            "gemini":    parse_google,
            "mistral":   parse_mistral,
            "cohere":    parse_cohere,
            "bedrock":   parse_bedrock,
            "aws":       parse_bedrock,
            "aws-bedrock": parse_bedrock,
        }
        if provider in parsers:
            return parsers[provider](response)

    # Auto-detect
    response_dict = response if isinstance(response, dict) else {}
    if hasattr(response, "__dict__"):
        response_dict = response.__dict__

    usage = (
        response_dict.get("usage") or
        (response_dict.get("usage_metadata")) or
        {}
    )
    if isinstance(usage, dict):
        if "prompt_tokens" in usage:
            return parse_openai(response)
        if "input_tokens" in usage and "promptTokenCount" not in usage:
            return parse_anthropic(response)

    if "usageMetadata" in response_dict or "usage_metadata" in response_dict:
        return parse_google(response)
    if "meta" in response_dict and isinstance(response_dict.get("meta"), dict):
        return parse_cohere(response)
    if "ResponseMetadata" in response_dict:
        return parse_bedrock(response)

    return parse_generic(response)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_usage(response: Any) -> dict | None:
    """Extract the 'usage' field from either a dict or SDK object."""
    if isinstance(response, dict):
        return response.get("usage")
    if hasattr(response, "usage"):
        u = response.usage
        if isinstance(u, dict):
            return u
        # SDK usage objects — convert to dict
        if hasattr(u, "__dict__"):
            return u.__dict__
        # Try common attribute access patterns
        return {
            "prompt_tokens":     getattr(u, "prompt_tokens",     getattr(u, "input_tokens", 0)),
            "completion_tokens": getattr(u, "completion_tokens", getattr(u, "output_tokens", 0)),
            "total_tokens":      getattr(u, "total_tokens", 0),
        }
    return None
