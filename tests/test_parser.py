"""
tests/test_parser.py

Tests for src/parser.py — parsing token counts from provider API responses.
"""
import pytest
from src.parser import (
    TokenUsage, parse_openai, parse_anthropic, parse_google,
    parse_mistral, parse_cohere, parse_bedrock, parse_generic, parse_response
)


class TestTokenUsage:

    def test_default_total_computed(self):
        usage = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=0)
        assert usage.total_tokens == 300

    def test_explicit_total_preserved(self):
        usage = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=350)
        assert usage.total_tokens == 350

    def test_repr(self):
        usage = TokenUsage(100, 200, 300, cached_tokens=50, reasoning_tokens=30)
        assert "in=100" in repr(usage)
        assert "cached=50" in repr(usage)
        assert "reasoning=30" in repr(usage)


class TestParseOpenAI:

    def test_standard_response(self):
        response = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 300,
                "total_tokens": 450,
            }
        }
        usage = parse_openai(response)
        assert usage.input_tokens  == 150
        assert usage.output_tokens == 300
        assert usage.total_tokens  == 450

    def test_with_cached_tokens(self):
        response = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 50},
            }
        }
        usage = parse_openai(response)
        assert usage.cached_tokens == 50

    def test_with_reasoning_tokens(self):
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 500,
                "total_tokens": 600,
                "completion_tokens_details": {"reasoning_tokens": 300},
            }
        }
        usage = parse_openai(response)
        assert usage.reasoning_tokens == 300

    def test_missing_usage_returns_zeros(self):
        usage = parse_openai({"choices": []})
        assert usage.input_tokens  == 0
        assert usage.output_tokens == 0

    def test_sdk_object_style(self):
        """Test with a mock SDK response object (attribute access)."""
        class MockUsage:
            prompt_tokens     = 200
            completion_tokens = 150
            total_tokens      = 350
        class MockResponse:
            usage = MockUsage()
        usage = parse_openai(MockResponse())
        assert usage.input_tokens  == 200
        assert usage.output_tokens == 150


class TestParseAnthropic:

    def test_standard_response(self):
        response = {
            "usage": {
                "input_tokens":  80,
                "output_tokens": 240,
            }
        }
        usage = parse_anthropic(response)
        assert usage.input_tokens  == 80
        assert usage.output_tokens == 240
        assert usage.total_tokens  == 320

    def test_with_cache_read_tokens(self):
        response = {
            "usage": {
                "input_tokens":            100,
                "output_tokens":           200,
                "cache_read_input_tokens":  60,
            }
        }
        usage = parse_anthropic(response)
        assert usage.cached_tokens == 60

    def test_missing_usage_returns_zeros(self):
        usage = parse_anthropic({"content": []})
        assert usage.input_tokens  == 0
        assert usage.output_tokens == 0


class TestParseGoogle:

    def test_rest_api_format(self):
        response = {
            "usageMetadata": {
                "promptTokenCount":     120,
                "candidatesTokenCount": 280,
                "totalTokenCount":      400,
            }
        }
        usage = parse_google(response)
        assert usage.input_tokens  == 120
        assert usage.output_tokens == 280
        assert usage.total_tokens  == 400

    def test_snake_case_format(self):
        response = {
            "usage_metadata": {
                "prompt_token_count":     100,
                "candidates_token_count": 200,
            }
        }
        usage = parse_google(response)
        assert usage.input_tokens  == 100
        assert usage.output_tokens == 200

    def test_empty_response(self):
        usage = parse_google({})
        assert usage.input_tokens == 0


class TestParseMistral:

    def test_standard_response(self):
        response = {
            "usage": {
                "prompt_tokens":     90,
                "completion_tokens": 180,
                "total_tokens":      270,
            }
        }
        usage = parse_mistral(response)
        assert usage.input_tokens  == 90
        assert usage.output_tokens == 180


class TestParseCohere:

    def test_meta_tokens_format(self):
        response = {
            "meta": {
                "tokens": {
                    "input_tokens":  110,
                    "output_tokens": 220,
                }
            }
        }
        usage = parse_cohere(response)
        assert usage.input_tokens  == 110
        assert usage.output_tokens == 220

    def test_billed_units_format(self):
        response = {
            "meta": {
                "billed_units": {
                    "input_tokens":  75,
                    "output_tokens": 150,
                }
            }
        }
        usage = parse_cohere(response)
        assert usage.input_tokens  == 75
        assert usage.output_tokens == 150


class TestParseBedrock:

    def test_claude_on_bedrock(self):
        response = {
            "usage": {
                "input_tokens":  200,
                "output_tokens": 400,
            }
        }
        usage = parse_bedrock(response)
        assert usage.input_tokens  == 200
        assert usage.output_tokens == 400

    def test_header_based_counting(self):
        response = {
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count":  "150",
                    "x-amzn-bedrock-output-token-count": "300",
                }
            }
        }
        usage = parse_bedrock(response)
        assert usage.input_tokens  == 150
        assert usage.output_tokens == 300


class TestParseGeneric:

    def test_openai_style_via_generic(self):
        response = {
            "usage": {
                "prompt_tokens":     50,
                "completion_tokens": 100,
                "total_tokens":      150,
            }
        }
        usage = parse_generic(response)
        assert usage.input_tokens  == 50
        assert usage.output_tokens == 100

    def test_anthropic_style_via_generic(self):
        response = {
            "usage": {
                "input_tokens":  70,
                "output_tokens": 130,
            }
        }
        usage = parse_generic(response)
        assert usage.input_tokens  == 70
        assert usage.output_tokens == 130


class TestParseResponse:

    def test_auto_detect_openai(self):
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}}
        usage = parse_response(response)
        assert usage.input_tokens  == 100
        assert usage.output_tokens == 200

    def test_auto_detect_anthropic(self):
        response = {"usage": {"input_tokens": 80, "output_tokens": 160}}
        usage = parse_response(response)
        assert usage.input_tokens  == 80
        assert usage.output_tokens == 160

    def test_provider_hint_openai(self):
        response = {"usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}}
        usage = parse_response(response, provider="openai")
        assert usage.input_tokens == 50

    def test_provider_hint_anthropic(self):
        response = {"usage": {"input_tokens": 60, "output_tokens": 120}}
        usage = parse_response(response, provider="anthropic")
        assert usage.input_tokens == 60

    def test_provider_hint_google(self):
        response = {"usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 180, "totalTokenCount": 270}}
        usage = parse_response(response, provider="google")
        assert usage.input_tokens == 90

    def test_auto_detect_google(self):
        response = {"usageMetadata": {"promptTokenCount": 110, "candidatesTokenCount": 220}}
        usage = parse_response(response)
        assert usage.input_tokens == 110

    def test_completely_unknown_format_returns_zeros(self):
        usage = parse_response({"random_field": "random_value"})
        # Should not raise — returns zeros gracefully
        assert isinstance(usage, TokenUsage)
