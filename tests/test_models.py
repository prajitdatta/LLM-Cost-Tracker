"""
tests/test_models.py

Tests for src/models.py — pricing registry, model lookup, aliases.
"""
import pytest
from src.models import (
    REGISTRY, get_model, list_models, list_providers,
    cheapest_model, most_expensive_model, compare_models
)


class TestRegistry:

    def test_registry_has_40_plus_models(self):
        assert len(REGISTRY) >= 40, f"Expected 40+ models, got {len(REGISTRY)}"

    def test_all_providers_covered(self):
        providers = list_providers()
        required = {"openai", "anthropic", "google", "mistral", "cohere", "meta", "deepseek"}
        missing  = required - set(providers)
        assert not missing, f"Missing providers: {missing}"

    def test_every_model_has_positive_prices(self):
        for mid, model in REGISTRY.items():
            assert model.input_per_1k  >= 0, f"{mid}: negative input price"
            assert model.output_per_1k >= 0, f"{mid}: negative output price"
            assert model.context_window > 0, f"{mid}: zero context window"

    def test_every_model_has_required_fields(self):
        for mid, model in REGISTRY.items():
            assert model.model_id,      f"{mid}: empty model_id"
            assert model.provider,      f"{mid}: empty provider"
            assert model.display_name,  f"{mid}: empty display_name"

    def test_output_more_expensive_than_input(self):
        """For most models, output tokens cost more than input tokens."""
        more_expensive_output = sum(
            1 for m in REGISTRY.values()
            if m.output_per_1k >= m.input_per_1k
        )
        ratio = more_expensive_output / len(REGISTRY)
        assert ratio >= 0.7, f"Expected 70%+ models with output >= input price, got {ratio:.0%}"


class TestModelLookup:

    def test_lookup_by_exact_id(self):
        model = get_model("gpt-4o")
        assert model.model_id == "gpt-4o"
        assert model.provider == "openai"

    def test_lookup_case_insensitive(self):
        assert get_model("GPT-4O").model_id == "gpt-4o"
        assert get_model("Claude-3-5-Sonnet-20241022").model_id == "claude-3-5-sonnet-20241022"

    def test_lookup_by_alias(self):
        # gpt-4o-2024-11-20 is an alias for gpt-4o
        model = get_model("gpt-4o-2024-11-20")
        assert model.model_id == "gpt-4o"

    def test_lookup_anthropic_aliases(self):
        model = get_model("claude-3-5-sonnet")
        assert "claude-3-5-sonnet" in model.model_id

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            get_model("gpt-99-ultra-pro-max")

    def test_empty_string_raises_key_error(self):
        with pytest.raises(KeyError):
            get_model("")


class TestPricingAccuracy:

    def test_gpt4o_mini_pricing(self):
        model = get_model("gpt-4o-mini")
        assert model.input_per_1k  == pytest.approx(0.000150, rel=0.01)
        assert model.output_per_1k == pytest.approx(0.000600, rel=0.01)

    def test_gpt4o_pricing(self):
        model = get_model("gpt-4o")
        assert model.input_per_1k  == pytest.approx(0.002500, rel=0.01)
        assert model.output_per_1k == pytest.approx(0.010000, rel=0.01)

    def test_claude_haiku_cheapest_anthropic(self):
        haiku  = get_model("claude-3-haiku-20240307")
        opus   = get_model("claude-3-opus-20240229")
        assert haiku.input_per_1k  < opus.input_per_1k
        assert haiku.output_per_1k < opus.output_per_1k

    def test_cost_for_tokens_calculation(self):
        model = get_model("gpt-4o-mini")
        # 1000 input @ $0.00015/1k + 500 output @ $0.0006/1k
        expected = 1.0 * 0.000150 + 0.5 * 0.000600
        assert model.cost_for_tokens(1000, 500) == pytest.approx(expected, rel=0.001)

    def test_zero_tokens_costs_nothing(self):
        model = get_model("gpt-4o")
        assert model.cost_for_tokens(0, 0) == 0.0

    def test_deepseek_cheaper_than_gpt4o(self):
        deepseek = get_model("deepseek-chat")
        gpt4o    = get_model("gpt-4o")
        deepseek_cost = deepseek.cost_for_tokens(1000, 500)
        gpt4o_cost    = gpt4o.cost_for_tokens(1000, 500)
        assert deepseek_cost < gpt4o_cost


class TestListAndFilter:

    def test_list_all_models_returns_all(self):
        models = list_models()
        assert len(models) == len(REGISTRY)

    def test_list_by_provider(self):
        openai_models = list_models("openai")
        assert all(m.provider == "openai" for m in openai_models)
        assert len(openai_models) >= 6

    def test_list_providers_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)

    def test_cheapest_model_overall(self):
        cheapest = cheapest_model()
        all_models = list_models()
        min_output = min(m.output_per_1k for m in all_models)
        assert cheapest.output_per_1k == min_output

    def test_cheapest_model_per_provider(self):
        for provider in list_providers():
            cheapest = cheapest_model(provider=provider)
            assert cheapest.provider.lower() == provider.lower()

    def test_most_expensive_model(self):
        most_exp = most_expensive_model()
        all_models = list_models()
        max_output = max(m.output_per_1k for m in all_models)
        assert most_exp.output_per_1k == max_output


class TestCompareModels:

    def test_compare_returns_sorted_by_cost(self):
        results = compare_models("gpt-4o-mini", "gpt-4o", "gpt-4-turbo",
                                 input_tokens=1000, output_tokens=500)
        costs = [r["total_cost"] for r in results]
        assert costs == sorted(costs)

    def test_compare_gpt4o_mini_cheapest(self):
        results = compare_models("gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
                                 input_tokens=1000, output_tokens=500)
        assert results[0]["model_id"] == "gpt-4o-mini"

    def test_compare_unknown_model_graceful(self):
        results = compare_models("gpt-4o-mini", "not-a-real-model",
                                 input_tokens=100, output_tokens=100)
        # Should not raise, just mark the unknown model
        assert any(r["model_id"] == "gpt-4o-mini" for r in results)
