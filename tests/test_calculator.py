"""
tests/test_calculator.py

Tests for src/calculator.py — cost calculations, batch estimates, projections.
"""
import pytest
from src.calculator import (
    calculate, estimate_from_text, batch_estimate,
    monthly_projection, break_even_analysis
)


class TestCalculate:

    def test_basic_calculation_gpt4o_mini(self):
        est = calculate("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert est.model_id  == "gpt-4o-mini"
        assert est.provider  == "openai"
        expected_input  = 1.0 * 0.000150
        expected_output = 0.5 * 0.000600
        assert est.input_cost   == pytest.approx(expected_input,  rel=0.001)
        assert est.output_cost  == pytest.approx(expected_output, rel=0.001)
        assert est.total_cost   == pytest.approx(expected_input + expected_output, rel=0.001)

    def test_calculate_zero_tokens(self):
        est = calculate("gpt-4o", 0, 0)
        assert est.total_cost == 0.0
        assert est.input_cost == 0.0
        assert est.output_cost == 0.0

    def test_calculate_only_input_tokens(self):
        est = calculate("gpt-4o-mini", input_tokens=1000, output_tokens=0)
        assert est.total_cost == pytest.approx(0.000150, rel=0.001)
        assert est.output_cost == 0.0

    def test_cost_per_1k_calls(self):
        est = calculate("gpt-4o-mini", 500, 200)
        assert est.cost_per_1k_calls == pytest.approx(est.total_cost * 1000, rel=0.001)

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError):
            calculate("gpt-4o", input_tokens=-1, output_tokens=100)
        with pytest.raises(ValueError):
            calculate("gpt-4o", input_tokens=100, output_tokens=-1)

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            calculate("not-a-model", 100, 100)

    def test_alias_resolution_in_calculate(self):
        est = calculate("gpt-4o-2024-11-20", 500, 200)
        assert est.model_id == "gpt-4o"

    def test_large_token_count(self):
        # 1M input + 100k output on GPT-4o
        est = calculate("gpt-4o", 1_000_000, 100_000)
        expected = (1000 * 0.002500) + (100 * 0.010000)
        assert est.total_cost == pytest.approx(expected, rel=0.001)
        assert est.total_cost > 0

    def test_estimate_summary_string(self):
        est = calculate("gpt-4o-mini", 500, 200)
        summary = est.summary()
        assert "gpt-4o-mini" in summary
        assert "openai" in summary
        assert "$" in summary

    def test_anthropic_model_calculation(self):
        est = calculate("claude-3-5-sonnet-20241022", 1000, 500)
        expected = (1.0 * 0.003000) + (0.5 * 0.015000)
        assert est.total_cost == pytest.approx(expected, rel=0.001)

    def test_claude_haiku_cheapest_anthropic(self):
        haiku   = calculate("claude-3-haiku-20240307",       1000, 500)
        sonnet  = calculate("claude-3-5-sonnet-20241022",    1000, 500)
        opus    = calculate("claude-3-opus-20240229",        1000, 500)
        assert haiku.total_cost < sonnet.total_cost < opus.total_cost


class TestEstimateFromText:

    def test_estimate_from_short_text(self):
        text = "What is the capital of France?"
        est  = estimate_from_text("gpt-4o-mini", text, estimated_output_tokens=100)
        assert est.input_tokens > 0
        assert est.output_tokens == 100
        assert est.total_cost > 0

    def test_token_estimate_scales_with_text_length(self):
        short_est = estimate_from_text("gpt-4o-mini", "Short text", 100)
        long_est  = estimate_from_text("gpt-4o-mini", "A" * 10000,  100)
        assert long_est.input_tokens > short_est.input_tokens
        assert long_est.total_cost   > short_est.total_cost

    def test_custom_chars_per_token(self):
        text = "Hello world " * 100  # 1200 chars
        est4 = estimate_from_text("gpt-4o", text, 100, chars_per_token=4.0)
        est2 = estimate_from_text("gpt-4o", text, 100, chars_per_token=2.0)
        # With 2 chars/token, we get more tokens → more cost
        assert est2.input_tokens > est4.input_tokens


class TestBatchEstimate:

    def test_batch_empty(self):
        result = batch_estimate("gpt-4o-mini", [])
        assert result["total_cost_usd"] == 0.0
        assert result["call_count"]     == 0

    def test_batch_single_call(self):
        calls  = [{"input_tokens": 500, "output_tokens": 200}]
        result = batch_estimate("gpt-4o-mini", calls)
        assert result["call_count"]       == 1
        assert result["total_input_tokens"]  == 500
        assert result["total_output_tokens"] == 200
        single_est = calculate("gpt-4o-mini", 500, 200)
        assert result["total_cost_usd"] == pytest.approx(single_est.total_cost, rel=0.001)

    def test_batch_multiple_calls(self):
        calls = [
            {"input_tokens": 500,  "output_tokens": 200},
            {"input_tokens": 1200, "output_tokens": 800},
            {"input_tokens": 300,  "output_tokens": 150},
        ]
        result = batch_estimate("gpt-4o", calls)
        assert result["call_count"] == 3
        assert result["total_input_tokens"]  == 2000
        assert result["total_output_tokens"] == 1150

        expected_total = sum(
            calculate("gpt-4o", c["input_tokens"], c["output_tokens"]).total_cost
            for c in calls
        )
        assert result["total_cost_usd"] == pytest.approx(expected_total, rel=0.001)

    def test_batch_statistics(self):
        calls = [
            {"input_tokens": 100, "output_tokens": 50},
            {"input_tokens": 500, "output_tokens": 500},
            {"input_tokens": 1000,"output_tokens": 1000},
        ]
        result = batch_estimate("gpt-4o-mini", calls)
        assert result["min_cost_per_call"] <= result["avg_cost_per_call"]
        assert result["avg_cost_per_call"] <= result["max_cost_per_call"]

    def test_batch_cost_matches_individual_sum(self):
        """Batch cost must equal sum of individual call costs."""
        calls = [{"input_tokens": i * 100, "output_tokens": i * 50} for i in range(1, 6)]
        batch  = batch_estimate("gpt-4o-mini", calls)
        manual = sum(calculate("gpt-4o-mini", c["input_tokens"], c["output_tokens"]).total_cost
                     for c in calls)
        assert batch["total_cost_usd"] == pytest.approx(manual, rel=0.001)


class TestMonthlyProjection:

    def test_projection_structure(self):
        proj = monthly_projection("gpt-4o-mini", calls_per_day=1000,
                                  avg_input_tokens=500, avg_output_tokens=200)
        assert "daily_cost"   in proj
        assert "weekly_cost"  in proj
        assert "monthly_cost" in proj
        assert "annual_cost"  in proj

    def test_projection_weekly_is_7x_daily(self):
        proj = monthly_projection("gpt-4o-mini", 100, 500, 200)
        assert proj["weekly_cost"] == pytest.approx(proj["daily_cost"] * 7, rel=0.001)

    def test_projection_monthly_is_30x_daily(self):
        proj = monthly_projection("gpt-4o-mini", 100, 500, 200)
        assert proj["monthly_cost"] == pytest.approx(proj["daily_cost"] * 30, rel=0.001)

    def test_higher_volume_costs_more(self):
        low  = monthly_projection("gpt-4o-mini", 100,   500, 200)
        high = monthly_projection("gpt-4o-mini", 10000, 500, 200)
        assert high["monthly_cost"] == pytest.approx(low["monthly_cost"] * 100, rel=0.001)


class TestBreakEvenAnalysis:

    def test_break_even_structure(self):
        result = break_even_analysis("gpt-4o", "gpt-4o-mini", 1000, 500)
        assert "cheaper_model"   in result
        assert "savings_per_call" in result
        assert "pct_cheaper"     in result

    def test_gpt4o_mini_cheaper_than_gpt4o(self):
        result = break_even_analysis("gpt-4o", "gpt-4o-mini", 1000, 500)
        assert result["cheaper_model"] == "gpt-4o-mini"
        assert result["savings_per_call"] > 0
        assert result["pct_cheaper"] > 0

    def test_savings_scale_correctly(self):
        single = break_even_analysis("gpt-4o", "gpt-4o-mini", 1000, 500)
        assert single["savings_per_1k_calls"] == pytest.approx(
            single["savings_per_call"] * 1000, rel=0.001
        )
        assert single["savings_per_1m_calls"] == pytest.approx(
            single["savings_per_call"] * 1_000_000, rel=0.001
        )

    def test_same_model_zero_savings(self):
        result = break_even_analysis("gpt-4o", "gpt-4o", 1000, 500)
        assert result["savings_per_call"] == pytest.approx(0.0, abs=1e-10)
        assert result["pct_cheaper"] == pytest.approx(0.0, abs=0.01)
