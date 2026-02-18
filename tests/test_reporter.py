"""
tests/test_reporter.py

Tests for src/reporter.py — report generation and analysis.
"""
import pytest
from src.tracker import CostTracker
from src.reporter import (
    model_breakdown, provider_breakdown, tag_breakdown,
    time_series, top_expensive_calls, efficiency_report,
    comparison_table, print_model_breakdown, print_comparison_table
)


def make_tracker() -> CostTracker:
    """Create a populated tracker for testing."""
    t = CostTracker()
    t.record("gpt-4o",      1000, 500, tags={"feature": "chat",    "env": "prod"})
    t.record("gpt-4o",       500, 200, tags={"feature": "search",  "env": "prod"})
    t.record("gpt-4o-mini", 2000, 800, tags={"feature": "chat",    "env": "staging"})
    t.record("gpt-4o-mini",  300, 150, tags={"feature": "summarise","env": "prod"})
    t.record("claude-3-5-sonnet-20241022", 800, 600, tags={"feature": "chat", "env": "prod"})
    t.record("claude-3-haiku-20240307",    400, 200, tags={"feature": "search","env": "staging"})
    return t


class TestModelBreakdown:

    def test_returns_list_of_dicts(self):
        t = make_tracker()
        rows = model_breakdown(t)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_all_models_present(self):
        t = make_tracker()
        rows = model_breakdown(t)
        model_ids = {r["model_id"] for r in rows}
        assert "gpt-4o"      in model_ids
        assert "gpt-4o-mini" in model_ids

    def test_sorted_by_cost_descending(self):
        t = make_tracker()
        rows = model_breakdown(t)
        costs = [r["total_cost"] for r in rows]
        assert costs == sorted(costs, reverse=True)

    def test_pct_of_total_sums_to_100(self):
        t = make_tracker()
        rows = model_breakdown(t)
        total_pct = sum(r["pct_of_total"] for r in rows)
        assert total_pct == pytest.approx(100.0, rel=0.01)

    def test_call_counts_correct(self):
        t = make_tracker()
        rows = model_breakdown(t)
        gpt4o_row = next(r for r in rows if r["model_id"] == "gpt-4o")
        assert gpt4o_row["call_count"] == 2

    def test_empty_tracker(self):
        t = CostTracker()
        rows = model_breakdown(t)
        assert rows == []


class TestProviderBreakdown:

    def test_providers_present(self):
        t = make_tracker()
        rows = provider_breakdown(t)
        providers = {r["provider"] for r in rows}
        assert "openai"    in providers
        assert "anthropic" in providers

    def test_sorted_by_cost_descending(self):
        t = make_tracker()
        rows = provider_breakdown(t)
        costs = [r["total_cost"] for r in rows]
        assert costs == sorted(costs, reverse=True)

    def test_models_used_field(self):
        t = make_tracker()
        rows = provider_breakdown(t)
        openai_row = next(r for r in rows if r["provider"] == "openai")
        assert "gpt-4o"      in openai_row["models_used"]
        assert "gpt-4o-mini" in openai_row["models_used"]


class TestTagBreakdown:

    def test_feature_breakdown(self):
        t = make_tracker()
        rows = tag_breakdown(t, "feature")
        tag_values = {r["tag_value"] for r in rows}
        assert "chat"     in tag_values
        assert "search"   in tag_values
        assert "summarise" in tag_values

    def test_env_breakdown(self):
        t = make_tracker()
        rows = tag_breakdown(t, "env")
        tag_values = {r["tag_value"] for r in rows}
        assert "prod"    in tag_values
        assert "staging" in tag_values

    def test_untagged_calls_handled(self):
        t = CostTracker()
        t.record("gpt-4o-mini", 100, 50)  # No tags
        t.record("gpt-4o-mini", 100, 50, tags={"feature": "chat"})
        rows = tag_breakdown(t, "feature")
        tag_values = {r["tag_value"] for r in rows}
        assert "(untagged)" in tag_values


class TestTimeSeries:

    def test_returns_list(self):
        t = make_tracker()
        rows = time_series(t)
        assert isinstance(rows, list)

    def test_bucket_labels_present(self):
        t = make_tracker()
        rows = time_series(t, granularity="hour")
        assert all("bucket_label" in r for r in rows)

    def test_call_counts_sum_to_total(self):
        t = make_tracker()
        rows = time_series(t)
        total_calls = sum(r["call_count"] for r in rows)
        assert total_calls == t.call_count()

    def test_cost_sums_match_total(self):
        t = make_tracker()
        rows = time_series(t)
        series_total = sum(r["total_cost"] for r in rows)
        assert series_total == pytest.approx(t.total_cost(), rel=0.001)

    def test_day_granularity(self):
        t = make_tracker()
        rows = time_series(t, granularity="day")
        assert all("bucket_label" in r for r in rows)


class TestTopExpensiveCalls:

    def test_returns_top_n(self):
        t = make_tracker()
        top = top_expensive_calls(t, n=3)
        assert len(top) <= 3

    def test_sorted_by_cost_descending(self):
        t = make_tracker()
        top = top_expensive_calls(t, n=10)
        costs = [r.total_cost for r in top]
        assert costs == sorted(costs, reverse=True)

    def test_top_1_is_most_expensive(self):
        t = make_tracker()
        top1 = top_expensive_calls(t, n=1)[0]
        all_costs = [r.total_cost for r in t.filter()]
        assert top1.total_cost == max(all_costs)


class TestEfficiencyReport:

    def test_returns_list(self):
        t = make_tracker()
        rows = efficiency_report(t)
        assert isinstance(rows, list)
        assert len(rows) == len(t.cost_by_model())

    def test_sorted_by_cost_per_output(self):
        t = make_tracker()
        rows = efficiency_report(t)
        costs = [r["cost_per_1k_output"] for r in rows]
        assert costs == sorted(costs)

    def test_fields_present(self):
        t = make_tracker()
        rows = efficiency_report(t)
        for row in rows:
            assert "model_id"           in row
            assert "output_input_ratio" in row
            assert "cost_per_1k_output" in row
            assert "cache_hit_rate"     in row


class TestComparisonTable:

    def test_sorted_by_cost(self):
        rows = comparison_table(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
        costs = [r["total_cost"] for r in rows]
        assert costs == sorted(costs)

    def test_gpt4o_mini_cheapest(self):
        rows = comparison_table(["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
        assert rows[0]["model_id"] == "gpt-4o-mini"

    def test_fields_present(self):
        rows = comparison_table(["gpt-4o-mini"])
        row = rows[0]
        assert "model_id"       in row
        assert "provider"       in row
        assert "total_cost"     in row
        assert "input_per_1k"   in row
        assert "output_per_1k"  in row
        assert "context_window" in row

    def test_unknown_model_handled(self):
        rows = comparison_table(["gpt-4o-mini", "not-a-model"])
        # Should not raise
        assert any(r["model_id"] == "gpt-4o-mini" for r in rows)


class TestPrettyPrint:

    def test_print_model_breakdown_no_error(self, capsys):
        t = make_tracker()
        print_model_breakdown(t)
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out

    def test_print_comparison_table_no_error(self, capsys):
        print_comparison_table(["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"])
        captured = capsys.readouterr()
        assert "gpt-4o-mini" in captured.out
        assert "$" in captured.out
