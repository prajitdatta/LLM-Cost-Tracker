"""
tests/test_tracker.py

Tests for src/tracker.py — CostTracker session tracking and aggregation.
"""
import json
import pytest
from src.tracker import CostTracker, CallRecord


class TestRecord:

    def setup_method(self):
        self.tracker = CostTracker()

    def test_record_returns_call_record(self):
        record = self.tracker.record("gpt-4o-mini", 500, 200)
        assert isinstance(record, CallRecord)

    def test_record_assigns_incremental_ids(self):
        r1 = self.tracker.record("gpt-4o-mini", 100, 50)
        r2 = self.tracker.record("gpt-4o-mini", 100, 50)
        r3 = self.tracker.record("gpt-4o-mini", 100, 50)
        assert r1.call_id == 1
        assert r2.call_id == 2
        assert r3.call_id == 3

    def test_record_computes_correct_cost(self):
        record = self.tracker.record("gpt-4o-mini", 1000, 500)
        # input: 1000 * 0.00015/1k = 0.00015
        # output: 500 * 0.0006/1k = 0.0003
        expected = 0.000150 + 0.000300
        assert record.total_cost == pytest.approx(expected, rel=0.001)

    def test_record_stores_provider(self):
        record = self.tracker.record("gpt-4o", 100, 50)
        assert record.provider == "openai"

    def test_record_stores_canonical_model_id(self):
        # Alias should resolve to canonical ID
        record = self.tracker.record("gpt-4o-2024-11-20", 100, 50)
        assert record.model_id == "gpt-4o"

    def test_record_with_tags(self):
        record = self.tracker.record("gpt-4o-mini", 500, 200,
                                     tags={"feature": "summarise", "env": "prod"})
        assert record.tags["feature"] == "summarise"
        assert record.tags["env"]     == "prod"

    def test_record_with_latency(self):
        record = self.tracker.record("gpt-4o-mini", 500, 200, latency_ms=342.5)
        assert record.latency_ms == pytest.approx(342.5)

    def test_record_with_cached_tokens(self):
        record = self.tracker.record("gpt-4o", 1000, 500, cached_tokens=200)
        assert record.cached_tokens == 200

    def test_record_unknown_model_raises(self):
        with pytest.raises(KeyError):
            self.tracker.record("not-a-real-model-xyz", 100, 50)

    def test_record_response_from_openai_format(self):
        response = {
            "usage": {"prompt_tokens": 150, "completion_tokens": 300, "total_tokens": 450}
        }
        record = self.tracker.record_response("gpt-4o-mini", response, provider="openai")
        assert record.input_tokens  == 150
        assert record.output_tokens == 300

    def test_record_response_from_anthropic_format(self):
        response = {"usage": {"input_tokens": 80, "output_tokens": 240}}
        record = self.tracker.record_response("claude-3-5-sonnet-20241022", response, provider="anthropic")
        assert record.input_tokens  == 80
        assert record.output_tokens == 240


class TestAggregation:

    def setup_method(self):
        self.tracker = CostTracker()
        self.tracker.record("gpt-4o-mini", 1000, 500)
        self.tracker.record("gpt-4o-mini", 800,  300)
        self.tracker.record("gpt-4o",      500,  200)
        self.tracker.record("claude-3-5-sonnet-20241022", 600, 400)

    def test_total_cost_sums_all_calls(self):
        total = self.tracker.total_cost()
        assert total > 0
        manual = sum(r.total_cost for r in self.tracker.filter())
        assert total == pytest.approx(manual, rel=0.001)

    def test_total_tokens(self):
        tok = self.tracker.total_tokens()
        assert tok["input"]  == 1000 + 800 + 500 + 600
        assert tok["output"] == 500  + 300 + 200 + 400
        assert tok["total"]  == tok["input"] + tok["output"]

    def test_call_count(self):
        assert self.tracker.call_count() == 4

    def test_cost_by_model(self):
        by_model = self.tracker.cost_by_model()
        assert "gpt-4o-mini" in by_model
        assert "gpt-4o"      in by_model
        assert "claude-3-5-sonnet-20241022" in by_model
        # Should be sorted by cost descending
        costs = list(by_model.values())
        assert costs == sorted(costs, reverse=True)

    def test_cost_by_provider(self):
        by_prov = self.tracker.cost_by_provider()
        assert "openai"    in by_prov
        assert "anthropic" in by_prov

    def test_avg_cost_per_call(self):
        avg = self.tracker.avg_cost_per_call()
        total = self.tracker.total_cost()
        assert avg == pytest.approx(total / 4, rel=0.001)


class TestFiltering:

    def setup_method(self):
        self.tracker = CostTracker()
        self.tracker.record("gpt-4o",     500, 200, tags={"feature": "chat",   "env": "prod"})
        self.tracker.record("gpt-4o",     300, 100, tags={"feature": "search", "env": "prod"})
        self.tracker.record("gpt-4o-mini",800, 400, tags={"feature": "chat",   "env": "staging"})
        self.tracker.record("claude-3-5-sonnet-20241022", 600, 300, tags={"feature": "summary", "env": "prod"})

    def test_filter_by_model(self):
        records = self.tracker.filter(model_id="gpt-4o")
        assert len(records) == 2
        assert all(r.model_id == "gpt-4o" for r in records)

    def test_filter_by_provider(self):
        records = self.tracker.filter(provider="openai")
        assert len(records) == 3

    def test_filter_by_tag(self):
        records = self.tracker.filter(tags={"feature": "chat"})
        assert len(records) == 2

    def test_filter_by_multiple_tags(self):
        records = self.tracker.filter(tags={"feature": "chat", "env": "prod"})
        assert len(records) == 1
        assert records[0].model_id == "gpt-4o"

    def test_filter_no_match_returns_empty(self):
        records = self.tracker.filter(model_id="gemini-1.5-pro")
        assert records == []

    def test_total_cost_with_filter(self):
        total_prod = self.tracker.total_cost(tags={"env": "prod"})
        total_all  = self.tracker.total_cost()
        assert 0 < total_prod < total_all

    def test_cost_by_tag(self):
        by_feature = self.tracker.cost_by_tag("feature")
        assert "chat"    in by_feature
        assert "search"  in by_feature
        assert "summary" in by_feature

    def test_untagged_calls_grouped(self):
        self.tracker.record("gpt-4o-mini", 100, 50)  # no tags
        by_feature = self.tracker.cost_by_tag("feature")
        assert "(untagged)" in by_feature


class TestExport:

    def setup_method(self):
        self.tracker = CostTracker()
        self.tracker.record("gpt-4o-mini", 500, 200, tags={"feature": "test"})
        self.tracker.record("gpt-4o",      300, 150)

    def test_to_json_valid(self):
        json_str = self.tracker.to_json()
        data = json.loads(json_str)
        assert "calls" in data
        assert len(data["calls"]) == 2
        assert "session_summary" in data

    def test_to_json_contains_costs(self):
        data = json.loads(self.tracker.to_json())
        assert data["session_summary"]["total_cost_usd"] > 0

    def test_to_csv_valid(self):
        csv_str = self.tracker.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
        assert "model_id" in lines[0]
        assert "total_cost" in lines[0]

    def test_to_csv_tag_columns(self):
        csv_str = self.tracker.to_csv()
        assert "tag_feature" in csv_str

    def test_empty_tracker_csv(self):
        empty = CostTracker()
        assert empty.to_csv() == ""

    def test_summary_string(self):
        summary = self.tracker.summary()
        assert "gpt-4o-mini" in summary
        assert "$" in summary

    def test_empty_tracker_summary(self):
        empty = CostTracker()
        summary = empty.summary()
        assert "No calls" in summary


class TestRepr:

    def test_repr(self):
        t = CostTracker()
        t.record("gpt-4o-mini", 100, 50)
        r = repr(t)
        assert "CostTracker" in r
        assert "calls=1" in r
