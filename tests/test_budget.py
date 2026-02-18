"""
tests/test_budget.py

Tests for src/budget.py — BudgetManager limits, alerts, and auto-cutoff.
"""
import pytest
from src.budget import BudgetManager, BudgetAlert, BudgetExceededError, AlertLevel


class TestBudgetSetup:

    def test_set_session_budget_returns_self(self):
        bm = BudgetManager()
        result = bm.set_session_budget(10.00)
        assert result is bm  # fluent interface

    def test_set_model_budget_returns_self(self):
        bm = BudgetManager()
        result = bm.set_model_budget("gpt-4o", 5.00)
        assert result is bm

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            BudgetManager().set_session_budget(10.00, warning_threshold=0.95, critical_threshold=0.80)

    def test_chained_setup(self):
        bm = (BudgetManager()
              .set_session_budget(100.00)
              .set_model_budget("gpt-4o", 50.00)
              .set_model_budget("gpt-4o-mini", 20.00))
        status = bm.status()
        assert "session"     in status["budgets"]
        assert "gpt-4o"      in status["budgets"]
        assert "gpt-4o-mini" in status["budgets"]


class TestAlerts:

    def test_no_alert_below_threshold(self):
        bm = BudgetManager().set_session_budget(10.00, warning_threshold=0.80)
        alerts = bm.record_spend(7.00)  # 70% — below 80% warning
        assert alerts == []

    def test_warning_at_80_percent(self):
        bm = BudgetManager().set_session_budget(10.00, warning_threshold=0.80)
        alerts = bm.record_spend(8.50)  # 85%
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING

    def test_critical_at_95_percent(self):
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(9.00)  # 90% — triggers warning
        alerts = bm.record_spend(0.60)  # 96% — triggers critical
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) == 1

    def test_exceeded_at_100_percent(self):
        bm = BudgetManager().set_session_budget(5.00, auto_cutoff=False)
        bm.record_spend(4.90)
        alerts = bm.record_spend(0.20)  # Goes over $5
        exceeded = [a for a in alerts if a.level == AlertLevel.EXCEEDED]
        assert len(exceeded) == 1

    def test_alert_fires_only_once_per_level(self):
        """The same alert level should not trigger twice for the same scope."""
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(8.50)  # Triggers WARNING
        alerts2 = bm.record_spend(0.10)  # Still in WARNING zone
        assert not any(a.level == AlertLevel.WARNING for a in alerts2)

    def test_on_alert_callback(self):
        received = []
        bm = BudgetManager().set_session_budget(
            10.00, on_alert=lambda a: received.append(a)
        )
        bm.record_spend(8.50)
        assert len(received) == 1
        assert received[0].level == AlertLevel.WARNING

    def test_model_specific_alert(self):
        bm = BudgetManager().set_model_budget("gpt-4o", 1.00)
        alerts = bm.record_spend(0.90, model_id="gpt-4o")
        assert any(a.scope == "gpt-4o" for a in alerts)

    def test_alert_remaining_usd(self):
        bm = BudgetManager().set_session_budget(10.00)
        alerts = bm.record_spend(8.50)
        alert = alerts[0]
        assert alert.remaining_usd == pytest.approx(10.00 - 8.50, rel=0.001)

    def test_get_alerts_returns_history(self):
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(8.50)
        bm.record_spend(1.00)
        history = bm.get_alerts()
        assert len(history) >= 1
        assert all(isinstance(a, BudgetAlert) for a in history)


class TestAutoCutoff:

    def test_auto_cutoff_raises_on_check(self):
        bm = BudgetManager().set_session_budget(1.00, auto_cutoff=True)
        bm.record_spend(1.10)  # Go over limit (no check yet)
        with pytest.raises(BudgetExceededError) as exc_info:
            bm.check()  # Now raises
        assert exc_info.value.alert.level == AlertLevel.EXCEEDED

    def test_no_error_when_under_budget(self):
        bm = BudgetManager().set_session_budget(10.00, auto_cutoff=True)
        bm.record_spend(5.00)
        bm.check()  # Should not raise

    def test_auto_cutoff_model_budget(self):
        bm = BudgetManager().set_model_budget("gpt-4o", 2.00, auto_cutoff=True)
        bm.record_spend(2.50, model_id="gpt-4o")
        with pytest.raises(BudgetExceededError):
            bm.check(model_id="gpt-4o")

    def test_budget_exceeded_error_message(self):
        bm = BudgetManager().set_session_budget(1.00, auto_cutoff=True)
        bm.record_spend(1.50)
        try:
            bm.check()
            assert False, "Should have raised"
        except BudgetExceededError as e:
            assert "exceeded" in str(e).lower()
            assert e.alert is not None


class TestStatus:

    def test_status_no_budgets(self):
        bm = BudgetManager()
        bm.record_spend(1.00)
        status = bm.status()
        assert status["session_spent"] == pytest.approx(1.00)
        assert status["budgets"] == {}

    def test_status_with_session_budget(self):
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(3.00)
        status = bm.status()
        session = status["budgets"]["session"]
        assert session["limit"]     == pytest.approx(10.00)
        assert session["spent"]     == pytest.approx(3.00)
        assert session["remaining"] == pytest.approx(7.00)
        assert session["pct_used"]  == pytest.approx(30.0)
        assert session["status"]    == "ok"

    def test_status_warning_state(self):
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(8.50)
        status = bm.status()
        assert status["budgets"]["session"]["status"] == "warning"

    def test_status_exceeded_state(self):
        bm = BudgetManager().set_session_budget(5.00, auto_cutoff=False)
        bm.record_spend(6.00)
        status = bm.status()
        assert status["budgets"]["session"]["status"] == "exceeded"

    def test_status_tracks_per_model_spend(self):
        bm = BudgetManager().set_model_budget("gpt-4o", 5.00)
        bm.record_spend(2.00, model_id="gpt-4o")
        bm.record_spend(1.00, model_id="gpt-4o-mini")
        status = bm.status()
        assert status["model_spent"]["gpt-4o"]      == pytest.approx(2.00)
        assert status["model_spent"]["gpt-4o-mini"] == pytest.approx(1.00)


class TestReset:

    def test_reset_session(self):
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(8.00)
        bm.reset("session")
        assert bm.status()["session_spent"] == pytest.approx(0.0)

    def test_reset_model(self):
        bm = BudgetManager().set_model_budget("gpt-4o", 5.00)
        bm.record_spend(3.00, model_id="gpt-4o")
        bm.reset("gpt-4o")
        assert bm.status()["model_spent"].get("gpt-4o", 0.0) == pytest.approx(0.0)

    def test_alerts_retriggerable_after_reset(self):
        """After reset, alerts should be triggerable again."""
        bm = BudgetManager().set_session_budget(10.00)
        bm.record_spend(8.50)  # Triggers WARNING
        bm.reset("session")
        alerts = bm.record_spend(8.50)  # Should trigger WARNING again
        warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warnings) == 1
