"""
src/budget.py

BudgetManager — set spending limits, receive alerts, auto-cut off spending.

Supports three budget scopes:
    session:  Entire tracking session
    daily:    Rolling 24-hour window
    per_model: Per-model limit (e.g. "no more than $5 on GPT-4o")

Alert levels:
    WARNING:  Configurable threshold (default 80% of budget)
    CRITICAL: 95% of budget
    EXCEEDED: Over budget — can optionally block further calls
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class AlertLevel(Enum):
    OK       = "ok"
    WARNING  = "warning"    # Approaching limit
    CRITICAL = "critical"   # Near limit
    EXCEEDED = "exceeded"   # Over limit


@dataclass
class BudgetAlert:
    """Triggered when spending crosses a threshold."""
    level:      AlertLevel
    scope:      str             # "session" | "daily" | model_id
    budget_usd: float
    spent_usd:  float
    pct_used:   float
    message:    str
    timestamp:  float = field(default_factory=time.time)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def __repr__(self) -> str:
        return f"BudgetAlert({self.level.value.upper()} | {self.scope} | {self.pct_used:.0f}% used | ${self.spent_usd:.4f}/${self.budget_usd:.4f})"


@dataclass
class BudgetConfig:
    """
    Budget configuration for a single scope.

    Args:
        limit_usd:          Hard spending limit in USD
        warning_threshold:  Fraction of limit that triggers WARNING (default 0.80)
        critical_threshold: Fraction of limit that triggers CRITICAL (default 0.95)
        auto_cutoff:        If True, raise BudgetExceededError when limit is hit
        on_alert:           Optional callback(BudgetAlert) invoked on each alert
    """
    limit_usd:           float
    warning_threshold:   float = 0.80
    critical_threshold:  float = 0.95
    auto_cutoff:         bool  = False
    on_alert:            Optional[Callable[[BudgetAlert], None]] = None

    def __post_init__(self):
        if not (0 < self.warning_threshold < self.critical_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < warning < critical <= 1.0. "
                f"Got warning={self.warning_threshold}, critical={self.critical_threshold}"
            )


class BudgetExceededError(Exception):
    """Raised when auto_cutoff=True and the budget limit is exceeded."""
    def __init__(self, alert: BudgetAlert):
        self.alert = alert
        super().__init__(
            f"Budget exceeded: ${alert.spent_usd:.4f} spent of "
            f"${alert.budget_usd:.4f} limit ({alert.scope})"
        )


class BudgetManager:
    """
    Manages spending budgets across multiple scopes.

    Usage:
        budget = BudgetManager()
        budget.set_session_budget(10.00)
        budget.set_model_budget("gpt-4o", 5.00)

        # Before each API call:
        budget.check("gpt-4o")  # raises BudgetExceededError if over limit

        # After each call:
        budget.record_spend(0.0023, model_id="gpt-4o")

        # Get status
        print(budget.status())
    """

    def __init__(self):
        self._session_config:  Optional[BudgetConfig] = None
        self._daily_config:    Optional[BudgetConfig] = None
        self._model_configs:   dict[str, BudgetConfig] = {}
        self._session_spent:   float = 0.0
        self._model_spent:     dict[str, float] = {}
        self._daily_spent:     float = 0.0
        self._daily_reset_at:  float = time.time()
        self._alerts:          list[BudgetAlert] = []
        self._alerted_levels:  dict[str, set[AlertLevel]] = {}  # scope → set of triggered levels

    # ── Budget setters ────────────────────────────────────────────────────────

    def set_session_budget(
        self,
        limit_usd: float,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.95,
        auto_cutoff: bool = False,
        on_alert: Optional[Callable] = None,
    ) -> "BudgetManager":
        """Set a budget for the entire session. Returns self for chaining."""
        self._session_config = BudgetConfig(
            limit_usd, warning_threshold, critical_threshold, auto_cutoff, on_alert
        )
        return self

    def set_daily_budget(
        self,
        limit_usd: float,
        warning_threshold: float = 0.80,
        auto_cutoff: bool = False,
        on_alert: Optional[Callable] = None,
    ) -> "BudgetManager":
        """Set a rolling 24-hour spending budget. Returns self for chaining."""
        self._daily_config = BudgetConfig(
            limit_usd, warning_threshold, 0.95, auto_cutoff, on_alert
        )
        return self

    def set_model_budget(
        self,
        model_id: str,
        limit_usd: float,
        warning_threshold: float = 0.80,
        auto_cutoff: bool = False,
        on_alert: Optional[Callable] = None,
    ) -> "BudgetManager":
        """Set a per-model spending budget. Returns self for chaining."""
        self._model_configs[model_id] = BudgetConfig(
            limit_usd, warning_threshold, 0.95, auto_cutoff, on_alert
        )
        return self

    # ── Spending recording ────────────────────────────────────────────────────

    def record_spend(self, amount_usd: float, model_id: Optional[str] = None) -> list[BudgetAlert]:
        """
        Record a spending event and check all applicable budgets.

        Args:
            amount_usd: Cost of this API call in USD
            model_id:   Model used (for per-model budget tracking)

        Returns:
            List of BudgetAlerts triggered by this spend (may be empty)

        Raises:
            BudgetExceededError: If auto_cutoff=True and a budget is exceeded
        """
        self._refresh_daily_if_needed()

        self._session_spent += amount_usd
        self._daily_spent   += amount_usd
        if model_id:
            self._model_spent[model_id] = self._model_spent.get(model_id, 0.0) + amount_usd

        triggered = []

        # Check session budget
        if self._session_config:
            alert = self._check_budget(
                "session", self._session_spent, self._session_config
            )
            if alert:
                triggered.append(alert)

        # Check daily budget
        if self._daily_config:
            alert = self._check_budget(
                "daily", self._daily_spent, self._daily_config
            )
            if alert:
                triggered.append(alert)

        # Check per-model budget
        if model_id and model_id in self._model_configs:
            spent = self._model_spent.get(model_id, 0.0)
            alert = self._check_budget(
                model_id, spent, self._model_configs[model_id]
            )
            if alert:
                triggered.append(alert)

        return triggered

    def check(self, model_id: Optional[str] = None):
        """
        Pre-flight check before making an API call.
        Raises BudgetExceededError if any auto_cutoff budget is already exceeded.
        """
        self._refresh_daily_if_needed()

        checks = []
        if self._session_config and self._session_config.auto_cutoff:
            checks.append(("session", self._session_spent, self._session_config))
        if self._daily_config and self._daily_config.auto_cutoff:
            checks.append(("daily", self._daily_spent, self._daily_config))
        if model_id and model_id in self._model_configs:
            cfg = self._model_configs[model_id]
            if cfg.auto_cutoff:
                checks.append((model_id, self._model_spent.get(model_id, 0.0), cfg))

        for scope, spent, cfg in checks:
            if spent >= cfg.limit_usd:
                alert = BudgetAlert(
                    level=AlertLevel.EXCEEDED,
                    scope=scope,
                    budget_usd=cfg.limit_usd,
                    spent_usd=spent,
                    pct_used=spent / cfg.limit_usd * 100,
                    message=f"Budget exceeded for {scope}: ${spent:.4f} of ${cfg.limit_usd:.4f}",
                )
                raise BudgetExceededError(alert)

    # ── Status & reporting ────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return current spending status across all configured budgets."""
        self._refresh_daily_if_needed()
        result = {
            "session_spent": self._session_spent,
            "daily_spent":   self._daily_spent,
            "model_spent":   dict(self._model_spent),
            "budgets":       {},
        }

        if self._session_config:
            cfg = self._session_config
            result["budgets"]["session"] = {
                "limit":     cfg.limit_usd,
                "spent":     self._session_spent,
                "remaining": max(0.0, cfg.limit_usd - self._session_spent),
                "pct_used":  min(100.0, self._session_spent / cfg.limit_usd * 100),
                "status":    self._alert_level("session", self._session_spent, cfg).value,
            }

        if self._daily_config:
            cfg = self._daily_config
            result["budgets"]["daily"] = {
                "limit":     cfg.limit_usd,
                "spent":     self._daily_spent,
                "remaining": max(0.0, cfg.limit_usd - self._daily_spent),
                "pct_used":  min(100.0, self._daily_spent / cfg.limit_usd * 100),
                "status":    self._alert_level("daily", self._daily_spent, cfg).value,
            }

        for model_id, cfg in self._model_configs.items():
            spent = self._model_spent.get(model_id, 0.0)
            result["budgets"][model_id] = {
                "limit":     cfg.limit_usd,
                "spent":     spent,
                "remaining": max(0.0, cfg.limit_usd - spent),
                "pct_used":  min(100.0, spent / cfg.limit_usd * 100),
                "status":    self._alert_level(model_id, spent, cfg).value,
            }

        result["total_alerts"] = len(self._alerts)
        return result

    def get_alerts(self) -> list[BudgetAlert]:
        """Return all alerts that have been triggered this session."""
        return list(self._alerts)

    def reset(self, scope: str = "session"):
        """Reset spending counter for a scope."""
        if scope == "session":
            self._session_spent = 0.0
            self._alerted_levels.pop("session", None)
        elif scope == "daily":
            self._daily_spent = 0.0
            self._daily_reset_at = time.time()
            self._alerted_levels.pop("daily", None)
        elif scope in self._model_spent:
            self._model_spent[scope] = 0.0
            self._alerted_levels.pop(scope, None)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_budget(
        self,
        scope: str,
        spent: float,
        cfg: BudgetConfig,
    ) -> Optional[BudgetAlert]:
        """Check spending against a budget config. Returns alert if threshold crossed."""
        level = self._alert_level(scope, spent, cfg)
        if level == AlertLevel.OK:
            return None

        # Only alert once per level per scope
        already_alerted = self._alerted_levels.get(scope, set())
        if level in already_alerted:
            return None

        pct = spent / cfg.limit_usd * 100
        messages = {
            AlertLevel.WARNING:  f"⚠️  Budget {pct:.0f}% used for {scope}: ${spent:.4f} of ${cfg.limit_usd:.4f}",
            AlertLevel.CRITICAL: f"🔴 Budget {pct:.0f}% used for {scope}: ${spent:.4f} of ${cfg.limit_usd:.4f}",
            AlertLevel.EXCEEDED: f"🚨 Budget EXCEEDED for {scope}: ${spent:.4f} of ${cfg.limit_usd:.4f}",
        }

        alert = BudgetAlert(
            level=level,
            scope=scope,
            budget_usd=cfg.limit_usd,
            spent_usd=spent,
            pct_used=pct,
            message=messages[level],
        )
        self._alerts.append(alert)
        self._alerted_levels.setdefault(scope, set()).add(level)

        if cfg.on_alert:
            cfg.on_alert(alert)

        if cfg.auto_cutoff and level == AlertLevel.EXCEEDED:
            raise BudgetExceededError(alert)

        return alert

    def _alert_level(self, scope: str, spent: float, cfg: BudgetConfig) -> AlertLevel:
        """Determine the current alert level for a scope."""
        pct = spent / cfg.limit_usd
        if pct >= 1.0:
            return AlertLevel.EXCEEDED
        if pct >= cfg.critical_threshold:
            return AlertLevel.CRITICAL
        if pct >= cfg.warning_threshold:
            return AlertLevel.WARNING
        return AlertLevel.OK

    def _refresh_daily_if_needed(self):
        """Reset daily counters if 24 hours have passed."""
        if time.time() - self._daily_reset_at >= 86400:
            self._daily_spent = 0.0
            self._daily_reset_at = time.time()
            self._alerted_levels.pop("daily", None)
