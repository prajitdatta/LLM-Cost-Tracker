"""
examples/budget_alerts.py

Set a session budget, watch alerts trigger at 80%/95%/100%, demo auto-cutoff.

Run:
    python examples/budget_alerts.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracker import CostTracker
from src.budget import BudgetManager, BudgetExceededError

def main():
    print("=" * 55)
    print("  LLM Cost Tracker — Budget Alerts Example")
    print("=" * 55)

    # ── 1. Session budget with alerts ────────────────────────
    alerts_received = []

    def on_alert(alert):
        alerts_received.append(alert)
        print(f"  🔔 ALERT: {alert.message}")

    tracker = CostTracker()
    tracker.budget.set_session_budget(
        0.01,                   # $0.01 budget
        warning_threshold=0.80,
        auto_cutoff=False,
        on_alert=on_alert,
    )

    print(f"\n[Budget: $0.01 — spending in increments]")
    spends = [0.004, 0.003, 0.002, 0.002]
    total  = 0.0
    for amount in spends:
        tracker.budget.record_spend(amount)
        total += amount
        status = tracker.budget.status()["budgets"].get("session", {})
        pct    = status.get("pct_used", 0)
        state  = status.get("status", "ok")
        print(f"  Spent ${total:.4f} / $0.01  ({pct:.0f}%)  [{state}]")

    print(f"\n  Total alerts received: {len(alerts_received)}")

    # ── 2. Auto-cutoff demo ───────────────────────────────────
    print(f"\n[Auto-cutoff demo — $0.005 budget, hard stop]")
    tracker2 = CostTracker(session_budget=0.005, auto_cutoff=True)

    # These calls cost roughly $0.0002 each
    for i in range(15):
        try:
            tracker2.budget.check()
            tracker2.budget.record_spend(0.0004)
            print(f"  Call {i+1}: OK  (spent ${tracker2.budget.status()['session_spent']:.4f})")
        except BudgetExceededError as e:
            print(f"  Call {i+1}: 🚨 BLOCKED — {e}")
            break

    # ── 3. Per-model budget ───────────────────────────────────
    print(f"\n[Per-model budgets]")
    bm = (BudgetManager()
          .set_session_budget(1.00)
          .set_model_budget("gpt-4o", 0.10)
          .set_model_budget("claude-3-opus-20240229", 0.05))

    bm.record_spend(0.09, model_id="gpt-4o")
    bm.record_spend(0.04, model_id="claude-3-opus-20240229")

    print(f"\n  Status:")
    for scope, data in bm.status()["budgets"].items():
        print(f"    {scope:<35} ${data['spent']:.4f}/${data['limit']:.4f} [{data['status']}]")

if __name__ == "__main__":
    main()
