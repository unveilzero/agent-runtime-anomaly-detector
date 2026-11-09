"""Runtime effect monitor for externally collected side effects.

The monitor is intentionally strict: it evaluates a trusted collector's event
stream against an allowlist. The agent itself should not be the source of truth.
"""

import time
from collections import defaultdict, deque

from src.audit import AuditLog


class RuntimeEffectMonitor:
    """Evaluate externally collected runtime effects against policy."""

    def __init__(self, policy, audit_log_path=".arad_effects_audit.jsonl",
                 blocked_action_sequences=None, correlation_window_seconds=300):
        self.policy = {action: frozenset(resources) for action, resources in policy.items()}
        self.audit_log = AuditLog(audit_log_path)
        self.blocked_action_sequences = tuple(blocked_action_sequences or ())
        self.correlation_window_seconds = correlation_window_seconds
        self._events_by_trace = defaultdict(deque)

    def evaluate_event(self, event):
        required = ("trace_id", "agent_id", "action", "resource")
        missing = [key for key in required if not event.get(key)]
        if missing:
            decision = "BLOCK"
            reason = "missing_required_fields:" + ",".join(missing)
        else:
            action = event["action"]
            resource = event["resource"]
            allowed = self.policy.get(action, frozenset())
            if resource not in allowed and "*" not in allowed:
                decision = "BLOCK"
                reason = "not_in_allowlist"
            else:
                decision = "ALLOW"
                reason = "allowed_by_policy"

        trace_id = event.get("trace_id")
        now = float(event.get("timestamp", time.time()))
        if trace_id:
            prior = self._events_by_trace[trace_id]
            while prior and now - prior[0][0] > self.correlation_window_seconds:
                prior.popleft()
            for first_action, second_action in self.blocked_action_sequences:
                if event.get("action") == second_action and any(item_action == first_action for _, item_action in prior):
                    decision = "BLOCK"
                    reason = f"correlated_actions:{first_action}->{second_action}"
                    break
            prior.append((now, event.get("action")))

        result = {"decision": decision, "reason": reason, "event": dict(event)}
        self.audit_log.append("runtime_effect_decision", result)
        return result

    def verify_audit_log(self):
        return self.audit_log.verify()
