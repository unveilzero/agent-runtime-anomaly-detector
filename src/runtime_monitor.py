"""Runtime event policy evaluator for externally collected effects."""

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping

from src.audit import AuditLog


class RuntimeEffectMonitor:
    """Evaluate process, file, and network events against exact allowlists."""

    def __init__(self, policy, audit_log_path=".arad_effects_audit.jsonl",
                 blocked_action_sequences=None, correlation_window_seconds=300):
        if not isinstance(policy, Mapping):
            raise TypeError("policy must map action names to resource collections")
        if not isinstance(correlation_window_seconds, (int, float)) or not math.isfinite(correlation_window_seconds) or correlation_window_seconds <= 0:
            raise ValueError("correlation_window_seconds must be finite and positive")
        self.policy = {}
        for action, resources in policy.items():
            if not isinstance(action, str) or not action:
                raise ValueError("Policy action names must be non-empty strings")
            resource_set = frozenset(resources)
            if any(not isinstance(resource, str) for resource in resource_set):
                raise ValueError("Policy resources must be strings")
            self.policy[action] = resource_set
        self.audit_log = AuditLog(audit_log_path)
        self.blocked_action_sequences = tuple(blocked_action_sequences or ())
        self.correlation_window_seconds = float(correlation_window_seconds)
        self._events_by_trace = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _event_fields(event):
        if not isinstance(event, Mapping):
            return {}, ["event_must_be_a_mapping"]
        fields = {name: event.get(name) for name in (
            "trace_id", "agent_id", "parent_agent_id", "action", "resource", "timestamp"
        ) if name in event}
        required = ("trace_id", "agent_id", "action", "resource")
        missing = [name for name in required if not isinstance(fields.get(name), str) or not fields[name]]
        return fields, missing

    def evaluate_event(self, event):
        fields, missing = self._event_fields(event)
        if missing:
            decision, reason, timestamp = "BLOCK", "invalid_required_fields:" + ",".join(missing), time.time()
        else:
            try:
                timestamp = float(fields.get("timestamp", time.time()))
            except (TypeError, ValueError):
                timestamp = float("nan")
            if not math.isfinite(timestamp):
                decision, reason = "BLOCK", "invalid_timestamp"
            else:
                allowed = self.policy.get(fields["action"], frozenset())
                if fields["resource"] not in allowed and "*" not in allowed:
                    decision, reason = "BLOCK", "not_in_allowlist"
                else:
                    decision, reason = "ALLOW", "allowed_by_policy"

        trace_id = fields.get("trace_id")
        if trace_id and math.isfinite(timestamp):
            with self._lock:
                prior = self._events_by_trace[trace_id]
                if prior and timestamp < prior[-1][0]:
                    decision, reason = "BLOCK", "non_monotonic_trace_timestamp"
                else:
                    while prior and timestamp - prior[0][0] > self.correlation_window_seconds:
                        prior.popleft()
                    if decision == "ALLOW":
                        for first_action, second_action in self.blocked_action_sequences:
                            if fields.get("action") == second_action and any(
                                action == first_action and outcome == "ALLOW"
                                for _, action, outcome in prior
                            ):
                                decision, reason = "BLOCK", f"correlated_actions:{first_action}->{second_action}"
                                break
                    prior.append((timestamp, fields.get("action"), decision))
        result = {"decision": decision, "reason": reason, "event": fields}
        self.audit_log.append("runtime_effect_decision", result)
        return result

    def verify_audit_log(self):
        return self.audit_log.verify()
