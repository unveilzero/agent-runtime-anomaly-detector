"""Runtime event policy evaluator for externally collected effects."""

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping

from src.audit import AuditLog

MAX_LINEAGE_DEPTH = 64


class RuntimeEffectMonitor:
    """Evaluate process, file, and network events against exact allowlists.

    Blocked action sequences are correlated by ``trace_id`` and by agent
    lineage (the root reached through ``parent_agent_id``). Tracking state is
    never evicted while it is inside the correlation window; when capacity is
    exhausted, new traces and agents are blocked instead: a flood of allowed
    events can therefore deny service, but cannot shed history. Lineage must be
    reported by a trusted collector, not by the agent. An agent seen without
    ``parent_agent_id`` has no known parent yet; the first parent reported for
    it is adopted and is fixed from then on. The correlation window is measured
    in the collector's event timestamps; set ``max_timestamp_skew_seconds`` in
    live deployments to reject timestamps far from this host's clock.
    """

    def __init__(self, policy, audit_log_path=".arad_effects_audit.jsonl",
                 blocked_action_sequences=None, correlation_window_seconds=300,
                 audit_key=None, max_tracked_keys=10000, clock=time.monotonic,
                 max_timestamp_skew_seconds=None):
        if not isinstance(policy, Mapping):
            raise TypeError("policy must map action names to resource collections")
        if not isinstance(correlation_window_seconds, (int, float)) or not math.isfinite(correlation_window_seconds) or correlation_window_seconds <= 0:
            raise ValueError("correlation_window_seconds must be finite and positive")
        if not isinstance(max_tracked_keys, int) or isinstance(max_tracked_keys, bool) or max_tracked_keys < 2:
            raise ValueError("max_tracked_keys must be an integer of at least 2")
        self.policy = {}
        for action, resources in policy.items():
            if not isinstance(action, str) or not action:
                raise ValueError("Policy action names must be non-empty strings")
            resource_set = frozenset(resources)
            if any(not isinstance(resource, str) for resource in resource_set):
                raise ValueError("Policy resources must be strings")
            self.policy[action] = resource_set
        if max_timestamp_skew_seconds is not None and (
                not isinstance(max_timestamp_skew_seconds, (int, float)) or isinstance(max_timestamp_skew_seconds, bool)
                or not math.isfinite(max_timestamp_skew_seconds) or max_timestamp_skew_seconds <= 0):
            raise ValueError("max_timestamp_skew_seconds must be finite and positive")
        self.max_timestamp_skew_seconds = max_timestamp_skew_seconds
        pairs = []
        for pair in blocked_action_sequences or ():
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or any(not isinstance(action, str) or not action for action in pair)):
                raise ValueError("blocked_action_sequences must contain (first_action, second_action) string pairs")
            pairs.append(tuple(pair))
        self.audit_log = AuditLog(audit_log_path, key=audit_key)
        self.blocked_action_sequences = tuple(pairs)
        self._first_actions = frozenset(first for first, _ in self.blocked_action_sequences)
        self.correlation_window_seconds = float(correlation_window_seconds)
        self.max_tracked_keys = max_tracked_keys
        self._clock = clock
        # Both tables are ordered by last arrival. Entries are dropped only
        # after a full correlation window without activity.
        self._history = OrderedDict()  # key -> {"last": {action: ts}, "max_ts", "seen"}
        self._agents = OrderedDict()   # agent_id -> {"parent", "seen"}
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
        parent = fields.get("parent_agent_id")
        if parent is not None and (not isinstance(parent, str) or not parent):
            missing.append("parent_agent_id")
        return fields, missing

    @staticmethod
    def _auditable(fields):
        """Copy event fields in a form the audit log can always serialize."""
        safe = {}
        for name, value in fields.items():
            if value is None or isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value)):
                safe[name] = value
            else:
                safe[name] = f"<{type(value).__name__}: {str(value)[:80]}>"
        return safe

    def _make_room(self, table, now):
        """Drop idle entries; return False if the table is still full."""
        cutoff = now - self.correlation_window_seconds
        while len(table) >= self.max_tracked_keys:
            oldest = next(iter(table))
            if table[oldest]["seen"] > cutoff:
                return False
            del table[oldest]
        return True

    def _bucket(self, key, now):
        bucket = self._history.get(key)
        if bucket is None:
            if not self._make_room(self._history, now):
                return None
            bucket = {"last": {}, "max_ts": float("-inf"), "seen": now}
            self._history[key] = bucket
        else:
            bucket["seen"] = now
            self._history.move_to_end(key)
        return bucket

    def _walk_to_root(self, start, now=None, forbidden=None):
        """Follow parents from ``start``; with ``now``, refresh every tracked agent on the path."""
        current, seen = start, {start}
        for _ in range(MAX_LINEAGE_DEPTH):
            if current == forbidden:
                return None, "lineage_cycle"
            entry = self._agents.get(current)
            if entry is not None and now is not None:
                entry["seen"] = now
                self._agents.move_to_end(current)
            if entry is None or entry["parent"] is None:
                return current, None
            current = entry["parent"]
            if current in seen:
                return None, "lineage_cycle"
            seen.add(current)
        return None, "lineage_too_deep"

    def _peek_lineage(self, agent_id, parent_id):
        """Resolve the root without changing any state."""
        entry = self._agents.get(agent_id)
        known = entry["parent"] if entry is not None else None
        if known is not None:
            return None if parent_id not in (None, known) else self._walk_to_root(agent_id)[0]
        if parent_id is None:
            return agent_id
        return self._walk_to_root(parent_id, forbidden=agent_id)[0]

    def _resolve_lineage(self, agent_id, parent_id, now):
        """Record the agent's parent; return (root, problem_reason, undo).

        ``undo`` reverses this call's registration and is None when nothing
        was recorded. Checks that can fail run before any state changes.
        """
        entry = self._agents.get(agent_id)
        known = entry["parent"] if entry is not None else None
        if known is not None and parent_id not in (None, known):
            return None, "lineage_conflict", None
        undo = None
        if parent_id is not None and known is None:
            _, problem = self._walk_to_root(parent_id, forbidden=agent_id)
            if problem:
                return None, problem, None
        if entry is None:
            if not self._make_room(self._agents, now):
                return None, "lineage_capacity_exceeded", None
            self._agents[agent_id] = {"parent": parent_id, "seen": now}
            undo = lambda: self._agents.pop(agent_id, None)
        elif parent_id is not None and known is None:
            # No parent was reported before: adopt this one. The caller folds
            # the agent's history as a root into the new root.
            entry["parent"] = parent_id
            undo = lambda: entry.__setitem__("parent", None)
        root, problem = self._walk_to_root(agent_id, now)
        if problem and undo:
            undo()
            undo = None
        return root, problem, undo

    def _lineage_bucket(self, root, agent_id, now):
        """Return the root's bucket, first folding in history the agent held as an implicit root."""
        orphan_key, root_key = ("lineage", agent_id), ("lineage", root)
        if orphan_key != root_key and orphan_key in self._history:
            orphan = self._history.pop(orphan_key)
            target = self._history.get(root_key)
            if target is None:
                # Re-key in place: the freed slot is reused, so this cannot fail at capacity.
                self._history[root_key] = orphan
            else:
                for action, ts in orphan["last"].items():
                    target["last"][action] = max(ts, target["last"].get(action, float("-inf")))
                target["max_ts"] = max(target["max_ts"], orphan["max_ts"])
        return self._bucket(root_key, now)

    def _sequence_violation(self, bucket, action, timestamp):
        # Matches on arrival order: an earlier-observed first action counts
        # even if this event claims an older timestamp.
        window_start = timestamp - self.correlation_window_seconds
        for first_action, second_action in self.blocked_action_sequences:
            if action == second_action and bucket["last"].get(first_action, float("-inf")) >= window_start:
                return f"correlated_actions:{first_action}->{second_action}"
        return None

    def _correlate(self, fields, timestamp):
        now = self._clock()
        agent_id, trace_key = fields["agent_id"], ("trace", fields["trace_id"])
        root, problem, undo_lineage = self._resolve_lineage(agent_id, fields.get("parent_agent_id"), now)
        if problem:
            return "BLOCK", problem, None
        new_trace = trace_key not in self._history
        trace_bucket = self._bucket(trace_key, now)
        lineage_bucket = self._lineage_bucket(root, agent_id, now) if trace_bucket is not None else None
        if lineage_bucket is None:
            # Out of capacity: undo this event's registrations so it leaves no trace.
            if undo_lineage:
                undo_lineage()
            if new_trace and trace_bucket is not None:
                del self._history[trace_key]
            return "BLOCK", "history_capacity_exceeded", root
        if timestamp < trace_bucket["max_ts"]:
            return "BLOCK", "non_monotonic_trace_timestamp", root
        if timestamp < lineage_bucket["max_ts"] - self.correlation_window_seconds:
            return "BLOCK", "stale_lineage_timestamp", root
        buckets = (("trace", trace_bucket), ("lineage", lineage_bucket))
        for scope, bucket in buckets:
            violation = self._sequence_violation(bucket, fields["action"], timestamp)
            if violation:
                return "BLOCK", f"{violation}:{scope}", root
        # Only allowed events advance a key's clock, so a rejected far-future
        # timestamp cannot age out real history.
        for _, bucket in buckets:
            bucket["max_ts"] = max(bucket["max_ts"], timestamp)
            if fields["action"] in self._first_actions:
                bucket["last"][fields["action"]] = max(
                    timestamp, bucket["last"].get(fields["action"], float("-inf")))
        return "ALLOW", "allowed_by_policy", root

    def evaluate_event(self, event):
        fields, missing = self._event_fields(event)
        timestamp = time.time()
        if missing:
            decision, reason = "BLOCK", "invalid_required_fields:" + ",".join(missing)
        else:
            wall_clock = time.time()
            timestamp = fields.get("timestamp", wall_clock)
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                timestamp = float("nan")
            timestamp = float(timestamp)
            if not math.isfinite(timestamp):
                decision, reason = "BLOCK", "invalid_timestamp"
            elif (self.max_timestamp_skew_seconds is not None
                    and abs(timestamp - wall_clock) > self.max_timestamp_skew_seconds):
                decision, reason, timestamp = "BLOCK", "timestamp_out_of_range", float("nan")
            else:
                allowed = self.policy.get(fields["action"], frozenset())
                if fields["resource"] not in allowed and "*" not in allowed:
                    decision, reason = "BLOCK", "not_in_allowlist"
                else:
                    decision, reason = "ALLOW", "allowed_by_policy"

        lineage_root = None
        if not missing and math.isfinite(timestamp):
            with self._lock:
                if decision == "BLOCK":
                    # Policy-denied events are reported but never change tracking state.
                    lineage_root = self._peek_lineage(fields["agent_id"], fields.get("parent_agent_id"))
                else:
                    decision, reason, lineage_root = self._correlate(fields, timestamp)

        result = {"decision": decision, "reason": reason, "event": self._auditable(fields),
                  "lineage_root": lineage_root}
        self.audit_log.append("runtime_effect_decision", result)
        return result

    def anchor_audit_log(self):
        return self.audit_log.anchor()

    def verify_audit_log(self, expected_anchor=None):
        return self.audit_log.verify(expected_anchor=expected_anchor)
