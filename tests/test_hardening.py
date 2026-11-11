"""Tests for lineage tracking, audit anchors, baseline integrity and calibration limits."""

import json
import time
import types

import numpy as np
import pytest

from src import CognitiveImmunityEngine, EnvironmentIntegrityValidator, RuntimeEffectMonitor
from src.audit import AuditLog
from src.environment_validator import _fingerprint_module

KEY = b"operator-held-secret-key-0123456789"


def _traces(count, dim=16, scale=0.001):
    return [np.eye(dim) + np.random.default_rng(seed).normal(0, scale, (dim, dim)) for seed in range(count)]


# Runtime monitor: lineage and tracking state

def _sequence_monitor(tmp_path):
    return RuntimeEffectMonitor(
        {"SENSITIVE_READ": {"/data"}, "OUTBOUND_CONNECT": {"svc"}},
        audit_log_path=tmp_path / "effects.jsonl",
        blocked_action_sequences=[("SENSITIVE_READ", "OUTBOUND_CONNECT")],
    )


def test_same_agent_new_trace_is_correlated(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "a", "action": "SENSITIVE_READ", "resource": "/data", "timestamp": 1})
    hop = monitor.evaluate_event({"trace_id": "t2", "agent_id": "a", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 2})
    assert hop["decision"] == "BLOCK"
    assert hop["reason"].endswith(":lineage")


def test_sub_agent_handoff_is_correlated(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "parent", "action": "SENSITIVE_READ", "resource": "/data", "timestamp": 1})
    child = monitor.evaluate_event({"trace_id": "t9", "agent_id": "child", "parent_agent_id": "parent",
                                    "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 2})
    assert child["decision"] == "BLOCK"
    assert child["lineage_root"] == "parent"


def test_unrelated_agents_are_not_correlated(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "a", "action": "SENSITIVE_READ", "resource": "/data", "timestamp": 1})
    other = monitor.evaluate_event({"trace_id": "t2", "agent_id": "b", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 2})
    assert other["decision"] == "ALLOW"


def test_reparenting_is_blocked(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "child", "parent_agent_id": "p1", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 1})
    moved = monitor.evaluate_event({"trace_id": "t1", "agent_id": "child", "parent_agent_id": "p2", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 2})
    assert moved == dict(moved, decision="BLOCK", reason="lineage_conflict")


def test_lineage_cycle_is_blocked(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "a", "parent_agent_id": "b", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 1})
    loop = monitor.evaluate_event({"trace_id": "t2", "agent_id": "b", "parent_agent_id": "a", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 2})
    assert loop["decision"] == "BLOCK" and loop["reason"] == "lineage_cycle"


def test_correlation_window_expires(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event({"trace_id": "t1", "agent_id": "a", "action": "SENSITIVE_READ", "resource": "/data", "timestamp": 0})
    late = monitor.evaluate_event({"trace_id": "t2", "agent_id": "a", "action": "OUTBOUND_CONNECT", "resource": "svc", "timestamp": 1000})
    assert late["decision"] == "ALLOW"


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _event(trace, agent, action, timestamp, **extra):
    resource = {"SENSITIVE_READ": "/data", "OUTBOUND_CONNECT": "svc", "NOOP": "x"}[action]
    return dict(trace_id=trace, agent_id=agent, action=action, resource=resource, timestamp=timestamp, **extra)


def _bounded_monitor(tmp_path, clock, max_tracked_keys=20):
    return RuntimeEffectMonitor(
        {"SENSITIVE_READ": {"/data"}, "OUTBOUND_CONNECT": {"svc"}, "NOOP": {"x"}},
        audit_log_path=tmp_path / "effects.jsonl",
        blocked_action_sequences=[("SENSITIVE_READ", "OUTBOUND_CONNECT")],
        max_tracked_keys=max_tracked_keys, clock=clock,
    )


def test_flooding_cannot_evict_live_history(tmp_path):
    monitor = _bounded_monitor(tmp_path, _Clock())
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 1))
    flood = [monitor.evaluate_event(_event(f"junk{i}", f"j{i}", "NOOP", 1)) for i in range(40)]
    assert flood[-1]["decision"] == "BLOCK"
    assert flood[-1]["reason"] in ("history_capacity_exceeded", "lineage_capacity_exceeded")
    hop = monitor.evaluate_event(_event("t1", "a", "OUTBOUND_CONNECT", 2))
    assert hop["decision"] == "BLOCK" and hop["reason"].startswith("correlated_actions")


def test_idle_entries_are_evicted_after_the_window(tmp_path):
    clock = _Clock()
    monitor = _bounded_monitor(tmp_path, clock)
    for i in range(40):
        monitor.evaluate_event(_event(f"junk{i}", f"j{i}", "NOOP", 1))
    clock.now = 301
    fresh = monitor.evaluate_event(_event("new", "n", "NOOP", 400))
    assert fresh["decision"] == "ALLOW"
    assert len(monitor._history) <= 20 and len(monitor._agents) <= 20


def test_rerooting_keeps_lineage_history(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 1, parent_agent_id="b"))
    assert monitor.evaluate_event(_event("t2", "b", "SENSITIVE_READ", 2, parent_agent_id="c"))["decision"] == "ALLOW"
    hop = monitor.evaluate_event(_event("t3", "a", "OUTBOUND_CONNECT", 3))
    assert hop["decision"] == "BLOCK" and hop["lineage_root"] == "c"


def test_backdated_timestamp_on_new_trace_is_still_correlated(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 100))
    hop = monitor.evaluate_event(_event("t2", "a", "OUTBOUND_CONNECT", 50))
    assert hop["decision"] == "BLOCK"


def test_far_future_timestamp_cannot_age_out_history(tmp_path):
    monitor = _bounded_monitor(tmp_path, _Clock())
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 100))
    monitor.evaluate_event(_event("t2", "a", "NOOP", 10**9))
    hop = monitor.evaluate_event(_event("t3", "a", "OUTBOUND_CONNECT", 101))
    assert hop["decision"] == "BLOCK"


def test_rejected_cycle_does_not_poison_lineage(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "a", "OUTBOUND_CONNECT", 1, parent_agent_id="b"))
    monitor.evaluate_event(_event("t2", "b", "OUTBOUND_CONNECT", 2, parent_agent_id="a"))
    again = monitor.evaluate_event(_event("t1", "a", "OUTBOUND_CONNECT", 3))
    assert again["decision"] == "ALLOW" and again["lineage_root"] == "b"


def test_active_lineage_root_is_not_evicted(tmp_path):
    clock = _Clock()
    monitor = _bounded_monitor(tmp_path, clock, max_tracked_keys=4)
    monitor.evaluate_event(_event("t0", "root", "NOOP", 0, parent_agent_id="top"))
    for step in range(1, 8):
        clock.now = step * 100
        monitor.evaluate_event(_event("t1", "child", "NOOP", step * 100, parent_agent_id="root"))
    for index in range(3):
        monitor.evaluate_event(_event("t1", f"x{index}", "NOOP", 700, parent_agent_id="root"))
    assert "root" in monitor._agents
    moved = monitor.evaluate_event(_event("t1", "root", "NOOP", 701, parent_agent_id="other"))
    assert moved["reason"] == "lineage_conflict"


def test_rerooting_at_capacity_keeps_history(tmp_path):
    monitor = _bounded_monitor(tmp_path, _Clock(), max_tracked_keys=4)
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 1, parent_agent_id="b"))
    monitor.evaluate_event(_event("t2", "z", "NOOP", 1))
    assert len(monitor._history) == 4
    assert monitor.evaluate_event(_event("t1", "b", "NOOP", 2, parent_agent_id="c"))["decision"] == "ALLOW"
    assert ("lineage", "b") not in monitor._history and len(monitor._history) == 4
    hop = monitor.evaluate_event(_event("t2", "a", "OUTBOUND_CONNECT", 3))
    assert hop["reason"].endswith(":lineage") and hop["lineage_root"] == "c"


def test_policy_denied_events_do_not_change_tracking_state(tmp_path):
    monitor = _bounded_monitor(tmp_path, _Clock())
    denied = dict(_event("t1", "victim", "NOOP", 1, parent_agent_id="evil"), resource="/not-allowed")
    for index in range(50):
        monitor.evaluate_event(dict(denied, trace_id=f"junk{index}", agent_id=f"j{index}"))
    assert monitor.evaluate_event(denied)["reason"] == "not_in_allowlist"
    assert not monitor._agents and not monitor._history
    claimed = monitor.evaluate_event(_event("t2", "victim", "NOOP", 2, parent_agent_id="real"))
    assert claimed["decision"] == "ALLOW" and claimed["lineage_root"] == "real"


def test_capacity_block_leaves_no_partial_state(tmp_path):
    monitor = _bounded_monitor(tmp_path, _Clock(), max_tracked_keys=3)
    monitor.evaluate_event(_event("t1", "a", "NOOP", 1))
    blocked = monitor.evaluate_event(_event("t2", "b", "NOOP", 1))
    assert blocked["reason"] == "history_capacity_exceeded"
    assert "b" not in monitor._agents and ("trace", "t2") not in monitor._history


def test_parent_reported_later_is_adopted_and_keeps_history(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "worker", "SENSITIVE_READ", 1))
    monitor.evaluate_event(_event("t2", "sibling", "SENSITIVE_READ", 1, parent_agent_id="boss"))
    late = monitor.evaluate_event(_event("t1", "worker", "SENSITIVE_READ", 2, parent_agent_id="boss"))
    assert late["decision"] == "ALLOW" and late["lineage_root"] == "boss"
    assert ("lineage", "worker") not in monitor._history
    hop = monitor.evaluate_event(_event("t3", "sibling", "OUTBOUND_CONNECT", 3))
    assert hop["reason"].endswith(":lineage")
    omitted = monitor.evaluate_event(_event("t4", "worker", "SENSITIVE_READ", 4))
    assert omitted["decision"] == "ALLOW" and omitted["lineage_root"] == "boss"
    changed = monitor.evaluate_event(_event("t4", "worker", "SENSITIVE_READ", 5, parent_agent_id="other"))
    assert changed["reason"] == "lineage_conflict"


def test_late_parent_cannot_shed_history_or_form_a_cycle(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 1))
    monitor.evaluate_event(_event("t2", "a", "SENSITIVE_READ", 2, parent_agent_id="fresh"))
    hop = monitor.evaluate_event(_event("t3", "a", "OUTBOUND_CONNECT", 3))
    assert hop["decision"] == "BLOCK" and hop["lineage_root"] == "fresh"
    monitor.evaluate_event(_event("t4", "child", "OUTBOUND_CONNECT", 4, parent_agent_id="parent"))
    loop = monitor.evaluate_event(_event("t5", "parent", "SENSITIVE_READ", 5, parent_agent_id="child"))
    assert loop["reason"] == "lineage_cycle"
    assert monitor._agents.get("parent", {}).get("parent") is None


def test_late_parent_on_a_capacity_blocked_event_is_rolled_back(tmp_path):
    clock = _Clock()
    monitor = _bounded_monitor(tmp_path, clock, max_tracked_keys=4)
    monitor.evaluate_event(_event("t1", "kid", "NOOP", 1, parent_agent_id="mid"))
    monitor.evaluate_event(_event("t1", "mid", "SENSITIVE_READ", 2))
    monitor.evaluate_event(_event("t2", "z", "NOOP", 2))
    blocked = monitor.evaluate_event(_event("t-new", "mid", "NOOP", 3, parent_agent_id="top"))
    assert blocked["reason"] == "history_capacity_exceeded"
    assert monitor._agents["mid"]["parent"] is None and ("lineage", "mid") in monitor._history
    hop = monitor.evaluate_event(_event("t2", "kid", "OUTBOUND_CONNECT", 4))
    assert hop["reason"].endswith(":lineage") and hop["lineage_root"] == "mid"


def test_rejected_lineage_claims_do_not_refresh_or_record(tmp_path):
    clock = _Clock()
    monitor = _bounded_monitor(tmp_path, clock)
    monitor.evaluate_event(_event("t1", "a", "NOOP", 0, parent_agent_id="b"))
    monitor.evaluate_event(_event("t1", "b", "NOOP", 0, parent_agent_id="c"))
    before = {agent: dict(entry) for agent, entry in monitor._agents.items()}
    clock.now = 50
    assert monitor.evaluate_event(_event("t2", "c", "NOOP", 1, parent_agent_id="a"))["reason"] == "lineage_cycle"
    assert monitor.evaluate_event(_event("t2", "a", "NOOP", 1, parent_agent_id="x"))["reason"] == "lineage_conflict"
    assert {agent: dict(entry) for agent, entry in monitor._agents.items()} == before
    assert ("trace", "t2") not in monitor._history


@pytest.mark.parametrize("override", [
    {"timestamp": float("nan")}, {"timestamp": float("inf")}, {"timestamp": object()}, {"timestamp": True},
    {"timestamp": "5"}, {"trace_id": b"raw"}, {"parent_agent_id": {1}}, {"resource": ["svc"]},
])
def test_malformed_events_are_blocked_and_audited_without_raising(tmp_path, override):
    monitor = _sequence_monitor(tmp_path)
    result = monitor.evaluate_event(dict(_event("t1", "a", "OUTBOUND_CONNECT", 1), **override))
    assert result["decision"] == "BLOCK"
    assert monitor.verify_audit_log()["records"] == 1
    assert not monitor._agents and not monitor._history


def test_timestamp_skew_limit_blocks_forward_dated_events(tmp_path):
    monitor = RuntimeEffectMonitor(
        {"SENSITIVE_READ": {"/data"}, "OUTBOUND_CONNECT": {"svc"}},
        audit_log_path=tmp_path / "effects.jsonl",
        blocked_action_sequences=[("SENSITIVE_READ", "OUTBOUND_CONNECT")],
        max_timestamp_skew_seconds=30,
    )
    now = time.time()
    assert monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", now))["decision"] == "ALLOW"
    jumped = monitor.evaluate_event(_event("t2", "a", "OUTBOUND_CONNECT", now + 301))
    assert jumped["reason"] == "timestamp_out_of_range"
    assert ("trace", "t2") not in monitor._history
    assert monitor.evaluate_event(_event("t2", "a", "OUTBOUND_CONNECT", now + 1))["reason"].endswith(":lineage")


@pytest.mark.parametrize("sequences", [["AB"], [("A",)], [("A", "")], [("A", 5)], [5]])
def test_blocked_sequences_must_be_string_pairs(tmp_path, sequences):
    with pytest.raises(ValueError, match="string pairs"):
        RuntimeEffectMonitor({}, audit_log_path=tmp_path / "effects.jsonl", blocked_action_sequences=sequences)


def _tracking_snapshot(monitor):
    return ({agent: entry["parent"] for agent, entry in monitor._agents.items()},
            {key: (dict(bucket["last"]), bucket["max_ts"]) for key, bucket in monitor._history.items()})


@pytest.mark.parametrize("rejected, reason", [
    (dict(trace="t1", agent="new", action="OUTBOUND_CONNECT", timestamp=101, parent_agent_id="a"),
     "correlated_actions:SENSITIVE_READ->OUTBOUND_CONNECT:trace"),
    (dict(trace="t-new", agent="new", action="OUTBOUND_CONNECT", timestamp=101, parent_agent_id="a"),
     "correlated_actions:SENSITIVE_READ->OUTBOUND_CONNECT:lineage"),
    (dict(trace="t1", agent="new", action="SENSITIVE_READ", timestamp=50, parent_agent_id="other"),
     "non_monotonic_trace_timestamp"),
    (dict(trace="t-new", agent="new", action="SENSITIVE_READ", timestamp=-900, parent_agent_id="a"),
     "stale_lineage_timestamp"),
    (dict(trace="t-new", agent="b", action="OUTBOUND_CONNECT", timestamp=101, parent_agent_id="a"),
     "correlated_actions:SENSITIVE_READ->OUTBOUND_CONNECT:lineage"),
])
def test_rejected_events_leave_tracking_state_unchanged(tmp_path, rejected, reason):
    monitor = _bounded_monitor(tmp_path, _Clock())
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 100))
    monitor.evaluate_event(_event("t-b", "b", "NOOP", 100))          # b: parent not known yet
    before = _tracking_snapshot(monitor)
    result = monitor.evaluate_event(_event(**rejected))
    assert (result["decision"], result["reason"]) == ("BLOCK", reason)
    assert _tracking_snapshot(monitor) == before
    # The rejected parent claim reserved nothing: the same agent can still report its real parent.
    later = monitor.evaluate_event(_event("t-later", rejected["agent"], "NOOP", 102, parent_agent_id="real"))
    assert later["decision"] == "ALLOW" and later["lineage_root"] == "real"


def test_history_held_before_a_late_parent_still_counts(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    monitor.evaluate_event(_event("t1", "a", "SENSITIVE_READ", 1))
    hop = monitor.evaluate_event(_event("t2", "a", "OUTBOUND_CONNECT", 2, parent_agent_id="fresh"))
    assert hop["reason"].endswith(":lineage")
    assert monitor._agents["a"]["parent"] is None and ("lineage", "a") in monitor._history


def test_oversized_timestamp_is_blocked_and_audited(tmp_path):
    monitor = _sequence_monitor(tmp_path)
    result = monitor.evaluate_event(_event("t1", "a", "OUTBOUND_CONNECT", 10 ** 400))
    assert (result["decision"], result["reason"]) == ("BLOCK", "invalid_timestamp")
    assert not monitor._agents and not monitor._history
    assert monitor.verify_audit_log()["records"] == 1


# Audit log: keys and anchors

def _log_with_records(path, count, key=None):
    log = AuditLog(path, key=key)
    for index in range(count):
        log.append("event", {"index": index})
    return log


def test_anchor_detects_truncation(tmp_path):
    path = tmp_path / "audit.jsonl"
    anchor = _log_with_records(path, 3).anchor()
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        AuditLog(path).verify(expected_anchor=anchor)


def test_anchor_detects_full_rewrite(tmp_path):
    path = tmp_path / "audit.jsonl"
    anchor = _log_with_records(path, 3).anchor()
    path.unlink()
    _log_with_records(path, 3)
    with pytest.raises(ValueError, match="rewritten"):
        AuditLog(path).verify(expected_anchor=anchor)


def test_anchor_survives_legitimate_appends(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = _log_with_records(path, 2)
    anchor = log.anchor()
    log.append("later", {"ok": True})
    assert log.verify(expected_anchor=anchor)["records"] == 3


def test_keyed_log_rejects_unkeyed_forgery(tmp_path):
    path = tmp_path / "audit.jsonl"
    _log_with_records(path, 2, key=KEY)
    assert AuditLog(path, key=KEY).verify()["keyed"] is True
    forged = tmp_path / "forged.jsonl"
    _log_with_records(forged, 2)
    with pytest.raises(ValueError, match="hash mismatch"):
        AuditLog(forged, key=KEY)


def test_verify_keeps_only_the_anchored_hash(tmp_path):
    log = _log_with_records(tmp_path / "audit.jsonl", 5)
    third = json.loads(log.path.read_text(encoding="utf-8").splitlines()[2])["chain_hash"]
    assert log._read_and_validate(3)[2] == third
    assert log._read_and_validate()[2] is None
    assert log.verify(expected_anchor={"sequence": 0, "head_hash": "0" * 64})["records"] == 5
    with pytest.raises(ValueError, match="Anchor must contain"):
        log.verify(expected_anchor={"sequence": "3", "head_hash": "x"})


def test_short_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="16 bytes"):
        AuditLog(tmp_path / "audit.jsonl", key=b"short")


# Detector: baseline profile

def test_expected_profile_equals_baseline(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    engine.calibrate_baseline_from_traces(_traces(50))
    np.testing.assert_allclose(engine.T_shared @ engine.Sing_X, engine.Sing_X)


def test_threshold_is_tight_and_clean_state_passes_with_margin(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    engine.calibrate_baseline_from_traces(_traces(50))
    assert engine.epsilon_threshold < 0.05
    clean = np.eye(16) + np.random.default_rng(999).normal(0, 0.001, (16, 16))
    assert engine.verify_state_invariance(clean) == (True, "NOMINAL_OR_BENIGN_ADAPTATION")
    drifted = np.eye(16) * 1.5
    assert engine.verify_state_invariance(drifted)[0] is False


# Detector: baseline integrity

def test_baseline_arrays_cannot_be_made_writable(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    engine.calibrate_baseline_from_traces(_traces(50))
    with pytest.raises(ValueError):
        engine.Sing_X.setflags(write=True)
    with pytest.raises(AttributeError):
        engine.Sing_X = np.zeros(16)


def test_threshold_tampering_is_detected(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    engine.calibrate_baseline_from_traces(_traces(50))
    engine.empirical_std_drift *= 1000
    with pytest.raises(RuntimeError, match="integrity"):
        engine.verify_state_invariance(np.eye(16))


def test_external_supervisor_hash_check(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    engine.calibrate_baseline_from_traces(_traces(50))
    supervisor_copy = engine.baseline_manifest["baseline_hash"]
    assert engine.verify_baseline(supervisor_copy)
    other_state = np.eye(16) * 2
    engine._sing_x_bytes = np.linalg.svd(other_state, compute_uv=False).tobytes()
    engine._baseline_hash = engine.compute_baseline_fingerprint()
    assert engine.verify_baseline(supervisor_copy) is False
    with pytest.raises(RuntimeError, match="supplied hash"):
        engine.verify_state_invariance(other_state, expected_baseline_hash=supervisor_copy)


def test_external_hash_can_be_required_on_every_evaluation(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl",
                                     require_external_baseline_hash=True)
    engine.calibrate_baseline_from_traces(_traces(50))
    state = np.eye(16) + np.random.default_rng(999).normal(0, 0.001, (16, 16))
    with pytest.raises(RuntimeError, match="required"):
        engine.verify_state_invariance(state)
    supervisor_copy = engine.baseline_manifest["baseline_hash"]
    assert engine.verify_state_invariance(state, expected_baseline_hash=supervisor_copy)[0] is True
    assert engine.get_calibration_report()["require_external_baseline_hash"] is True


# Detector: calibration limits

def test_minimum_trace_floor_restored(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    assert engine.min_calibration_traces == 50
    with pytest.raises(ValueError, match="Insufficient calibration data"):
        engine.calibrate_baseline_from_traces(_traces(1))
    with pytest.raises(ValueError, match="Insufficient calibration data"):
        engine.calibrate_baseline_from_traces(_traces(49))
    assert not engine.calibrated


def test_floor_cannot_be_set_below_two(tmp_path):
    with pytest.raises(ValueError, match="min_calibration_traces"):
        CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl", min_calibration_traces=1)


def test_optional_epsilon_limit_rejects_contaminated_calibration(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl", max_epsilon=0.05)
    noisy = _traces(25) + _traces(25, scale=0.5)
    with pytest.raises(ValueError, match="max_epsilon"):
        engine.calibrate_baseline_from_traces(noisy)
    assert not engine.calibrated


def test_calibration_rejects_identical_traces_and_bad_limits(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "d.jsonl")
    with pytest.raises(ValueError, match="no variation"):
        engine.calibrate_baseline_from_traces(np.eye(16) for _ in range(50))
    assert not engine.calibrated
    for bad in ("x", True, float("nan"), 0):
        with pytest.raises(ValueError, match="max_epsilon"):
            CognitiveImmunityEngine(embedding_dim=16, audit_log_path=tmp_path / "e.jsonl", max_epsilon=bad)


# Environment validator

def test_package_fingerprint_sees_submodule_changes(tmp_path):
    package = tmp_path / "fakepkg"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (package / "sub" / "core.py").write_text("y = 2\n", encoding="utf-8")
    module = types.SimpleNamespace(__name__="fakepkg", __path__=[str(package)], __file__=str(package / "__init__.py"))
    _, before, count = _fingerprint_module(module)
    assert count == 2
    (package / "sub" / "core.py").write_text("y = 3\n", encoding="utf-8")
    _, after, _ = _fingerprint_module(module)
    assert before != after


def test_bytecode_cache_does_not_change_fingerprint(tmp_path):
    package = tmp_path / "fakepkg"
    package.mkdir()
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    module = types.SimpleNamespace(__name__="fakepkg", __path__=[str(package)], __file__=str(package / "__init__.py"))
    _, before, _ = _fingerprint_module(module)
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "__init__.cpython-311.pyc").write_bytes(b"cache")
    assert _fingerprint_module(module)[1] == before


def test_environment_validator_round_trip(tmp_path):
    validator = EnvironmentIntegrityValidator(audit_log_path=tmp_path / "env.jsonl", critical_modules=("numpy",))
    manifest = validator.snapshot_environment()
    assert manifest["module_fingerprints"]["numpy"]["files"] > 1
    assert validator.verify_environment() is True
