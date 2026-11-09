"""Tests for audit integrity and runtime event policy evaluation."""

import json

import pytest

from src import CognitiveImmunityEngine, RuntimeEffectMonitor
from src.audit import AuditLog


def test_audit_chain_detects_tampering_and_resumes(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append("start", {"ok": True})
    second = AuditLog(path)
    second.append("next", {"ok": True})
    assert second.verify()["records"] == 2
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["details"]["ok"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        AuditLog(path)


def test_audit_rejects_non_json_values(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "audit.jsonl").append("bad", {"not_finite": float("nan")})


def test_runtime_effect_allowlist_and_deny(tmp_path):
    monitor = RuntimeEffectMonitor({"FILE_READ": {"/docs"}}, audit_log_path=tmp_path / "effects.jsonl")
    base = {"trace_id": "t1", "agent_id": "a1"}
    allowed = monitor.evaluate_event(dict(base, action="FILE_READ", resource="/docs", timestamp=1))
    denied = monitor.evaluate_event(dict(base, action="FILE_WRITE", resource="/docs", timestamp=2))
    assert allowed["decision"] == "ALLOW"
    assert denied["decision"] == "BLOCK"


def test_runtime_effect_correlation_is_trace_scoped(tmp_path):
    monitor = RuntimeEffectMonitor(
        {"SENSITIVE_READ": {"/data"}, "OUTBOUND_CONNECT": {"service"}},
        audit_log_path=tmp_path / "effects.jsonl",
        blocked_action_sequences=[("SENSITIVE_READ", "OUTBOUND_CONNECT")],
    )
    first = monitor.evaluate_event({"trace_id": "trace-a", "agent_id": "a", "action": "SENSITIVE_READ", "resource": "/data", "timestamp": 1})
    other = monitor.evaluate_event({"trace_id": "trace-b", "agent_id": "b", "action": "OUTBOUND_CONNECT", "resource": "service", "timestamp": 2})
    correlated = monitor.evaluate_event({"trace_id": "trace-a", "agent_id": "c", "action": "OUTBOUND_CONNECT", "resource": "service", "timestamp": 3})
    assert first["decision"] == other["decision"] == "ALLOW"
    assert correlated["decision"] == "BLOCK"


def test_runtime_monitor_rejects_missing_fields_and_bad_time(tmp_path):
    monitor = RuntimeEffectMonitor({}, audit_log_path=tmp_path / "effects.jsonl")
    assert monitor.evaluate_event({"trace_id": "x"})["decision"] == "BLOCK"
    invalid = monitor.evaluate_event({"trace_id": "x", "agent_id": "a", "action": "READ", "resource": "r", "timestamp": "nan"})
    assert invalid["decision"] == "BLOCK"


def test_detector_rejects_invalid_dimension(tmp_path):
    with pytest.raises(ValueError, match="embedding_dim"):
        CognitiveImmunityEngine(embedding_dim=True, audit_log_path=tmp_path / "bad.jsonl")
