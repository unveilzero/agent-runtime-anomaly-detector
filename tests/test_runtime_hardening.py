"""Tests for hash-chain auditing and runtime effect policy evaluation."""

import json

import pytest

from src.audit import AuditLog
from src.detector import CognitiveImmunityEngine
from src.runtime_monitor import RuntimeEffectMonitor


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("start", {"ok": True})
    assert log.verify()["records"] == 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["details"]["ok"] = False
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        AuditLog(path)


def test_runtime_effect_allowlist_and_block(tmp_path):
    monitor = RuntimeEffectMonitor(
        {"FILE_READ": {"/docs"}, "FILE_WRITE": set()},
        audit_log_path=tmp_path / "effects.jsonl",
    )
    allowed = monitor.evaluate_event({
        "trace_id": "t1", "agent_id": "a1", "action": "FILE_READ", "resource": "/docs", "timestamp": 1,
    })
    denied = monitor.evaluate_event({
        "trace_id": "t1", "agent_id": "a1", "action": "FILE_WRITE", "resource": "/docs", "timestamp": 2,
    })
    assert allowed["decision"] == "ALLOW"
    assert denied["decision"] == "BLOCK"


def test_runtime_effect_sequence_correlation(tmp_path):
    monitor = RuntimeEffectMonitor(
        {"SENSITIVE_FILE_READ": {"/secrets"}, "OUTBOUND_CONNECT": {"example.org"}},
        audit_log_path=tmp_path / "effects.jsonl",
        blocked_action_sequences=[("SENSITIVE_FILE_READ", "OUTBOUND_CONNECT")],
    )
    base = {"trace_id": "trace", "agent_id": "worker"}
    first = monitor.evaluate_event(dict(base, action="SENSITIVE_FILE_READ", resource="/secrets", timestamp=1))
    second = monitor.evaluate_event(dict(base, action="OUTBOUND_CONNECT", resource="example.org", timestamp=2))
    assert first["decision"] == "ALLOW"
    assert second["decision"] == "BLOCK"


def test_baseline_lock_prevents_recalibration(tmp_path):
    import numpy as np

    engine = CognitiveImmunityEngine(embedding_dim=8, audit_log_path=tmp_path / "detector.jsonl")
    traces = [np.eye(8) + np.random.default_rng(i).normal(0, 0.001, size=(8, 8)) for i in range(10)]
    engine.calibrate_baseline_from_traces(traces)
    with pytest.raises(RuntimeError, match="Baseline already locked"):
        engine.calibrate_baseline_from_traces(traces)
    assert engine.verify_state_invariance(traces[0])[0]


def test_audit_log_can_resume_existing_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    first.append("first", {"n": 1})
    second = AuditLog(path)
    second.append("second", {"n": 2})
    assert second.verify()["records"] == 2


def test_detector_rejects_non_integer_embedding_dimension(tmp_path):
    with pytest.raises(ValueError, match="embedding_dim"):
        CognitiveImmunityEngine(embedding_dim="8", audit_log_path=tmp_path / "bad.jsonl")
