"""Deterministic tests for ARAD state evaluation."""

import numpy as np
import pytest

from src import CognitiveImmunityEngine


@pytest.fixture
def calibrated_engine(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=8, audit_log_path=tmp_path / "detector.jsonl")
    traces = [np.eye(8) + np.random.default_rng(seed).normal(0, 0.001, (8, 8)) for seed in range(10)]
    engine.calibrate_baseline_from_traces(traces)
    return engine, traces


def test_calibration_locks_and_reports_thresholds(calibrated_engine):
    engine, _ = calibrated_engine
    report = engine.get_calibration_report()
    assert engine.calibrated and engine.baseline_locked
    assert report["baseline_manifest"]["baseline_hash"]
    assert "energy_ratio_floor" in report


def test_recalibration_requires_new_engine(calibrated_engine):
    engine, traces = calibrated_engine
    with pytest.raises(RuntimeError, match="Baseline already locked"):
        engine.calibrate_baseline_from_traces(traces)


def test_nominal_state_passes(calibrated_engine):
    safe, profile = calibrated_engine[0].verify_state_invariance(np.eye(8))
    assert safe and profile == "NOMINAL_OR_BENIGN_ADAPTATION"


def test_contracted_state_is_flagged(calibrated_engine):
    safe, profile = calibrated_engine[0].verify_state_invariance(np.diag([1.0] * 5 + [0.001] * 3))
    assert not safe and profile == "ADVERSARIAL_CONTRACTION_EVASION"


def test_input_shape_and_values_are_validated(calibrated_engine):
    engine, _ = calibrated_engine
    with pytest.raises(ValueError, match="shape"):
        engine.verify_state_invariance(np.eye(4))
    invalid = np.eye(8)
    invalid[0, 0] = np.nan
    assert engine.verify_state_invariance(invalid) == (False, "INVALID_INPUT")


def test_calibration_rejects_bad_traces(tmp_path):
    engine = CognitiveImmunityEngine(embedding_dim=8, audit_log_path=tmp_path / "detector.jsonl")
    with pytest.raises(ValueError, match="shape"):
        engine.calibrate_baseline_from_traces([np.eye(4)])
    invalid = np.eye(8)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        engine.calibrate_baseline_from_traces([invalid])
