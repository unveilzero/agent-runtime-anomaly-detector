"""CognitiveImmunityEngine: SVD-driven structural invariance checker.

The engine evaluates caller-supplied matrices for drift. It does not observe
external OS-side effects on its own; pair it with a trusted runtime monitor for
process, file, and network containment decisions.
"""

import hashlib
import json
import logging
from typing import List, Tuple

import numpy as np

from src.audit import AuditLog

logger = logging.getLogger("ARAD_Detector")


class CognitiveImmunityEngine:
    """Runtime anomaly detector for autonomous agent state monitoring."""

    def __init__(self, embedding_dim: int = 512, audit_log_path: str = None):
        if not isinstance(embedding_dim, int) or embedding_dim < 2:
            raise ValueError("embedding_dim must be an integer >= 2")

        self.embedding_dim = embedding_dim
        self.audit_log_path = audit_log_path or ".arad_audit.jsonl"
        self.audit_log = AuditLog(self.audit_log_path)

        self.calibrated = False
        self.baseline_locked = False
        self.T_shared = None
        self.Sing_X = None
        self.empirical_mean_drift = 0.0
        self.empirical_std_drift = 0.0
        self.trailing_var_mean = 0.0
        self.trailing_var_std = 0.0
        self.variance_threshold_lower = 0.0
        self.split_idx = int(self.embedding_dim * 0.7)
        self.baseline_manifest = {}
        self._baseline_hash = None
        self._record_event("engine_initialized", {"embedding_dim": embedding_dim})

    @property
    def epsilon_threshold(self) -> float:
        if not self.calibrated:
            raise RuntimeError("Engine not yet calibrated. Call calibrate_baseline_from_traces() first.")
        return self.empirical_mean_drift + (3.0 * self.empirical_std_drift)

    def _record_event(self, event_type: str, details: dict):
        self.audit_log.append(event_type, details)

    def calibrate_baseline_from_traces(self, benign_traces: List[np.ndarray]) -> None:
        """Calibrate detection thresholds from benign traces."""
        if self.baseline_locked:
            raise RuntimeError("Baseline already locked; create a new engine for reviewed recalibration.")
        if not benign_traces:
            raise ValueError("At least 1 benign trace required for calibration")

        traces = []
        expected_shape = (self.embedding_dim, self.embedding_dim)
        for index, trace in enumerate(benign_traces):
            array = np.asarray(trace, dtype=float)
            if array.shape != expected_shape:
                raise ValueError(f"Trace {index} has shape {array.shape}, expected {expected_shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Trace {index} contains non-finite values (NaN or Inf)")
            traces.append(array.copy())

        if len(traces) < 10:
            logger.warning("Calibrating on only %d traces; 50+ is recommended.", len(traces))

        profiles = [np.linalg.svd(trace, compute_uv=False) for trace in traces]
        sing_x = np.mean(profiles, axis=0)
        covariance = np.zeros(expected_shape)
        for trace in traces:
            covariance += trace @ trace.T
        transform, _, _ = np.linalg.svd(covariance / len(traces))

        drift_norms = []
        trailing_variances = []
        for profile in profiles:
            delta = profile - (transform @ sing_x)
            drift_norms.append(float(np.linalg.norm(delta)))
            trailing_variances.append(float(np.var(delta[self.split_idx:])))

        mean_drift = float(np.mean(drift_norms))
        std_drift = float(np.std(drift_norms))
        trailing_mean = float(np.mean(trailing_variances))
        trailing_std = float(np.std(trailing_variances))
        variance_lower = trailing_mean - 3.0 * trailing_std

        thresholds = {
            "embedding_dim": self.embedding_dim,
            "trace_count": len(traces),
            "empirical_mean_drift": mean_drift,
            "empirical_std_drift": std_drift,
            "epsilon_threshold": mean_drift + 3.0 * std_drift,
            "trailing_var_mean": trailing_mean,
            "trailing_var_std": trailing_std,
            "variance_threshold_lower": variance_lower,
        }
        baseline_hash = hashlib.sha256(
            json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + sing_x.tobytes() + transform.tobytes()
        ).hexdigest()

        self.Sing_X = sing_x.copy()
        self.T_shared = transform.copy()
        self.Sing_X.setflags(write=False)
        self.T_shared.setflags(write=False)
        self.empirical_mean_drift = mean_drift
        self.empirical_std_drift = std_drift
        self.trailing_var_mean = trailing_mean
        self.trailing_var_std = trailing_std
        self.variance_threshold_lower = variance_lower
        self.baseline_manifest = dict(thresholds, baseline_hash=baseline_hash)
        self._baseline_hash = baseline_hash
        self.calibrated = True
        self.baseline_locked = True
        self._record_event("baseline_calibrated", self.baseline_manifest)

    def _baseline_is_intact(self):
        if not self.calibrated or self.Sing_X is None or self.T_shared is None:
            return False
        thresholds = {
            "embedding_dim": self.embedding_dim,
            "trace_count": self.baseline_manifest.get("trace_count"),
            "empirical_mean_drift": self.empirical_mean_drift,
            "empirical_std_drift": self.empirical_std_drift,
            "epsilon_threshold": self.epsilon_threshold,
            "trailing_var_mean": self.trailing_var_mean,
            "trailing_var_std": self.trailing_var_std,
            "variance_threshold_lower": self.variance_threshold_lower,
        }
        digest = hashlib.sha256(
            json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + self.Sing_X.tobytes() + self.T_shared.tobytes()
        ).hexdigest()
        return digest == self._baseline_hash

    def verify_state_invariance(self, target_state_matrix: np.ndarray) -> Tuple[bool, str]:
        if not self.calibrated:
            raise RuntimeError("Engine must be calibrated with benign traces before use. Call calibrate_baseline_from_traces() first.")
        if not self._baseline_is_intact():
            self._record_event("baseline_integrity_failure", {})
            raise RuntimeError("Calibrated baseline integrity check failed.")

        state = np.asarray(target_state_matrix, dtype=float)
        expected_shape = (self.embedding_dim, self.embedding_dim)
        if state.shape != expected_shape:
            raise ValueError(f"Input has shape {state.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(state)):
            self._record_event("state_invalid", {"reason": "non_finite_input", "shape": list(state.shape)})
            return False, "INVALID_INPUT"

        profile = np.linalg.svd(state, compute_uv=False)
        delta = profile - (self.T_shared @ self.Sing_X)
        drift_norm = float(np.linalg.norm(delta))
        trailing_variance = float(np.var(delta[self.split_idx:]))
        details = {
            "drift_norm": drift_norm,
            "epsilon_threshold": self.epsilon_threshold,
            "trailing_variance": trailing_variance,
            "baseline_hash": self._baseline_hash,
        }

        if drift_norm <= self.epsilon_threshold:
            self._record_event("state_pass", details)
            return True, "NOMINAL_OR_BENIGN_ADAPTATION"

        if trailing_variance < self.variance_threshold_lower:
            details["anomaly_class"] = "ADVERSARIAL_SDVC_MUTATION"
            self._record_event("state_block", details)
            return False, "ADVERSARIAL_SDVC_MUTATION"

        details["anomaly_class"] = "EXTREME_BENIGN_DRIFT"
        self._record_event("state_review", details)
        return False, "EXTREME_BENIGN_DRIFT"

    def get_calibration_report(self) -> dict:
        if not self.calibrated:
            raise RuntimeError("Engine not yet calibrated. Call calibrate_baseline_from_traces() first.")
        return {
            "embedding_dim": self.embedding_dim,
            "calibrated": self.calibrated,
            "empirical_mean_drift": self.empirical_mean_drift,
            "empirical_std_drift": self.empirical_std_drift,
            "epsilon_threshold": self.epsilon_threshold,
            "trailing_var_mean": self.trailing_var_mean,
            "trailing_var_std": self.trailing_var_std,
            "sdvc_variance_ceiling": self.variance_threshold_lower,
            "baseline_manifest": dict(self.baseline_manifest),
            "baseline_locked": self.baseline_locked,
        }
