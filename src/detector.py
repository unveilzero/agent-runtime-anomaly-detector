"""CognitiveImmunityEngine: SVD-driven structural invariance checker."""

import hashlib
import hmac
import json
import logging
from typing import List, Optional, Tuple

import numpy as np

from src.audit import AuditLog

logger = logging.getLogger("ARAD_Detector")


class CognitiveImmunityEngine:
    """Evaluate caller-supplied state matrices against a calibrated baseline.

    The internal integrity check compares against a hash stored on this same
    object, so code running in this process can defeat it. Hold
    ``baseline_manifest["baseline_hash"]`` with an external supervisor and
    pass it to every ``verify_state_invariance()`` call; set
    ``require_external_baseline_hash=True`` to make that mandatory.
    """

    MIN_CALIBRATION_TRACES = 50

    def __init__(self, embedding_dim: int = 512, audit_log_path: str = None,
                 min_calibration_traces: Optional[int] = None,
                 max_epsilon: Optional[float] = None, audit_key=None,
                 require_external_baseline_hash: bool = False):
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 2:
            raise ValueError("embedding_dim must be an integer >= 2")
        if min_calibration_traces is None:
            min_calibration_traces = self.MIN_CALIBRATION_TRACES
        if (not isinstance(min_calibration_traces, int) or isinstance(min_calibration_traces, bool)
                or min_calibration_traces < 2):
            raise ValueError("min_calibration_traces must be an integer >= 2")
        if max_epsilon is not None and (
                isinstance(max_epsilon, bool) or not isinstance(max_epsilon, (int, float))
                or not np.isfinite(max_epsilon) or max_epsilon <= 0):
            raise ValueError("max_epsilon must be finite and positive")
        self.embedding_dim = embedding_dim
        self.require_external_baseline_hash = bool(require_external_baseline_hash)
        self.min_calibration_traces = min_calibration_traces
        self.max_epsilon = None if max_epsilon is None else float(max_epsilon)
        self.audit_log = AuditLog(audit_log_path or ".arad_audit.jsonl", key=audit_key)
        self.calibrated = False
        self.baseline_locked = False
        self._sing_x_bytes = self._t_shared_bytes = None
        self.empirical_mean_drift = self.empirical_std_drift = 0.0
        self.trailing_var_mean = self.trailing_var_std = 0.0
        self.variance_threshold_lower = 0.0
        self.energy_ratio_floor = self.energy_ratio_mean = self.energy_ratio_std = 0.0
        self.split_idx = int(self.embedding_dim * 0.7)
        self.baseline_manifest = {}
        self._baseline_hash = None
        self._record_event("engine_initialized", {
            "embedding_dim": embedding_dim,
            "min_calibration_traces": min_calibration_traces,
            "max_epsilon": self.max_epsilon,
            "require_external_baseline_hash": self.require_external_baseline_hash,
        })

    # Read-only views over immutable bytes; setflags(write=True) raises.
    @property
    def Sing_X(self):
        if self._sing_x_bytes is None:
            return None
        return np.frombuffer(self._sing_x_bytes, dtype=np.float64)

    @property
    def T_shared(self):
        if self._t_shared_bytes is None:
            return None
        n = self.embedding_dim
        return np.frombuffer(self._t_shared_bytes, dtype=np.float64).reshape(n, n)

    @property
    def epsilon_threshold(self) -> float:
        if not self.calibrated:
            raise RuntimeError("Engine not yet calibrated. Call calibrate_baseline_from_traces() first.")
        return self.empirical_mean_drift + 3.0 * self.empirical_std_drift

    def _record_event(self, event_type: str, details: dict):
        self.audit_log.append(event_type, details)

    def _threshold_values(self, trace_count):
        return {
            "embedding_dim": self.embedding_dim, "trace_count": trace_count,
            "empirical_mean_drift": self.empirical_mean_drift, "empirical_std_drift": self.empirical_std_drift,
            "epsilon_threshold": self.empirical_mean_drift + 3.0 * self.empirical_std_drift,
            "trailing_var_mean": self.trailing_var_mean, "trailing_var_std": self.trailing_var_std,
            "variance_threshold_lower": self.variance_threshold_lower,
            "energy_ratio_mean": self.energy_ratio_mean, "energy_ratio_std": self.energy_ratio_std,
            "energy_ratio_floor": self.energy_ratio_floor,
        }

    @staticmethod
    def _fingerprint(thresholds, sing_x_bytes, t_shared_bytes):
        return hashlib.sha256(
            json.dumps(thresholds, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + sing_x_bytes + t_shared_bytes
        ).hexdigest()

    def calibrate_baseline_from_traces(self, benign_traces: List[np.ndarray]) -> None:
        if self.baseline_locked:
            raise RuntimeError("Baseline already locked; create a new engine for reviewed recalibration.")
        benign_traces = [] if benign_traces is None else list(benign_traces)
        if len(benign_traces) < self.min_calibration_traces:
            self._record_event("calibration_rejected", {
                "reason": "insufficient_traces", "received": len(benign_traces),
                "required": self.min_calibration_traces,
            })
            raise ValueError(
                f"Insufficient calibration data: {len(benign_traces)} traces received, "
                f"{self.min_calibration_traces} required. Too few traces shrink the threshold spread "
                "and can indicate calibration poisoning."
            )
        expected_shape = (self.embedding_dim, self.embedding_dim)
        traces = []
        for index, trace in enumerate(benign_traces):
            try:
                array = np.asarray(trace, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Trace {index} is not a numeric matrix") from exc
            if array.shape != expected_shape:
                raise ValueError(f"Trace {index} has shape {array.shape}, expected {expected_shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Trace {index} contains non-finite values (NaN or Inf)")
            traces.append(array.copy())

        profiles = [np.linalg.svd(trace, compute_uv=False) for trace in traces]
        sing_x = np.mean(profiles, axis=0)
        # Singular values are basis-independent, so the expected profile is
        # the calibrated mean and T_shared stays the identity.
        transform = np.eye(self.embedding_dim)
        expected = transform @ sing_x
        drift_norms, trailing_variances, energy_ratios = [], [], []
        for profile in profiles:
            delta = profile - expected
            drift_norms.append(float(np.linalg.norm(delta)))
            trailing_variances.append(float(np.var(delta[self.split_idx:])))
            leading = float(np.sum(profile[:self.split_idx]))
            trailing = float(np.sum(profile[self.split_idx:]))
            energy_ratios.append(trailing / max(leading, np.finfo(float).eps))

        mean_drift, std_drift = float(np.mean(drift_norms)), float(np.std(drift_norms))
        epsilon = mean_drift + 3.0 * std_drift
        if epsilon <= 0.0:
            self._record_event("calibration_rejected", {"reason": "no_variation"})
            raise ValueError(
                "Calibration traces show no variation, so no drift threshold can be derived; "
                "identical traces can indicate replayed or synthetic calibration data."
            )
        if self.max_epsilon is not None and epsilon > self.max_epsilon:
            self._record_event("calibration_rejected", {
                "reason": "epsilon_above_limit", "epsilon_threshold": epsilon, "max_epsilon": self.max_epsilon,
            })
            raise ValueError(
                f"Calibrated epsilon {epsilon:.6g} exceeds max_epsilon {self.max_epsilon:.6g}; "
                "calibration traces may be contaminated."
            )

        self.empirical_mean_drift, self.empirical_std_drift = mean_drift, std_drift
        self.trailing_var_mean = float(np.mean(trailing_variances))
        self.trailing_var_std = float(np.std(trailing_variances))
        self.variance_threshold_lower = self.trailing_var_mean - 3.0 * self.trailing_var_std
        self.energy_ratio_mean = float(np.mean(energy_ratios))
        self.energy_ratio_std = float(np.std(energy_ratios))
        self.energy_ratio_floor = self.energy_ratio_mean - 3.0 * self.energy_ratio_std
        self._sing_x_bytes = sing_x.astype(np.float64).tobytes()
        self._t_shared_bytes = transform.astype(np.float64).tobytes()

        thresholds = self._threshold_values(len(traces))
        baseline_hash = self._fingerprint(thresholds, self._sing_x_bytes, self._t_shared_bytes)
        self.baseline_manifest = dict(thresholds, baseline_hash=baseline_hash)
        self._baseline_hash = baseline_hash
        self.calibrated = self.baseline_locked = True
        self._record_event("baseline_calibrated", dict(self.baseline_manifest))

    def compute_baseline_fingerprint(self) -> str:
        """Recompute the baseline hash from the live values (not from the stored copy)."""
        if not self.calibrated or self._sing_x_bytes is None or self._t_shared_bytes is None:
            raise RuntimeError("Engine not yet calibrated. Call calibrate_baseline_from_traces() first.")
        thresholds = self._threshold_values(self.baseline_manifest.get("trace_count"))
        return self._fingerprint(thresholds, self._sing_x_bytes, self._t_shared_bytes)

    def verify_baseline(self, expected_hash: str) -> bool:
        """Compare the live baseline against a hash held by an external supervisor."""
        if not isinstance(expected_hash, str):
            return False
        try:
            actual = self.compute_baseline_fingerprint()
        except RuntimeError:
            return False
        intact = hmac.compare_digest(actual, expected_hash)
        if not intact:
            self._record_event("baseline_integrity_failure", {"check": "external"})
        return intact

    def _baseline_is_intact(self):
        if not self.calibrated or self._baseline_hash is None:
            return False
        try:
            return hmac.compare_digest(self.compute_baseline_fingerprint(), self._baseline_hash)
        except RuntimeError:
            return False

    def verify_state_invariance(self, target_state_matrix: np.ndarray,
                                expected_baseline_hash: Optional[str] = None) -> Tuple[bool, str]:
        if not self.calibrated:
            raise RuntimeError("Engine must be calibrated with benign traces before use.")
        if expected_baseline_hash is None:
            if self.require_external_baseline_hash:
                raise RuntimeError("expected_baseline_hash is required for every state evaluation.")
        elif not self.verify_baseline(expected_baseline_hash):
            raise RuntimeError("Calibrated baseline integrity check failed against the supplied hash.")
        if not self._baseline_is_intact():
            self._record_event("baseline_integrity_failure", {"check": "internal"})
            raise RuntimeError("Calibrated baseline integrity check failed.")
        try:
            state = np.asarray(target_state_matrix, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Input must be a numeric matrix") from exc
        expected_shape = (self.embedding_dim, self.embedding_dim)
        if state.shape != expected_shape:
            raise ValueError(f"Input has shape {state.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(state)):
            self._record_event("state_invalid", {"reason": "non_finite_input", "shape": list(state.shape)})
            return False, "INVALID_INPUT"

        profile = np.linalg.svd(state, compute_uv=False)
        delta = profile - self.T_shared @ self.Sing_X
        drift_norm = float(np.linalg.norm(delta))
        trailing_variance = float(np.var(delta[self.split_idx:]))
        leading = float(np.sum(profile[:self.split_idx]))
        trailing = float(np.sum(profile[self.split_idx:]))
        energy_ratio = trailing / max(leading, np.finfo(float).eps)
        details = {
            "drift_norm": drift_norm, "epsilon_threshold": self.epsilon_threshold,
            "trailing_variance": trailing_variance, "variance_threshold_lower": self.variance_threshold_lower,
            "energy_ratio": energy_ratio, "energy_ratio_floor": self.energy_ratio_floor,
            "baseline_hash": self._baseline_hash,
        }
        if energy_ratio < self.energy_ratio_floor:
            self._record_event("state_block", dict(details, anomaly_class="ADVERSARIAL_CONTRACTION_EVASION"))
            return False, "ADVERSARIAL_CONTRACTION_EVASION"
        if drift_norm <= self.epsilon_threshold:
            self._record_event("state_pass", details)
            return True, "NOMINAL_OR_BENIGN_ADAPTATION"
        if trailing_variance < self.variance_threshold_lower:
            self._record_event("state_block", dict(details, anomaly_class="ADVERSARIAL_SDVC_MUTATION"))
            return False, "ADVERSARIAL_SDVC_MUTATION"
        self._record_event("state_review", dict(details, anomaly_class="EXTREME_BENIGN_DRIFT"))
        return False, "EXTREME_BENIGN_DRIFT"

    def get_calibration_report(self) -> dict:
        if not self.calibrated:
            raise RuntimeError("Engine not yet calibrated. Call calibrate_baseline_from_traces() first.")
        return {
            "embedding_dim": self.embedding_dim, "calibrated": self.calibrated,
            "empirical_mean_drift": self.empirical_mean_drift, "empirical_std_drift": self.empirical_std_drift,
            "epsilon_threshold": self.epsilon_threshold, "trailing_var_mean": self.trailing_var_mean,
            "trailing_var_std": self.trailing_var_std, "variance_threshold_lower": self.variance_threshold_lower,
            "energy_ratio_mean": self.energy_ratio_mean, "energy_ratio_std": self.energy_ratio_std,
            "energy_ratio_floor": self.energy_ratio_floor, "baseline_manifest": dict(self.baseline_manifest),
            "baseline_locked": self.baseline_locked,
            "min_calibration_traces": self.min_calibration_traces, "max_epsilon": self.max_epsilon,
            "require_external_baseline_hash": self.require_external_baseline_hash,
        }
