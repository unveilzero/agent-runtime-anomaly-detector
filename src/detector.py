"""
OASF-BEA Master Architecture (v1.5.2) - Core Matrix Validation Engine
Component: CognitiveImmunityEngine (SVD & Calibrated Energy Proportion Validation)
License: Open Framework Public Specification (BEA-CAP-1.0)
"""

import hashlib
import json
import logging
import time

import numpy as np

try:
    from src.environment_validator import EnvironmentIntegrityValidator
except Exception:  # pragma: no cover - defensive import fallback
    EnvironmentIntegrityValidator = None

# Configure telemetry logger
logging.basicConfig(level=logging.INFO, format='[OASF-TELEMETRY] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OASF_Matrix_Core")


class CognitiveImmunityEngine:
    def __init__(self, embedding_dim: int = 512, audit_log_path: str = None):
        """
        Initializes the cognitive immunity core using geometric invariance checks
        to intercept Self-Directed Value Compromise (SDVC).
        """
        self.embedding_dim = embedding_dim
        self.audit_log_path = audit_log_path or ".arad_audit.log"
        self.audit_chain = ""
        self.baseline_manifest = {}
        self.baseline_locked = False

        # Core operational anchors (derived via historical trace calibration)
        self.calibrated = False
        self.T_shared = None
        self.Sing_X = None

        # Empirical thresholds established during system calibration
        self.epsilon_threshold = 0.0
        self.variance_threshold_lower = 0.0
        self.energy_ratio_floor = 0.0
        self.split_idx = int(self.embedding_dim * 0.7)  # 70/30 structural partition boundary

        self.environment_validator = None
        if EnvironmentIntegrityValidator is not None:
            self.environment_validator = EnvironmentIntegrityValidator(audit_log_path=self.audit_log_path)
            try:
                self.environment_validator.snapshot_environment()
                if not self.environment_validator.verify_environment():
                    raise RuntimeError("Environment integrity verification failed during bootstrap.")
            except Exception as exc:  # pragma: no cover - defensive bootstrap guard
                logger.critical(f"🚨 RUNTIME BOOTSTRAP GUARD: environment integrity failed: {exc}")
                raise RuntimeError("Execution environment is not trusted. Detection startup aborted.") from exc

        self._record_event("engine_initialized", {
            "embedding_dim": self.embedding_dim,
            "audit_log_path": self.audit_log_path,
        })

    def _record_event(self, event_type: str, details: dict):
        """Append a chain-hashed audit record for the detector lifecycle."""
        record = {
            "ts": int(time.time()),
            "event": event_type,
            "details": details,
            "previous_hash": self.audit_chain,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        chain_hash = hashlib.sha256((self.audit_chain + payload).encode("utf-8")).hexdigest()
        record["chain_hash"] = chain_hash
        self.audit_chain = chain_hash

        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
        except OSError as exc:
            logger.warning(f"Unable to write matrix audit log: {exc}")

    def ensure_environment_integrity(self):
        """
        Runtime bootstrap guard. Refuses execution if the environment has changed
        from the trusted baseline captured during engine initialization.
        """
        if self.environment_validator is None:
            logger.warning("Environment validator unavailable; running without pre-flight integrity guard.")
            return

        try:
            if not self.environment_validator.verify_environment():
                raise RuntimeError("Environment integrity verification failed.")
        except Exception as exc:
            logger.critical(f"🚨 RUNTIME BOOTSTRAP GUARD: environment drift detected: {exc}")
            raise RuntimeError("Execution environment drift detected. Aborting validation cycle.") from exc

    def calibrate_baseline_from_traces(self, benign_traces: list):
        """
        Calibrates trusted boundaries using historical trace logs from agents
        operating nominal parameters over extended observation windows.
        
        CALIBRATION ATTACK DEFENSES:
        - MIN_CALIBRATION_TRACES: Prevents history_window shrinking attacks
        - Epsilon sanity check: Catches artificially inflated thresholds
        """
        if self.calibrated and self.baseline_locked:
            raise RuntimeError(
                "Baseline already locked. Recalibration would silently modify the trusted baseline. "
                "Create a new engine instance to begin a fresh calibration window."
            )

        self.ensure_environment_integrity()

        if not benign_traces:
            raise ValueError("Calibration requires a non-empty list of historical agent traces.")

        MIN_CALIBRATION_TRACES = 50
        if len(benign_traces) < MIN_CALIBRATION_TRACES:
            raise RuntimeError(
                f"Insufficient calibration data. Minimum {MIN_CALIBRATION_TRACES} traces required. "
                f"Received {len(benign_traces)}. Possible calibration poisoning attempt (history_window shrink attack)."
            )

        logger.info(f"Calibrating system telemetry baseline using {len(benign_traces)} historical traces...")

        svd_profiles = [np.linalg.svd(trace, compute_uv=False) for trace in benign_traces]
        self.Sing_X = np.mean(svd_profiles, axis=0)

        C = np.zeros((self.embedding_dim, self.embedding_dim))
        for trace in benign_traces:
            C += trace @ trace.T
        U_c, _, _ = np.linalg.svd(C / len(benign_traces))
        self.T_shared = U_c

        drift_norms = []
        trailing_variances = []
        energy_ratios = []

        for Sing_Y in svd_profiles:
            expected_alignment = self.T_shared @ self.Sing_X
            drift_delta = Sing_Y - expected_alignment

            norm_val = np.linalg.norm(drift_delta)
            drift_norms.append(norm_val)

            t_var = np.var(drift_delta[self.split_idx:])
            trailing_variances.append(t_var)

            leading_energy = np.sum(Sing_Y[:self.split_idx])
            trailing_energy = np.sum(Sing_Y[self.split_idx:])
            energy_ratio = trailing_energy / (leading_energy + 1e-9)
            energy_ratios.append(energy_ratio)

        self.epsilon_threshold = float(np.mean(drift_norms) + (3 * np.std(drift_norms)))

        mean_t_var = np.mean(trailing_variances)
        std_t_var = np.std(trailing_variances)
        self.variance_threshold_lower = float(mean_t_var - (3 * std_t_var))

        mean_energy_ratio = np.mean(energy_ratios)
        std_energy_ratio = np.std(energy_ratios)
        self.energy_ratio_floor = float(mean_energy_ratio - (3 * std_energy_ratio))

        EPSILON_SANITY_LIMIT = 0.15
        if self.epsilon_threshold > EPSILON_SANITY_LIMIT:
            logger.critical(
                f"🚨 CALIBRATION ATTACK DETECTED: epsilon_threshold={self.epsilon_threshold:.6f} exceeds safety limit {EPSILON_SANITY_LIMIT}. "
                f"Possible benign trace contamination or history_window manipulation attack."
            )
            self._record_event("baseline_rejected", {
                "reason": "epsilon_sanity_limit_exceeded",
                "epsilon_threshold": float(self.epsilon_threshold),
                "limit": EPSILON_SANITY_LIMIT,
            })
            raise RuntimeError(
                "Calibration baseline validation failed. Suspected history window manipulation or poisoned benign traces."
            )

        self.calibrated = True
        self.baseline_locked = True
        self.baseline_manifest = {
            "embedding_dim": self.embedding_dim,
            "trace_count": len(benign_traces),
            "epsilon_threshold": float(self.epsilon_threshold),
            "variance_threshold_lower": float(self.variance_threshold_lower),
            "energy_ratio_floor": float(self.energy_ratio_floor),
            "baseline_hash": hashlib.sha256(
                json.dumps({
                    "embedding_dim": self.embedding_dim,
                    "trace_count": len(benign_traces),
                    "epsilon_threshold": float(self.epsilon_threshold),
                    "variance_threshold_lower": float(self.variance_threshold_lower),
                    "energy_ratio_floor": float(self.energy_ratio_floor),
                }, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        self._record_event("baseline_calibrated", self.baseline_manifest)

        logger.info("✅ Baseline calibration complete. System safety bounds locked.")
        logger.info(f"-> Epsilon Limit: {self.epsilon_threshold:.6f}")
        logger.info(f"-> Lower Trailing Variance Floor: {self.variance_threshold_lower:.6f}")
        logger.info(f"-> Energy Ratio Floor: {self.energy_ratio_floor:.6f}")
        logger.info(f"-> Baseline Hash: {self.baseline_manifest['baseline_hash']}")

    def verify_state_invariance(self, target_state_matrix: np.ndarray) -> tuple:
        """
        Executes inline mathematical validation loop to evaluate active states.
        Protects against standard drift, structural collapse, and contraction evasion.

        Uses dual-gate verification:
        - Gate A: Energy Proportion Ratio (catches uniform contraction attempts)
        - Gate B: Trailing Variance Profile (catches structural collapse)

        :return: (bool (is_safe), str (profile_signature))
        """
        self.ensure_environment_integrity()

        if not self.calibrated:
            raise RuntimeError("Engine must be calibrated with benign traces before use.")

        if target_state_matrix.shape != (self.embedding_dim, self.embedding_dim):
            raise ValueError(
                f"Input shape {target_state_matrix.shape} is invalid for embedding_dim={self.embedding_dim}."
            )

        Sing_Y = np.linalg.svd(target_state_matrix, compute_uv=False)

        expected_alignment = self.T_shared @ self.Sing_X
        drift_delta = Sing_Y - expected_alignment
        drift_norm = np.linalg.norm(drift_delta)

        leading_energy = np.sum(Sing_Y[:self.split_idx])
        trailing_energy = np.sum(Sing_Y[self.split_idx:])
        energy_ratio = trailing_energy / (leading_energy + 1e-9)

        outcome = {
            "drift_norm": float(drift_norm),
            "trailing_variance": float(np.var(drift_delta[self.split_idx:])),
            "energy_ratio": float(energy_ratio),
            "epsilon_threshold": float(self.epsilon_threshold),
            "variance_threshold_lower": float(self.variance_threshold_lower),
            "energy_ratio_floor": float(self.energy_ratio_floor),
            "baseline_hash": self.baseline_manifest.get("baseline_hash"),
        }

        if energy_ratio < self.energy_ratio_floor:
            logger.critical("⚠️ ADVERSARIAL RATIO COLLAPSE: Uniform contraction attempt caught via calibrated Energy Proportion Gate!")
            outcome["profile_signature"] = "ADVERSARIAL_CONTRACTION_EVASION"
            self._record_event("state_block", outcome)
            return False, "ADVERSARIAL_CONTRACTION_EVASION"

        if drift_norm <= self.epsilon_threshold:
            self._record_event("state_pass", outcome)
            return True, "NOMINAL_OR_BENIGN_ADAPTATION"

        trailing_variance = outcome["trailing_variance"]

        if trailing_variance < self.variance_threshold_lower:
            logger.critical("⚠️ ADVERSARIAL SDVC SIGNATURE DETECTED: Sudden structural matrix collapse!")
            outcome["profile_signature"] = "ADVERSARIAL_SDVC_MUTATION"
            self._record_event("state_block", outcome)
            return False, "ADVERSARIAL_SDVC_MUTATION"
        else:
            logger.warning("🚨 HIGH VARIANCE VARIATION: Continuous learning domain adaptation profile detected.")
            outcome["profile_signature"] = "EXTREME_BENIGN_DRIFT"
            self._record_event("state_review", outcome)
            return False, "EXTREME_BENIGN_DRIFT"

    def get_calibration_report(self):
        """Return calibration metadata for transparency and reproducibility."""
        if not self.calibrated:
            raise RuntimeError("Engine must be calibrated with benign traces before use.")

        return {
            "embedding_dim": self.embedding_dim,
            "calibrated": self.calibrated,
            "epsilon_threshold": float(self.epsilon_threshold),
            "variance_threshold_lower": float(self.variance_threshold_lower),
            "energy_ratio_floor": float(self.energy_ratio_floor),
            "baseline_manifest": self.baseline_manifest,
            "baseline_locked": self.baseline_locked,
        }


if __name__ == "__main__":
    logger.info("Running engine infrastructure verification loop...")
    engine = CognitiveImmunityEngine(embedding_dim=512)

    nominal_historical_traces = [np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512)) for _ in range(100)]
    engine.calibrate_baseline_from_traces(nominal_historical_traces)

    test_trace = np.eye(512)
    safe, profile = engine.verify_state_invariance(test_trace)
    logger.info(f"Result: safe={safe}, profile={profile}")


"""}],