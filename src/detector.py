"""
CognitiveImmunityEngine: Core anomaly detection module
Based on SVD-driven structural invariance checking
"""

import numpy as np
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='[ARAD] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ARAD_Detector")


class CognitiveImmunityEngine:
    """
    Runtime anomaly detector for autonomous agent state monitoring.
    Detects structural deviations that indicate compromise or self-modification.
    """

    def __init__(self, embedding_dim: int = 512):
        """
        Initialize the detector.
        
        Args:
            embedding_dim: Dimensionality of agent state matrices (default 512)
        """
        self.embedding_dim = embedding_dim
        
        # Calibration state
        self.calibrated = False
        self.T_shared = None
        self.Sing_X = None
        
        # Statistical thresholds (derived during calibration)
        self.epsilon_threshold = 0.0
        self.variance_threshold_lower = 0.0
        self.split_idx = int(self.embedding_dim * 0.7)  # 70/30 split for variance analysis
        
        # Calibration history for transparency
        self.calibration_metadata = {
            "num_traces": 0,
            "mean_drift": None,
            "std_drift": None,
            "mean_trailing_variance": None,
            "std_trailing_variance": None
        }

    def calibrate_baseline_from_traces(self, benign_traces: List[np.ndarray]) -> None:
        """
        Calibrate detection thresholds from real benign agent traces.
        This MUST be called before verify_state_invariance().
        
        Args:
            benign_traces: List of (embedding_dim, embedding_dim) state matrices from clean operation
            
        Raises:
            ValueError: If traces have invalid shape or insufficient data
        """
        if not benign_traces:
            raise ValueError("At least 1 benign trace required for calibration")
        
        if len(benign_traces) < 10:
            logger.warning(f"Calibrating on only {len(benign_traces)} traces. " 
                          f"Recommend 50+ for robust thresholds.")
        
        # Validate all traces
        for i, trace in enumerate(benign_traces):
            if trace.shape != (self.embedding_dim, self.embedding_dim):
                raise ValueError(
                    f"Trace {i} has shape {trace.shape}, expected "
                    f"({self.embedding_dim}, {self.embedding_dim})"
                )
            if not np.all(np.isfinite(trace)):
                raise ValueError(f"Trace {i} contains non-finite values (NaN or Inf)")
        
        logger.info(f"Calibrating on {len(benign_traces)} benign traces...")
        
        # 1. Derive Sing_X: average singular value profile from benign operation
        svd_profiles = [np.linalg.svd(trace, compute_uv=False) for trace in benign_traces]
        self.Sing_X = np.mean(svd_profiles, axis=0)
        
        # 2. Compute T_shared: transformation matrix from covariance structure
        C = np.zeros((self.embedding_dim, self.embedding_dim))
        for trace in benign_traces:
            C += trace @ trace.T
        C /= len(benign_traces)
        
        U_c, _, _ = np.linalg.svd(C)
        self.T_shared = U_c  # Use left singular vectors as transformation
        
        # 3. Compute drift statistics for threshold calibration
        drift_norms = []
        trailing_variances = []
        
        for Sing_Y in svd_profiles:
            expected_alignment = self.T_shared @ self.Sing_X
            drift_delta = Sing_Y - expected_alignment
            
            # Euclidean norm of drift vector
            norm_val = np.linalg.norm(drift_delta)
            drift_norms.append(norm_val)
            
            # Variance in trailing dimensions (used to detect structural collapse)
            t_var = np.var(drift_delta[self.split_idx:])
            trailing_variances.append(t_var)
        
        # 4. Apply 3-sigma rule: mean ± 3*std captures 99.73% of benign variation
        mean_drift = np.mean(drift_norms)
        std_drift = np.std(drift_norms)
        self.epsilon_threshold = float(mean_drift + (3 * std_drift))
        
        # Lower bound for SDVC structural collapse detection
        mean_t_var = np.mean(trailing_variances)
        std_t_var = np.std(trailing_variances)
        self.variance_threshold_lower = float(mean_t_var - (3 * std_t_var))
        
        # Store metadata for transparency
        self.calibration_metadata = {
            "num_traces": len(benign_traces),
            "mean_drift": float(mean_drift),
            "std_drift": float(std_drift),
            "mean_trailing_variance": float(mean_t_var),
            "std_trailing_variance": float(std_t_var)
        }
        
        self.calibrated = True
        logger.info(f"✅ Calibration complete")
        logger.info(f"   Epsilon threshold: {self.epsilon_threshold:.6f}")
        logger.info(f"   Variance lower bound: {self.variance_threshold_lower:.6f}")

    def verify_state_invariance(self, target_state_matrix: np.ndarray) -> Tuple[bool, str]:
        """
        Evaluate agent runtime state for anomalies.
        
        Args:
            target_state_matrix: Current (embedding_dim, embedding_dim) agent state matrix
            
        Returns:
            (is_safe, anomaly_class) where:
            - is_safe: bool, True if state passes all checks
            - anomaly_class: str, classification of deviation type
                * "NOMINAL_OR_BENIGN_ADAPTATION" - within expected variance
                * "EXTREME_BENIGN_DRIFT" - high variance but not malicious
                * "ADVERSARIAL_SDVC_MUTATION" - structural collapse detected
                
        Raises:
            RuntimeError: If called before calibration
            ValueError: If input has invalid shape or non-finite values
        """
        if not self.calibrated:
            raise RuntimeError(
                "Engine must be calibrated with benign traces before use. "
                "Call calibrate_baseline_from_traces() first."
            )
        
        # Validate input
        if target_state_matrix.shape != (self.embedding_dim, self.embedding_dim):
            raise ValueError(
                f"Input has shape {target_state_matrix.shape}, "
                f"expected ({self.embedding_dim}, {self.embedding_dim})"
            )
        
        if not np.all(np.isfinite(target_state_matrix)):
            logger.error("Input contains non-finite values (NaN or Inf)")
            return False, "INVALID_INPUT"
        
        # Compute singular values
        Sing_Y = np.linalg.svd(target_state_matrix, compute_uv=False)
        
        # Calculate structural drift
        expected_alignment = self.T_shared @ self.Sing_X
        drift_delta = Sing_Y - expected_alignment
        
        # Frobenius norm (Euclidean distance)
        drift_norm = np.linalg.norm(drift_delta)
        
        # Primary check: overall drift magnitude
        if drift_norm <= self.epsilon_threshold:
            return True, "NOMINAL_OR_BENIGN_ADAPTATION"
        
        # Secondary check: trailing variance (indicates structural collapse)
        trailing_variance = np.var(drift_delta[self.split_idx:])
        
        if trailing_variance < self.variance_threshold_lower:
            logger.critical(
                f"⚠️  ANOMALY DETECTED: Structural collapse signature\n"
                f"   Drift norm: {drift_norm:.6f} (threshold: {self.epsilon_threshold:.6f})\n"
                f"   Trailing variance: {trailing_variance:.6f} "
                f"(lower bound: {self.variance_threshold_lower:.6f})"
            )
            return False, "ADVERSARIAL_SDVC_MUTATION"
        else:
            logger.warning(
                f"⚠️  HIGH DRIFT detected but not classified as malicious\n"
                f"   Drift norm: {drift_norm:.6f} (threshold: {self.epsilon_threshold:.6f})\n"
                f"   Trailing variance: {trailing_variance:.6f} (normal)"
            )
            return False, "EXTREME_BENIGN_DRIFT"

    def get_calibration_report(self) -> dict:
        """
        Return calibration metadata for transparency and reproducibility.
        """
        return self.calibration_metadata.copy()
