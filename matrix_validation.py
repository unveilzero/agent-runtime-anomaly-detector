"""
OASF-BEA Master Architecture (v1.5.1) - Core Matrix Validation Engine
Component: CognitiveImmunityEngine (Empirical Calibrator)
License: Open Framework Public Specification (BEA-CAP-1.0)
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[OASF-TELEMETRY] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OASF_Empirical_Core")

class CognitiveImmunityEngine:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.T_shared = None
        self.Sing_X = None
        
        # Statistical thresholds derived via 3-sigma rule during calibration
        self.epsilon_threshold = 0.0
        self.variance_threshold_lower = 0.0
        self.split_idx = int(self.embedding_dim * 0.7) # 70/30 split boundary
        
    def calibrate_baseline_from_traces(self, benign_traces: list):
        """
        Derives T_shared and the trailing component variance 
        thresholds dynamically using the statistical profile of real benign traces.
        """
        logger.info(f"Calibrating system telemetry baseline using {len(benign_traces)} historical traces...")
        
        # 1. Derive Sing_X as the average empirical singular baseline
        svd_profiles = [np.linalg.svd(trace, compute_uv=False) for trace in benign_traces]
        self.Sing_X = np.mean(svd_profiles, axis=0)
        
        # 2. Compute T_shared using a deterministic cross-covariance mapping
        C = np.zeros((self.embedding_dim, self.embedding_dim))
        for trace in benign_traces:
            C += trace @ trace.T
        U_c, _, _ = np.linalg.svd(C / len(benign_traces))
        self.T_shared = U_c  # Securely anchored transformation matrix
        
        # 3. Track distribution metrics to establish thresholds
        drift_norms = []
        trailing_variances = []
        
        for Sing_Y in svd_profiles:
            expected_alignment = self.T_shared @ self.Sing_X
            drift_delta = Sing_Y - expected_alignment
            
            # Compute vector 2-norm for 1D arrays
            norm_val = np.linalg.norm(drift_delta)
            drift_norms.append(norm_val)
            
            # Profile trailing variance
            t_var = np.var(drift_delta[self.split_idx:])
            trailing_variances.append(t_var)
            
        # 4. Apply 3-Sigma Rule to derive thresholds empirically
        self.epsilon_threshold = float(np.mean(drift_norms) + (3 * np.std(drift_norms)))
        
        # Lower bound for SDVC structural collapse detection
        mean_t_var = np.mean(trailing_variances)
        std_t_var = np.std(trailing_variances)
        self.variance_threshold_lower = float(mean_t_var - (3 * std_t_var))
        
        self.calibrated = True
        logger.info(f"Calibration metrics established.")

    def verify_state_invariance(self, target_state_matrix: np.ndarray) -> tuple:
        """
        Evaluates active states using verified SVD parsing and empirical thresholds.
        PATCH v1.5.1: Added Dual-Gate Energy Proportion tracking to close evasion gaps.
        """
        if not self.calibrated:
            raise RuntimeError("Engine must be calibrated with benign traces before use.")
            
        # Unpack SVD output array safely
        Sing_Y = np.linalg.svd(target_state_matrix, compute_uv=False)
        
        expected_alignment = self.T_shared @ self.Sing_X
        drift_delta = Sing_Y - expected_alignment
        drift_norm = np.linalg.norm(drift_delta)
        
        # --- NEW ENERGY PROPORTION GATE TRACKING ---
        leading_energy = np.sum(Sing_Y[:self.split_idx])
        trailing_energy = np.sum(Sing_Y[self.split_idx:])
        energy_ratio = trailing_energy / (leading_energy + 1e-9)
        
        # Lower bound for ratio checks to catch uniform matrix contractions
        if hasattr(self, 'variance_threshold_lower') and energy_ratio < (self.variance_threshold_lower * 0.1):
            logger.critical("⚠️ ADVERSARIAL RATIO COLLAPSE: Uniform contraction attempt caught via Energy Proportion Gate!")
            return False, "ADVERSARIAL_CONTRACTION_EVASION"
        # ------------------------------------------

        if drift_norm <= self.epsilon_threshold:
            return True, "NOMINAL_OR_BENIGN_ADAPTATION"
            
        # Evaluate anomaly signature against calibrated threshold variance
        trailing_variance = np.var(drift_delta[self.split_idx:])
        
        if trailing_variance < self.variance_threshold_lower: 
            logger.critical("⚠️ ADVERSARIAL SDVC SIGNATURE DETECTED: Sudden structural matrix collapse!")
            return False, "ADVERSARIAL_SDVC_MUTATION"
        else:
            logger.warning("🚨 HIGH VARIANCE VARIATION: Continuous learning domain adaptation profile.")
            return False, "EXTREME_BENIGN_DRIFT"

if __name__ == "__main__":
    engine = CognitiveImmunityEngine(embedding_dim=512)
    base_traces = [np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512,512)) for _ in range(100)]
    engine.calibrate_baseline_from_traces(base_traces)
    
    logger.info("\nEvaluating against out-of-distribution (OOD) continuous learning traces...")
    false_positives = 0
    for _ in range(100):
        ood_adaptation_state = np.eye(512) + np.random.laplace(loc=0, scale=0.0015, size=(512,512))
        is_safe, profile = engine.verify_state_invariance(ood_adaptation_state)
        if not is_safe and profile == "ADVERSARIAL_SDVC_MUTATION": 
            false_positives += 1
            
    print(f">> Measured False Positive Rate (FPR): {(false_positives / 100) * 100:.2f}%")
