"""
Synthetic validation test suite for ARAD detector.
Use this to validate the detector on your local system.
Run: pytest tests/test_synthetic_validation.py -v
"""

import numpy as np
import pytest
from src.detector import CognitiveImmunityEngine


class TestCalibration:
    """Test calibration pipeline"""
    
    def test_calibration_basic(self):
        """Test that calibration completes without error"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        
        # Generate benign traces
        benign_traces = [
            np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512))
            for _ in range(50)
        ]
        
        # Should complete without error
        engine.calibrate_baseline_from_traces(benign_traces)
        assert engine.calibrated
        assert engine.epsilon_threshold > 0
        
    def test_calibration_requires_traces(self):
        """Test that calibration rejects empty trace list"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        
        with pytest.raises(ValueError):
            engine.calibrate_baseline_from_traces([])
    
    def test_calibration_validates_shape(self):
        """Test that calibration rejects wrong-shaped traces"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        
        # Wrong shape
        bad_traces = [np.random.normal(size=(256, 256)) for _ in range(10)]
        
        with pytest.raises(ValueError):
            engine.calibrate_baseline_from_traces(bad_traces)
    
    def test_calibration_rejects_nans(self):
        """Test that calibration rejects non-finite values"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        
        benign_traces = [
            np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512))
            for _ in range(10)
        ]
        
        # Inject NaN
        benign_traces[0][0, 0] = np.nan
        
        with pytest.raises(ValueError):
            engine.calibrate_baseline_from_traces(benign_traces)


class TestDetection:
    """Test anomaly detection on calibrated engine"""
    
    @pytest.fixture
    def calibrated_engine(self):
        """Provide a calibrated engine for testing"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        
        benign_traces = [
            np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512))
            for _ in range(100)
        ]
        
        engine.calibrate_baseline_from_traces(benign_traces)
        return engine
    
    def test_detection_requires_calibration(self):
        """Test that detection rejects uncalibrated engine"""
        engine = CognitiveImmunityEngine(embedding_dim=512)
        state = np.eye(512)
        
        with pytest.raises(RuntimeError):
            engine.verify_state_invariance(state)
    
    def test_benign_ood_passes(self, calibrated_engine):
        """Test that benign out-of-distribution data passes detection"""
        # Different noise distribution (Laplace instead of Gaussian)
        ood_state = np.eye(512) + np.random.laplace(loc=0, scale=0.0015, size=(512, 512))
        
        is_safe, profile = calibrated_engine.verify_state_invariance(ood_state)
        assert is_safe, f"Benign OOD should pass detection, got profile: {profile}"
        assert profile == "NOMINAL_OR_BENIGN_ADAPTATION"
    
    def test_structural_collapse_detected(self, calibrated_engine):
        """Test that structural truncation is detected as SDVC"""
        # Simulate structural collapse: zero out trailing dimensions
        collapsed_state = np.eye(512)
        collapsed_state[350:, 350:] *= 0.001  # Truncate
        
        is_safe, profile = calibrated_engine.verify_state_invariance(collapsed_state)
        assert not is_safe, "Structural collapse should be detected"
        assert profile == "ADVERSARIAL_SDVC_MUTATION"
    
    def test_input_validation(self, calibrated_engine):
        """Test that detection validates input shape and content"""
        # Wrong shape
        with pytest.raises(ValueError):
            calibrated_engine.verify_state_invariance(np.random.normal(size=(256, 256)))
        
        # NaN values
        bad_state = np.eye(512)
        bad_state[0, 0] = np.nan
        is_safe, _ = calibrated_engine.verify_state_invariance(bad_state)
        assert not is_safe


class TestFalsePositiveRate:
    """Measure false positive rate on benign domain adaptation"""
    
    def test_false_positive_rate_on_ood(self):
        """Measure FPR when domain shifts"""
        # Calibrate on Gaussian noise
        engine = CognitiveImmunityEngine(embedding_dim=512)
        np.random.seed(42)  # Reproducible
        
        benign_traces = [
            np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512))
            for _ in range(100)
        ]
        engine.calibrate_baseline_from_traces(benign_traces)
        
        # Test on different distribution
        false_positives = 0
        test_samples = 100
        
        for _ in range(test_samples):
            ood_state = np.eye(512) + np.random.laplace(loc=0, scale=0.0015, size=(512, 512))
            is_safe, profile = engine.verify_state_invariance(ood_state)
            
            # Count as FP only if misclassified as malicious SDVC
            if not is_safe and profile == "ADVERSARIAL_SDVC_MUTATION":
                false_positives += 1
        
        fpr = (false_positives / test_samples) * 100
        print(f"\n>>> False Positive Rate (OOD domain shift): {fpr:.2f}%")
        
        # Expect <5% FPR
        assert fpr < 5.0, f"FPR too high: {fpr}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
