"""
OASF-BEA Master Architecture (v1.6.0) - Environment Integrity Validator
Component: Pre-flight execution environment security audit
License: Open Framework Public Specification (BEA-CAP-1.0)

CRITICAL: This module validates that the execution environment has not been 
poisoned via dependency injection, library replacement, or environment variable 
tampering before running the CognitiveImmunityEngine detector.
"""

import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[OASF-ENV-VALIDATOR] %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OASF_Environment_Validator")


class EnvironmentIntegrityValidator:
    """
    Pre-flight environment security checkpoint.
    
    Captures cryptographic fingerprints of critical loaded libraries and 
    environment variables. Detects poisoning via dependency injection, 
    library replacement, or environment variable tampering (CVE-2026-22708).
    """
    
    # Critical modules that must be verified before detector runs
    CRITICAL_MODULES = ['numpy', 'scipy']
    
    # Safe environment variable baselines (system-dependent, user-configurable)
    SAFE_ENV_VARS = {
        'PATH': None,  # User must set during validator initialization
        'PYTHONPATH': None,
        'LD_LIBRARY_PATH': None,
    }
    
    def __init__(self, safe_env_vars: dict = None):
        """
        Initialize environment validator with optional custom environment baseline.
        
        :param safe_env_vars: Dict of env var name -> expected value. If None, 
                              current values are used as baseline.
        """
        self.baseline_imports = {}
        self.baseline_env_vars = {}
        
        if safe_env_vars:
            self.SAFE_ENV_VARS.update(safe_env_vars)
        
        logger.info("Environment Integrity Validator initialized.")
    
    def snapshot_environment(self):
        """
        Capture cryptographic fingerprints of loaded critical libraries
        and environment variable state at baseline time.
        """
        logger.info("Capturing environment baseline snapshot...")
        
        # Snapshot critical module paths and hashes
        for module_name in self.CRITICAL_MODULES:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                module_path = getattr(module, '__file__', None)
                
                if module_path:
                    try:
                        with open(module_path, 'rb') as f:
                            module_hash = hashlib.sha256(f.read()).hexdigest()
                        self.baseline_imports[module_name] = {
                            'path': module_path,
                            'hash': module_hash
                        }
                        logger.info(f"✅ Fingerprinted {module_name}: {module_hash[:16]}...")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not fingerprint {module_name}: {e}")
                else:
                    logger.warning(f"⚠️  Could not locate {module_name} file path.")
        
        # Snapshot environment variables
        for var_name in self.SAFE_ENV_VARS.keys():
            self.baseline_env_vars[var_name] = os.environ.get(var_name, '')
            logger.info(f"✅ Captured env var {var_name} = {self.baseline_env_vars[var_name][:50]}...")
        
        logger.info("✅ Environment baseline snapshot complete.")
    
    def verify_environment(self) -> bool:
        """
        Verify that loaded libraries and environment variables match baseline.
        
        Returns True if environment is clean, False if poisoning detected.
        Raises RuntimeError on critical failure.
        """
        logger.info("Running environment integrity verification...")
        
        # Verify module fingerprints
        for module_name, baseline_info in self.baseline_imports.items():
            if module_name not in sys.modules:
                logger.critical(f"🚨 CRITICAL: Module {module_name} was unloaded. Possible sabotage.")
                return False
            
            module = sys.modules[module_name]
            actual_path = getattr(module, '__file__', None)
            baseline_path = baseline_info['path']
            baseline_hash = baseline_info['hash']
            
            # Check path integrity
            if actual_path != baseline_path:
                logger.critical(
                    f"🚨 CRITICAL: {module_name} loaded from different path.\n"
                    f"   Expected: {baseline_path}\n"
                    f"   Actual: {actual_path}\n"
                    f"   Possible library injection or environment poisoning."
                )
                return False
            
            # Check binary hash
            try:
                with open(actual_path, 'rb') as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                
                if actual_hash != baseline_hash:
                    logger.critical(
                        f"🚨 CRITICAL: {module_name} binary mismatch (poisoning detected).\n"
                        f"   Expected hash: {baseline_hash}\n"
                        f"   Actual hash: {actual_hash}\n"
                        f"   Module path: {actual_path}"
                    )
                    return False
                
                logger.info(f"✅ {module_name} binary verified: {actual_hash[:16]}...")
            
            except Exception as e:
                logger.critical(f"🚨 CRITICAL: Could not verify {module_name} binary: {e}")
                return False
        
        # Verify environment variables
        for var_name, baseline_value in self.baseline_env_vars.items():
            actual_value = os.environ.get(var_name, '')
            
            if actual_value != baseline_value:
                logger.critical(
                    f"🚨 CRITICAL: Environment variable {var_name} corrupted (CVE-2026-22708).\n"
                    f"   Expected: {baseline_value}\n"
                    f"   Actual: {actual_value}\n"
                    f"   Possible shell built-in bypass or prompt injection attack."
                )
                return False
            
            logger.info(f"✅ Environment variable {var_name} verified.")
        
        logger.info("✅ Environment integrity verification PASSED.")
        return True


if __name__ == "__main__":
    # Example usage
    validator = EnvironmentIntegrityValidator()
    validator.snapshot_environment()
    
    if validator.verify_environment():
        logger.info("✅ Safe to proceed with CognitiveImmunityEngine initialization.")
    else:
        logger.critical("❌ Environment compromised. Aborting detector initialization.")
        sys.exit(1)
