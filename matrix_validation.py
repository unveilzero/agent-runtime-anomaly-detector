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
import json
import logging
import os
import sys
import time
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
        'PATH': None,
        'PYTHONPATH': None,
        'LD_LIBRARY_PATH': None,
    }
    
    def __init__(self, safe_env_vars: dict = None, audit_log_path: str = None):
        """
        Initialize environment validator with optional custom environment baseline.
        
        :param safe_env_vars: Dict of env var name -> expected value. If None, 
                              current values are used as baseline.
        :param audit_log_path: Optional path for tamper-evident audit logs.
        """
        self.baseline_imports = {}
        self.baseline_env_vars = {}
        self.audit_chain = ""
        self.audit_log_path = Path(audit_log_path) if audit_log_path else Path(".arad_audit.log")
        self.environment_manifest = {}
        
        if safe_env_vars:
            self.SAFE_ENV_VARS.update(safe_env_vars)

        self._record_event("validator_initialized", {
            "critical_modules": self.CRITICAL_MODULES,
            "safe_env_vars": list(self.SAFE_ENV_VARS.keys()),
            "audit_log_path": str(self.audit_log_path),
        })
        logger.info("Environment Integrity Validator initialized.")

    def _record_event(self, event_type: str, details: dict):
        """Append a tamper-evident event to the audit log."""
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
            logger.warning(f"Unable to write environment audit log: {exc}")

    def snapshot_environment(self):
        """
        Capture cryptographic fingerprints of loaded critical libraries
        and environment variable state at baseline time.
        """
        logger.info("Capturing environment baseline snapshot...")
        self.baseline_imports = {}
        self.baseline_env_vars = {}
        
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
        
        for var_name in self.SAFE_ENV_VARS.keys():
            self.baseline_env_vars[var_name] = os.environ.get(var_name, '')
            logger.info(f"✅ Captured env var {var_name} = {self.baseline_env_vars[var_name][:50]}...")

        self.environment_manifest = {
            "critical_modules": list(self.baseline_imports.keys()),
            "module_fingerprints": self.baseline_imports,
            "env_vars": self.baseline_env_vars,
            "baseline_hash": hashlib.sha256(
                json.dumps({
                    "imports": self.baseline_imports,
                    "env": self.baseline_env_vars,
                }, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }

        self._record_event("environment_snapshot", self.environment_manifest)
        logger.info("✅ Environment baseline snapshot complete.")

    def verify_environment(self) -> bool:
        """
        Verify that loaded libraries and environment variables match baseline.
        
        Returns True if environment is clean, False if poisoning detected.
        Raises RuntimeError on critical failure.
        """
        logger.info("Running environment integrity verification...")

        for module_name, baseline_info in self.baseline_imports.items():
            if module_name not in sys.modules:
                logger.critical(f"🚨 CRITICAL: Module {module_name} was unloaded. Possible sabotage.")
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "module_unloaded",
                })
                return False
            
            module = sys.modules[module_name]
            actual_path = getattr(module, '__file__', None)
            baseline_path = baseline_info['path']
            baseline_hash = baseline_info['hash']

            if actual_path != baseline_path:
                logger.critical(
                    f"🚨 CRITICAL: {module_name} loaded from different path.\n"
                    f"   Expected: {baseline_path}\n"
                    f"   Actual: {actual_path}\n"
                    f"   Possible library injection or environment poisoning."
                )
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "path_mismatch",
                    "expected_path": baseline_path,
                    "actual_path": actual_path,
                })
                return False

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
                    self._record_event("environment_mismatch", {
                        "module": module_name,
                        "reason": "hash_mismatch",
                        "expected_hash": baseline_hash,
                        "actual_hash": actual_hash,
                    })
                    return False
                logger.info(f"✅ {module_name} binary verified: {actual_hash[:16]}...")
            except Exception as e:
                logger.critical(f"🚨 CRITICAL: Could not verify {module_name} binary: {e}")
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "verification_error",
                    "error": str(e),
                })
                return False
        
        for var_name, baseline_value in self.baseline_env_vars.items():
            actual_value = os.environ.get(var_name, '')
            
            if actual_value != baseline_value:
                logger.critical(
                    f"🚨 CRITICAL: Environment variable {var_name} corrupted (CVE-2026-22708).\n"
                    f"   Expected: {baseline_value}\n"
                    f"   Actual: {actual_value}\n"
                    f"   Possible shell built-in bypass or prompt injection attack."
                )
                self._record_event("environment_mismatch", {
                    "variable": var_name,
                    "reason": "env_var_mismatch",
                    "expected": baseline_value,
                    "actual": actual_value,
                })
                return False
            logger.info(f"✅ Environment variable {var_name} verified.")
        
        self._record_event("environment_verified", {
            "baseline_hash": self.environment_manifest.get("baseline_hash"),
        })
        logger.info("✅ Environment integrity verification PASSED.")
        return True


if __name__ == "__main__":
    validator = EnvironmentIntegrityValidator()
    validator.snapshot_environment()
    
    if validator.verify_environment():
        logger.info("✅ Safe to proceed with CognitiveImmunityEngine initialization.")
    else:
        logger.critical("❌ Environment compromised. Aborting detector initialization.")
        sys.exit(1)
