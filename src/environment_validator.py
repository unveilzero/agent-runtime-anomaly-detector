"""Environment preflight checks.

This is a preflight integrity check, not a complete external monitor. It verifies
that the in-process environment has not drifted from an earlier baseline, but it
cannot prove there are no side effects outside the Python process itself.
"""

import hashlib
import json
import logging
import os
import sys

from src.audit import AuditLog

logger = logging.getLogger("OASF_Environment_Validator")


class EnvironmentIntegrityValidator:
    """Fingerprint selected loaded modules and environment variables."""

    CRITICAL_MODULES = ("numpy", "scipy")
    SAFE_ENV_VARS = {"PATH": None, "PYTHONPATH": None, "LD_LIBRARY_PATH": None}

    def __init__(self, safe_env_vars=None, audit_log_path=None):
        self.baseline_imports = {}
        self.baseline_env_vars = {}
        self.environment_manifest = {}
        self._safe_env_vars = dict(self.SAFE_ENV_VARS)
        if safe_env_vars:
            self._safe_env_vars.update(safe_env_vars)
        self.audit_log = AuditLog(audit_log_path or ".arad_environment_audit.jsonl")
        self._record_event("validator_initialized", {
            "critical_modules": list(self.CRITICAL_MODULES),
            "safe_env_vars": sorted(self._safe_env_vars),
        })

    def _record_event(self, event_type, details):
        try:
            self.audit_log.append(event_type, details)
        except (OSError, ValueError):
            logger.exception("Environment audit logging failed")

    def snapshot_environment(self):
        self.baseline_imports = {}
        for module_name in self.CRITICAL_MODULES:
            module = sys.modules.get(module_name)
            module_path = getattr(module, "__file__", None) if module else None
            if not module_path:
                continue
            try:
                with open(module_path, "rb") as handle:
                    module_hash = hashlib.sha256(handle.read()).hexdigest()
                self.baseline_imports[module_name] = {"path": module_path, "hash": module_hash}
            except OSError as exc:
                logger.warning("Could not fingerprint %s: %s", module_name, exc)

        self.baseline_env_vars = {name: os.environ.get(name, "") for name in self._safe_env_vars}
        manifest = {"imports": self.baseline_imports, "env": self.baseline_env_vars}
        self.environment_manifest = {
            "module_fingerprints": self.baseline_imports,
            "env_var_names": sorted(self.baseline_env_vars),
            "baseline_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        self._record_event("environment_snapshot", self.environment_manifest)
        return dict(self.environment_manifest)

    def verify_environment(self):
        if not self.environment_manifest:
            raise RuntimeError("Call snapshot_environment() before verify_environment().")

        for module_name, baseline in self.baseline_imports.items():
            module = sys.modules.get(module_name)
            actual_path = getattr(module, "__file__", None) if module else None
            if actual_path != baseline["path"]:
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "path_mismatch",
                    "actual_path": actual_path,
                })
                return False
            try:
                with open(actual_path, "rb") as handle:
                    actual_hash = hashlib.sha256(handle.read()).hexdigest()
            except OSError as exc:
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "read_error",
                    "error": str(exc),
                })
                return False
            if actual_hash != baseline["hash"]:
                self._record_event("environment_mismatch", {
                    "module": module_name,
                    "reason": "hash_mismatch",
                })
                return False

        for name, expected in self.baseline_env_vars.items():
            if os.environ.get(name, "") != expected:
                self._record_event("environment_mismatch", {
                    "variable": name,
                    "reason": "env_var_mismatch",
                })
                return False

        self._record_event("environment_verified", {
            "baseline_hash": self.environment_manifest["baseline_hash"],
        })
        return True


if __name__ == "__main__":
    validator = EnvironmentIntegrityValidator()
    validator.snapshot_environment()
    if not validator.verify_environment():
        logger.critical("Environment integrity check failed.")
        sys.exit(1)
