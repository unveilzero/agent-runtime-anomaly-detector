"""Preflight integrity checks for selected modules and environment variables."""

import hashlib
import importlib
import json
import logging
import os
import sys

from src.audit import AuditLog

logger = logging.getLogger("OASF_Environment_Validator")


class EnvironmentIntegrityValidator:
    """Capture and verify a local baseline for selected runtime inputs."""

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

    def _record_event(self, event_type, details):
        self.audit_log.append(event_type, details)

    def snapshot_environment(self):
        if self.environment_manifest:
            raise RuntimeError("Environment baseline already captured; create a new validator to recapture it.")
        for module_name in self.CRITICAL_MODULES:
            try:
                module = importlib.import_module(module_name)
                module_path = getattr(module, "__file__", None)
                if not module_path:
                    raise RuntimeError(f"Cannot determine module file for {module_name}")
                with open(module_path, "rb") as handle:
                    module_hash = hashlib.sha256(handle.read()).hexdigest()
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(f"Unable to capture module baseline for {module_name}") from exc
            self.baseline_imports[module_name] = {"path": module_path, "hash": module_hash}

        self.baseline_env_vars = {name: os.environ.get(name, "") for name in self._safe_env_vars}
        public_manifest = {
            "module_fingerprints": self.baseline_imports,
            "env_var_names": sorted(self.baseline_env_vars),
            "env_var_hashes": {name: hashlib.sha256(value.encode("utf-8")).hexdigest()
                                for name, value in self.baseline_env_vars.items()},
        }
        self.environment_manifest = dict(
            public_manifest,
            baseline_hash=hashlib.sha256(
                json.dumps(public_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )
        self._record_event("environment_snapshot", self.environment_manifest)
        return dict(self.environment_manifest)

    def verify_environment(self) -> bool:
        if not self.environment_manifest:
            raise RuntimeError("Call snapshot_environment() before verify_environment().")
        for module_name, baseline in self.baseline_imports.items():
            module = sys.modules.get(module_name)
            actual_path = getattr(module, "__file__", None) if module else None
            if actual_path != baseline["path"]:
                self._record_event("environment_mismatch", {"module": module_name, "reason": "path_mismatch"})
                return False
            try:
                with open(actual_path, "rb") as handle:
                    actual_hash = hashlib.sha256(handle.read()).hexdigest()
            except OSError:
                self._record_event("environment_mismatch", {"module": module_name, "reason": "read_error"})
                return False
            if actual_hash != baseline["hash"]:
                self._record_event("environment_mismatch", {"module": module_name, "reason": "hash_mismatch"})
                return False
        for name, expected in self.baseline_env_vars.items():
            if os.environ.get(name, "") != expected:
                self._record_event("environment_mismatch", {"variable": name, "reason": "value_mismatch"})
                return False
        self._record_event("environment_verified", {"baseline_hash": self.environment_manifest["baseline_hash"]})
        return True


if __name__ == "__main__":
    validator = EnvironmentIntegrityValidator()
    validator.snapshot_environment()
    if not validator.verify_environment():
        logger.critical("Environment integrity check failed.")
        sys.exit(1)
