"""Preflight integrity checks for selected modules and environment variables."""

import hashlib
import importlib
import json
import logging
import os
import sys
from pathlib import Path

from src.audit import AuditLog

logger = logging.getLogger("OASF_Environment_Validator")

# Bytecode caches change during normal imports, so they are excluded.
_SKIP_DIRS = {"__pycache__"}
_SKIP_SUFFIXES = (".pyc", ".pyo")


def _fingerprint_module(module):
    """Hash every file of a package or module; returns (root, sha256_hex, file_count)."""
    package_paths = list(getattr(module, "__path__", []) or [])
    module_file = getattr(module, "__file__", None)
    if package_paths:
        root = Path(package_paths[0]).resolve()
        files = []
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in _SKIP_DIRS)
            for name in sorted(names):
                if not name.endswith(_SKIP_SUFFIXES):
                    files.append(Path(directory) / name)
    elif module_file:
        root = Path(module_file).resolve()
        files = [root]
    else:
        raise RuntimeError(f"Cannot locate files for module {module.__name__}")

    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix() if path != root else path.name
        digest.update(relative.encode("utf-8") + b"\0")
        file_hash = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                file_hash.update(chunk)
        digest.update(file_hash.digest())
    return str(root), digest.hexdigest(), len(files)


class EnvironmentIntegrityValidator:
    """Capture and verify a local baseline for selected runtime inputs."""

    CRITICAL_MODULES = ("numpy", "scipy")
    SAFE_ENV_VARS = {"PATH": None, "PYTHONPATH": None, "LD_LIBRARY_PATH": None}

    def __init__(self, safe_env_vars=None, audit_log_path=None, critical_modules=None, audit_key=None):
        self.baseline_imports = {}
        self.baseline_env_vars = {}
        self.environment_manifest = {}
        self.critical_modules = tuple(critical_modules or self.CRITICAL_MODULES)
        self._safe_env_vars = dict(self.SAFE_ENV_VARS)
        if safe_env_vars:
            self._safe_env_vars.update(safe_env_vars)
        self.audit_log = AuditLog(audit_log_path or ".arad_environment_audit.jsonl", key=audit_key)

    def _record_event(self, event_type, details):
        self.audit_log.append(event_type, details)

    def snapshot_environment(self):
        if self.environment_manifest:
            raise RuntimeError("Environment baseline already captured; create a new validator to recapture it.")
        for module_name in self.critical_modules:
            try:
                module = importlib.import_module(module_name)
                root, module_hash, file_count = _fingerprint_module(module)
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(f"Unable to capture module baseline for {module_name}") from exc
            self.baseline_imports[module_name] = {"path": root, "hash": module_hash, "files": file_count}

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
            if module is None:
                self._record_event("environment_mismatch", {"module": module_name, "reason": "not_loaded"})
                return False
            try:
                root, actual_hash, _ = _fingerprint_module(module)
            except (OSError, RuntimeError):
                self._record_event("environment_mismatch", {"module": module_name, "reason": "read_error"})
                return False
            if root != baseline["path"]:
                self._record_event("environment_mismatch", {"module": module_name, "reason": "path_mismatch"})
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
