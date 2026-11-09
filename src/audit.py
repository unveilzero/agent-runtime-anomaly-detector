"""Append-only hash-chained audit records.

A hash chain detects modification or reordering of retained records, but it does
not prove that the final log segment was not removed. For strong evidence, export
periodic checkpoints to external append-only storage and keep one writer per log.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path


class AuditLog:
    """A small JSONL audit log with a hash chain and validation."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._sequence = 0
        self._head_hash = "0" * 64
        self._sequence, self._head_hash = self._read_and_validate()

    @staticmethod
    def _canonical(record):
        return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _read_and_validate(self):
        if not self.path.exists():
            return 0, "0" * 64

        previous_hash = "0" * 64
        sequence = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid audit record at line {line_number}") from exc

                chain_hash = payload.get("chain_hash")
                if chain_hash is None:
                    raise ValueError(f"Audit log entry missing chain_hash at line {line_number}")

                record = dict(payload)
                record.pop("chain_hash", None)
                expected_hash = hashlib.sha256(self._canonical(record).encode("utf-8")).hexdigest()
                if not hashlib.compare_digest(chain_hash, expected_hash):
                    raise ValueError(f"Audit hash mismatch at line {line_number}")
                if record.get("previous_hash") != previous_hash:
                    raise ValueError(f"Broken audit chain at line {line_number}")
                if record.get("sequence") != sequence + 1:
                    raise ValueError(f"Sequence mismatch at line {line_number}")

                previous_hash = chain_hash
                sequence += 1

        return sequence, previous_hash

    def append(self, event, details):
        with self._lock:
            record = {
                "sequence": self._sequence + 1,
                "timestamp": time.time(),
                "event": str(event),
                "details": details,
                "previous_hash": self._head_hash,
            }
            chain_hash = hashlib.sha256(self._canonical(record).encode("utf-8")).hexdigest()
            output = dict(record, chain_hash=chain_hash)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(self._canonical(output) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._sequence += 1
            self._head_hash = chain_hash
            return chain_hash

    def verify(self):
        sequence, head_hash = self._read_and_validate()
        return {"valid": True, "records": sequence, "head_hash": head_hash}
