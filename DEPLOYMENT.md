# Deployment

ARAD provides state-matrix evaluation, local environment checks, and runtime-event policy evaluation. Integrate these components with the deployment's trusted telemetry collector and enforcement supervisor.

## State and environment checks

```python
from src import CognitiveImmunityEngine, EnvironmentIntegrityValidator

validator = EnvironmentIntegrityValidator()
validator.snapshot_environment()
if not validator.verify_environment():
    raise RuntimeError("Environment integrity check failed")

engine = CognitiveImmunityEngine(embedding_dim=512, require_external_baseline_hash=True)
engine.calibrate_baseline_from_traces(benign_traces)
baseline_hash = engine.baseline_manifest["baseline_hash"]  # store with the supervisor
safe, profile = engine.verify_state_invariance(target_state_matrix, expected_baseline_hash=baseline_hash)
if not safe:
    raise RuntimeError(f"State evaluation requires handling: {profile}")
```

Use a reviewed calibration set and retain the generated baseline report with deployment records. Recalibration requires a new engine instance.

## Runtime event policy

```python
from src import RuntimeEffectMonitor

monitor = RuntimeEffectMonitor(
    policy={
        "FILE_READ": {"/srv/agent/docs"},
        "FILE_WRITE": set(),
        "OUTBOUND_CONNECT": {"api.example.org"},
    },
)
result = monitor.evaluate_event({
    "trace_id": "trace-from-supervisor",
    "agent_id": "worker-1",
    "action": "FILE_READ",
    "resource": "/srv/agent/docs",
})
if result["decision"] == "BLOCK":
    raise RuntimeError(result["reason"])
```

Supply events from an independently operated collector, not solely from agent-reported telemetry. Normalize resource identities in that collector before policy evaluation. `RuntimeEffectMonitor` evaluates submitted events; collection and enforcement belong to the integrating runtime.

## Operational notes

- Keep policy and collector configuration under operator control.
- Use distinct audit paths for concurrently running components.
- Restrict write access to audit storage and export records to protected storage when required.
- Run tests with `python -m pytest -q`.

The local environment validator is a consistency check, not a replacement for operating-system isolation or external runtime monitoring.

## Supervisor configuration

Run ARAD in a supervisor process that is separate from the agent, and keep keys, anchors and the baseline hash in storage only the supervisor can access.

- **Keyed audit logs.** Pass `audit_key=` (at least 16 bytes) to each component. Records are chained with HMAC-SHA256.
- **Anchors.** Call `anchor()` on an `AuditLog` (or `anchor_audit_log()` on the runtime monitor) and store the returned `{"sequence", "head_hash"}`. Pass it to `verify(expected_anchor=...)` to confirm the log still contains that record. Normal appends pass.
- **Baseline hash.** After calibration, store `baseline_manifest["baseline_hash"]`. Pass it on every call as `verify_state_invariance(state, expected_baseline_hash=stored_hash)`, and construct the engine with `require_external_baseline_hash=True` so the hash is mandatory. The setting is recorded in the audit log and the calibration report. `verify_baseline(stored_hash)` runs the same comparison on its own.
- **Calibration limits.** At least 50 traces are required by default (`min_calibration_traces=`, minimum 2), and the traces must vary. Set `max_epsilon=` to cap the accepted drift threshold.
- **Agent lineage.** Have the collector send `parent_agent_id` with runtime events when an agent spawns another. Blocked action sequences are matched per trace and per lineage root. The first parent reported for an agent is fixed; a different one returns `lineage_conflict`.
- **Tracking capacity.** `max_tracked_keys` bounds the tracked traces, lineages and agents. At the limit, events from new traces or agents return `history_capacity_exceeded` or `lineage_capacity_exceeded` until idle entries age out. Size the limit for peak concurrency and alert on these two reasons.
- **Event timestamps.** `timestamp` must be a number in the collector's clock. In live deployments set `max_timestamp_skew_seconds=` to the largest acceptable difference from the monitor host's clock; events outside it return `timestamp_out_of_range`. Leave it unset when replaying historical events.
- **Package fingerprints.** The environment validator hashes every file in each critical package, excluding bytecode caches.
