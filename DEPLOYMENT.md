# Deployment

ARAD provides state-matrix evaluation, local environment checks, and runtime-event policy evaluation. Integrate these components with the deployment's trusted telemetry collector and enforcement supervisor.

## State and environment checks

```python
from src import CognitiveImmunityEngine, EnvironmentIntegrityValidator

validator = EnvironmentIntegrityValidator()
validator.snapshot_environment()
if not validator.verify_environment():
    raise RuntimeError("Environment integrity check failed")

engine = CognitiveImmunityEngine(embedding_dim=512)
engine.calibrate_baseline_from_traces(benign_traces)
safe, profile = engine.verify_state_invariance(target_state_matrix)
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

## Keeping the checks out of the agent's reach

An agent that can read or write this process can act around any check that lives only inside it. These settings move the trust anchor to a supervisor the agent cannot touch.

- **Keyed audit logs.** Pass `audit_key=` (at least 16 bytes) to each component. Records are then chained with HMAC-SHA256, so a log rebuilt from scratch without the key fails verification. Keep the key outside the agent's environment.
- **Anchors.** Call `anchor()` on an `AuditLog` (or `anchor_audit_log()` on the runtime monitor) and store the returned `{"sequence", "head_hash"}` with the supervisor. `verify(expected_anchor=...)` raises if the log was cut short or rewritten past that point. Normal appends still pass.
- **Baseline hash.** After calibration, store `baseline_manifest["baseline_hash"]` with the supervisor and call `verify_baseline(stored_hash)` from there. Better, pass it on every call as `verify_state_invariance(state, expected_baseline_hash=stored_hash)`, and construct the engine with `require_external_baseline_hash=True` so a call without it fails. Treat that flag as required whenever the agent's runtime is not fully trusted; it is recorded in the audit log and the calibration report so a supervisor can confirm it. The internal check compares against a hash stored on the same object, so it only catches accidental or partial edits. None of this stops code that runs inside the engine's own process from patching the engine itself; for an adversarial agent, the engine has to run in a separate process.
- **Calibration floor.** At least 50 traces are required by default (`min_calibration_traces=`, minimum 2). Traces with no variation at all are rejected. Set `max_epsilon=` to reject calibration sets whose drift spread is wider than expected.
- **Agent lineage.** Send `parent_agent_id` with runtime events when an agent spawns another. Blocked action sequences are matched per trace and per lineage root, so moving to a new `trace_id` or handing the second step to a sub-agent no longer resets correlation. An agent first seen without a parent is treated as "parent not known yet": the first parent reported for it is adopted, and its earlier history moves to the new root. Once a parent is known, a different one is blocked as `lineage_conflict`. Lineage must come from the collector, not from the agent.
- **Tracking capacity.** The monitor never drops trace or lineage state that is still inside the correlation window. Events denied by policy never change tracking state. When `max_tracked_keys` is reached, events from new traces or agents are blocked (`history_capacity_exceeded`, `lineage_capacity_exceeded`) until idle entries age out. This is a deliberate denial-of-service tradeoff: a source that can generate many allowed events with new trace or agent IDs can get new work blocked, but cannot make the monitor forget history. Size the limit for peak concurrency and alert on these two reasons. An event blocked by policy or capacity leaves no tracking state behind. An event dated more than one window behind its lineage is blocked as `stale_lineage_timestamp`.
- **Event timestamps.** The correlation window is measured in the timestamps the collector supplies, and `timestamp` must be a number. A source that can forward-date an event past the window can separate two halves of a blocked sequence, so in live deployments set `max_timestamp_skew_seconds=` to block events dated too far from the monitor host's clock (`timestamp_out_of_range`). Leave it unset only when replaying historical events.
- **Package fingerprints.** The environment validator hashes every file in each critical package (bytecode caches excluded), not just `__init__.py`.
