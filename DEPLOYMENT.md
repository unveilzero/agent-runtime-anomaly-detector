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
