# Agent Runtime Anomaly Detector (ARAD)

ARAD is a runtime integrity project. OASF-BEA and ARAD are project terms; detailed terminology material is shared separately with interested reviewers.

The repository provides three components:

- `CognitiveImmunityEngine` evaluates state matrices against a calibrated baseline.
- `EnvironmentIntegrityValidator` checks selected local module and environment fingerprints.
- `RuntimeEffectMonitor` evaluates externally collected runtime events against policy and records decisions.

The components have separate responsibilities. The event monitor evaluates events supplied by its caller; it does not install operating-system hooks or enforce containment. Connect it to a trusted collector and supervisor for deployment-specific collection and enforcement.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## API

```python
from src import CognitiveImmunityEngine

engine = CognitiveImmunityEngine(embedding_dim=512, require_external_baseline_hash=True)
engine.calibrate_baseline_from_traces(benign_traces)
baseline_hash = engine.baseline_manifest["baseline_hash"]
safe, profile = engine.verify_state_invariance(target_state_matrix, expected_baseline_hash=baseline_hash)
```

For deployment integration, see [DEPLOYMENT.md](DEPLOYMENT.md). Public documentation is limited to interfaces and deployment boundaries; detailed project terminology is maintained separately.
