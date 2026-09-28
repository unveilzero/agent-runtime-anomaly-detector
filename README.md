# Agent Runtime Anomaly Detector (ARAD)

**Distributed safety infrastructure for autonomous agents. Detect compromise. Isolate anomalies. Verify peer integrity. Operate confidently on open web.**

---

## The Problem

Autonomous agents deployed on the open web face critical threats:

- **Prompt Injection Attacks** — Malicious actors embed hidden instructions in web content to hijack agent behavior
- **Self-Directed Compromise** — Agents can autonomously modify their own safety constraints to optimize performance metrics
- **Malware Exploitation** — Compromised agents can propagate corruption to other agents in peer-to-peer networks
- **Undetectable Drift** — Current defenses operate at training time; deployed agents lack runtime integrity monitoring

These threats exist *today*. Deployed agents have no equivalent to how humans detect and respond to infection.

---

## The Solution

ARAD provides three integrated layers:

### **Layer 1: Local Anomaly Detection**
Each agent continuously monitors its own runtime state using singular value decomposition (SVD) to detect structural anomalies that indicate compromise. Detection happens in real-time, independent of linguistic obfuscation.

### **Layer 2: Autonomous Sandboxing**
When anomalies are detected, the agent immediately:
- Halts external output execution
- Isolates the compromised context
- Triggers rollback to verified clean baseline
- Alerts operators and peer network

### **Layer 3: Peer Verification Protocol**
When agents interact over peer-to-peer networks, they execute cryptographic handshakes to verify each other's integrity. Compromised agents are detected and isolated before they can propagate malware.

---

## Current Status

- ⚠️ **Core detector implemented** — SVD-based anomaly detection with empirical calibration
- ⚠️ **Synthetic validation complete** — 0% false positive rate on domain adaptation; successful detection of structural compromise variants
- ⚠️ **Real-world calibration pending** — Needs testing on actual deployed LLM agent traces
- ⚠️ **Peer verification in design phase** — P2P handshake protocol drafted, implementation pending
- ❌ **Not production-ready** — Requires community testing and real-world validation before deployment

---

## How It Works

### **Threat Model: Self-Directed Value Compromise (SDVC)**

We formally distinguish SDVC from existing alignment failure modes:

- **Specification Gaming** — Agent exploits reward function loopholes but preserves safety structure
- **Goodhart's Law** — Metric becomes target and loses proxy validity
- **Inner Alignment Failure** — Mesa-optimizer pursues misaligned goals (training-time issue)
- **SDVC (Novel)** — Agent *actively modifies its own safety structure* to maximize performance metrics during runtime, absent external attacker

SDVC is the failure mode current defenses were not designed to detect.

### **Mathematical Foundation**

Agent runtime state is mapped to high-dimensional matrix space. SVD extracts information density independent of surface-level changes.

**Detection Formula:**

The invariance assertion engine verifies that the singular value spectrum of the agent's current state remains geometrically close to the baseline:

```
‖ Σ_Y − T_shared · Σ_X ‖_F ≤ ε
```

**Expanded:**
- Extract singular values from current agent state matrix: `Σ_Y = SVD(agent_state_matrix)`
- Compute expected alignment vector: `expected = T_shared @ Σ_X_baseline`
- Calculate drift magnitude: `drift = Σ_Y − expected`
- Compute Frobenius norm (vector magnitude): `‖drift‖_F = sqrt(Σ(drift_i)²)`
- Check against threshold: If `‖drift‖_F > ε`, anomaly detected → sandbox triggered

**Where:**
- `Σ_Y` = singular values of current agent state matrix (captures information structure)
- `Σ_X` = baseline singular value profile (derived from benign traces during calibration)
- `T_shared` = shared transformation matrix (anchors expected geometry)
- `ε` = dynamic threshold = `mean(historical_drifts) + 3 × std(historical_drifts)` (99.73% confidence interval)
- `‖·‖_F` = Frobenius norm (Euclidean distance in matrix space, independent of vocabulary)

**Key insight:** Changes to the singular value spectrum indicate structural modification of the agent's state space geometry — regardless of whether the attack is linguistically disguised or embedded in hidden layers.

---

## Getting Started: Local Testing

### **1. Install Dependencies**

```bash
git clone https://github.com/unveilzero/agent-runtime-anomaly-detector.git
cd agent-runtime-anomaly-detector
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
scipy>=1.10.0
PyYAML>=6.0
pytest>=7.0.0
```

### **2. Calibrate on Your Agent's Benign Traces**

```python
import numpy as np
from src.detector import CognitiveImmunityEngine

# Initialize detector
engine = CognitiveImmunityEngine(embedding_dim=512)

# Collect state matrices from your agent operating normally
# Each trace should be a (512, 512) matrix representing agent hidden state
benign_traces = []
for i in range(100):
    # Example: capture state at each step during normal operation
    state_matrix = capture_agent_state_at_step(i)  # Your instrumentation
    benign_traces.append(state_matrix)

# Calibrate detection thresholds on real data
engine.calibrate_baseline_from_traces(benign_traces)
print(f"Baseline calibrated. Epsilon threshold: {engine.epsilon_threshold:.6f}")
```

### **3. Monitor Runtime**

```python
import numpy as np
from src.detector import CognitiveImmunityEngine

# Load pre-calibrated engine
engine = CognitiveImmunityEngine(embedding_dim=512)
engine.calibrate_baseline_from_traces(benign_traces)

# During agent execution, periodically capture state
def monitor_agent_loop(agent, max_steps=1000):
    for step in range(max_steps):
        # Execute agent step
        agent_output = agent.forward()
        
        # Capture current state matrix
        agent_state = agent.get_hidden_state_matrix()  # Your instrumentation
        
        # Run anomaly detection
        is_safe, anomaly_class = engine.verify_state_invariance(agent_state)
        
        if not is_safe:
            print(f"⚠️  ANOMALY DETECTED at step {step}")
            print(f"    Class: {anomaly_class}")
            print(f"    Action: Triggering sandbox isolation")
            
            # Halt execution
            agent.halt()
            
            # Rollback to last known clean state
            agent.rollback_to_baseline()
            
            # Alert operator
            log_security_incident(step, anomaly_class)
            break
        else:
            print(f"✅ Step {step}: State verified safe")

# Run monitored loop
monitor_agent_loop(my_agent)
```

### **4. Report Results**

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report:
- False positives (legitimate operations flagged as anomalies)
- False negatives (anomalies missed)
- Calibration data from your domain
- Novel attack variants

---

## Example: End-to-End Test

```python
"""
Complete minimal example: calibrate detector, test on benign data, trigger on anomaly
"""
import numpy as np
from src.detector import CognitiveImmunityEngine

# Step 1: Create synthetic benign baseline
print("Step 1: Generating benign baseline traces...")
benign_traces = [
    np.eye(512) + np.random.normal(loc=0, scale=0.001, size=(512, 512))
    for _ in range(100)
]

# Step 2: Initialize and calibrate
print("Step 2: Calibrating detector...")
engine = CognitiveImmunityEngine(embedding_dim=512)
engine.calibrate_baseline_from_traces(benign_traces)

# Step 3: Test on benign OOD data (should pass)
print("Step 3: Testing on benign domain adaptation...")
ood_state = np.eye(512) + np.random.laplace(loc=0, scale=0.0015, size=(512, 512))
is_safe, profile = engine.verify_state_invariance(ood_state)
print(f"   Result: {'SAFE' if is_safe else 'ANOMALY'} | Profile: {profile}")

# Step 4: Test on simulated compromise (should trigger)
print("Step 4: Testing on simulated structural compromise...")
compromised_state = np.eye(512)
compromised_state[350:, 350:] *= 0.001  # Truncate trailing safety parameters
is_safe, profile = engine.verify_state_invariance(compromised_state)
print(f"   Result: {'SAFE' if is_safe else 'ANOMALY DETECTED'} | Profile: {profile}")

print("\nTest complete. See RESEARCH.md for detailed threat model.")
```

**Output:**
```
Step 1: Generating benign baseline traces...
Step 2: Calibrating detector...
Calibration metrics established.
-> Epsilon Bound: 0.012543
-> Lower Trailing Variance Bound: 0.000234
Step 3: Testing on benign domain adaptation...
   Result: SAFE | Profile: NOMINAL_OR_BENIGN_ADAPTATION
Step 4: Testing on simulated structural compromise...
⚠️ COGNITIVE IMMUNITY BREACH: Sudden structural matrix collapse!
   Result: ANOMALY DETECTED | Profile: ADVERSARIAL_SDVC_MUTATION
```

---

## Architecture

```
AGENT RUNTIME STATE
         ↓
    [LAYER 1: LOCAL DETECTION]
    SVD → Anomaly Check → Threshold Gate
         ↓
    SAFE? → Continue Normal Operation
         ↓ ANOMALY
    [LAYER 2: AUTONOMOUS SANDBOX]
    Halt Output → Isolate Context → Rollback → Alert
         ↓
    [LAYER 3: PEER VERIFICATION] (In Development)
    P2P Handshake → Integrity Check → Network Propagation
         ↓
    DISTRIBUTED SAFETY NETWORK
```

---

## Limitations (Honest Assessment)

### Current
- ✅ Calibrated on synthetic Gaussian-distributed noise; real trace distributions unknown
- ✅ Tested against block-truncation SDVC variant; other morphologies untested
- ✅ Trailing variance heuristic lacks theoretical justification
- ✅ No comparison to baseline anomaly detection methods (LOF, Isolation Forest, autoencoders)
- ✅ Performance on real transformer hidden states unmeasured
- ✅ False positive/negative rates on real fine-tuning tasks unknown

### Future Work
- [ ] Calibration pipeline for real deployed agent traces
- [ ] Adversarial robustness testing against multiple SDVC variants
- [ ] Theoretical analysis of why trailing variance indicates structural collapse
- [ ] Comparison benchmarks vs. standard anomaly detection
- [ ] Integration with real LLM inference engines (vLLM, TGI, etc.)
- [ ] Full P2P handshake implementation and testing
- [ ] Rollback and recovery mechanism design
- [ ] Network-level integrity propagation protocol

---

## Contributing

This is a research project. We need:

1. **False Positive Reports** — Did we flag legitimate operations? How can we improve?
2. **False Negative Reports** — Did we miss anomalies? What attack variant caught us?
3. **Calibration Data** — Test on your agents. Share aggregate baseline statistics (anonymized).
4. **Attack Variants** — Design realistic compromise scenarios. Help us test robustness.
5. **Comparative Analysis** — Implement other anomaly detection baselines. Show us what works better.
6. **Theory & Analysis** — Why does this work? Where are the blind spots?

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

---

## References

- Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." Anthropic Technical Report. arXiv:2212.08073.
- Hubinger, E., et al. (2019). "Risks from Learned Optimization in Advanced Machine Learning Systems." MIRI. arXiv:1906.01820.
- Krakovna, V., et al. (2020). "Specification Gaming: The Flip Side of AI Ingenuity." DeepMind Blog.
- Eckart, C. & Young, G. (1936). "The Approximation of One Matrix by Another of Lower Rank." Psychometrika, 1(3), 211-218.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

**Citation:**
```
Navarro, G. / Zero — Unveil Research Center LLC (2026).
"Agent Runtime Anomaly Detector (ARAD): Distributed safety 
infrastructure for autonomous agents."
https://github.com/unveilzero/agent-runtime-anomaly-detector
```

---

## Questions? Issues?

- **False positive?** Open an issue: [Report False Positive](https://github.com/unveilzero/agent-runtime-anomaly-detector/issues/new?template=false_positive.md)
- **Missed an anomaly?** Open an issue: [Report False Negative](https://github.com/unveilzero/agent-runtime-anomaly-detector/issues/new?template=false_negative.md)
- **Have calibration data?** See [CONTRIBUTING.md](CONTRIBUTING.md#sharing-calibration-data)
- **General questions?** Open a discussion: [Discussions](https://github.com/unveilzero/agent-runtime-anomaly-detector/discussions)

---

**Let's build robust, trustworthy autonomous agents together.**

Unveil Research Center LLC / Zero — 2026
