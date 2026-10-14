# Adaptive Temporal Semantics Learning (ATSL)

> **Full Research Idea Document**
> **Status**: Active development
> **Last Updated**: 2026-10-14

---

## 1. Executive Summary

### Title
**Task- and Patient-Conditioned Temporal Representation Learning for Asynchronous EHR**

### One-Sentence Summary
The model learns *how far back to look* based on the current patient state and the prediction task, instead of using a fixed temporal window.

### Core Hypothesis
> Clinical temporal semantics is **not fixed**. It varies with:
> - Patient physiological state
> - Disease progression stage
> - Downstream prediction task

---

## 2. Motivation

### Clinical Observation

Consider two ICU patients:

**Patient A — Stable Post-Op**
```
Time:  0h────4h────8h────12h────16h────20h
State: stable  stable  stable  stable  stable
Relevant history: 12-24h (long-term recovery trend)
```

**Patient B — Septic Shock**
```
Time:  0h────4h────8h────12h────16h────20h
State: stable  declining  shock  shock  recovery
Relevant history: 2-4h (acute deterioration dynamics)
```

> A fixed 5h window is **too short** for Patient A and **too long** for Patient B.

### Technical Observation

Current EHR models use:
- Fixed temporal windows (1h, 6h, 12h, 24h)
- Fixed positional encodings
- Fixed attention patterns
- Task-agnostic temporal encoders

None of these adapt to **who the patient is** or **what the task is**.

---

## 3. Core Concept: Adaptive Temporal Kernel

### From Fixed Window to Learned Kernel

**Traditional Fixed Window**:
```
if |t_i - t_j| < 5h:    # fixed threshold
    attend(t_i, t_j)
```

**Adaptive Temporal Kernel**:
```
σ_t = f(patient_state, task)     # learned bandwidth
K(Δt) = exp(-Δt² / (2σ_t²))     # Gaussian kernel
```

### What Does σ_t Represent?

```
σ_t is the "temporal receptive field" — how far back the model looks.

Large σ_t (e.g., 12h):  "I need long-term history"
Small σ_t (e.g., 2h):   "I only need recent dynamics"
```

### How Is σ_t Learned?

```
σ_t = f(patient_state, task)

where:
  patient_state = current clinical state (vitals, labs, medications, acuity)
  task = prediction objective (mortality, decompensation, phenotyping, LOS)

f is a neural network that outputs a positive scalar (via softplus or exp)
```

---

## 4. The Three Research Questions

### RQ1: Does Fixed Temporal Granularity Limit EHR Representation?

**Hypothesis**: No single window size is optimal across all patients and tasks.

**Experiment**: Compare fixed windows (1h, 3h, 5h, 12h, 24h) across all tasks.

**Expected Result**: Different tasks prefer different windows; no universal optimum.

### RQ2: Is Temporal Relevance Patient-Dependent?

**Hypothesis**: The learned temporal receptive field (σ_t) correlates with clinical state.

**Correlates**:
- SOFA score change rate
- Vital sign instability
- Medication escalation

**Visualization**:
```
Patient Timeline:
0h        20h       40h       60h
stable    stable    shock     recovery
σ_t=12h   σ_t=12h   σ_t=2h    σ_t=6h
```

### RQ3: Should Downstream Tasks Reshape Temporal Representation?

**Hypothesis**: Different tasks need different temporal evidence.

| Task | What it needs | Expected σ_t pattern |
|------|---------------|---------------------|
| Mortality | Acute deterioration + long-term burden | Bimodal (short + long) |
| Decompensation | Recent rapid dynamics | Short (1-3h) |
| Phenotyping | Longer-term disease trajectory | Long (12-24h) |
| LOS | Overall trajectory understanding | Medium (6-12h) |

---

## 5. Method Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ATSL Architecture                         │
│                                                             │
│  EHR Events ──► Temporal Encoder ──► Patient State h_t     │
│                                        │                    │
│                                        ▼                    │
│                              ┌─────────────────┐            │
│                              │  σ_t = f(h_t,   │            │
│                              │    task_emb)    │            │
│                              └────────┬────────┘            │
│                                       │                     │
│                                       ▼                     │
│                              ┌─────────────────┐            │
│                              │  K(Δt) =        │            │
│                              │  exp(-Δt²/2σ_t²)│            │
│                              └────────┬────────┘            │
│                                       │                     │
│                                       ▼                     │
│                              Weighted Event Aggregation     │
│                                       │                     │
│                                       ▼                     │
│                              Task-Specific Prediction       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component 1: Temporal Encoder

- Input: Irregular EHR events with timestamps
- Output: Patient state representation h_t
- Choice: GRU-D (for missing data handling) or Transformer

### Component 2: Adaptive Bandwidth Network

```python
class AdaptiveBandwidth(nn.Module):
    def __init__(self, state_dim, task_dim, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensure σ_t > 0
        )

    def forward(self, patient_state, task_embedding):
        combined = torch.cat([patient_state, task_embedding], dim=-1)
        sigma_t = self.net(combined)
        return sigma_t
```

### Component 3: Kernel-Based Aggregation

```python
def adaptive_kernel_aggregation(events, timestamps, sigma_t):
    """
    events: [N, D] — N events with D-dimensional features
    timestamps: [N] — time of each event
    sigma_t: scalar — learned bandwidth
    """
    current_time = timestamps[-1]
    delta_t = current_time - timestamps  # [N]

    # Gaussian kernel weights
    weights = torch.exp(-delta_t**2 / (2 * sigma_t**2))  # [N]

    # Weighted aggregation
    aggregated = torch.sum(weights.unsqueeze(-1) * events, dim=0)
    return aggregated
```

### Component 4: Task Embedding

Each task gets a learnable embedding vector:
```python
task_embeddings = nn.Embedding(num_tasks, task_dim)
# e.g., task 0 = mortality, task 1 = decompensation, etc.
```

---

## 6. Ablation Study Design

```
A1: Fixed window (baseline)
    └── Single fixed window size for all patients/tasks

A2: Multi-scale (no adaptation)
    └── Multiple fixed windows combined with equal weights

A3: Patient-adaptive
    └── σ_t = f(patient_state) — adapts to patient but not task

A4: Task-adaptive
    └── σ_t = f(task) — adapts to task but not patient

A5: Full model (patient + task adaptive)
    └── σ_t = f(patient_state, task) — full adaptation
```

**Key comparison**: A3 vs. A4 vs. A5 shows whether patient conditioning, task conditioning, or both are necessary.

---

## 7. Evaluation Strategy

### Beyond AUROC

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| AUROC / AUPRC | Prediction accuracy | Standard |
| σ_t distribution | Learned temporal scale | Histogram of σ_t values |
| σ_t vs. SOFA correlation | Clinical relevance | Pearson/Spearman correlation |
| Event order sensitivity | Temporal reasoning | Shuffle events, measure prediction change |
| Temporal perturbation robustness | Stability | Add noise to timestamps, measure AUROC drop |

### Temporal Behavior Analysis

**Event Order Test**:
```
Original:  vasopressor → hypotension
Shuffled:  hypotension → vasopressor

Question: Does the model change its evidence selection?
Expected: Yes, if the model truly understands temporal semantics
```

### Learned Temporal Scale Visualization

```
Patient Timeline with learned σ_t:

0h        20h       40h       60h
stable    stable    shock     recovery
σ_t=12h   σ_t=12h   σ_t=2h    σ_t=6h

Interpretation: The model "knows" to look far back when stable,
                and focus on recent events during shock.
```

---

## 8. Expected Contributions

1. **Method**: Novel adaptive temporal kernel for EHR representation
2. **Theory**: Formalization of patient-conditioned and task-conditioned temporal semantics
3. **Empirical**: Comprehensive evaluation showing adaptive > fixed
4. **Clinical**: Interpretable σ_t values that correlate with clinical state

---

## 9. Paper Outline (Target: Top Conference/Journal)

```
1. Introduction
   - Clinical motivation (stable vs. crashing patient)
   - Limitations of fixed temporal windows
   - Our hypothesis: adaptive temporal semantics

2. Related Work
   - EHR representation learning
   - Temporal modeling in healthcare
   - Attention mechanisms for EHR
   - LLMs for clinical prediction

3. Method: Adaptive Temporal Semantics Learning
   - Problem formulation
   - Adaptive temporal kernel
   - Patient-conditioned bandwidth
   - Task-conditioned adaptation
   - Architecture details

4. Experiments
   - Setup: MIMIC-IV, 4 tasks, baselines
   - RQ1: Fixed window comparison
   - RQ2: Patient-dependent analysis
   - RQ3: Task-dependent analysis
   - Ablation study
   - Temporal behavior evaluation

5. Discussion
   - When does adaptation help most?
   - Limitations and future work
   - Clinical implications

6. Conclusion
```

---

## 10. Open Questions

1. How to handle the cold-start problem (new patient with no history)?
2. Should σ_t be continuous or discrete (multi-scale)?
3. How to ensure σ_t doesn't collapse to trivial solutions?
4. Can we extend to multi-modal EHR (text + time series)?
5. How to validate σ_t against clinician judgments?

---

## 11. References

- GRU-D: Che et al., "Recurrent Neural Networks for Multivariate Timeseries with Missing Values", 2018
- RETAIN: Choi et al., "RETAIN: An Interpretable Predictive Model for Healthcare", NeurIPS 2016
- LLM4EHR: Zhu et al., "Prompting LLMs for Zero-Shot Clinical Prediction", 2024
- Transformer-EHR: Various surveys and benchmarks
