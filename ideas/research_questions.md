# Research Questions: Adaptive Temporal Semantics

> **Detailed formulation of RQ1, RQ2, RQ3**

---

## RQ1: Does Fixed Temporal Granularity Limit EHR Representation?

### Motivation

Every EHR model must decide: **how much history to use?**

Current practice: Choose a fixed window (1h, 6h, 12h, 24h) based on intuition or grid search.

**Problem**: This choice is:
- Arbitrary
- Not patient-specific
- Not task-specific
- Not data-driven

### Hypothesis

> No single fixed window size is optimal across all patients, clinical states, and prediction tasks.

### Experiment Design

**Setup**:
- Train the same model architecture with different fixed windows: 1h, 3h, 5h, 12h, 24h
- Evaluate on all 4 tasks: mortality, decompensation, phenotyping, LOS
- Use MIMIC-IV dataset

**Analysis**:
1. For each task, plot AUROC vs. window size
2. Check if the optimal window varies by task
3. Check if the optimal window varies by patient subgroup (stable vs. unstable)

**Expected Result**:
```
Task            | Best Window | Why
----------------|-------------|--------------------------
Decompensation  | 1-3h        | Acute dynamics matter
Mortality       | 12-24h      | Long-term burden matters
Phenotyping     | 12-24h      | Disease trajectory patterns
LOS             | 6-12h       | Overall trajectory
```

### Success Criteria

- [ ] Show that different tasks have different optimal windows
- [ ] Show that no single window is best for all tasks
- [ ] Show that the optimal window varies by patient state

---

## RQ2: Is Temporal Relevance Patient-Dependent?

### Motivation

A stable patient and a crashing patient have **different information needs**:
- Stable: Long history shows trends
- Crashing: Recent events are critical

### Hypothesis

> The learned temporal receptive field (σ_t) correlates with clinical state:
> - SOFA score change rate
> - Vital sign instability
> - Medication escalation

### Experiment Design

**Setup**:
- Train the full ATSL model
- Extract learned σ_t values for each patient at each prediction time
- Compute clinical state indicators:
  - SOFA change rate: ΔSOFA / Δt
  - Vital instability: variance of HR, BP, SpO2 in recent window
  - Medication escalation: new vasopressors, antibiotics, etc.

**Analysis**:
1. Correlation: σ_t vs. SOFA change rate (Pearson/Spearman)
2. Correlation: σ_t vs. vital instability
3. Visualization: Patient timeline with σ_t values

**Expected Visualization**:
```
Patient Timeline:
0h        20h       40h       60h
stable    stable    shock     recovery
σ_t=12h   σ_t=12h   σ_t=2h    σ_t=6h

Correlation: σ_t ∝ 1 / (SOFA change rate)
             (higher acuity → smaller σ_t → focus on recent)
```

### Success Criteria

- [ ] Show significant correlation between σ_t and clinical state
- [ ] Show that σ_t decreases during acute deterioration
- [ ] Show that σ_t increases during stable periods
- [ ] Validate with clinician assessment (if possible)

---

## RQ3: Should Downstream Tasks Reshape Temporal Representation?

### Motivation

Different clinical questions need different temporal evidence:
- "Will this patient die?" → long-term burden + acute events
- "Will this patient crash soon?" → recent rapid dynamics
- "What disease does this patient have?" → long-term patterns

### Hypothesis

> Different tasks should induce different temporal representations.
> A task-conditioned model can learn to adjust its temporal receptive field.

### Experiment Design

**Setup**:
- Train separate models for each task (task-specific)
- Train a shared model with task conditioning (our approach)
- Compare: task-specific vs. task-conditioned

**Analysis**:
1. For each task, visualize the distribution of learned σ_t
2. Check if σ_t distributions differ significantly across tasks
3. Compare task-conditioned model vs. task-specific models

**Expected Result**:
```
Task            | Expected σ_t Pattern
----------------|----------------------------------
Decompensation  | Short (1-3h), tight distribution
Mortality       | Bimodal: short (acute) + long (burden)
Phenotyping     | Long (12-24h), wide distribution
LOS             | Medium (6-12h), moderate distribution
```

### Success Criteria

- [ ] Show that σ_t distributions differ across tasks
- [ ] Show that task-conditioned model matches or exceeds task-specific models
- [ ] Show that the model learns to "switch" temporal scales based on task

---

## Summary Table

| RQ | Question | Key Metric | Expected Outcome |
|----|----------|------------|------------------|
| RQ1 | Fixed window limitation | AUROC vs. window size | No universal optimal window |
| RQ2 | Patient-dependent relevance | σ_t vs. SOFA correlation | Significant negative correlation |
| RQ3 | Task-conditioned representation | σ_t distribution by task | Distinct distributions per task |

---

## Discussion Prompts

1. If RQ1 is confirmed, what does this mean for current EHR model design?
2. If RQ2 is confirmed, can σ_t be used as a clinical acuity indicator?
3. If RQ3 is confirmed, should we always use task-conditioned models?
4. How do these findings translate to clinical practice?
