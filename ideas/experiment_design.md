# Experiment Design: Paper-Level Plan

> **Comprehensive experiment plan for the ATSL paper**

---

## 1. Dataset

### Primary: MIMIC-IV

| Characteristic | Value |
|----------------|-------|
| Patients | ~500,000+ |
| ICU stays | ~70,000+ |
| Time range | 2008-2019 |
| Data types | Structured (vitals, labs, medications, procedures) + Notes |

**Preprocessing**:
- Extract ICU stays with > 24h duration
- Create regular prediction windows (every 6h)
- Handle missing data with GRU-D-style masks
- Normalize timestamps to hours since ICU admission

### Optional Extensions

| Dataset | Purpose |
|---------|---------|
| MIMIC-CXR | Multimodal (imaging + EHR) |
| PhysioNet 2012 | Generalization to different hospital |
| eICU | Multi-center validation |

---

## 2. Tasks

### Task 1: Decompensation Prediction (Acute)

**Definition**: Predict within next 6h whether the patient will experience:
- Vasopressor initiation
- Mechanical ventilation
- ICU transfer from floor

**Why**: Proves that short-term dynamics matter; tests RQ1.

**Label**: Binary (0/1)

### Task 2: Mortality Prediction (Longitudinal)

**Definition**: Predict in-hospital mortality within next 48h.

**Why**: Proves that long-term accumulation matters; tests RQ1 and RQ3.

**Label**: Binary (0/1)

### Task 3: Phenotyping (Pattern)

**Definition**: Predict the presence of specific disease categories (e.g., sepsis, AKI, respiratory failure).

**Why**: Proves that disease trajectory patterns matter; tests RQ3.

**Label**: Multi-label

### Task 4: Length of Stay (Trajectory)

**Definition**: Predict whether LOS will exceed 7 days.

**Why**: Proves overall trajectory understanding; tests RQ1.

**Label**: Binary (0/1)

---

## 3. Baselines

### Category 1: Fixed Window Models

| Model | Window | Description |
|-------|--------|-------------|
| LSTM-1h | 1h | LSTM with 1h fixed window |
| LSTM-3h | 3h | LSTM with 3h fixed window |
| LSTM-5h | 5h | LSTM with 5h fixed window |
| LSTM-12h | 12h | LSTM with 12h fixed window |
| LSTM-24h | 24h | LSTM with 24h fixed window |

### Category 2: Existing EHR Models

| Model | Description |
|-------|-------------|
| GRU-D | Handles missing data with decay |
| RETAIN | Interpretable attention |
| Transformer-EHR | Self-attention over events |
| BEHRT | BERT-style pre-training |

### Category 3: LLM-Guided Models

| Model | Description |
|-------|-------------|
| LLM4EHR | Zero-shot LLM prediction |
| LLM4EHR-finetuned | Fine-tuned LLM |

### Category 4: Multi-Scale Models

| Model | Description |
|-------|-------------|
| Multi-scale LSTM | Multiple fixed windows combined |
| Multi-scale Attention | Attention over multiple scales |

---

## 4. Ablation Study

### Ablation Pipeline

```
A1: Fixed window (5h)
    └── Simplest baseline

A2: Multi-scale (1h + 3h + 5h + 12h + 24h)
    └── Multiple fixed windows, equal weights

A3: Patient-adaptive (σ_t = f(patient_state))
    └── Adapts to patient but not task

A4: Task-adaptive (σ_t = f(task))
    └── Adapts to task but not patient

A5: Full model (σ_t = f(patient_state, task))
    └── Full adaptation
```

### Key Comparisons

| Comparison | What it shows |
|------------|---------------|
| A1 vs. A2 | Does multi-scale help? |
| A2 vs. A3 | Does patient adaptation help? |
| A2 vs. A4 | Does task adaptation help? |
| A3 vs. A5 | Does task conditioning add value? |
| A4 vs. A5 | Does patient conditioning add value? |
| A1 vs. A5 | Overall improvement from full model |

---

## 5. Evaluation Metrics

### Primary Metrics

| Metric | Task | Description |
|--------|------|-------------|
| AUROC | All | Area under ROC curve |
| AUPRC | All | Area under precision-recall curve |
| F1-score | All | Harmonic mean of precision and recall |

### Temporal Behavior Metrics

| Metric | Description | How to compute |
|--------|-------------|----------------|
| Event order sensitivity | Does prediction change when event order changes? | Shuffle events, measure prediction difference |
| Temporal perturbation robustness | How sensitive is the model to timestamp noise? | Add Gaussian noise to timestamps, measure AUROC drop |
| σ_t distribution | What temporal scales does the model learn? | Histogram of σ_t values |
| σ_t vs. SOFA correlation | Does σ_t track clinical acuity? | Pearson correlation |

### Interpretability Metrics

| Metric | Description |
|--------|-------------|
| Attention weight analysis | Which events receive highest attention? |
| σ_t visualization | How does σ_t vary over the patient timeline? |
| Clinician agreement | Do clinicians agree with the model's temporal focus? |

---

## 6. Detailed Experiment Plans

### Experiment 1: RQ1 — Fixed Window Limitation

**Setup**:
- Train LSTM with windows: 1h, 3h, 5h, 12h, 24h
- Evaluate on all 4 tasks

**Analysis**:
1. Plot AUROC vs. window size for each task
2. Identify optimal window per task
3. Show that no single window is best for all tasks

**Expected Result**:
```
Task            | Best Window | AUROC
----------------|-------------|-------
Decompensation  | 1-3h        | 0.85
Mortality       | 12-24h      | 0.82
Phenotyping     | 12-24h      | 0.78
LOS             | 6-12h       | 0.75
```

### Experiment 2: RQ2 — Patient-Dependent Relevance

**Setup**:
- Train full ATSL model
- Extract σ_t values for each patient
- Compute SOFA change rate, vital instability

**Analysis**:
1. Correlation: σ_t vs. SOFA change rate
2. Visualization: Patient timeline with σ_t
3. Group analysis: σ_t distribution for stable vs. unstable patients

**Expected Result**:
```
Correlation: σ_t vs. SOFA change rate = -0.6 to -0.8
(more acute → smaller σ_t → focus on recent)
```

### Experiment 3: RQ3 — Task-Conditioned Representation

**Setup**:
- Train task-conditioned model
- Extract σ_t distributions per task

**Analysis**:
1. Plot σ_t distribution for each task
2. Statistical test: Are distributions different? (KS test)
3. Compare task-conditioned vs. task-specific models

**Expected Result**:
```
Task            | Mean σ_t | Std σ_t
----------------|----------|--------
Decompensation  | 2.5h     | 1.2h
Mortality       | 10.2h    | 5.8h
Phenotyping     | 15.3h    | 6.1h
LOS             | 7.8h     | 3.4h
```

### Experiment 4: Temporal Behavior Analysis

**Setup**:
- Take trained model
- Create test cases with shuffled event order

**Test Cases**:
```
Case 1: Original
  vasopressor → hypotension → lactate rise

Case 2: Shuffled
  hypotension → lactate rise → vasopressor

Case 3: Reversed
  lactate rise → hypotension → vasopressor
```

**Analysis**:
1. Measure prediction difference between original and shuffled
2. Visualize attention weights for each case
3. Show that the model changes evidence selection based on order

### Experiment 5: Ablation Study

**Setup**:
- Train all 5 ablation variants (A1-A5)
- Evaluate on all 4 tasks

**Analysis**:
1. Performance comparison table
2. Show incremental improvement from each component
3. Identify which component contributes most

---

## 7. Results Visualization Plan

### Figure 1: Fixed Window Comparison
- X-axis: Window size (1h, 3h, 5h, 12h, 24h)
- Y-axis: AUROC
- Lines: One per task

### Figure 2: σ_t Distribution by Task
- X-axis: σ_t (hours)
- Y-axis: Density
- Panels: One per task

### Figure 3: Patient Timeline with σ_t
- X-axis: Time (hours)
- Y-axis: Clinical indicators (SOFA, HR, BP)
- Overlay: σ_t values

### Figure 4: Ablation Results
- X-axis: Ablation variant (A1-A5)
- Y-axis: AUROC
- Grouped by task

### Figure 5: Event Order Sensitivity
- Heatmap: Attention weights for original vs. shuffled
- Shows which events the model focuses on

---

## 8. Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Data preparation | 2 weeks | MIMIC-IV extraction, preprocessing |
| Baseline implementation | 2 weeks | Fixed window models, GRU-D, RETAIN |
| ATSL implementation | 3 weeks | Adaptive kernel, bandwidth network |
| Experiments | 3 weeks | RQ1-RQ3, ablation, temporal analysis |
| Analysis & visualization | 2 weeks | Figures, tables, statistics |
| Writing | 4 weeks | Paper draft, revision |
| **Total** | **16 weeks** | |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| σ_t collapses to trivial values | Regularization, curriculum learning |
| Overfitting to MIMIC-IV | Validate on eICU or PhysioNet |
| Computational cost | Use efficient kernel implementation |
| Unclear clinical relevance | Collaborate with clinicians for validation |
