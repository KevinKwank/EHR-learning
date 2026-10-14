# Research Ideas Index

This directory contains our research ideas, with a focus on **Adaptive Temporal Semantics Learning** for EHR representation.

## Active Ideas

| Idea | Status | Document |
|------|--------|----------|
| Adaptive Temporal Semantics Learning (ATSL) | 🔬 Active | [adaptive_temporal_semantics.md](adaptive_temporal_semantics.md) |
| Research Questions (RQ1-RQ3) | 📝 Draft | [research_questions.md](research_questions.md) |
| Method Design | 📝 Draft | [method_design.md](method_design.md) |
| Experiment Design | 📝 Draft | [experiment_design.md](experiment_design.md) |

## Core Idea Summary

**Title**: Task- and Patient-Conditioned Temporal Representation Learning for Asynchronous EHR

**One-liner**: The model learns *how far back to look* based on the current patient state and the prediction task, instead of using a fixed temporal window.

**Key Formula**:
```
K(Δt) = exp(-Δt² / (2σ_t²))
where σ_t = f(patient_state, task)
```

## Idea Evolution

```
Phase 1: Observation
  "Different patients seem to need different amounts of history"

Phase 2: Hypothesis
  "Clinical temporal semantics is not fixed; it varies with patient state and task"

Phase 3: Method
  "Adaptive Temporal Kernel with learned bandwidth σ_t"

Phase 4: Validation
  "Three research questions (RQ1-RQ3) with corresponding experiments"
```

## How to Contribute

1. Add new idea files to this directory
2. Update this index
3. Link related papers from `../papers/`
4. Update the main [README](../README.md) if the idea becomes a main direction
