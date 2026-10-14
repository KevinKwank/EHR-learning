
# Experiment Plan

## Dataset

Primary:
- MIMIC-IV
- MIMIC-CXR (if multimodal extension)

## Tasks

Clinical prediction:
- mortality
- decompensation
- length of stay
- phenotyping

## Baselines

1. Fixed window TS encoder
2. Existing EHR representation models
3. LLM-guided EHR representation methods
4. Multi-scale temporal models

## Ablations

A1 fixed window
A2 multi-scale without adaptation
A3 patient-conditioned adaptation
A4 task-conditioned adaptation
A5 full model

## Additional evaluation

Not only AUROC/AUPRC.

Need evaluate:
- learned temporal scale distribution;
- evidence localization;
- robustness to temporal perturbation;
- sensitivity to event delays.
