# Figures Index

This directory contains key figures used in group meeting presentations and paper explanations.

## Figure Catalog

### Core Hypothesis Figures

| File | Description | Used In |
|------|-------------|---------|
| `core_hypothesis.png` | Diagram showing patient trajectory → temporal encoder → task-conditioned learner | Main README, ideas/adaptive_temporal_semantics.md |
| `fixed_vs_adaptive.png` | Comparison of fixed window vs. adaptive kernel | ideas/adaptive_temporal_semantics.md |
| `patient_timeline.png` | Patient timeline showing σ_t variation | ideas/research_questions.md |

### Method Figures

| File | Description | Used In |
|------|-------------|---------|
| `atsl_architecture.png` | Full ATSL model architecture diagram | ideas/method_design.md |
| `adaptive_kernel.png` | Gaussian kernel with different σ_t values | ideas/method_design.md |
| `bandwidth_network.png` | Adaptive bandwidth network structure | ideas/method_design.md |

### Experiment Figures

| File | Description | Used In |
|------|-------------|---------|
| `experiment_pipeline.png` | Experiment pipeline and ablation design | ideas/experiment_design.md |
| `results_template.png` | Expected results visualization template | ideas/experiment_design.md |
| `sigma_distribution.png` | σ_t distribution across tasks (expected) | ideas/experiment_design.md |

### Paper Figures

| File | Description | Used In |
|------|-------------|---------|
| `gru_d_cell.png` | GRU-D cell structure | papers/gru_d/GRU_D_notes.md |
| `retain_architecture.png` | RETAIN two-level attention | papers/retain/RETAIN_notes.md |
| `llm4ehr_prompt.png` | LLM4EHR prompt template | papers/llm4ehr/LLM4EHR_full_notes.md |

## Figure Creation Guidelines

### For Group Meeting Presentations

1. **Use clear labels** — Every axis, curve, and annotation should be labeled
2. **Consistent color scheme** — Same colors for same concepts across figures
3. **Large fonts** — Minimum 14pt for readability in presentations
4. **Export formats** — PNG (for slides) + PDF (for paper)

### For Paper Submissions

1. **Vector format** — PDF or EPS for scalability
2. **Color-blind friendly** — Use patterns + colors, not colors alone
3. **Self-contained** — Each figure should be understandable without reading the caption
4. **Resolution** — Minimum 300 DPI for raster images

## Placeholder Images

The following placeholder files have been created for figure positions:

- `core_hypothesis.png` — Placeholder for core hypothesis diagram
- `adaptive_kernel.png` — Placeholder for adaptive kernel illustration
- `experiment_pipeline.png` — Placeholder for experiment pipeline
- `results_template.png` — Placeholder for expected results format

> **Note**: Replace these placeholders with actual figures as they are created.

## Figure Workflow

```
1. Sketch → Draw figure in draw.io / Figma / PowerPoint
2. Refine → Clean up, add labels, ensure readability
3. Export → Save as PNG (300 DPI) and PDF
4. Document → Add description to this README
5. Use → Reference in paper notes and presentations
```

## Tools

- **Diagrams**: draw.io, Figma, PowerPoint
- **Plots**: matplotlib, seaborn, ggplot2
- **Paper figures**: Adobe Illustrator, Inkscape
- **Slides**: PowerPoint, Google Slides
