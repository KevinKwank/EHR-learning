# Glossary: EHR Temporal Representation Learning

> Clinical and technical terms used across this repository

---

## Clinical Terms

| Term | Definition |
|------|------------|
| **EHR** | Electronic Health Record — digital patient health information |
| **ICU** | Intensive Care Unit — specialized hospital ward for critically ill patients |
| **SOFA** | Sequential Organ Failure Assessment — score measuring organ function (0-24, higher = worse) |
| **Acuity** | How sick a patient is; higher acuity = more unstable |
| **Decompensation** | Sudden worsening of a patient's clinical condition |
| **Vasopressor** | Medication that raises blood pressure (e.g., norepinephrine) |
| **Phenotyping** | Identifying disease patterns from clinical data |
| **LOS** | Length of Stay — how long a patient is hospitalized |
| **Mortality** | Death; in-hospital mortality = death during hospital stay |
| **Admission** | When a patient enters the hospital/ICU |
| **Triage** | Process of assessing patient priority on arrival |

## Technical Terms

| Term | Definition |
|------|------------|
| **Temporal encoding** | Representing time information in a model |
| **Fixed window** | Using a constant time interval (e.g., always look at last 5 hours) |
| **Adaptive kernel** | A learned function that weights events by temporal distance |
| **σ_t (sigma_t)** | The learned temporal bandwidth — how far back the model looks |
| **Receptive field** | The range of inputs that influence a prediction |
| **Irregular sampling** | Data points not equally spaced in time |
| **Missing data** | Values that should exist but weren't recorded |
| **GRU-D** | Gated Recurrent Unit with Decay — handles missing clinical data |
| **RETAIN** | Reverse Time Attention Model — interpretable EHR predictions |
| **AUROC** | Area Under the Receiver Operating Characteristic curve — prediction quality metric |
| **AUPRC** | Area Under the Precision-Recall curve — prediction quality metric |
| **Zero-shot** | Making predictions without task-specific training |
| **Fine-tuning** | Adapting a pre-trained model to a specific task |
| **Ablation study** | Removing components one at a time to measure each one's contribution |

## Abbreviations

| Abbr. | Full Form |
|-------|-----------|
| EHR | Electronic Health Record |
| ICU | Intensive Care Unit |
| TS | Time Series |
| DL | Deep Learning |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| LLM | Large Language Model |
| LOS | Length of Stay |
| SOFA | Sequential Organ Failure Assessment |
| HR | Heart Rate |
| BP | Blood Pressure |
| SpO2 | Oxygen Saturation |
| WBC | White Blood Cell count |
| CRP | C-Reactive Protein (inflammation marker) |
