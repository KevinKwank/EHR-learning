# Dataset Guide: MIMIC-IV

> Quick reference for using MIMIC-IV in experiments

---

## Overview

| Field | Value |
|-------|-------|
| Full name | Medical Information Mart for Intensive Care IV |
| Version | v2.2 (latest as of 2024) |
| Patients | ~500,000 unique patients |
| ICU stays | ~70,000 ICU admissions |
| Time range | 2008-2019 |
| Hospital | Beth Israel Deaconess Medical Center (Boston, MA) |
| Access | PhysioNet (requires credentialed access) |

---

## Core Tables

### patients
- `subject_id`: Unique patient identifier
- `gender`: Patient gender
- `anchor_year`: Shifted anchor year for de-identification

### admissions
- `hadm_id`: Hospital admission ID
- `subject_id`: Patient ID
- `admittime`: Admission time
- `dischtime`: Discharge time
- `admission_type`: Elective, emergency, urgent
- `hospital_expire_flag`: In-hospital mortality

### icustays
- `stay_id`: ICU stay ID
- `subject_id`: Patient ID
- `hadm_id`: Hospital admission ID
- `intime`: ICU admission time
- `outtime`: ICU discharge time
- `los`: Length of stay (days)

### chartevents (ICU measurements)
- `stay_id`, `charttime`, `itemid`, `value`, `valuenum`
- Vital signs: HR, BP, SpO2, Temp, RR
- **Key itemids**:
  - 220045: Heart Rate
  - 220179: Non-invasive BP systolic
  - 220180: Non-invasive BP diastolic
  - 220277: Oxygen saturation (SpO2)

### labevents (Laboratory results)
- `hadm_id`, `charttime`, `itemid`, `value`, `valuenum`
- Labs: WBC, creatinine, lactate, bilirubin, etc.
- **Key itemids**:
  - 50818: Bicarbonate
  - 50912: Creatinine
  - 50983: Sodium
  - 51221: Hematocrit
  - 51265: Platelet count

### prescriptions (Medications)
- `stay_id`, `starttime`, `stoptime`, `drug`, `route`
- Key medications: vasopressors, antibiotics, sedatives

### procedures
- `stay_id`, `starttime`, `itemid`
- Procedures: mechanical ventilation, dialysis, etc.

---

## Data Preprocessing Pipeline

### Step 1: Extract ICU Stays
```sql
SELECT stay_id, subject_id, intime, outtime, los
FROM icustays
WHERE los > 1  -- at least 24 hours
```

### Step 2: Create Prediction Windows
```python
# Create prediction windows every 6 hours
for stay in icu_stays:
    for t in range(6, stay.los * 24, 6):
        create_prediction_window(stay, t)
```

### Step 3: Extract Features per Window
```python
# For each prediction window [t-24h, t]:
# - Vital signs: mean, std, min, max, count
# - Labs: mean, std, min, max, count
# - Medications: binary flags (given/not given)
# - Demographics: age, gender
```

### Step 4: Handle Missing Data
```python
# Create observation masks
mask = 1 if observed, 0 if missing

# Normalize timestamps to hours since ICU admission
timestamps = (timestamps - admission_time) / 3600  # hours
```

### Step 5: Create Labels
```python
# Decompensation: vasopressor initiation within next 6h
# Mortality: death within next 48h
# Phenotyping: disease category from ICD codes
# LOS: total LOS > 7 days
```

---

## Access Requirements

1. Create account on PhysioNet: https://physionet.org/
2. Complete CITI Data Training (researcher credentialing)
3. Sign MIMIC-IV Data Use Agreement
4. Access via: `pip install physionet` or direct download

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Duplicate rows | Use `DISTINCT` or deduplicate by timestamp |
| Out-of-range values | Apply clinical range filters (e.g., HR 20-250) |
| Time zone issues | MIMIC-IV uses EST; convert carefully |
| Missing stay_id | Some labevents lack `stay_id`; join via `hadm_id` |
| De-identification shifts | Dates are shifted; use relative time from admission |

---

## Example Queries

### Get all vitals for a stay
```sql
SELECT itemid, charttime, valuenum
FROM chartevents
WHERE stay_id = 'XXX'
  AND itemid IN (220045, 220179, 220180, 220277)
ORDER BY charttime
```

### Get SOFA score components
```sql
-- Respiration: PaO2/FiO2
-- Coagulation: Platelets
-- Liver: Bilirubin
-- Cardiovascular: MAP + vasopressor use
-- CNS: GCS
-- Renal: Creatinine + urine output
```

---

## Citation

> If using MIMIC-IV, cite:
> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
