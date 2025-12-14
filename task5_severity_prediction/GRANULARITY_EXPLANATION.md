# Granularity and Interaction Features: Explanation

This document explains the granular feature engineering approach and interaction terms used in the CRS mortality prediction model.

---

## Granularity Concept

Rather than using only aggregate counts, we create **granular features** that capture specific clinical thresholds and meaningful cutoffs.

---

## Granular Features vs. Aggregate Counts

### Not Granular (Aggregate Only)
- "Number of drugs" (raw count)
- "Number of comorbidities" (raw count)
- "Age" (continuous only)

### Granular (Specific Cutoffs + Interactions)
- Age >65, >70, >75 (specific thresholds)
- BMI categories: obese (>30), underweight (<18.5)
- Specific drug classes: steroids, antibiotics, chemo
- Drug combinations: steroid + antibiotic
- Cancer stage: Stage I/II vs. Stage III/IV

---

## Granular Feature Categories

### 1. Age Stratification

**Continuous**: `age_years`
**Binary Thresholds**:
- `age_gt_65` - Age > 65 years
- `age_gt_70` - Age > 70 years
- `age_gt_75` - Age > 75 years

**Why Granular?**
- Specific age thresholds (65, 70, 75) are clinically meaningful
- Risk increases non-linearly with age
- Thresholds help identify high-risk subgroups

### 2. BMI Stratification

**Continuous**: `bmi`
**Categorical Buckets**:
- `bmi_underweight` - BMI < 18.5 (anorexic-like)
- `bmi_normal` - BMI 18.5-25
- `bmi_overweight` - BMI 25-30
- `bmi_obese` - BMI > 30

**Why Granular?**
- Both extremes (underweight and obese) are risk factors
- Captures non-linear relationships
- Clinically interpretable categories

### 3. Polypharmacy Granularity

**Continuous**: `num_drugs`
**Binary/Categorical**:
- `polypharmacy` - >1 drug
- `high_polypharmacy` - >5 drugs
- `very_high_polypharmacy` - >10 drugs

**Why Granular?**
- Not just "many drugs" but specific thresholds
- Different risk levels for different polypharmacy levels

### 4. Drug Class Granularity

**Not**: "Number of drug classes"
**Instead**: Specific classes:
- `has_steroid` - Receiving steroids
- `has_antibiotic` - Receiving antibiotics
- `has_chemo` - Receiving chemotherapy
- `has_antiviral` - Receiving antivirals

**Why Granular?**
- Specific drug classes have different clinical meanings
- Allows identification of high-risk combinations

---

## Interaction Features

Interaction terms capture synergistic effects between features.

### Example: Steroid + Antibiotic

```python
# Not just individual flags:
has_steroid = 1
has_antibiotic = 1

# But also interaction:
steroid_plus_antibiotic = (has_steroid == 1) & (has_antibiotic == 1)
```

**Clinical Interpretation**:
- Steroid + Antibiotic combination suggests complex clinical situation
- May indicate infection requiring both immunosuppression and antimicrobial treatment
- Associated with 93.5% death rate in CRS patients (29/31 cases)

### Other Potential Interactions

- Age >70 + High Polypharmacy
- BMI Obese + Diabetes
- Infection AE + Comorbidity
- Cancer Stage III/IV + Chemo

---

## Visualization: Granularity and Interactions

```
┌─────────────────────────────────────────────────────────┐
│           GRANULAR FEATURE ENGINEERING                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Continuous Variables          Binary/Categorical        │
│  ────────────────              ───────────────────      │
│                                                          │
│  age_years ────────────────→  age_gt_65                 │
│                               age_gt_70                 │
│                               age_gt_75                 │
│                                                          │
│  num_drugs ────────────────→  polypharmacy              │
│                               high_polypharmacy         │
│                                                          │
│  bmi ──────────────────────→  bmi_obese                 │
│                               bmi_underweight           │
│                                                          │
│                                                          │
│  INTERACTION FEATURES                                    │
│  ────────────────────                                    │
│                                                          │
│  has_steroid ────┐                                        │
│                  ├──→ steroid_plus_antibiotic           │
│  has_antibiotic ─┘                                        │
│                                                          │
│  age_gt_70 ──────┐                                        │
│                  ├──→ high_risk_combination             │
│  high_polypharm ─┘                                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Why Granularity Matters

1. **Clinical Interpretability**: Specific thresholds (e.g., age >70) are easier to explain than continuous values
2. **Non-linear Relationships**: Captures threshold effects (e.g., BMI >30 = obese risk)
3. **Actionable Insights**: "Patients >70 with steroid+antibiotic" is more actionable than "high-risk score"
4. **Interaction Effects**: Identifies high-risk combinations (e.g., steroid+antibiotic = 93.5% death rate)

---

## Key Findings from Granular Analysis

- **Age >65**: 83.5% death rate vs. 67.4% for ≤65
- **Steroid + Antibiotic**: 93.5% death rate (29/31 patients)
- **Stage III-IV**: Higher death rate than Stage I-II
- **High Polypharmacy (>5 drugs)**: Associated with increased risk

---

## Summary

**Granularity** = Specific cutoffs and meaningful thresholds
**Interactions** = Synergistic effects between features

This approach provides:
- Clinically interpretable features
- Actionable risk stratification
- Clear identification of high-risk combinations
- Plain language explanations for clinicians

