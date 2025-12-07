# Biomarker Integration Strategy

## Overview

This document outlines how biomarker data (cytokines, chemokines, etc.) can be integrated into the existing survival analysis pipeline if such data becomes available. This is based on the CAR-T biomarker paper approach and demonstrates the pipeline's flexibility for future enhancements.

---

## Biomarker Mapping Table

### Cytokines and Chemokines from CAR-T Literature

Based on biomarker studies in CAR-T and bispecific antibody literature, the following biomarkers are relevant for CRS prediction:

| Biomarker | Type | Unit | Integration Method | Preprocessing | Expected Relationship with CRS |
|-----------|------|------|-------------------|---------------|-------------------------------|
| **IL-6** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL6+1) | ↑ IL-6 → ↑ CRS risk (strong) |
| **IL-7** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL7+1) | ↑ IL-7 → ↑ CRS risk (moderate) |
| **IL-21** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL21+1) | ↑ IL-21 → ↑ CRS risk (moderate) |
| **IL-2** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL2+1) | ↑ IL-2 → ↑ CRS risk (moderate) |
| **IL-15** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL15+1) | ↑ IL-15 → ↑ CRS risk (moderate) |
| **IL-10** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IL10+1) | ↑ IL-10 → ↑ CRS severity |
| **IFN-γ** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(IFNg+1) | ↑ IFN-γ → ↑ CRS risk (strong) |
| **TNF-α** | Cytokine | pg/mL | Continuous lab value | Log-transform: log10(TNFa+1) | ↑ TNF-α → ↑ CRS risk (strong) |
| **CCL17** | Chemokine | pg/mL | Continuous lab value | Log-transform: log10(CCL17+1) | ↑ CCL17 → ↑ CRS risk |
| **CCL13** | Chemokine | pg/mL | Continuous lab value | Log-transform: log10(CCL13+1) | ↑ CCL13 → ↑ CRS risk |
| **CCL22** | Chemokine | pg/mL | Continuous lab value | Log-transform: log10(CCL22+1) | ↑ CCL22 → ↑ CRS risk |
| **Ferritin** | Acute phase | ng/mL | Continuous lab value | Log-transform: log10(ferritin+1) | ↑ Ferritin → ↑ CRS severity (strong) |
| **CRP** | Acute phase | mg/L | Continuous lab value | Log-transform: log10(CRP+1) | ↑ CRP → ↑ CRS severity (strong) |
| **TGF-β1** | Growth factor | pg/mL | Continuous lab value | Normalization: z-score | ↓ TGF-β1 → ↑ CRS risk (protective) |
| **MCP-1** | Chemokine | pg/mL | Continuous lab value | Log-transform: log10(MCP1+1) | ↑ MCP-1 → ↑ CRS risk |

---

## Integration Approach

### Conceptual Pipeline Enhancement

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED PIPELINE WITH BIOMARKERS              │
└─────────────────────────────────────────────────────────────────┘

CURRENT FEATURES                    NEW BIOMARKER FEATURES
─────────────────                   ──────────────────────
• Demographics                      • Baseline Cytokines
  - Age                               - IL-6 (log)
  - Weight                            - IL-7 (log)
  - Sex                               - IFN-γ (log)
                                      - TNF-α (log)
• Clinical                          
  - Hospitalization                 • Peak Cytokines (Day 1-3)
  - Life-threatening                  - IL-6 peak
  - Serious                           - Ferritin peak
                                      - CRP peak
• Drug Exposure                     
  - Total drugs                     • Chemokines
  - Polypharmacy                      - CCL17 (log)
  - Concomitant                       - CCL13 (log)
                                      - CCL22 (log)
                ↓                                ↓
        ┌───────────────────────────────────────┐
        │   COMBINED FEATURE MATRIX             │
        │   (Demographics + Clinical + Biomarkers)│
        └───────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │   COX PROPORTIONAL HAZARDS MODEL      │
        │   + Feature Selection                  │
        │   + SHAP Interpretation                │
        └───────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────┐
        │   ENHANCED CRS RISK PREDICTION        │
        │   (Improved C-index expected)          │
        └───────────────────────────────────────┘
```

---

## Implementation Details

### 1. Data Structure

**Biomarker data would be added as additional columns:**

```python
# Current data structure
df = pd.DataFrame({
    'patient_id': [...],
    'patient_age': [...],
    'patient_weight': [...],
    'has_crs': [...],
    'time_to_event': [...]
})

# Enhanced with biomarkers
df_enhanced = pd.DataFrame({
    # Existing features
    'patient_id': [...],
    'patient_age': [...],
    'patient_weight': [...],
    
    # NEW: Baseline biomarkers (pre-treatment)
    'IL6_baseline': [...],       # pg/mL
    'IL7_baseline': [...],       # pg/mL
    'IFNg_baseline': [...],      # pg/mL
    'TNFa_baseline': [...],      # pg/mL
    'ferritin_baseline': [...],  # ng/mL
    'CRP_baseline': [...],       # mg/L
    
    # NEW: Peak biomarkers (Day 1-3)
    'IL6_peak': [...],
    'ferritin_peak': [...],
    
    # Outcome
    'has_crs': [...],
    'time_to_event': [...]
})
```

### 2. Preprocessing Steps

**Same as other continuous variables, following existing pipeline:**

```python
def preprocess_biomarkers(df):
    """
    Preprocess biomarker values using the same approach as other numeric features.
    
    This function would be added to the existing pipeline.
    """
    biomarkers = ['IL6_baseline', 'IL7_baseline', 'IFNg_baseline', 
                  'TNFa_baseline', 'ferritin_baseline', 'CRP_baseline']
    
    for biomarker in biomarkers:
        if biomarker in df.columns:
            # Step 1: Log-transformation (handle skewed distributions)
            df[f'{biomarker}_log'] = np.log10(df[biomarker] + 1)
            
            # Step 2: Z-score normalization
            mean_val = df[f'{biomarker}_log'].mean()
            std_val = df[f'{biomarker}_log'].std()
            df[f'{biomarker}_zscore'] = (df[f'{biomarker}_log'] - mean_val) / std_val
            
            # Step 3: Create binary threshold (for interpretation)
            # Example: IL-6 > 5 pg/mL is clinically significant
            thresholds = {
                'IL6_baseline': 5,
                'ferritin_baseline': 500,
                'CRP_baseline': 10
            }
            if biomarker in thresholds:
                df[f'{biomarker}_high'] = (df[biomarker] > thresholds[biomarker]).astype(int)
    
    return df

# Example output for Patient A:
# IL6_baseline = 12.5 pg/mL
# IL6_baseline_log = 1.097
# IL6_baseline_zscore = 0.68
# IL6_baseline_high = 1 (yes, >5 pg/mL)
```

### 3. Feature Selection with Biomarkers

**Biomarkers would be included in the same ensemble feature selection:**

```python
# Extended feature list
features_extended = [
    # Existing features
    'patient_age', 'patient_weight', 'total_drugs',
    'concomitant_drugs', 'polypharmacy',
    'is_lifethreatening', 'is_hospitalization',
    
    # NEW: Biomarker features
    'IL6_baseline_zscore',
    'IL7_baseline_zscore',
    'IFNg_baseline_zscore',
    'ferritin_baseline_zscore',
    'CRP_baseline_zscore',
    'CCL17_baseline_zscore'
]

# Run the same feature selection methods
# 1. F-test
# 2. Mutual Information
# 3. Random Forest
# (code remains identical, just with extended feature list)
```

### 4. Cox Model with Biomarkers

**Cox model would automatically include biomarkers:**

```python
# Fit Cox model (same code as before)
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df[features_extended + ['time_adjusted', 'event_occurred']], 
        duration_col='time_adjusted', 
        event_col='event_occurred')

# Expected output (example):
# Feature                    HR      95% CI          p-value
# ─────────────────────────────────────────────────────────
# patient_weight            0.992   [0.985-1.000]   0.037*
# IL6_baseline_zscore       1.450   [1.320-1.595]   <0.001***
# ferritin_baseline_zscore  1.280   [1.150-1.425]   <0.001***
# IFNg_baseline_zscore      1.180   [1.050-1.325]   0.006**
# patient_age               0.995   [0.984-1.006]   0.347
```

**Interpretation for IL-6:**
```
HR = 1.450 means:
- For every 1 standard deviation increase in log(IL-6), CRS risk increases by 45%
- Example: Patient with IL-6 = 20 pg/mL vs 5 pg/mL
  → log10(20) - log10(5) = 0.60 difference
  → If this equals 1 SD, then 45% higher risk
```

---

## Expected Performance Improvement

### Predicted Model Enhancement

| Metric | Current Model | With Biomarkers (Expected) |
|--------|--------------|---------------------------|
| **C-index** | 0.5796 | **0.75-0.85** (substantial improvement) |
| **Top Risk Factor** | Weight (HR=0.992) | **IL-6 (HR~1.4-1.8)** |
| **Sensitivity** | Moderate | **High** (better early detection) |
| **Specificity** | Moderate | **Moderate-High** |

**Rationale:**
- CAR-T studies show baseline IL-6, ferritin, and IFN-γ are strong CRS predictors
- C-index improvements of 0.15-0.25 are typical when adding lab values
- Biomarkers capture biological mechanisms better than demographics alone

---

## Example Clinical Use Case

### Patient Risk Assessment with Biomarkers

**Scenario:** Assessing CRS risk before Epcoritamab dose 2

**Patient Profile:**
```python
patient = {
    # Demographics
    'age': 68,
    'weight': 92,  # kg
    
    # Clinical
    'polypharmacy': 1,
    'prior_crs': 0,  # no CRS on dose 1
    
    # NEW: Biomarkers (measured pre-dose 2)
    'IL6_baseline': 8.5,      # pg/mL (elevated)
    'ferritin_baseline': 650,  # ng/mL (elevated)
    'CRP_baseline': 15,        # mg/L (elevated)
    'IFNg_baseline': 3.2       # pg/mL (normal)
}
```

**Risk Calculation:**

```python
# Current model prediction (demographics only):
risk_demographic = predict_crs_risk(age=68, weight=92, polypharmacy=1)
# Output: 32% CRS risk

# Enhanced model prediction (with biomarkers):
risk_enhanced = predict_crs_risk_with_biomarkers(
    age=68, weight=92, polypharmacy=1,
    IL6=8.5, ferritin=650, CRP=15, IFNg=3.2
)
# Output: 58% CRS risk (HIGH RISK)

# Recommendation:
# - Prophylactic monitoring intensification
# - Consider prophylactic tocilizumab (off-label)
# - Inpatient administration mandatory
# - ICU bed reserved
```

**SHAP Interpretation (if available):**
```
Feature Contribution to Risk:
  IL6_baseline_zscore:     +0.15 (↑ pushes risk UP, major contributor)
  ferritin_baseline_zscore: +0.08 (↑ pushes risk UP)
  CRP_baseline_zscore:     +0.05 (↑ pushes risk UP)
  weight:                  -0.03 (↓ pushes risk DOWN, protective)
  age:                     +0.01 (minimal effect)
  
Interpretation: 
"Elevated IL-6 (8.5 pg/mL, 68% above normal) is the main driver 
of this patient's high CRS risk. Consider dose reduction or 
prophylactic intervention."
```

---

## Data Requirements

### If Biomarker Data Becomes Available

**Minimum Requirements:**
1. **Timing:** Baseline (pre-treatment) measurements
2. **Frequency:** At least IL-6, ferritin, CRP for all patients
3. **Quality:** Standardized assays, validated lab methods
4. **Format:** Numeric values in standard units (pg/mL, ng/mL)

**Optimal Dataset:**
```
patient_id | IL6 | IL7 | IL21 | IFNg | TNFa | ferritin | CRP | ... | has_crs
─────────────────────────────────────────────────────────────────────────────
001        | 3.2 | 1.5 | 0.8  | 2.1  | 4.5  | 320      | 5   | ... | 0
002        | 12.5| 4.2 | 2.1  | 8.3  | 15.2 | 680      | 18  | ... | 1
003        | 5.1 | 2.0 | 1.2  | 3.5  | 6.1  | 420      | 8   | ... | 0
...
```

---

## Validation Strategy

### If Biomarker Data is Obtained

**Step 1: Retrospective Validation**
- Collect biomarker data from past Epcoritamab patients
- Re-run pipeline with biomarkers included
- Compare C-index improvement

**Step 2: Prospective Validation**
- Enroll new patients with baseline biomarker measurement
- Predict CRS risk using enhanced model
- Validate predictions against actual CRS occurrence

**Step 3: Clinical Trial Integration**
- Use enhanced model for patient stratification
- Test prophylactic interventions in high biomarker-risk patients

---

## Code Integration Example

### How to Add Biomarkers to Existing Pipeline

**Minimal code changes required:**

```python
# In run_survival_analysis.py

# BEFORE (current):
features = ['patient_age', 'patient_weight', 'total_drugs',
           'concomitant_drugs', 'polypharmacy',
           'is_lifethreatening', 'is_hospitalization']

# AFTER (with biomarkers):
features = ['patient_age', 'patient_weight', 'total_drugs',
           'concomitant_drugs', 'polypharmacy',
           'is_lifethreatening', 'is_hospitalization',
           # NEW: Just add biomarker columns
           'IL6_baseline_zscore', 'ferritin_baseline_zscore',
           'CRP_baseline_zscore', 'IFNg_baseline_zscore']

# Rest of the code remains IDENTICAL
# Cox model, feature selection, risk stratification all work automatically
```

**That's the entire change!** The pipeline is designed to be flexible.

---

## Summary

### Key Points

1. ✅ **Pipeline is flexible:** Biomarkers can be added as new columns with minimal code changes
2. ✅ **Processing is standardized:** Biomarkers use same preprocessing as other continuous variables (log-transform, z-score)
3. ✅ **Expected improvement:** C-index from 0.58 → 0.75-0.85 with biomarkers
4. ✅ **Clinical relevance:** IL-6, ferritin, CRP are established CRS biomarkers
5. ✅ **Based on literature:** Approach mirrors successful CAR-T biomarker studies

### Next Steps if Biomarker Data Becomes Available

1. Obtain biomarker measurements for subset of patients
2. Add columns to existing data files
3. Re-run `run_pipeline()` with extended feature list
4. Validate improved predictions
5. Update clinical risk stratification thresholds

---

## References

### CAR-T and Bispecific Antibody Biomarker Studies

1. Hay, K. A., et al. (2017). "Kinetics and biomarkers of severe cytokine release syndrome after CD19 chimeric antigen receptor–modified T-cell therapy." *Blood*, 130(21), 2295-2306.

2. Teachey, D. T., et al. (2016). "Identification of predictive biomarkers for cytokine release syndrome after chimeric antigen receptor T-cell therapy for acute lymphoblastic leukemia." *Cancer Discovery*, 6(6), 664-679.

3. Thieblemont, C., et al. (2022). "Epcoritamab, a Novel, Subcutaneous CD3xCD20 Bispecific T-Cell-Engaging Antibody in Relapsed or Refractory Large B-Cell Lymphoma." *Journal of Clinical Oncology*, 40(21), 2238-2247.

4. Lee, D. W., et al. (2019). "ASTCT Consensus Grading for Cytokine Release Syndrome and Neurologic Toxicity Associated with Immune Effector Cells." *Biology of Blood and Marrow Transplantation*, 25(4), 625-638.

---

**Last Updated:** 2025-11-18  
**Version:** 1.0  
**Status:** Conceptual (awaiting biomarker data availability)

