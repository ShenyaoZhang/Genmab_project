# Biomarker Integration: Conceptual Design

This document illustrates how biomarkers (e.g., IL-6, CRP, ferritin) can be integrated into the CRS mortality prediction pipeline.

---

## Overview

Biomarkers are not currently available in FAERS data. This document provides a conceptual framework for integrating biomarker data when it becomes available from:
- Lab results
- Electronic health records (EHR)
- Clinical trial databases
- Integrated clinical datasets

---

## Biomarker Mapping Table

| Biomarker | Type | Typical Units | Clinical Significance | Integration Method |
|-----------|------|---------------|----------------------|-------------------|
| **IL-6** (Interleukin-6) | Continuous | pg/mL | Key cytokine in CRS; elevated levels indicate severity | Continuous feature; threshold flags (e.g., >40 pg/mL = severe) |
| **CRP** (C-Reactive Protein) | Continuous | mg/L | Inflammation marker; elevated in CRS | Continuous feature; threshold flags (e.g., >10 mg/L = high) |
| **Ferritin** | Continuous | ng/mL | Elevated in cytokine storm | Continuous feature; threshold flags (e.g., >500 ng/mL = high) |
| **D-dimer** | Continuous | μg/mL | Coagulation marker; elevated in severe CRS | Continuous feature; threshold flags |
| **Procalcitonin** | Continuous | ng/mL | Infection/sepsis marker | Continuous feature; threshold flags |
| **IL-10** (Interleukin-10) | Continuous | pg/mL | Anti-inflammatory cytokine | Continuous feature; threshold flags |
| **TNF-α** (Tumor Necrosis Factor-alpha) | Continuous | pg/mL | Pro-inflammatory cytokine | Continuous feature; threshold flags |

---

## Integration Architecture

### Current Pipeline Structure

```
FAERS Data → Feature Engineering → Model Training → Predictions
```

### Enhanced Pipeline with Biomarkers

```
FAERS Data ──┐
             ├→ Feature Engineering → Model Training → Predictions
Biomarker DB ─┘
```

### Integration Points

1. **Data Collection Stage** (`01_extract_data.py`)
   - Add biomarker extraction if available in data source
   - Merge biomarker data with FAERS records by patient ID

2. **Preprocessing Stage** (`03_preprocess_data.py`)
   - Add biomarker features to feature engineering pipeline
   - Handle missing biomarker data (imputation or flags)

3. **Feature Engineering** (`12_crs_model_training.py`)
   - Create biomarker-derived features (high/low flags, ratios)
   - Add biomarker interaction terms

---

## Conceptual Integration Example

See `add_future_biomarkers.py` for a working example of biomarker integration.

### Example Usage

```python
from add_future_biomarkers import add_biomarkers, create_biomarker_features

# Sample biomarker data (from lab results or EHR)
biomarker_data = {
    'IL6': [45.2, 12.5, 38.9, 28.3, 15.7],      # pg/mL
    'CRP': [15.8, 8.2, 12.5, 9.1, 6.3],         # mg/L
    'ferritin': [520, 180, 450, 320, 210]       # ng/mL
}

# Add raw biomarkers to dataset
df_with_biomarkers = add_biomarkers(df, biomarker_data)

# Create derived features (high/low flags, combined scores)
df_with_features = create_biomarker_features(df_with_biomarkers)

# Features created:
# - IL6_high (>40 pg/mL)
# - CRP_high (>10 mg/L)
# - ferritin_high (>500 ng/mL)
# - biomarker_severity_score (combined high flags)
```

---

## Biomarker Feature Engineering

### Continuous Features
- Raw biomarker values (e.g., `IL6`, `CRP`) kept as continuous variables
- No scaling needed for tree-based models (XGBoost, Random Forest)

### Derived Binary Features
- High/low threshold flags (e.g., `IL6_high`, `CRP_elevated`)
- Clinically meaningful cutoffs based on literature

### Interaction Features
- Combined biomarker scores (e.g., `biomarker_severity_score`)
- Ratios (e.g., `IL6_to_IL10_ratio`)

---

## Future Data Sources

The pipeline is designed to accept biomarker data from:

1. **Clinical Trials**: Structured lab results with patient IDs
2. **Electronic Health Records**: Lab values from hospital systems
3. **Integrated Databases**: Merged FAERS + clinical lab databases
4. **Research Studies**: Epcoritamab-specific biomarker studies

### Data Format Expected

```csv
safetyreportid,IL6,CRP,ferritin,d_dimer
12345678,45.2,15.8,520,0.8
12345679,12.5,8.2,180,0.3
...
```

---

## Clinical Relevance

Biomarkers are particularly important for CRS:

- **IL-6**: Directly related to CRS pathophysiology
- **CRP**: General inflammation marker, often elevated in CRS
- **Ferritin**: Hyperferritinemia associated with cytokine storm

Once integrated, these biomarkers are expected to be among the top predictive features for CRS-related mortality.

---

## Implementation Status

- ✅ **Conceptual framework**: Complete (see `add_future_biomarkers.py`)
- ✅ **Integration points**: Identified and documented
- ⏳ **Actual biomarker data**: Not yet available in FAERS
- ✅ **Pipeline readiness**: Code structure supports biomarker addition

---

## Summary

The pipeline architecture supports biomarker integration without major structural changes. When biomarker data becomes available:

1. Add biomarker columns to the dataset
2. Run existing feature engineering (automatically includes biomarkers)
3. Retrain model with enhanced feature set
4. Biomarkers will appear in feature importance rankings

No code modifications needed - the pipeline is designed for extensibility.

