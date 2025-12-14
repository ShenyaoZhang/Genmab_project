# CRS Model: Variable List and Data Sources

This document provides a comprehensive table of all variables used in the CRS mortality prediction model, their categories, descriptions, and availability in FAERS.

---

## Variable Inventory Table

| Variable Name | Category | Description | Available in FAERS | Data Source Field | Processing Method |
|--------------|----------|-------------|-------------------|-------------------|-------------------|
| **Demographics** |
| `age_years` | Demographic | Age at time of report (normalized to years) | Yes | `patient.patientonsetage` + `patientonsetageunit` | Standardized to years from various units (800=decade, 801=year, 802=month, etc.) |
| `age_gt_65` | Demographic | Age > 65 years (binary indicator) | Yes | Derived from `age_years` | Binary threshold |
| `age_gt_70` | Demographic | Age > 70 years (binary indicator) | Yes | Derived from `age_years` | Binary threshold |
| `age_gt_75` | Demographic | Age > 75 years (binary indicator) | Yes | Derived from `age_years` | Binary threshold |
| `patientsex` | Demographic | Patient sex (1=male, 2=female, 0=unknown) | Yes | `patient.patientsex` | Direct from FAERS |
| `sex_male` | Demographic | Male gender (binary indicator) | Yes | Derived from `patientsex` | Binary encoding |
| `sex_female` | Demographic | Female gender (binary indicator) | Yes | Derived from `patientsex` | Binary encoding |
| `patientweight` | Demographic | Patient weight in kg | Yes | `patient.patientweight` | Continuous variable (median imputation if missing) |
| `bmi` | Demographic | Body Mass Index (calculated from weight) | Yes | Calculated from `patientweight` | Weight/(height²), default height=1.65m |
| `bmi_obese` | Demographic | Obese (BMI >30) | Yes | Derived from `bmi` | Binary threshold |
| `bmi_overweight` | Demographic | Overweight (BMI 25-30) | Yes | Derived from `bmi` | Binary threshold |
| `bmi_underweight` | Demographic | Underweight (BMI <18.5) | Yes | Derived from `bmi` | Binary threshold |
| **Medications** |
| `num_drugs` | Medication | Number of concurrent medications | Yes | Count of `patient.drug` array | Direct count |
| `polypharmacy` | Medication | Multiple medications (>1 drug) | Yes | Derived from `num_drugs` | Binary threshold |
| `high_polypharmacy` | Medication | High polypharmacy (>5 drugs) | Yes | Derived from `num_drugs` | Binary threshold |
| `has_steroid` | Medication | Receiving steroid medications | Yes | `patient.drug.openfda.generic_name` | Keyword matching (PREDNISONE, DEXAMETHASONE, etc.) |
| `has_antibiotic` | Medication | Receiving antibiotic medications | Yes | `patient.drug.openfda.generic_name` | Keyword matching (CIPROFLOXACIN, VANCOMYCIN, etc.) |
| `has_antiviral` | Medication | Receiving antiviral medications | Yes | `patient.drug.openfda.generic_name` | Keyword matching (ACYCLOVIR, GANCICLOVIR, etc.) |
| `has_chemo` | Medication | Receiving chemotherapy | Yes | `patient.drug.openfda.generic_name` | Keyword matching (CYCLOPHOSPHAMIDE, DOXORUBICIN, etc.) |
| `has_targeted` | Medication | Receiving targeted therapy | Yes | `patient.drug.openfda.generic_name` | Keyword matching (RITUXIMAB, LENALIDOMIDE, etc.) |
| `has_antifungal` | Medication | Receiving antifungal medications | Yes | `patient.drug.openfda.generic_name` | Keyword matching (FLUCONAZOLE, VORICONAZOLE, etc.) |
| `steroid_plus_antibiotic` | Medication | Combination of steroid + antibiotic | Yes | Derived from `has_steroid` and `has_antibiotic` | Interaction term (has_steroid AND has_antibiotic) |
| **Adverse Events** |
| `num_reactions` | Adverse Event | Number of adverse reactions reported | Yes | Count of `patient.reaction` array | Direct count |
| `multiple_reactions` | Adverse Event | Multiple adverse reactions (>1) | Yes | Derived from `num_reactions` | Binary threshold |
| `has_infection_ae` | Adverse Event | Infection-related adverse event | Yes | `patient.reaction.reactionmeddrapt` | Keyword matching (INFECTION, SEPSIS, PNEUMONIA, etc.) |
| **Comorbidities** |
| `comorbidity_diabetes` | Comorbidity | History of diabetes | Yes | `patient.drug.drugindication` | Keyword matching from free-text |
| `comorbidity_hypertension` | Comorbidity | History of hypertension | Yes | `patient.drug.drugindication` | Keyword matching from free-text |
| `comorbidity_cardiac` | Comorbidity | History of cardiac disease | Yes | `patient.drug.drugindication` | Keyword matching from free-text |
| **Cancer Stage** |
| `cancer_stage_I` | Cancer Stage | DLBCL Stage I | Partial | `patient.drug.drugindication` (free-text) | Pattern matching from free-text (imperfect extraction) |
| `cancer_stage_II` | Cancer Stage | DLBCL Stage II | Partial | `patient.drug.drugindication` (free-text) | Pattern matching from free-text (imperfect extraction) |
| `cancer_stage_III` | Cancer Stage | DLBCL Stage III | Partial | `patient.drug.drugindication` (free-text) | Pattern matching from free-text (imperfect extraction) |
| `cancer_stage_IV` | Cancer Stage | DLBCL Stage IV | Partial | `patient.drug.drugindication` (free-text) | Pattern matching from free-text (imperfect extraction) |
| **Data Quality** |
| `age_missing` | Data Quality | Age missing indicator | Yes | Derived | Binary flag for missing age |
| `weight_missing` | Data Quality | Weight missing indicator | Yes | Derived | Binary flag for missing weight |

---

## Data Source Summary

- **Primary Data Source**: FAERS (FDA Adverse Event Reporting System) - United States
- **Data Access**: OpenFDA API (`https://api.fda.gov/drug/event.json`)
- **Time Period**: Reports collected from 2004-2024 (as available via API)

---

## Notes

1. **Cancer Stage Limitation**: DLBCL stage is not available as a structured variable in FAERS. Current extraction uses pattern matching from free-text fields, which may miss many cases. See `CANCER_STAGE_DOCUMENTATION.md` for details.

2. **Future Data Sources**: The pipeline is designed to accept structured stage data from future datasets or integrated clinical databases.

3. **Biomarker Data**: Currently not available in FAERS. See `BIOMARKER_INTEGRATION.md` for conceptual integration plan.

---

## Feature Categories Summary

- **Demographics**: 11 variables (age, sex, weight, BMI)
- **Medications**: 10 variables (polypharmacy, drug classes, combinations)
- **Adverse Events**: 3 variables (reaction counts, infection AE)
- **Comorbidities**: 3 variables (diabetes, hypertension, cardiac)
- **Cancer Stage**: 4 variables (Stage I-IV, imperfect extraction)
- **Data Quality**: 2 variables (missing indicators)

**Total**: 33 features used in CRS mortality prediction model

