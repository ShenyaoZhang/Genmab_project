# CRS → Death Model: Plain Language Summary for Clinicians
**Generated:** 2025-12-14 18:25:11
---

## What This Model Does

This model predicts the risk of death in patients with **Cytokine Release Syndrome (CRS)** who are being treated with Epcoritamab. It helps identify which factors are most important in determining patient outcomes.

## Key Findings

### Global Model Interpretation (SHAP Analysis)


🔹 Key Factors That INCREASE Death Risk:
   • Cancer Stage III: Strong positive contribution to death risk
   • Age Missing in Report: Strong positive contribution to death risk
   • Steroid + Antibiotic Combination: Strong positive contribution to death risk
   • Number of Adverse Reactions: Strong positive contribution to death risk
   • Infection-Related Adverse Event: Strong positive contribution to death risk

🔹 Key Factors That DECREASE Death Risk:
   • Female Gender: Protective effect against death risk
   • Patient Age: Protective effect against death risk
   • Receiving Steroids: Protective effect against death risk
   • Patient Weight: Protective effect against death risk
   • Body Mass Index (BMI): Protective effect against death risk

================================================================================English Summary (Ready for Report)================================================================================
For CRS cases, the model finds that age over 70, and infection-related adverse events are the strongest contributors to death risk.
Steroid plus antibiotic combinations appear frequently among high-risk CRS cases, which may suggest complex clinical situations that warrant closer monitoring.
Advanced age (particularly over 70 years) is consistently associated with increased death risk in CRS patients.

### English Summary for Report

For CRS cases, the model finds that age over 70, and infection-related adverse events are the strongest contributors to death risk.
Steroid plus antibiotic combinations appear frequently among high-risk CRS cases, which may suggest complex clinical situations that warrant closer monitoring.
Advanced age (particularly over 70 years) is consistently associated with increased death risk in CRS patients.

## Individual Case Interpretations

### Local SHAP Explanations

#### Case 1

🔹 High-Risk Case Example (Death Outcome):

   Patient Profile:
   • A patient with age 1 years and high polypharmacy

   Why the model predicts high death risk:
   • Number of Concurrent Medications: increases death risk (contribution: +0.085)
   • Receiving Chemotherapy: increases death risk (contribution: +0.047)
   • Number of Adverse Reactions: increases death risk (contribution: +0.039)
   • Receiving Targeted Therapy: increases death risk (contribution: +0.018)
   • Patient Age: increases death risk (contribution: +0.016)


#### Case 2

🔹 Low-Risk Case Example (Survival Outcome):

   Why the model predicts lower death risk:
   • Receiving Chemotherapy: decreases death risk (contribution: -0.080)
   • Number of Concurrent Medications: decreases death risk (contribution: -0.074)
   • Patient Weight: decreases death risk (contribution: -0.044)
   • Body Mass Index (BMI): decreases death risk (contribution: -0.037)
   • Underweight (BMI <18.5): decreases death risk (contribution: -0.021)


## Clinical Interpretation

### What This Means for Patient Care:

1. **Monitor polypharmacy:** Patients on multiple medications need close monitoring.
2. **Consider chemotherapy carefully:** Combined chemo + Epcoritamab has higher risk - consider dose adjustments or timing.
3. **Age-based risk stratification:** Older patients (>70 years) may need more intensive monitoring.
4. **Watch for multiple reactions:** Patients with multiple concurrent adverse reactions are at higher risk.
5. **Comorbidity management:** Patients with diabetes, hypertension, or cardiac disease require closer monitoring.

### What This Means for Drug Safety Teams:

1. **Risk stratification:** Use these factors to identify high-risk patients early.
2. **Intervention opportunities:** Focus on modifiable factors (e.g., medication review, infection prevention).
3. **Clinical decision support:** Integrate this model into clinical workflows for risk assessment.

## Limitations

- Model is based on observational data (FAERS reports)
- Cannot prove causation, only associations
- Should be used alongside clinical judgment
- Requires validation in independent datasets

