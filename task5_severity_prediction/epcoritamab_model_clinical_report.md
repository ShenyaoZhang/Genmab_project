# CRS → Death Model: Clinical Interpretation Report
**Generated:** 2025-12-07 13:36:25
---
## Executive Summary
This report presents a machine learning model trained specifically on **CRS (Cytokine Release Syndrome) patients** to predict death outcomes. The model uses structured clinical features and achieves:
- **PR-AUC:** 0.896
- **ROC-AUC:** 0.624
- **F1-Score:** 0.767
- **Best Model:** Random Forest
## Dataset
- **Total CRS patients:** 185
- **Deaths:** 150 (81.1%)
- **Survivors:** 35 (18.9%)

## Model Performance

The Random Forest model achieved the following performance:

- **ROC-AUC:** 0.624 (Area under the receiver operating characteristic curve)
- **PR-AUC:** 0.896 (Area under the precision-recall curve, important for imbalanced data)
- **F1-Score:** 0.767 (Balanced measure of precision and recall)
- **Accuracy:** 0.643
- **Precision:** 0.805
- **Recall:** 0.733

## Clinical Implications

1. **Risk Stratification:** This model can help identify CRS patients at highest risk of death.
2. **Monitoring Focus:** Variables identified as important should be closely monitored.
3. **Treatment Decisions:** Drug combinations and comorbidities should be considered in treatment planning.

## Limitations

1. **Sample Size:** Model trained on limited number of CRS cases.
2. **Observational Data:** FAERS data is observational and may have biases.
3. **External Validation:** Model should be validated on independent datasets.
4. **Clinical Context:** Model predictions should be interpreted in clinical context.
