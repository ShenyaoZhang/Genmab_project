# Task 5: Adverse Event Severity Prediction

Machine learning pipeline for predicting adverse event severity (death) using OpenFDA FAERS data.

---

## Overview

This pipeline performs end-to-end analysis from data extraction to model deployment:

1. **Data Extraction** - Collect FAERS reports for oncology drugs
2. **Data Inspection** - Quality checks and descriptive statistics
3. **Data Preprocessing** - Feature engineering and train-test splitting
4. **Model Training** - Train multiple ML models with class imbalance handling
5. **Feature Analysis** - Analyze feature importance
6. **Visualization** - Generate performance and distribution plots
7. **Explainability** - SHAP and LIME explanations
8. **Model Testing** - Evaluate on independent test sets
9. **CRS-Specific Analysis** - Granular analysis for CRS → Death prediction

---

## Quick Start

### Basic Pipeline (Steps 01-09)

```bash
# Step 1: Extract data from OpenFDA API
python 01_extract_data.py

# Step 2: Inspect data quality
python 02_inspect_data.py

# Step 3: Preprocess and engineer features
python 03_preprocess_data.py

# Step 4: Train multiple ML models
python 04_train_models.py

# Step 5: Analyze feature importance
python 05_analyze_features.py

# Step 6: Visualize results
python 06_visualize_results.py

# Step 7: Generate explainability analysis
python 07_explainability.py

# Step 8: Test models on independent set
python 08_test_models.py

# Step 9: Test on specific drug (Epcoritamab)
python 09_test_epcoritamab.py
```

### CRS-Specific Pipeline (Steps 11-13)

```bash
# Step 11: Granular CRS analysis with specific cutoffs
python 11_granular_crs_analysis.py

# Step 12: Train CRS-specific mortality model
python 12_crs_model_training.py

# Step 13: SHAP analysis for CRS model
python 13_crs_shap_analysis.py
```

---

## Parameterized Pipeline Usage

### CRS Mortality Prediction Pipeline

The pipeline is fully parameterized to support different drugs and adverse events:

```python
from task5_severity_prediction.crs_model_training import run_crs_mortality_pipeline

# Default: Epcoritamab + CRS
results = run_crs_mortality_pipeline()

# Custom drug + adverse event
results = run_crs_mortality_pipeline(
    drug_name="tafasitamab",
    ae_keyword_list=["ICANS", "IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY"],
    input_csv="main_data.csv",
    output_dir="./results_tafasitamab"
)
```

### Command Line Usage

```bash
# Run CRS pipeline with default parameters
python -c "from 12_crs_model_training import run_crs_mortality_pipeline; run_crs_mortality_pipeline()"

# Or modify the script to accept command-line arguments
```

---

## CRS → Death Model

### Overview

The CRS-specific model predicts the probability that a CRS case will result in death, given:
- Demographics (age, sex, weight, BMI)
- Comorbidities (diabetes, hypertension, cardiac disease)
- Drug exposure (polypharmacy, specific drug classes, combinations)
- Cancer stage (if available)

### Features

See `crs_feature_inventory.csv` for complete variable list with data sources.

Key feature categories:
- **Demographics**: Age, sex, weight, BMI (11 variables)
- **Medications**: Polypharmacy, drug classes, combinations (10 variables)
- **Adverse Events**: Reaction counts, infection AE (3 variables)
- **Comorbidities**: Diabetes, hypertension, cardiac (3 variables)
- **Cancer Stage**: Stage I-IV (4 variables, imperfect extraction)
- **Data Quality**: Missing indicators (2 variables)

**Total**: 33 features

### Cancer Stage Limitations

**DLBCL stage is NOT available as a structured variable in FAERS.**

- Current: Extracted from free-text `drug_indication` field (imperfect)
- Future: Pipeline ready to accept structured stage data (numeric 1-4)
- See `CANCER_STAGE_DOCUMENTATION.md` for details

### Usage Example

```python
import pickle
import pandas as pd

# Load trained model
with open('crs_model_best.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']

# Prepare feature vector for a new patient
features = {
    'age_years': 72,
    'sex_male': 1,
    'num_drugs': 6,
    'has_steroid': 1,
    'has_antibiotic': 1,
    'steroid_plus_antibiotic': 1,
    'bmi_obese': 1,
    # ... other features
}

# Ensure all features are present
feature_vector = pd.DataFrame([features])
for feat in feature_names:
    if feat not in feature_vector.columns:
        feature_vector[feat] = 0

# Predict death risk
death_probability = model.predict_proba(feature_vector[feature_names])[0][1]
print(f"Predicted death risk: {death_probability:.2%}")
```

---

## Output Files

### Data Files
- `main_data.csv` - Raw extracted data
- `preprocessed_data.csv` - Preprocessed dataset with features
- `X_train.csv`, `y_train.csv` - Training set
- `X_test.csv`, `y_test.csv` - Test set

### Model Files
- `model_comparison.csv` - Model performance comparison
- `trained_model_xgboost.pkl` - Best trained model
- `feature_importance_complete.csv` - Full feature importance list
- `training_meta.json` - Training metadata

### CRS-Specific Files
- `crs_model_best.pkl` - CRS mortality prediction model
- `crs_feature_inventory.csv` - Variable list + data sources
- `granular_crs_report.md` - Detailed CRS analysis report
- `crs_missingness_summary.csv` - Missing data summary
- `crs_plain_language_summary.md` - Plain language explanations

### Visualization Files
- `model_performance_comparison.png`
- `top_features.png`
- `advanced_evaluation_curves.png` (PR/ROC/Calibration)
- `shap_summary_plot.png`
- `crs_age_stratification.png`
- `crs_bmi_stratification.png`

---

## Documentation

- **Feature Inventory**: `FEATURE_VARIABLE_TABLE.md` - Complete variable list with data sources
- **Cancer Stage**: `CANCER_STAGE_DOCUMENTATION.md` - Stage data limitations and future integration
- **Biomarker Integration**: `BIOMARKER_INTEGRATION.md` - Conceptual integration plan
- **Granularity & Interactions**: `GRANULARITY_EXPLANATION.md` - Explanation of granular features and interactions

---

## Requirements

See `requirements.txt` for Python dependencies. Key libraries:
- pandas, numpy
- scikit-learn
- xgboost
- shap, lime
- matplotlib, seaborn

---

## Key Features

- ✅ **Exponential backoff** for API rate limiting
- ✅ **4-field OR search** (generic_name, medicinalproduct, brand_name, activesubstance)
- ✅ **Data deduplication** (within drug and globally)
- ✅ **Age standardization** to years
- ✅ **Severity normalization** to 0/1
- ✅ **Class imbalance handling** (class_weight, scale_pos_weight)
- ✅ **PR-AUC as primary metric** for imbalanced data
- ✅ **Probability calibration** for reliable predictions
- ✅ **SHAP explainability** with plain language mappings
- ✅ **Granular analysis** with specific cutoffs
- ✅ **Parameterized pipeline** for different drugs/AEs

---

## Model Performance

### General Severity Model (All Drugs)
- **PR-AUC**: ~0.44 (imbalanced data)
- **ROC-AUC**: ~0.66
- **Best Model**: Gradient Boosting / XGBoost

### CRS Mortality Model (Epcoritamab + CRS)
- **PR-AUC**: Varies by dataset size
- **ROC-AUC**: Varies by dataset size
- **Top Features**: num_drugs, has_chemo, age_years, cancer_stage_III

---

## License

See project root LICENSE file.

---

## Contact

For questions about this pipeline, refer to project documentation or contact the development team.

