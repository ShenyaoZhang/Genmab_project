# CRS Risk Analysis Pipeline

**Pharmacovigilance Analysis for Cytokine Release Syndrome (CRS) in Epcoritamab-Treated Patients**

A scalable, extensible pipeline for analyzing adverse events from multiple pharmacovigilance databases (FAERS, EudraVigilance, JADER).

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Usage Examples](#usage-examples)
6. [Module Documentation](#module-documentation)
7. [Expected Outputs](#expected-outputs)
8. [Adding New Drugs/AEs/Datasets](#adding-new-drugsaesdatasets)
9. [Model Interpretation](#model-interpretation)
10. [File Structure](#file-structure)

---

## Overview

This pipeline analyzes safety data from multiple pharmacovigilance databases to:

- **Detect signals** for adverse events (known and unexpected)
- **Predict risk** of serious outcomes (CRS severity, mortality)
- **Identify causal relationships** vs. correlations
- **Extract features** from narrative text using NLP

### Key Features

- **Scalable**: Works for any drug or adverse event
- **Multi-source**: Integrates FAERS, EudraVigilance, JADER
- **Interpretable**: SHAP-based explanations for safety physicians
- **Extensible**: Easy to add biomarkers or new data sources

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/MengqiLiu-9543/capstone_project-33.git
cd capstone_project-33/Task1/Part1

# 2. Set up environment
conda create -n crs_analysis python=3.10
conda activate crs_analysis
pip install -r requirements.txt

# 3. Run the complete pipeline
python main.py --all

# 4. Or analyze a different drug/AE
python main.py --all --drug tafasitamab --ae ICANS

# 5. Quick signal check
python main.py --check-signal --drug epcoritamab --ae neutropenia
```

---

## Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip

### Environment Setup

**Option 1: Using Conda (Recommended)**

```bash
conda create -n crs_analysis python=3.10
conda activate crs_analysis
pip install -r requirements.txt
```

**Option 2: Using pip**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
scikit-learn>=1.1.0
requests>=2.28.0
openpyxl>=3.0.0
transformers>=4.25.0  # Optional: for BERT
torch>=1.13.0         # Optional: for BERT
streamlit>=1.20.0     # For dashboard
plotly>=5.13.0        # For visualizations
```

---

## Pipeline Architecture

```
+-----------------------------------------------------------------------------+
|                         INPUT PARAMETERS                                     |
|                    drug="epcoritamab", ae="CRS"                              |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
| STEP 1: DATA EXTRACTION                                                      |
|   - Query FAERS API with drug + AE filters                                  |
|   - Load EudraVigilance CSV (if available)                                  |
|   - Load JADER data (if available)                                          |
|   - Output: multi_source_crs_data.json                                      |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
| STEP 2: FEATURE EXTRACTION                                                   |
|   - Demographics (age, sex, weight)                                         |
|   - Clinical variables (seriousness, outcome)                               |
|   - Drug exposure (dose, frequency, co-medications)                         |
|   - NLP features from narratives                                            |
|   - Output: extracted_features.json                                         |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
| STEP 3: MODEL TRAINING                                                       |
|   - Rare AE Model: Detect unexpected AE patterns                            |
|   - Risk Model: Predict probability of target AE                            |
|   - Mortality Model: Predict risk of AE-related death                       |
|   - Output: model_results.json, feature_importances.json                    |
+-----------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------+
| STEP 4: CAUSAL ANALYSIS                                                      |
|   - DAG-based framework                                                     |
|   - Propensity score analysis                                               |
|   - Sensitivity analysis (E-values)                                         |
|   - Output: causal_analysis_results.json                                    |
+-----------------------------------------------------------------------------+
```

---

## Usage Examples

### Example 1: Run Full Pipeline

```python
from main import run_pipeline

# Run for default drug (epcoritamab) and AE (CRS)
results = run_pipeline(drug="epcoritamab", adverse_event="CRS")

# Run for a different drug/AE combination
results = run_pipeline(drug="tafasitamab", adverse_event="ICANS")
```

### Example 2: Signal Detection

```python
from main import check_signal

# Quick signal check
result = check_signal("epcoritamab", "neutropenia")

# Example output:
# {
#     'assessment': 'Expected. Known label AE with moderate signal.',
#     'on_label': True,
#     'signal_strength': 'moderate',
#     'recommendation': 'Continue routine monitoring per label requirements.'
# }
```

### Example 3: Command Line Usage

```bash
# Full pipeline
python main.py --all --drug epcoritamab --ae CRS

# Signal check only
python main.py --check-signal --drug epcoritamab --ae neutropenia

# Individual steps
python main.py --extract --drug epcoritamab --ae CRS
python main.py --causal
python main.py --nlp
python main.py --dashboard
```

### Example 4: Generate Data Summary

```python
from data_summary import DatasetSummary

summary = DatasetSummary("multi_source_crs_data.json")
report = summary.generate_full_report()
print(report)
```

---

## Module Documentation

### `main.py` - Pipeline Orchestrator

**Key Functions:**

| Function | Purpose | Example |
|----------|---------|---------|
| `run_pipeline()` | Run complete analysis pipeline | `run_pipeline(drug="epcoritamab", ae="CRS")` |
| `check_signal()` | Quick signal detection check | `check_signal("epcoritamab", "neutropenia")` |

### `data_summary.py` - Dataset Summaries

Generates:
- Dataset counts by source
- Missingness tables
- Variable availability across databases
- Polypharmacy analysis with drug classes

### `preprocessing.py` - Variable Preprocessing

Documents and implements:
- Z-score normalization (age, weight)
- Log transformation (dose)
- BMI bucketing (<25 Normal, 25-30 Overweight, >30 Obese)
- Disease stage ordinal encoding (if available)

### `biomarker_integration.py` - Future Biomarker Support

Shows how biomarkers (IL-6, IL-7, IL-21, CCL17, etc.) would be integrated:
- Processing methods (z-score, log transform)
- Clinical cutoffs
- Integration into feature matrix

### `rare_ae_detection.py` - Unexpected AE Detection

Implements 5-step detection:
1. Collect all AE pairs
2. Remove known label AEs
3. Remove high-frequency AEs
4. Flag remaining rare AEs
5. Prioritize by signal strength

### `causal_analysis.py` - Causal Inference

- DAG-based framework
- Propensity score analysis
- Sensitivity analysis (E-values)
- Causal vs. correlational interpretation

### `nlp_analysis.py` - NLP Features

- Rule-based feature extraction
- BERT embeddings (optional)
- Severity prediction from narratives

---

## Expected Outputs

### Files Generated

| File | Description |
|------|-------------|
| `multi_source_crs_data.json` | Combined dataset from all sources |
| `crs_extracted_data.json` | Structured FAERS data |
| `causal_analysis_results.json` | Causal analysis output |
| `causal_analysis_report.txt` | Human-readable causal report |
| `narrative_features.json` | NLP-extracted features |
| `nlp_analysis_report.txt` | NLP analysis report |
| `executive_summary.txt` | Executive summary |
| `data_summary_report.txt` | Dataset summary with missingness |
| `rare_ae_report.txt` | Rare AE detection results |

### Example Output: Signal Check

```
==============================================================
SIGNAL CHECK: EPCORITAMAB + NEUTROPENIA
==============================================================

Assessment: Expected. Known label AE with moderate signal.
On Label: Yes
Signal Strength: moderate

Case Counts by Database:
  FAERS: 23 cases (Detected)
  JADER: 15 cases (Detected)
  EUDRAVIGILANCE: 8 cases (Detected)

Recommendation: Continue routine monitoring per label requirements.
```

---

## Adding New Drugs/AEs/Datasets

### Adding a New Drug

```python
# Just change the drug parameter
results = run_pipeline(drug="your_new_drug", adverse_event="CRS")

# Or from command line
python main.py --all --drug your_new_drug --ae CRS
```

### Adding a New Adverse Event

```python
# Change the adverse_event parameter
results = run_pipeline(drug="epcoritamab", adverse_event="ICANS")

# Or from command line  
python main.py --all --drug epcoritamab --ae ICANS
```

### Adding a New Dataset

1. **Prepare your data** in the expected format:

```json
{
  "report_id": "12345",
  "source": "your_database",
  "is_crs": true,
  "crs_outcome": "recovered",
  "age": 65,
  "sex": "male",
  "weight": 75,
  "epcoritamab_doses": [{"dose_mg": 24}],
  "co_medications": ["DRUG1", "DRUG2"]
}
```

2. **Add extractor** in `data_extractors.py`:

```python
class YourDatabaseExtractor:
    def load_data(self, file_path):
        # Your extraction logic
        pass
    
    def to_unified_format(self, df):
        # Convert to standard format
        pass
```

3. **Update** `create_multi_source_data()` to include new source.

---

## Model Interpretation

### Reading SHAP Values

```
For Patient 203:
  Feature              | Raw Value | SHAP Value | Effect on Prediction
  ---------------------|-----------|------------|----------------------
  Weight               | 92 kg     | -0.08      | Reduces mortality risk
  Age                  | 72 years  | +0.15      | Increases mortality risk  
  Steroid premedication| Yes       | -0.22      | Reduces CRS severity
  Dose (full, 48mg)    | 48 mg     | +0.18      | Increases CRS risk

Interpretation Guide:
  - Positive SHAP: Feature pushes risk UPWARD (worse outcome)
  - Negative SHAP: Feature pushes risk DOWNWARD (better outcome)
  - Magnitude: Strength of effect
```

### Model Purposes

| Model | Purpose |
|-------|---------|
| **Rare AE Model** | Detects unexpected AE patterns not on drug label |
| **Risk Model** | Predicts probability of target AE (e.g., CRS) |
| **Mortality Model** | Predicts risk of AE-related death |
| **Severity Model** | Predicts severity grade (e.g., CRS Grade 1-4) |

---

## File Structure

```
capstone_part1/
|-- main.py                     # Pipeline orchestrator (scalable entry point)
|-- data_extractors.py          # FAERS/EudraVigilance/JADER extractors
|-- extract_crs_data.py         # CRS-specific variable extraction
|-- causal_analysis.py          # Causal inference analysis
|-- nlp_analysis.py             # NLP/BERT narrative analysis
|-- interactive_dashboard.py    # Streamlit dashboard
|-- data_summary.py             # Dataset summaries & missingness
|-- preprocessing.py            # Variable preprocessing documentation
|-- biomarker_integration.py    # Future biomarker integration
|-- rare_ae_detection.py        # Rare/unexpected AE detection
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
|-- METHODOLOGY.md              # Detailed methodology documentation
|
|-- figures/                    # Generated visualizations
|   |-- causal_summary.png
|   |-- dag_framework.png
|   +-- ...
|
+-- [Generated Data Files]
    |-- fda_drug_events.json
    |-- crs_extracted_data.json
    |-- multi_source_crs_data.json
    |-- causal_analysis_results.json
    |-- narrative_features.json
    +-- ...
```

---

## Support

For questions about this pipeline, contact the development team or refer to the `METHODOLOGY.md` for detailed technical documentation.

---

*Developed for NYU Capstone Project in collaboration with Genmab*
