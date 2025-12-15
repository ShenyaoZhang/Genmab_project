# Pharmacovigilance Analysis Pipeline

**AI-Powered Drug Safety Signal Detection and Risk Prediction**

A comprehensive pharmacovigilance pipeline for analyzing adverse events from FDA FAERS data, with support for signal detection, survival analysis, rare AE identification, and severity prediction.

---

## Project Structure

```
capstone_project-33/
├── Task1/          # CRS Risk Analysis & Signal Detection
├── Task2/          # Survival Analysis Pipeline
├── Task3/          # Rare & Unexpected AE Detection
├── Task4/          # Severity Prediction Pipeline
├── poster/         # Presentation materials
├── README.md       # This file
└── requirements.txt # Combined dependencies
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd capstone_project-33

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run Any Task

Each task can be run independently. See task-specific sections below.

---

## Task Overview

| Task | Description | Key Function |
|------|-------------|--------------|
| **Task1** | CRS Risk Analysis & Causal Inference | `run_pipeline(drug, ae)` |
| **Task2** | Survival Analysis (Cox PH, Kaplan-Meier) | `run_pipeline(drug, ae)` |
| **Task3** | Rare & Unexpected AE Detection | `check_any_combo(drug, ae)` |
| **Task4** | Severity/Mortality Prediction | `run_pipeline(drug, ae)` |

---

## Task1: CRS Risk Analysis Pipeline

Analyzes safety data from multiple pharmacovigilance databases (FAERS, EudraVigilance, JADER) to detect signals, predict risk, and identify causal relationships.

### Quick Start

```bash
# ⚠️ First: Unzip JADER data (required)
cd Task1
unzip jader_data.zip
cd Part1

# Run full pipeline
python3 main.py --all --drug epcoritamab --ae CRS

# Signal check
python3 main.py --check-signal --drug epcoritamab --ae neutropenia

# Launch dashboard
python3 main.py --dashboard
```

### Python API

```python
from main import run_pipeline, check_signal

# Full analysis
results = run_pipeline(drug="epcoritamab", adverse_event="CRS")

# Quick signal check
signal = check_signal("epcoritamab", "neutropenia")
```

### Key Features

- Multi-source data integration (FAERS, EV, JADER)
- Causal inference with DAG framework
- NLP feature extraction from narratives
- SHAP-based model interpretability

---

## Task2: Survival Analysis Pipeline

Scalable survival analysis for any drug-adverse event pair using Cox Proportional Hazards and Kaplan-Meier methods.

### Quick Start

```bash
cd Task2

# Simplest: one-command analysis
python3 run_epcoritamab_analysis_simple.py

# Or specify drug/AE
python3 run_survival_analysis.py --drug epcoritamab --adverse_event "cytokine release syndrome"

# Different drug-AE pair
python3 run_survival_analysis.py --drug tafasitamab --adverse_event ICANS
```

### Python API

```python
from run_survival_analysis import run_pipeline

results = run_pipeline(
    drug="epcoritamab",
    adverse_event="cytokine release syndrome"
)

print(f"AE Rate: {results['data_summary']['ae_rate']:.1f}%")
print(f"C-index: {results['cox_model']['c_index']:.4f}")
```

### Key Features

- Cox Proportional Hazards modeling
- Kaplan-Meier survival curves
- Feature selection (F-test, MI, Random Forest)
- Hazard ratio interpretation

---

## Task3: Rare & Unexpected AE Detection

Identifies rare and unexpected drug-adverse event combinations using Isolation Forest anomaly detection and FDA label filtering.

### Quick Start

```bash
cd Task3

# Step 1: Collect data (required first)
python3 task3_data_collector.py

# Step 2: Run detection pipeline
python3 task3_improved_pipeline.py

# Step 3: Interactive query
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Haemorrhagic gastroenteritis"
```

### Python API

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()

# Check if combination is rare & unexpected
result = query.check_any_combo("Epcoritamab", "Neutropenia")
print(f"Status: {result['conclusion']}")
```

### Example Output

```
Drug-Event Query: Epcoritamab + Haemorrhagic gastroenteritis
Status: RARE & UNEXPECTED
Report Count: 1
PRR: 111.69 (threshold: >2)
```

### Key Features

- Isolation Forest anomaly detection
- FDA label AE filtering
- Statistical tests (PRR, IC025, Chi-square)
- Clinical risk factor analysis

---

## Task4: Severity Prediction Pipeline

Machine learning pipeline for predicting adverse event severity (death) with SHAP explainability.

### Quick Start

```bash
cd Task4

# Scalable pipeline (recommended)
python3 scalable_pipeline.py --drug epcoritamab --ae CRS
python3 scalable_pipeline.py --drug tafasitamab --ae ICANS

# Signal check
python3 scalable_pipeline.py --check epcoritamab neutropenia

# Database summary
python3 scalable_pipeline.py --summary

# Polypharmacy analysis
python3 scalable_pipeline.py --polypharmacy CRS
```

### Python API

```python
from scalable_pipeline import run_pipeline, check_signal

# Run full pipeline
results = run_pipeline(drug="epcoritamab", adverse_event="CRS")

# Signal check
signal = check_signal("epcoritamab", "neutropenia")
# Output: "Expected. Listed on drug label. Frequency: 11.6%"
```

### Pipeline Flow

```
Input drug → Filter dataset → Extract features → Model training → Results
```

### Key Features

- Fully parameterized (any drug, any AE)
- SHAP explainability with physician-friendly interpretations
- Polypharmacy analysis by drug class
- Biomarker integration ready

---

## Model Purpose Table

| Model | Purpose | Example Output |
|-------|---------|----------------|
| **Rare AE Model** | Detects unexpected AE patterns | "Rare signal. Not on label." |
| **CRS Model** | Predicts probability of CRS | "CRS probability: 0.34" |
| **Mortality Model** | Predicts death risk | "Death risk: 0.12. Top factor: age > 70" |
| **Severity Model** | Predicts overall AE severity | "Severity score: 0.67" |

---

## Data Sources

| Source | Description | Used In |
|--------|-------------|---------|
| **FAERS** | FDA Adverse Event Reporting System | All tasks |
| **EudraVigilance** | European pharmacovigilance database | Task1 |
| **JADER** | Japanese adverse event database | Task1 |
| **OpenFDA API** | Programmatic access to FAERS | All tasks |

---

## SHAP Interpretation Guide

**How to read SHAP outputs (for physicians):**

- **Positive SHAP value**: Feature INCREASES predicted risk
- **Negative SHAP value**: Feature DECREASES predicted risk

**Example:**
```
Feature: age_years
SHAP value: +0.15
Interpretation: "Older age increased the predicted mortality risk for this patient"

Feature: has_steroid
SHAP value: -0.22
Interpretation: "Steroid premedication decreased the predicted CRS severity"
```

---

## Requirements

See `requirements.txt` for all dependencies. Key libraries:

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
lifelines>=0.27.0      # Survival analysis
xgboost>=2.0.0         # Gradient boosting
shap>=0.42.0           # Model explainability
matplotlib>=3.5.0
seaborn>=0.11.0
requests>=2.28.0       # API calls
```

### Optional Dependencies

```
transformers>=4.25.0   # BERT NLP features
torch>=1.13.0          # PyTorch for BERT
streamlit>=1.20.0      # Interactive dashboard
plotly>=5.13.0         # Interactive plots
```

---

## Installation

### Option 1: Full Installation

```bash
pip install -r requirements.txt
```

### Option 2: Minimal Installation

```bash
pip install pandas numpy scipy scikit-learn requests matplotlib seaborn
```

### Option 3: Task-Specific Installation

```bash
# Task1
pip install -r Task1/Part1/requirements.txt

# Task2
pip install -r Task2/requirements.txt

# Task3
pip install -r Task3/requirements.txt

# Task4
pip install -r Task4/requirements.txt
```

---

## Example Workflow

### Analyzing a New Drug-AE Combination

```bash
# 1. Check if it's a known or rare signal (Task3)
cd Task3
python3 task3_interactive_query.py --drug "YourDrug" --adverse_event "YourAE"

# 2. Run survival analysis (Task2)
cd ../Task2
python3 run_survival_analysis.py --drug YourDrug --adverse_event "YourAE"

# 3. Predict severity (Task4)
cd ../Task4
python3 scalable_pipeline.py --drug YourDrug --ae YourAE

# 4. Full causal analysis (Task1)
cd ../Task1/Part1
python3 main.py --all --drug YourDrug --ae YourAE
```

---

## Troubleshooting

### "command not found: python"

Use `python3` instead of `python`:
```bash
python3 scalable_pipeline.py --drug epcoritamab --ae CRS
```

### "ModuleNotFoundError"

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### API Rate Limiting

The pipelines include exponential backoff. If you encounter rate limits, wait a few minutes and retry.

---

## Citation

```bibtex
@article{pharmacovigilance_pipeline_2024,
  title={AI-Powered Pharmacovigilance Pipeline for Drug Safety Analysis},
  author={Capstone Team},
  year={2024},
  institution={NYU Center for Data Science}
}
```

---

## License

See LICENSE file in project root.

---

## Contact

For questions about this project, refer to task-specific documentation or contact the development team.

---

*Developed for NYU Capstone Project in collaboration with Genmab*

