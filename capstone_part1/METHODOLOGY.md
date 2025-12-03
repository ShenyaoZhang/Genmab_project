# CRS Risk Analysis Methodology

## Cytokine Release Syndrome (CRS) Risk Modeling for Epcoritamab
### A Causal Inference Approach Using Pharmacovigilance Data

---

## Table of Contents
1. [Overview](#1-overview)
2. [Data Sources](#2-data-sources)
3. [Data Extraction Pipeline](#3-data-extraction-pipeline)
4. [Causal Inference Framework](#4-causal-inference-framework)
5. [NLP Analysis](#5-nlp-analysis)
6. [Interactive Analysis](#6-interactive-analysis)
7. [Key Findings](#7-key-findings)
8. [Limitations](#8-limitations)

---

## 1. Overview

This project analyzes risk factors for Cytokine Release Syndrome (CRS) in patients treated with Epcoritamab, a bispecific T-cell engager used in oncology. The key goal is to distinguish **causal** risk factors from merely **correlated** variables.

### Project Structure

```
capstone_part1/
├── test.py                    # Initial FAERS API data retrieval
├── extract_crs_data.py        # Structured variable extraction from FAERS
├── data_extractors.py         # Eudravigilance & JADER extraction modules
├── causal_analysis.py         # Causal inference analysis
├── nlp_analysis.py            # NLP/BERT narrative analysis
├── interactive_dashboard.py   # Streamlit interactive dashboard
├── main.py                    # Pipeline orchestrator
├── requirements.txt           # Python dependencies
│
├── fda_drug_events.json       # Raw FAERS API response
├── crs_extracted_data.json    # Extracted structured FAERS data
├── multi_source_crs_data.json # Combined multi-source dataset
├── causal_analysis_results.json
├── narrative_features.json
│
├── causal_analysis_report.txt
├── nlp_analysis_report.txt
├── executive_summary.txt
└── METHODOLOGY.md             # This file
```

---

## 2. Data Sources

### 2.1 FAERS (FDA Adverse Event Reporting System)
- **Access**: Public API at `https://api.fda.gov/drug/event.json`
- **Status**: ✅ Fully implemented
- **Coverage**: United States, global submissions

```python
# API Query
params = {
    "search": 'patient.drug.medicinalproduct:"epcoritamab" AND '
              'patient.reaction.reactionmeddrapt:"cytokine release syndrome"',
    "limit": 100
}
```

### 2.2 Eudravigilance (European Medicines Agency)
- **Access**: Manual download from https://www.adrreports.eu/
- **Status**: 📋 Extraction code ready, requires manual data download
- **Coverage**: European Union member states

### 2.3 JADER (Japanese Adverse Drug Event Report)
- **Access**: Public CSV files from PMDA website
- **Status**: 📋 Extraction code ready, requires manual data download
- **Coverage**: Japan

---

## 3. Data Extraction Pipeline

### 3.1 Raw Data Structure (FAERS JSON)

```json
{
  "safetyreportid": "19134472",
  "serious": "1",
  "seriousnesshospitalization": "1",
  "patient": {
    "patientonsetage": "79",
    "patientonsetageunit": "801",
    "patientsex": "2",
    "patientweight": "87.9",
    "reaction": [
      {
        "reactionmeddrapt": "Cytokine release syndrome",
        "reactionoutcome": "1"
      }
    ],
    "drug": [
      {
        "drugcharacterization": "1",
        "medicinalproduct": "EPCORITAMAB.",
        "drugstructuredosagenumb": "24",
        "drugstructuredosageunit": "003",
        "drugdosagetext": "FULL DOSE: 24 MG, WEEKLY",
        "drugindication": "B-CELL LYMPHOMA STAGE IV"
      }
    ]
  }
}
```

### 3.2 Key Extraction Logic

#### Drug Characterization Codes
| Code | Meaning | Interpretation |
|------|---------|----------------|
| 1 | Suspect | Drug suspected to cause the adverse event |
| 2 | Concomitant | Drug taken alongside but not suspected |
| 3 | Interacting | Drug that interacted with suspect drug |

#### Outcome Codes
| Code | Meaning |
|------|---------|
| 1 | Recovered/resolved |
| 2 | Recovering/resolving |
| 3 | Not recovered/not resolved |
| 4 | Recovered with sequelae |
| 5 | Fatal |
| 6 | Unknown |

#### Age Unit Codes
| Code | Unit | Conversion to Years |
|------|------|---------------------|
| 800 | Decade | × 10 |
| 801 | Year | × 1 |
| 802 | Month | × (1/12) |
| 803 | Week | × (1/52) |
| 804 | Day | × (1/365) |

### 3.3 Extracted Variables Schema

```json
{
  "report_id": "19134472",
  
  "is_crs": true,
  "crs_outcome": "recovered|recovering|not_recovered|fatal|unknown",
  "serious": true,
  "hospitalized": true,
  "death": false,
  "life_threatening": false,
  
  "epcoritamab_exposure": true,
  "epcoritamab_suspect": true,
  "epcoritamab_doses": [
    {"dose_mg": 24.0, "date": "2021-03-22", "dose_type": "full"}
  ],
  "co_medications": ["RITUXIMAB", "PREDNISOLONE"],
  "indication": "B-CELL LYMPHOMA STAGE IV",
  
  "age": 79.0,
  "sex": "female",
  "weight": 87.9,
  "country": "DK",
  
  "crs_onset_date": null,
  "first_epcoritamab_date": "2021-03-08",
  "dose_to_crs_interval_days": null,
  
  "narrative_text": "CASE EVENT DATE: 20210323"
}
```

---

## 4. Causal Inference Framework

### 4.1 Directed Acyclic Graph (DAG)

The DAG represents our theoretical understanding of causal relationships:

```
                    ┌─────────────────────────────────────┐
                    │         CONFOUNDERS                 │
                    │   (Age, Disease Stage, Prior Tx)    │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────┐
            │  Epcoritamab  │             │  CRS Severity │
            │     Dose      │────────────▶│   (Outcome)   │
            └───────┬───────┘             └───────▲───────┘
                    │                             │
                    │     CAUSAL PATHWAY          │
                    ▼                             │
            ┌───────────────┐             ┌───────────────┐
            │   T-cell      │────────────▶│   Cytokine    │
            │  Activation   │             │   Release     │
            └───────────────┘             └───────────────┘
                                                  │
                    ┌─────────────────────────────┘
                    ▼
            ┌───────────────────────────────────────────┐
            │           EFFECT MODIFIERS                │
            │   (Steroids, Tocilizumab - can block      │
            │    the causal pathway)                    │
            └───────────────────────────────────────────┘
```

### 4.2 Variable Classification

| Variable | Type | Rationale |
|----------|------|-----------|
| Epcoritamab dose | **Exposure** | Primary treatment of interest |
| CRS severity | **Outcome** | What we're trying to predict/explain |
| Age | **Confounder** | Affects both treatment decisions AND outcomes |
| Disease stage | **Confounder** | Sicker patients get different doses AND have worse outcomes |
| Steroids | **Effect Modifier** | Blocks the causal pathway (anti-inflammatory) |
| Tocilizumab | **Effect Modifier** | Blocks IL-6 pathway |
| Hospitalization | **Collider** | Caused by BOTH exposure and outcome - DO NOT adjust |
| # Co-medications | **Correlational** | Marker of disease severity, not a cause |

### 4.3 Statistical Association Analysis

#### Continuous Variables (Age, Dose, Weight)
```python
from scipy import stats

# Point-biserial correlation (continuous vs binary outcome)
correlation, p_value = stats.pointbiserialr(
    outcome_binary,  # e.g., severe_crs (0 or 1)
    continuous_var   # e.g., age
)

# Interpretation:
# - correlation > 0: positive association
# - correlation < 0: negative association  
# - p_value < 0.05: statistically significant
```

#### Categorical Variables (Sex, Steroid Use)
```python
# Create 2x2 contingency table
#                  Outcome=0    Outcome=1
# Variable=0          d            c
# Variable=1          b            a

contingency = pd.crosstab(variable, outcome)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

# Odds Ratio
odds_ratio = (a * d) / (b * c)

# Interpretation:
# - OR > 1: variable associated with HIGHER outcome risk
# - OR < 1: variable associated with LOWER outcome risk
# - OR = 1: no association
```

### 4.4 Propensity Score Analysis

**Purpose**: Estimate causal effects from observational data by controlling for confounding.

#### Step 1: Estimate Propensity Scores
```python
from sklearn.linear_model import LogisticRegression

# Propensity score = P(Treatment | Confounders)
# e.g., P(Steroids | Age, Sex, Disease Severity)

confounders = ['age', 'sex_male', 'n_co_medications']
X = df[confounders].values
T = df['has_steroids'].values  # Treatment indicator

ps_model = LogisticRegression()
ps_model.fit(X, T)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# Clip to avoid extreme weights
propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
```

#### Step 2: Calculate Inverse Probability Weights (IPW)
```python
# For treated patients: weight = 1 / PS
# For control patients: weight = 1 / (1 - PS)

weights = np.where(
    T == 1,
    1 / propensity_scores,
    1 / (1 - propensity_scores)
)
```

#### Step 3: Estimate Average Treatment Effect (ATE)
```python
# Weighted outcome means
treated_outcome = np.average(Y[T == 1], weights=weights[T == 1])
control_outcome = np.average(Y[T == 0], weights=weights[T == 0])

# Average Treatment Effect
ATE = treated_outcome - control_outcome

# Interpretation:
# ATE = -0.10 means treatment REDUCES outcome risk by 10 percentage points
# ATE = +0.05 means treatment INCREASES outcome risk by 5 percentage points
```

#### Step 4: Bootstrap Confidence Intervals
```python
n_bootstrap = 1000
ate_bootstrap = []

for _ in range(n_bootstrap):
    # Resample with replacement
    idx = np.random.choice(len(df), size=len(df), replace=True)
    # Recalculate ATE on bootstrap sample
    ate_boot = calculate_ate(df.iloc[idx])
    ate_bootstrap.append(ate_boot)

ci_lower = np.percentile(ate_bootstrap, 2.5)
ci_upper = np.percentile(ate_bootstrap, 97.5)

# If CI excludes 0, effect is statistically significant
```

### 4.5 Sensitivity Analysis (E-value)

**Purpose**: Assess how robust findings are to unmeasured confounding.

```python
# E-value: How strong would an unmeasured confounder need to be
# to explain away the observed association?

observed_OR = 2.21  # From our analysis

if observed_OR >= 1:
    E_value = observed_OR + np.sqrt(observed_OR * (observed_OR - 1))
else:
    E_value = 1/observed_OR + np.sqrt((1/observed_OR) * (1/observed_OR - 1))

# Result: E_value = 3.85
# 
# Interpretation: An unmeasured confounder would need to be associated
# with BOTH exposure AND outcome by a risk ratio of at least 3.85
# to fully explain away the observed effect.
#
# Higher E-value = More robust finding = Stronger evidence for causation
```

---

## 5. NLP Analysis

### 5.1 Rule-Based Feature Extraction

```python
def extract_features(text):
    text_lower = text.lower()
    
    features = {
        # Severity indicators
        'mentions_fever': bool(re.search(r'fever|pyrexia|febrile', text_lower)),
        'mentions_hypotension': bool(re.search(r'hypotension|low blood pressure', text_lower)),
        'mentions_hypoxia': bool(re.search(r'hypoxia|oxygen|desaturation', text_lower)),
        'mentions_icu': bool(re.search(r'icu|intensive care', text_lower)),
        'mentions_intubation': bool(re.search(r'intubat|ventilat', text_lower)),
        'mentions_vasopressor': bool(re.search(r'vasopressor|norepinephrine', text_lower)),
        
        # Treatment indicators
        'mentions_tocilizumab': bool(re.search(r'tocilizumab|actemra', text_lower)),
        'mentions_steroids': bool(re.search(r'steroid|dexamethasone|predniso', text_lower)),
        
        # CRS grade extraction
        'crs_grade': extract_grade(text_lower),  # Returns 1-4 or None
        
        # Time to onset
        'time_to_onset_hours': extract_onset_time(text_lower)
    }
    
    # Composite severity score
    severity_indicators = ['mentions_hypotension', 'mentions_hypoxia', 
                          'mentions_icu', 'mentions_intubation', 'mentions_vasopressor']
    features['severity_score'] = sum(features[ind] for ind in severity_indicators)
    
    return features
```

### 5.2 BERT-Based Analysis (When Available)

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load BioBERT (trained on biomedical text)
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_bert_embedding(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', 
                       truncation=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token as sentence embedding (768 dimensions)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

# Use embeddings as features for severity prediction
embeddings = [get_bert_embedding(text) for text in narratives]
X = np.vstack(embeddings)
y = severity_labels

# Train classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
classifier.fit(X, y)
```

---

## 6. Interactive Analysis

### 6.1 Streamlit Dashboard

```bash
# Launch the dashboard
streamlit run interactive_dashboard.py
```

**Features:**
1. **Outcome Selection**: Choose any outcome (severe CRS, death, recovery, hospitalization)
2. **Data Filtering**: Filter by data source, age range, etc.
3. **Association Analysis**: View statistical associations with selected outcome
4. **Causal Analysis**: Run propensity score analysis on any treatment
5. **Subgroup Analysis**: Explore outcome rates across patient subgroups

### 6.2 Terminal Interface

```bash
# For environments without Streamlit
python interactive_dashboard.py --terminal
```

---

## 7. Key Findings

### 7.1 Causal Risk Factors

| Factor | Direction | Evidence | Mechanism |
|--------|-----------|----------|-----------|
| **Epcoritamab Dose** | ↑ Risk | OR=2.21 (high vs low), E-value=3.85 | Dose-dependent T-cell activation |
| **Steroid Premedication** | ↓ Risk (Protective) | Propensity score analysis | Anti-inflammatory, cytokine suppression |
| **Tocilizumab** | ↓ Risk (Protective) | Known CRS treatment | IL-6 receptor blockade |

### 7.2 Confounders (Must Control For)

| Factor | Why It's a Confounder |
|--------|----------------------|
| **Age** | Older patients may receive lower doses (caution) AND have worse immune dysregulation |
| **Disease Stage** | Advanced disease → more aggressive treatment AND worse outcomes |
| **Prior Therapies** | More prior therapies → different dosing AND different baseline risk |

### 7.3 Correlational (Not Causal)

| Factor | Why It's Not Causal |
|--------|---------------------|
| **# Co-medications** | Marker of how sick the patient is, not a direct cause of CRS |
| **Data Source** | Reflects reporting practices, not biological differences |

---

## 8. Limitations

### 8.1 Data Limitations
- **Reporting Bias**: FAERS is voluntary; severe cases more likely reported
- **Missing Data**: Weights, exact doses, onset times often missing
- **Confounding by Indication**: Sicker patients may receive different treatments

### 8.2 Methodological Limitations
- **Unmeasured Confounding**: Cannot fully control for all confounders
- **Cross-sectional Data**: Limited temporal information
- **Simulated Data**: Eudravigilance and JADER data simulated for demonstration

### 8.3 Recommendations for Improvement
1. Obtain real Eudravigilance and JADER data for multi-database validation
2. Link with clinical trial data for better dose-response characterization
3. Use instrumental variables if available (e.g., prescribing physician preferences)
4. Conduct negative control outcome analysis to detect residual confounding

---

## Appendix: Running the Pipeline

```bash
# 1. Setup environment
conda activate capstone
pip install -r requirements.txt

# 2. Run complete pipeline
python main.py --all

# 3. Run individual components
python main.py --extract    # Data extraction only
python main.py --causal     # Causal analysis only
python main.py --nlp        # NLP analysis only
python main.py --dashboard  # Launch interactive dashboard
python main.py --summary    # Generate executive summary

# 4. For real EU/JP data, see instructions in:
python -c "from data_extractors import EudravigilanceExtractor; EudravigilanceExtractor().download_line_listing('epcoritamab')"
python -c "from data_extractors import JADERExtractor; JADERExtractor().download_instructions()"
```

---

*Document generated for NYU Capstone Project - CRS Risk Analysis*

