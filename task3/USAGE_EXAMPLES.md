# Task 3 Usage Examples

This document provides practical examples of how to use the Task 3 interactive query system.

---

## Quick Start

### Basic Query: Check a Specific Drug-Event Combination

```python
from task3_interactive_query import InteractiveAnomalyQuery

# Initialize query system
query = InteractiveAnomalyQuery()

# Check if a combination is rare & unexpected
result = query.check_any_combo("Epcoritamab", "Neutropenia")

# Display results
print(f"Drug: {result['drug']}")
print(f"Adverse Event: {result['adverse_event']}")
print(f"Status: {result['conclusion']}")
print(f"Report Count: {result.get('report_count', 'N/A')}")
```

**Example Output:**
```
Drug: Epcoritamab
Adverse Event: Neutropenia
Status: ✅ RARE & UNEXPECTED (passed IF + all 3 statistical tests)
Report Count: 2
Statistics:
  - PRR: 15.3
  - IC025: 2.1
  - Chi-square: 8.5
Clinical Impact:
  - Death Rate: 0.0%
  - Hospitalization Rate: 50.0%
  - Serious Rate: 100.0%
```

---

## Example 1: Check "Epcoritamab + Neutropenia"

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()
result = query.check_any_combo("Epcoritamab", "Neutropenia")

if result['found']:
    print("=" * 70)
    print(f"Drug-Event Combination: {result['drug']} + {result['adverse_event']}")
    print("=" * 70)
    print(f"\nReport Count: {result['report_count']}")
    print(f"\nStatistical Metrics:")
    stats = result['statistics']
    print(f"  - PRR (Proportional Reporting Ratio): {stats['prr']}")
    print(f"  - IC025 (Information Component): {stats['ic025']}")
    print(f"  - Chi-square: {stats['chi2']}")
    
    print(f"\nClinical Impact:")
    clinical = result['clinical']
    print(f"  - Death Rate: {clinical['death_rate']}")
    print(f"  - Hospitalization Rate: {clinical['hosp_rate']}")
    print(f"  - Serious Rate: {clinical['serious_rate']}")
    
    print(f"\nAssessment:")
    assessment = result['assessment']
    print(f"  - In No-Cap Results: {assessment['in_no_cap_results']}")
    print(f"  - Passes PRR Test (>2): {assessment['passes_prr']}")
    print(f"  - Passes IC025 Test (>0): {assessment['passes_ic025']}")
    print(f"  - Passes Chi² Test (>4): {assessment['passes_chi2']}")
    
    print(f"\nConclusion: {result['conclusion']}")
else:
    print(f"❌ {result['message']}")
```

**Output:**
```
======================================================================
Drug-Event Combination: Epcoritamab + Neutropenia
======================================================================

Report Count: 2

Statistical Metrics:
  - PRR (Proportional Reporting Ratio): 15.3
  - IC025 (Information Component): 2.1
  - Chi-square: 8.5

Clinical Impact:
  - Death Rate: 0.0%
  - Hospitalization Rate: 50.0%
  - Serious Rate: 100.0%

Assessment:
  - In No-Cap Results: True
  - Passes PRR Test (>2): True
  - Passes IC025 Test (>0): True
  - Passes Chi² Test (>4): True

Conclusion: ✅ RARE & UNEXPECTED (passed IF + all 3 statistical tests)
```

---

## Example 2: Get Top Anomalies for a Drug

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()

# Get top 10 rare & unexpected AEs for Epcoritamab
top_aes = query.get_top_events_for_drug("Epcoritamab", n=10)

if top_aes is not None and len(top_aes) > 0:
    print(f"Top 10 Rare & Unexpected AEs for Epcoritamab:")
    print("=" * 70)
    for idx, row in top_aes.iterrows():
        print(f"\n{idx + 1}. {row['adverse_event']}")
        print(f"   Anomaly Score: {row['anomaly_score']:.3f}")
        print(f"   Count: {row['count']}")
        print(f"   PRR: {row.get('prr', 'N/A')}")
else:
    print("No results found for Epcoritamab")
```

**Output:**
```
Top 10 Rare & Unexpected AEs for Epcoritamab:
======================================================================

1. Renal impairment
   Anomaly Score: 0.782
   Count: 2
   PRR: 15.3

2. Neutropenia
   Anomaly Score: 0.768
   Count: 2
   PRR: 12.5

3. Thrombocytopenia
   Anomaly Score: 0.745
   Count: 3
   PRR: 8.2

...
```

---

## Example 3: Compare Multiple Drugs

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()

# Compare Epcoritamab, Glofitamab, and Mosunetuzumab
drugs = ["Epcoritamab", "Glofitamab", "Mosunetuzumab"]
comparison = query.compare_drugs(drugs)

if comparison is not None:
    print("Drug Comparison: Rare & Unexpected AE Counts")
    print("=" * 70)
    print(comparison[['drug', 'adverse_event', 'count', 'anomaly_score']].to_string(index=False))
```

**Output:**
```
Drug Comparison: Rare & Unexpected AE Counts
======================================================================
         drug      adverse_event  count  anomaly_score
Epcoritamab    Renal impairment      2          0.782
Epcoritamab        Neutropenia      2          0.768
Glofitamab     Cytokine release      5          0.812
Mosunetuzumab      Thrombocytopenia      3          0.745
...
```

---

## Example 4: Search by Adverse Event Keyword

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()

# Search for all drugs associated with "infection"
results = query.search_by_adverse_event("infection")

if results is not None and len(results) > 0:
    print(f"Found {len(results)} drug-event pairs with 'infection':")
    print("=" * 70)
    for idx, row in results.head(10).iterrows():
        print(f"{row['drug']} + {row['adverse_event']} (Count: {row['count']})")
else:
    print("No results found for 'infection'")
```

---

## Example 5: Command-Line Usage

### Check a Specific Combination

```bash
cd task3
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Neutropenia"
```

**Output:**
```
✓ Loaded 150 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (12,450 reports) for arbitrary queries

======================================================================
Drug-Event Query: Epcoritamab + Neutropenia
======================================================================

Found in Database: Yes
Report Count: 2

Statistical Metrics:
  PRR: 15.3
  IC025: 2.1
  Chi-square: 8.5

Clinical Impact:
  Death Rate: 0.0%
  Hospitalization Rate: 50.0%
  Serious Rate: 100.0%

Assessment:
  In No-Cap Results: True
  Passes PRR Test (>2): True
  Passes IC025 Test (>0): True
  Passes Chi² Test (>4): True

Conclusion: ✅ RARE & UNEXPECTED (passed IF + all 3 statistical tests)
```

---

## Example 6: Real-World Usage Scenario

**Scenario:** A safety physician wants to check if "Epcoritamab + Renal impairment" is a rare and unexpected signal.

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()
result = query.check_any_combo("Epcoritamab", "Renal impairment")

# Simple interpretation
if result['is_unexpected']:
    print("⚠️ SIGNAL DETECTED: Rare & Unexpected")
    print(f"   Observed: {result['report_count']} reports in FAERS")
    print(f"   Status: Not listed in FDA drug label")
    print(f"   Statistical Evidence: PRR={result['statistics']['prr']}, IC025={result['statistics']['ic025']}")
    print(f"   Clinical Impact: {result['clinical']['serious_rate']} serious cases")
else:
    print("✓ No rare/unexpected signal detected")
    print(f"   Reason: {result['conclusion']}")
```

**Output:**
```
⚠️ SIGNAL DETECTED: Rare & Unexpected
   Observed: 2 reports in FAERS
   Status: Not listed in FDA drug label
   Statistical Evidence: PRR=15.3, IC025=2.1
   Clinical Impact: 100.0% serious cases
```

---

## UI Mock (Conceptual)

For presentation purposes, here's a conceptual UI mock:

```
┌─────────────────────────────────────────────────────────────┐
│  Task 3: Rare & Unexpected AE Detection                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Drug Selection:                                            │
│  ┌─────────────────────────────┐                          │
│  │ [Epcoritamab ▼]            │                          │
│  └─────────────────────────────┘                          │
│                                                             │
│  Adverse Event Selection:                                   │
│  ┌─────────────────────────────┐                          │
│  │ [Neutropenia ▼]            │                          │
│  └─────────────────────────────┘                          │
│                                                             │
│  ┌─────────────────┐                                       │
│  │   [Run Query]   │                                       │
│  └─────────────────┘                                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Results Panel:                                             │
│                                                             │
│  Status: ✅ RARE & UNEXPECTED                              │
│                                                             │
│  Report Count: 2                                           │
│                                                             │
│  Statistical Metrics:                                      │
│    • PRR: 15.3                                             │
│    • IC025: 2.1                                            │
│    • Chi-square: 8.5                                       │
│                                                             │
│  Clinical Impact:                                           │
│    • Death Rate: 0.0%                                      │
│    • Hospitalization: 50.0%                                │
│    • Serious: 100.0%                                       │
│                                                             │
│  Top Risk Factors:                                          │
│    1. Age > 65 (SHAP: +0.15)                              │
│    2. Male sex (SHAP: +0.08)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Expected Output Formats

### Format 1: Simple Status Check

```
Drug: Epcoritamab
Adverse Event: Neutropenia
Status: ✅ RARE & UNEXPECTED
Observed in: FAERS
```

### Format 2: Detailed Analysis

```
Drug: Epcoritamab
Adverse Event: Neutropenia
Status: ✅ RARE & UNEXPECTED (passed IF + all 3 statistical tests)

Report Count: 2
Mean Threshold: 3.24 (2 < 3.24 → RARE)

Statistical Evidence:
  - PRR: 15.3 (>2 threshold) ✓
  - IC025: 2.1 (>0 threshold) ✓
  - Chi-square: 8.5 (>4 threshold) ✓

Clinical Impact:
  - Death Rate: 0.0%
  - Hospitalization Rate: 50.0%
  - Serious Rate: 100.0%

FDA Label Check: NOT listed (unexpected)
Indication Check: NOT an indication
Frequency Check: Below mean (rare)
```

---

## Example 7: BERT Clinical Feature Analysis

Analyze clinical risk factors that influence or cause a specific adverse event.

### Command-Line Usage

```bash
cd task3
python3 task3_bert_clinical_features.py "Epcoritamab" "Neutropenia"
```

**Output:**
```
======================================================================
Task 3: BERT Clinical Feature Analysis
======================================================================

Analyzing: Epcoritamab + Neutropenia
----------------------------------------------------------------------
Fetching reports for Epcoritamab + Neutropenia...
✓ Found 24 reports

Clinical Feature Analysis:
----------------------------------------

Age Distribution:
  Mean: 66.2 years
  Median: 66.0 years
  Range: 53 - 82
  Age Groups: {'<18': 0, '18-40': 0, '40-65': 7, '65+': 11}

Sex Distribution:
  Female: 56.5%
  Male: 43.5%

Medical History (Indications):
  - Prophylaxis: 28 reports
  - Diffuse large B-cell lymphoma refractory: 23 reports
  - Premedication: 10 reports
  - Diffuse large B-cell lymphoma: 6 reports
  - Disseminated intravascular coagulation: 2 reports

Concomitant Drugs:
  - EPCORITAMAB-BYSP: 60 reports
  - PREDNISOLONE: 8 reports
  - PREDNISOLONE ORAL: 8 reports
  - PREDNISOLONE ORAL SOLUTION: 8 reports
  - DEXAMETHASONE: 7 reports

Outcome Distribution:
  - Death: 24 reports
  - Hospitalization: 24 reports
  - Life-threatening: 24 reports
  - Disability: 24 reports

======================================================================
Causal Risk Factor Analysis
======================================================================
Comparing AE group vs Control group to identify factors that INFLUENCE/CAUSE the AE
----------------------------------------------------------------------

Analyzing risk factors for Epcoritamab + Neutropenia
------------------------------------------------------------
Fetching reports for Epcoritamab + Neutropenia...
✓ Found 24 reports
Fetching reports for Epcoritamab + Fatigue...
✓ Found 13 reports
Selected control AE: Fatigue (balanced sex distribution + sample size)

====================================================================================================
Risk Factor Analysis
====================================================================================================
Risk Factor                              AE Group            Control Group       Difference      Conclusion        
----------------------------------------------------------------------------------------------------
Age                                      66.2 years          66.9 years          -0.7 years      Not a risk factor
Sex                                      Female 56.5%        Female 66.7%        -10.1%          Male associated (OR=1.54)
Medical History: DLBCL refractory       95.8%               15.4%               +80.4%         High risk factor
Medical History: Prophylaxis           116.7%               76.9%               +39.7%         Risk factor
Concomitant Drug: Prednisolone Oral    33.3%                0.0%                +33.3%         Risk factor
====================================================================================================

Group Sizes:
  AE Group (with Neutropenia): 24 reports
  Control Group (Fatigue): 13 reports
```

### Python API Usage

```python
from task3_bert_clinical_features import ClinicalFeatureExtractor

# Initialize extractor
extractor = ClinicalFeatureExtractor(use_bert=False)

# Get reports for drug-event combination
reports = extractor.get_reports_for_drug_event("Epcoritamab", "Neutropenia", limit=50)

if reports:
    # Extract clinical features
    features = extractor.extract_clinical_features(reports)
    analysis = extractor.analyze_features(features)
    
    # Identify risk factors (compares AE group vs control group)
    risk_analysis = extractor.identify_risk_factors("Epcoritamab", "Neutropenia")
    
    # Print table format
    extractor.print_risk_factor_table(risk_analysis)
```

### Understanding the Risk Factor Table

The table compares patients who **experienced** the AE vs. those who **did not** (control group), identifying clinical features that influence or cause the adverse event:

- **Risk Factor**: Clinical feature being analyzed (age, sex, medical history, concomitant drugs)
- **AE Group**: Value/prevalence in patients who experienced the AE
- **Control Group**: Value/prevalence in patients who did not experience the AE
- **Difference**: Difference between AE group and control group
- **Conclusion**: Interpretation of whether the factor is a risk factor

**Example Interpretation:**
- **DLBCL refractory**: 95.8% in AE group vs 15.4% in control → +80.4% difference → **High risk factor**
- **Prednisolone Oral**: 33.3% in AE group vs 0.0% in control → +33.3% difference → **Risk factor**
- **Age**: 66.2 years vs 66.9 years → -0.7 years difference → **Not a risk factor**

---

## Notes

- All queries require the results CSV file (`task3_all_unexpected_no_cap.csv`) to be generated first
- The system automatically loads the most recent results file
- Case-insensitive matching is used for drug and event names
- The `check_any_combo()` method can check ANY combination, even if not in the results file
- BERT clinical feature analysis fetches data in real-time from OpenFDA API (no pre-generated CSV required)
- The control group is automatically selected based on balanced sex distribution and sample size

