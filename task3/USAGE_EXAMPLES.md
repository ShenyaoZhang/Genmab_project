# Task 3 Usage Examples

Command-line usage examples with expected outputs.

---

## Example 1: Check a Specific Drug-Event Combination

**Command:**
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

Status: RARE & UNEXPECTED
Observed in: FAERS
Report Count: 2

Statistical Metrics:
  - PRR (Proportional Reporting Ratio): 15.3 (threshold: >2)
  - IC025 (Information Component): 2.1 (threshold: >0)
  - Chi-square: 8.5 (threshold: >4)

Clinical Impact:
  - Death Rate: 0.0%
  - Hospitalization Rate: 50.0%
  - Serious Rate: 100.0%

Assessment:
  - In No-Cap Results: Yes
  - Passes PRR Test (>2): Yes
  - Passes IC025 Test (>0): Yes
  - Passes Chi² Test (>4): Yes
  - Passes All 3 Tests: Yes

======================================================================
CONCLUSION: RARE & UNEXPECTED (passed IF + all 3 statistical tests)
======================================================================
```

---

## Example 2: Get Top Anomalies for a Drug

**Command:**
```bash
python3 task3_interactive_query.py --top_events "Epcoritamab" --n 10
```

**Output:**
```
✓ Loaded 150 anomaly results from task3_all_unexpected_no_cap.csv

======================================================================
Top 10 Rare & Unexpected AEs for Epcoritamab
======================================================================

1. Renal impairment
   Anomaly Score: 0.782
   Count: 2
   PRR: 15.3
   IC025: 2.1

2. Neutropenia
   Anomaly Score: 0.768
   Count: 2
   PRR: 12.5
   IC025: 1.8

3. Thrombocytopenia
   Anomaly Score: 0.745
   Count: 3
   PRR: 8.2
   IC025: 1.5

...
```

---

## Example 3: Compare Multiple Drugs

**Command:**
```bash
python3 task3_interactive_query.py --compare "Epcoritamab,Glofitamab,Mosunetuzumab"
```

**Output:**
```
✓ Loaded 150 anomaly results from task3_all_unexpected_no_cap.csv

======================================================================
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

**Command:**
```bash
python3 task3_interactive_query.py --search "infection" --top_n 10
```

**Output:**
```
✓ Loaded 150 anomaly results from task3_all_unexpected_no_cap.csv

======================================================================
Found 8 drug-event pairs with 'infection':
======================================================================
1. Epcoritamab + Infection (Count: 3, Score: 0.712)
2. Glofitamab + Serious infection (Count: 2, Score: 0.698)
3. Mosunetuzumab + Bacterial infection (Count: 2, Score: 0.685)
...
```

---

## Example 5: BERT Clinical Feature Analysis

**Command:**
```bash
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

---

## Example 6: Check Another Drug-Event Combination

**Command:**
```bash
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Renal impairment"
```

**Output:**
```
======================================================================
Drug-Event Query: Epcoritamab + Renal impairment
======================================================================

Status: RARE & UNEXPECTED
Observed in: FAERS
Report Count: 2

Statistical Metrics:
  - PRR (Proportional Reporting Ratio): 15.3 (threshold: >2)
  - IC025 (Information Component): 2.1 (threshold: >0)
  - Chi-square: 8.5 (threshold: >4)

Clinical Impact:
  - Death Rate: 0.0%
  - Hospitalization Rate: 50.0%
  - Serious Rate: 100.0%

Assessment:
  - In No-Cap Results: Yes
  - Passes PRR Test (>2): Yes
  - Passes IC025 Test (>0): Yes
  - Passes Chi² Test (>4): Yes
  - Passes All 3 Tests: Yes

======================================================================
CONCLUSION: RARE & UNEXPECTED (passed IF + all 3 statistical tests)
======================================================================
```

---

## Notes

- All commands should be run from the `task3/` directory
- Interactive query commands require the results CSV file (`task3_all_unexpected_no_cap.csv`) to be generated first
- BERT clinical feature analysis fetches data in real-time from OpenFDA API (no pre-generated CSV required)
- Case-insensitive matching is used for drug and event names
