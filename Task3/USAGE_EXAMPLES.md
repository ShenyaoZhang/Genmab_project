# Task 3 Usage Examples

Command-line usage examples with expected outputs.

---

## Example 1: Check a Specific Drug-Event Combination

**Command:**
```bash
cd task3
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Haemorrhagic gastroenteritis"
```

**Output:**
```
✓ Loaded 1386 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (58,296 reports) for arbitrary queries
======================================================================
Drug-Event Query: Epcoritamab + Haemorrhagic gastroenteritis
======================================================================

Status: RARE & UNEXPECTED
Observed in: FAERS
Report Count: 1

Statistical Metrics:
  - PRR (Proportional Reporting Ratio): 111.69 (threshold: >2)
  - IC025 (Information Component): 3.602 (threshold: >0)
  - Chi-square: 41.14 (threshold: >4)

Clinical Impact:
  - Death Rate: 100.0%
  - Hospitalization Rate: 100.0%
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
✓ Loaded 1386 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (58,296 reports) for arbitrary queries
======================================================================
Top 10 Rare & Unexpected AEs for Epcoritamab
======================================================================

1. Haemorrhagic gastroenteritis
   Anomaly Score: 0.689
   Count: 1
   PRR: 111.69
   IC025: 3.602

2. Psoas abscess
   Anomaly Score: 0.689
   Count: 1
   PRR: 111.69
   IC025: 3.602

3. Cytomegalovirus enterocolitis
   Anomaly Score: 0.662
   Count: 2
   PRR: 62.05
   IC025: 3.531

4. Adrenomegaly
   Anomaly Score: 0.640
   Count: 1
   PRR: 111.69
   IC025: 3.602

5. Enterobacter bacteraemia
   Anomaly Score: 0.640
   Count: 1
   PRR: 111.69
   IC025: 3.602

6. Spinal stenosis
   Anomaly Score: 0.631
   Count: 3
   PRR: 148.46
   IC025: 4.140

7. Hyperferritinaemia
   Anomaly Score: 0.629
   Count: 1
   PRR: 111.69
   IC025: 3.602

8. Peritoneal haematoma
   Anomaly Score: 0.620
   Count: 1
   PRR: 111.69
   IC025: 3.602

9. Abdominal wall haematoma
   Anomaly Score: 0.620
   Count: 1
   PRR: 111.69
   IC025: 3.602

10. Pulmonary infarction
   Anomaly Score: 0.612
   Count: 1
   PRR: 111.69
   IC025: 3.602
```

---

## Example 3: Compare Multiple Drugs

**Command:**
```bash
python3 task3_interactive_query.py --compare "Epcoritamab,Glofitamab,Mosunetuzumab"
```

**Output:**
```
✓ Loaded 1386 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (58,296 reports) for arbitrary queries
======================================================================
Drug Comparison: Rare & Unexpected AE Counts
======================================================================
         drug                                   adverse_event  count  anomaly_score
  Epcoritamab                    Haemorrhagic gastroenteritis      1       0.689063
  Epcoritamab                                   Psoas abscess      1       0.689063
  Epcoritamab                   Cytomegalovirus enterocolitis      2       0.662268
  Epcoritamab                                    Adrenomegaly      1       0.639748
  Epcoritamab                        Enterobacter bacteraemia      1       0.639748
  Epcoritamab                                 Spinal stenosis      3       0.630767
   Glofitamab                Blood immunoglobulin M decreased      3       0.644422
   Glofitamab                Blood immunoglobulin A decreased      3       0.644422
   Glofitamab                    Choroidal neovascularisation      3       0.644422
Mosunetuzumab                       CD4 lymphocytes decreased      1       0.660105
Mosunetuzumab                        Gastrointestinal fistula      1       0.654325
Mosunetuzumab                  Wrong patient received product      2       0.652358
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
✓ Loaded 1386 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (58,296 reports) for arbitrary queries
======================================================================
Found 10 drug-event pairs with 'infection':
======================================================================
1. Atezolizumab + Infective exacerbation of chronic obstructive airways disease (Count: 1, Score: 0.643)
2. Durvalumab + Cholangitis infective (Count: 2, Score: 0.631)
3. Venetoclax + Infective exacerbation of bronchiectasis (Count: 1, Score: 0.626)
4. Pembrolizumab + Meningitis noninfective (Count: 1, Score: 0.582)
5. Carfilzomib + Phlebitis infective (Count: 1, Score: 0.563)
6. Bevacizumab + Infected skin ulcer (Count: 3, Score: 0.542)
7. Abemaciclib + Infected bite (Count: 1, Score: 0.541)
8. Epcoritamab + Infectious pleural effusion (Count: 1, Score: 0.539)
9. Pomalidomide + Arthritis infective (Count: 2, Score: 0.537)
10. Atezolizumab + Infectious pleural effusion (Count: 1, Score: 0.516)
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

## Example 6: Check Another Drug-Event Combination (NOT Rare/Unexpected)

**Command:**
```bash
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Renal impairment"
```

**Output:**
```
✓ Loaded 1386 anomaly results from task3_all_unexpected_no_cap.csv
✓ Loaded raw data (58,296 reports) for arbitrary queries
======================================================================
Drug-Event Query: Epcoritamab + Renal impairment
======================================================================

Status: NOT RARE/UNEXPECTED
Observed in: FAERS
Report Count: 8

Statistical Metrics:
  - PRR (Proportional Reporting Ratio): 2.84 (threshold: >2)
  - IC025 (Information Component): 0.562 (threshold: >0)
  - Chi-square: 7.68 (threshold: >4)

Clinical Impact:
  - Death Rate: 137.5%
  - Hospitalization Rate: 175.0%
  - Serious Rate: 100.0%

Assessment:
  - In No-Cap Results: No
  - Passes PRR Test (>2): Yes
  - Passes IC025 Test (>0): Yes
  - Passes Chi² Test (>4): Yes
  - Passes All 3 Tests: Yes

======================================================================
CONCLUSION: NOT RARE/UNEXPECTED (did not pass Isolation Forest)
======================================================================
```

**Note:** This example shows a case where the drug-event combination passes all statistical tests but did not pass the Isolation Forest anomaly detection step, so it is not classified as rare & unexpected.

---

## Notes

- All commands should be run from the `task3/` directory
- Interactive query commands require the results CSV file (`task3_all_unexpected_no_cap.csv`) to be generated first
- BERT clinical feature analysis fetches data in real-time from OpenFDA API (no pre-generated CSV required)
- Case-insensitive matching is used for drug and event names
