# Task 3: Rare & Unexpected Adverse Event Detection

Detect rare and unexpected drug-adverse event relationships using Isolation Forest anomaly detection and FDA label filtering.

---

## Overview

Task 3 identifies drug-adverse event combinations that are:
- **Rare**: Low frequency in FAERS reports (count < mean threshold)
- **Unexpected**: Not listed in FDA drug labels
- **Statistically Significant**: Pass multiple disproportionality tests (PRR, IC025, Chi-square)

---

## Detection Pipeline Flowchart

```
All Drug-Event Pairs (from FAERS)
    ↓
Step 1: Isolation Forest Anomaly Detection
    ↓ (Identify statistical anomalies)
Anomaly Scores for All Pairs
    ↓
Step 2: Remove Known Label AEs
    ↓ (Filter against FDA drug labels)
Step 3: Remove Indication-Related Terms
    ↓ (Filter disease terms that are drug indications)
Step 4: Remove High-Frequency AEs
    ↓ (Filter count >= mean count threshold: 3.24)
    ↓
Flagged Rare & Unexpected AEs
```

**For detailed step-by-step explanation, see [DETECTION_STEPS.md](DETECTION_STEPS.md)**

---

## Example: "Epcoritamab + Renal Impairment"

**Why it's flagged as rare & unexpected:**

1. ✅ **Isolation Forest**: Anomaly score 0.78 (unusual pattern)
2. ✅ **Known AE Check**: Renal impairment NOT in Epcoritamab FDA label
3. ✅ **Indication Check**: NOT an indication (it's a condition)
4. ✅ **Frequency Check**: Count = 2, Mean = 3.24 → 2 < 3.24 (rare)

**Result:** ✅ **RARE & UNEXPECTED**

**Reasoning:** Appeared only **twice** in FAERS, is **not** on the drug label, and is below frequency thresholds, so it is flagged as unexpected.

---

## File Structure

- `task3_improved_pipeline.py` - Main pipeline: trains Isolation Forest, generates results
- `task3_drug_label_filter.py` - Filters known AEs and indications, includes MedDRA synonym matching
- `task3_interactive_query.py` - Interactive query system for specific drug-event combinations
- `task3_bert_clinical_features.py` - Clinical feature & risk factor analysis (optional)
- `task3_data_collector.py` - Collects raw drug-event data from OpenFDA
- `task3_data_and_model.py` - Data preprocessing and model construction
- `task3_show_results.py` - Display generated result files
- `config_task3.py` - Configuration parameters (thresholds, top_k, etc.)

---

## Quick Start

### 1. Setup Data Directory

Create a `data/` directory and place:
- Raw drug-event pairs: `task3_oncology_drug_event_pairs.csv`
- (Optional) Model files: `data/models/task3_if_model.joblib`, `task3_scaler.joblib`

### 2. Run Full Pipeline

```bash
cd task3
python3 task3_improved_pipeline.py
```

**Output Files:**
- `data/task3_all_unexpected_no_cap.csv` - All rare & unexpected AEs (no limit)
- `data/task3_unexpected_anomalies.csv` - Top-K results with per-drug cap (max 5 per drug)

### 3. Interactive Query

Check if a specific drug-event combination is rare & unexpected:

```bash
python3 task3_interactive_query.py --drug "Epcoritamab" --adverse_event "Neutropenia"
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
```

**For more usage examples, see [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)**

### 4. Clinical Feature Analysis (Optional)

Analyze clinical risk factors for a specific drug-event combination:

```bash
python3 task3_bert_clinical_features.py "Epcoritamab" "Neutropenia"
```

**Example Output:**
```
====================================================================================================
Risk Factor Analysis
====================================================================================================
Risk Factor                              AE Group            Control Group       Difference      Conclusion        
----------------------------------------------------------------------------------------------------
Age                                      66.2 years          66.9 years          -0.7 years      Not a risk factor
Sex                                      Female 56.5%        Female 66.7%        -10.1%          Male associated (OR=0.65)
Medical History: DLBCL refractory       95.8%               15.4%               +80.4%         High risk factor
Medical History: Prophylaxis           116.7%               76.9%               +39.7%         Risk factor
Concomitant Drug: Prednisolone Oral    33.3%                0.0%                +33.3%         Risk factor
====================================================================================================

Group Sizes:
  AE Group (with Neutropenia): 24 reports
  Control Group (Fatigue): 12 reports
```

This analysis compares patients who experienced the AE vs. those who did not, identifying clinical features (age, sex, medical history, concomitant drugs) that influence or cause the adverse event.

---

## Python API Usage

### Check a Specific Drug-Event Combination

```python
from task3_interactive_query import InteractiveAnomalyQuery

# Initialize query system
query = InteractiveAnomalyQuery()

# Check if combination is rare & unexpected
result = query.check_any_combo("Epcoritamab", "Neutropenia")

# Display results
print(f"Status: {result['conclusion']}")
print(f"Report Count: {result['report_count']}")
print(f"PRR: {result['statistics']['prr']}")
```

### Get Top Anomalies for a Drug

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()
top_aes = query.get_top_events_for_drug("Epcoritamab", n=10)

for idx, row in top_aes.iterrows():
    print(f"{row['adverse_event']}: Count={row['count']}, Score={row['anomaly_score']:.3f}")
```

### Compare Multiple Drugs

```python
from task3_interactive_query import InteractiveAnomalyQuery

query = InteractiveAnomalyQuery()
comparison = query.compare_drugs(["Epcoritamab", "Glofitamab", "Mosunetuzumab"])
print(comparison)
```

---

## Detection Steps Summary

1. **Isolation Forest Anomaly Detection**
   - Identifies statistically unusual drug-event relationships
   - Contamination parameter: 0.15 (15% flagged as anomalies)

2. **Remove Known Label AEs**
   - Fetches FDA drug labels via OpenFDA API
   - Filters out AEs listed in drug labels
   - Uses MedDRA synonym matching for accuracy

3. **Remove Indication-Related Terms**
   - Filters out disease terms that are drug indications
   - Example: "DLBCL" removed (it's an indication, not an AE)

4. **Remove High-Frequency AEs**
   - Mean count threshold: 3.24 (calculated from original raw data)
   - Removes pairs with count >= threshold

**All steps use AND logic** - a pair must pass ALL steps to be flagged as rare & unexpected.

---

## Key Parameters

- **Isolation Forest Contamination:** 0.15
- **Mean Count Threshold:** 3.24
- **Statistical Test Thresholds:**
  - PRR > 2
  - IC025 > 0
  - Chi-square > 4

---

## Output Format

### Simple Status Check

```
Drug: Epcoritamab
Adverse Event: Neutropenia
Status: ✅ RARE & UNEXPECTED
Observed in: FAERS
```

### Detailed Analysis

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

## Documentation

- **[DETECTION_STEPS.md](DETECTION_STEPS.md)** - Detailed step-by-step detection process with examples
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Comprehensive usage examples and code snippets
- **[FEEDBACK_IMPROVEMENT_PLAN.md](FEEDBACK_IMPROVEMENT_PLAN.md)** - Feedback analysis and improvement plan

---

## Notes

- All filtering steps are **additive** (AND logic)
- Pipeline uses **FAERS data only** (FDA Adverse Event Reporting System)
- MedDRA synonym matching ensures better accuracy in known AE filtering
- Mean count threshold (3.24) is calculated from **original raw data**, not filtered results
- No data files included - place your data in `data/` directory

---

## Requirements

See `requirements.txt` for Python dependencies. Key libraries:
- pandas, numpy
- scikit-learn (Isolation Forest)
- scipy (statistical tests)
- requests (OpenFDA API)

---

## License

See project root LICENSE file.
