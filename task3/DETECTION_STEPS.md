# Rare & Unexpected AE Detection Steps

This document explains the step-by-step process for detecting rare and unexpected drug-adverse event relationships in Task 3.

---

## Overview Flowchart

```
All Drug-Event Pairs (from FAERS)
    ↓
Step 1: Isolation Forest Anomaly Detection
    ↓
Anomaly Scores for All Pairs
    ↓
Step 2: Remove Known Label AEs
    ↓ (Filter against FDA drug labels)
Step 3: Remove Indication-Related Terms
    ↓ (Filter disease terms that are drug indications)
Step 4: Remove High-Frequency AEs
    ↓ (Filter count >= mean count threshold)
    ↓
Flagged Rare & Unexpected AEs
```

---

## Detailed Steps

### Step 1: Isolation Forest Anomaly Detection

**Purpose:** Identify statistically unusual drug-event relationships using unsupervised machine learning.

**Process:**
- Input: All drug-event pairs from FAERS with statistical features (PRR, IC025, Chi-square, count)
- Model: Isolation Forest (contamination=0.15)
- Output: Anomaly scores for each drug-event pair
- **Rationale:** Isolation Forest identifies patterns that deviate from the norm, flagging rare combinations

**Example:**
- Input: 10,000+ drug-event pairs
- Output: ~1,500 pairs flagged as anomalies (top 15%)

---

### Step 2: Remove Known Label AEs

**Purpose:** Filter out adverse events that are already documented in FDA drug labels.

**Process:**
- Fetch FDA drug labels via OpenFDA API
- Extract adverse reactions sections
- Match AE terms using:
  - Exact matching
  - MedDRA synonym matching (e.g., "thrombocytopenia" ↔ "platelet count decreased")
- Remove any pairs where the AE is listed in the drug's label

**Example:**
- **Input:** "Epcoritamab + Cytokine release syndrome" (anomaly score: 0.85)
- **FDA Label Check:** CRS is listed in Epcoritamab label
- **Result:** **REMOVED** (known AE, not unexpected)

**Counter-Example:**
- **Input:** "Epcoritamab + Renal impairment" (anomaly score: 0.78)
- **FDA Label Check:** Renal impairment NOT listed in Epcoritamab label
- **Result:** **KEPT** (proceeds to next step)

---

### Step 3: Remove Indication-Related Terms

**Purpose:** Filter out terms that are drug indications (diseases treated) rather than adverse events.

**Process:**
- Identify common indication terms (e.g., "DLBCL", "lymphoma", "cancer")
- Remove drug-event pairs where the "event" is actually a disease indication
- **Rationale:** These are not adverse events - they are the conditions the drug is meant to treat

**Example:**
- **Input:** "Epcoritamab + DLBCL" (anomaly score: 0.72)
- **Indication Check:** DLBCL is the indication for Epcoritamab
- **Result:** **REMOVED** (indication, not an AE)

**Counter-Example:**
- **Input:** "Epcoritamab + Neutropenia" (anomaly score: 0.68)
- **Indication Check:** Neutropenia is NOT an indication
- **Result:** **KEPT** (proceeds to next step)

---

### Step 4: Remove High-Frequency AEs

**Purpose:** Ensure only truly rare events are flagged (low report counts).

**Process:**
- Calculate mean count from original raw data: **mean = 3.24**
- Remove any pairs with count >= mean threshold
- **Rationale:** High-frequency events are not "rare" by definition

**Example:**
- **Input:** "Epcoritamab + Fatigue" (count: 45, anomaly score: 0.65)
- **Frequency Check:** 45 >= 3.24 (mean threshold)
- **Result:** **REMOVED** (too common, not rare)

**Counter-Example:**
- **Input:** "Epcoritamab + Renal impairment" (count: 2, anomaly score: 0.78)
- **Frequency Check:** 2 < 3.24 (mean threshold)
- **Result:** **KEPT** (rare frequency)

---

## Complete Example: "Epcoritamab + Renal Impairment"

Let's trace a complete example through all steps:

### Initial State
- **Drug:** Epcoritamab
- **Adverse Event:** Renal impairment
- **Report Count:** 2
- **Anomaly Score:** 0.78 (from Isolation Forest)

### Step 1: Isolation Forest
- Passed: Anomaly score 0.78 indicates unusual pattern
- **Status:** Flagged as anomaly

### Step 2: Known Label AEs
- Checked FDA label for Epcoritamab
- Renal impairment NOT found in label
- **Status:** Not a known AE → **KEPT**

### Step 3: Indication Filter
- Checked if "Renal impairment" is an indication
- It is NOT an indication (it's a disease/condition)
- **Status:** Not an indication → **KEPT**

### Step 4: Frequency Filter
- Count = 2
- Mean threshold = 3.24
- 2 < 3.24 → **RARE**
- **Status:** Below frequency threshold → **KEPT**

### Final Result
**FLAGGED AS RARE & UNEXPECTED**

**Reasoning:**
- Appeared only **twice** in FAERS (rare frequency)
- Is **not** on the drug label (unexpected)
- Is **not** an indication term
- Passed Isolation Forest anomaly detection
- Passed all three statistical tests (PRR > 2, IC025 > 0, Chi² > 4)

---

## Summary Table

| Step | Filter | Purpose | Example Removal |
|------|--------|---------|-----------------|
| 1 | Isolation Forest | Detect statistical anomalies | Low anomaly scores removed |
| 2 | Known Label AEs | Remove documented AEs | "CRS" removed (in label) |
| 3 | Indication Terms | Remove disease indications | "DLBCL" removed (indication) |
| 4 | Frequency Threshold | Remove common events | "Fatigue" removed (count=45) |

---

## Key Parameters

- **Isolation Forest Contamination:** 0.15 (15% flagged as anomalies)
- **Mean Count Threshold:** 3.24 (calculated from original raw data)
- **Statistical Test Thresholds:**
  - PRR > 2
  - IC025 > 0
  - Chi-square > 4

---

## Output Files

After running the pipeline, two output files are generated:

1. **`task3_all_unexpected_no_cap.csv`**
   - All rare & unexpected AEs (no per-drug limit)
   - Complete results for comprehensive analysis

2. **`task3_unexpected_anomalies.csv`**
   - Top-K results with per-drug cap (max 5 per drug)
   - Curated results for presentation/reporting

---

## Notes

- All filtering steps are **additive** (AND logic) - a pair must pass ALL steps to be flagged
- The pipeline uses **FAERS data only** (FDA Adverse Event Reporting System)
- MedDRA synonym matching ensures better accuracy in Step 2 (known AE filtering)
- The mean count threshold (3.24) is calculated from the **original raw data**, not filtered results


