# Complete Variable List and Data Sources

## Overview

This document lists all variables used in the survival analysis pipeline, their data types, processing methods, and availability across different databases.

---

## Variable Categories

### 1. Demographic Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **patient_age** | Continuous | Patient age in years | z-score normalization OR bucketing (<50, 50-65, >65) | ✅ Yes | ✅ Yes | ✅ Yes |
| **patient_weight** | Continuous | Patient weight in kg | z-score normalization: (x-μ)/σ | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **patient_sex** | Categorical | Patient sex (M/F/Unknown) | One-hot encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **weight_group** | Ordinal | Weight categories | Bucketed: <60kg, 60-80kg, 80-100kg, >100kg | ✅ Derived | ⚠️ Limited | ⚠️ Limited |
| **age_group** | Ordinal | Age categories | Bucketed: <50, 50-65, >65 | ✅ Derived | ✅ Derived | ✅ Derived |

**Processing Example for Weight:**
```python
# Z-score normalization
weight_mean = df['patient_weight'].mean()  # e.g., 75.3 kg
weight_std = df['patient_weight'].std()    # e.g., 18.2 kg
df['weight_zscore'] = (df['patient_weight'] - weight_mean) / weight_std

# For Patient 203 with weight=92 kg:
# weight_zscore = (92 - 75.3) / 18.2 = 0.92
# Interpretation: 0.92 standard deviations above mean
```

**Processing Example for Weight Groups:**
```python
# Bucketing
df['weight_group'] = pd.cut(df['patient_weight'],
                            bins=[0, 60, 80, 100, 200],
                            labels=['<60kg', '60-80kg', '80-100kg', '>100kg'])

# For Patient 203 with weight=92 kg:
# weight_group = '80-100kg'
```

---

### 2. Clinical Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **is_serious** | Binary | Serious adverse event flag | 0/1 encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **is_lifethreatening** | Binary | Life-threatening outcome | 0/1 encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **is_hospitalization** | Binary | Required hospitalization | 0/1 encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **is_death** | Binary | Death outcome | 0/1 encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **is_disabling** | Binary | Disabling outcome | 0/1 encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **event_occurred** | Binary | Target AE occurred | 0/1 encoding | ✅ Derived | ✅ Derived | ✅ Derived |

---

### 3. Adverse Event Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **has_crs** | Binary | Cytokine Release Syndrome | Search in reactions using keywords | ✅ Yes | ✅ Yes | ✅ Yes |
| **has_severe_crs** | Binary | Severe CRS (Grade 3-4) | Based on seriousness codes | ✅ Derived | ✅ Derived | ✅ Derived |
| **reactions** | Text | All reported reactions | Text search/NLP | ✅ Yes | ⚠️ Japanese text | ✅ Limited |
| **time_to_event_days** | Continuous | Days from drug start to AE | Calculated from dates | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **time_adjusted** | Continuous | Adjusted time (0→0.5) | time_adjusted = max(time_to_event, 0.5) | ✅ Derived | ✅ Derived | ✅ Derived |

**Processing Example for Time-to-Event:**
```python
# Handle zero or negative times (reporting artifacts)
df['time_adjusted'] = df['time_to_event_days'].fillna(0.5)
df.loc[df['time_adjusted'] <= 0, 'time_adjusted'] = 0.5

# For Patient A with time_to_event_days = 0:
# time_adjusted = 0.5 days (same-day event)
```

---

### 4. Drug Exposure Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **total_drugs** | Count | Total number of drugs | Count from drug records | ✅ Yes | ✅ Yes | ✅ Yes |
| **concomitant_drugs** | Count | Number of concomitant drugs | Count where drug_characterization=2 | ✅ Yes | ✅ Yes | ✅ Yes |
| **polypharmacy** | Binary | ≥3 drugs flag | 0 if <3 drugs, 1 if ≥3 drugs | ✅ Derived | ✅ Derived | ✅ Derived |
| **drug_characterization** | Categorical | Primary suspect (1) vs concomitant (2) | Categorical encoding | ✅ Yes | ✅ Yes | ✅ Yes |
| **drug_start_date** | Date | Drug start date | Date parsing (YYYYMMDD) | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **drug_end_date** | Date | Drug end date | Date parsing (YYYYMMDD) | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **dose** | Continuous | Drug dose | Parsed from text OR separate field | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| **dose_changes** | Binary | Dose modification occurred | Derived from dose unit/text | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |

**Processing Example for Polypharmacy:**
```python
# Binary classification
df['polypharmacy'] = (df['total_drugs'] >= 3).astype(int)

# For Patient B with 5 drugs:
# polypharmacy = 1 (yes)
# Clinical interpretation: Higher complexity, potential drug interactions
```

---

### 5. Comorbidity Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **has_hypertension** | Binary | Hypertension diagnosis | Search in indications/medical history | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| **has_cardiac_disease** | Binary | Cardiac disease | Search in indications/medical history | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| **has_diabetes** | Binary | Diabetes diagnosis | Search in indications/medical history | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| **has_renal_impairment** | Binary | Renal impairment | Search in indications/medical history | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |

**Note:** FAERS has limited structured comorbidity data. These are often extracted from narrative text or drug indications.

---

### 6. Disease-Specific Variables

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **cancer_type** | Categorical | Type of cancer | Extracted from indication field | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| **disease_stage** | Ordinal | Cancer stage (I, II, III, IV) | Ordinal encoding: I=1, II=2, III=3, IV=4 | ❌ No | ❌ No | ❌ No |
| **prior_lines_therapy** | Count | Number of prior treatments | Extracted from narrative if available | ❌ No | ❌ No | ❌ No |
| **ECOG_status** | Ordinal | Performance status (0-4) | Ordinal encoding | ❌ No | ❌ No | ❌ No |

**How these would be integrated if available:**
```python
# Example: DLBCL Stage integration
stage_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df['disease_stage_numeric'] = df['disease_stage'].map(stage_mapping)

# Then use in Cox model:
# Higher stage → higher risk (typically)
```

---

### 7. NLP-Derived Features

| Variable | Type | Description | Processing | FAERS | JADER | EudraVigilance |
|----------|------|-------------|------------|-------|-------|----------------|
| **narrative_text** | Text | Case narrative | NLP: sentiment, entity extraction | ✅ Yes | ❌ Japanese | ✅ Limited |
| **narrative_length** | Count | Word count of narrative | len(text.split()) | ✅ Derived | ❌ No | ✅ Derived |
| **mentions_cytokines** | Binary | Mentions IL-6, IL-2, etc. | Keyword search in narrative | ✅ Derived | ❌ No | ✅ Derived |
| **sentiment_score** | Continuous | Severity sentiment (-1 to 1) | NLP sentiment analysis | ✅ Derived | ❌ No | ✅ Derived |

**Note:** Narrative text processing:
- **FAERS:** English text, available for most cases
- **JADER:** Japanese text, requires translation
- **EudraVigilance:** Limited narrative availability

---

## Data Availability Summary

### FAERS (FDA Adverse Event Reporting System)
- ✅ **Strengths:** Comprehensive demographics, drug exposure data, narrative text
- ⚠️ **Limitations:** Limited dose information, no lab values, no disease staging
- 📊 **Completeness:** ~85% have core demographics, ~60% have weight data

### JADER (Japanese Adverse Drug Event Report)
- ✅ **Strengths:** Asian population data, drug exposure
- ⚠️ **Limitations:** Japanese language narratives, limited temporal data
- 📊 **Completeness:** ~80% have core demographics, ~30% have weight data

### EudraVigilance (European Medicines Agency)
- ✅ **Strengths:** European population, standardized reporting
- ⚠️ **Limitations:** More restricted access, limited narratives
- 📊 **Completeness:** ~90% have core demographics, ~40% have weight data

---

## Missing Data Handling

### Strategy by Variable Type

**Continuous Variables (age, weight):**
```python
# Option 1: Median imputation
df['patient_age'].fillna(df['patient_age'].median(), inplace=True)

# Option 2: Drop missing (used in this analysis)
df_clean = df.dropna(subset=['patient_age', 'patient_weight'])
```

**Categorical Variables:**
```python
# Create "Unknown" category
df['patient_sex'].fillna('Unknown', inplace=True)
```

**Binary Variables:**
```python
# Treat missing as False (conservative)
df['is_hospitalization'].fillna(0, inplace=True)
```

---

## Example: Patient Feature Vector

**Patient ID: 203**

```python
{
    # Demographics
    'patient_age': 68,                    # years
    'patient_weight': 92,                 # kg
    'weight_zscore': 0.92,                # normalized
    'patient_sex': 'M',                   # male
    'age_group': '>65',                   # bucketed
    'weight_group': '80-100kg',           # bucketed
    
    # Clinical
    'is_serious': 1,                      # yes
    'is_lifethreatening': 0,              # no
    'is_hospitalization': 1,              # yes
    'is_death': 0,                        # no
    
    # Drug Exposure
    'total_drugs': 5,                     # count
    'concomitant_drugs': 4,               # count
    'polypharmacy': 1,                    # yes (≥3)
    
    # Adverse Event
    'has_crs': 1,                         # yes
    'time_to_event_days': 1,              # day 1
    'time_adjusted': 1.0,                 # adjusted
    
    # Derived (if available)
    'disease_stage_numeric': 3,           # Stage III (if available)
    'mentions_cytokines': 1               # narrative mentions IL-6
}
```

**Clinical Interpretation for Patient 203:**
- 68-year-old male with moderate weight
- Received 5 drugs (polypharmacy)
- Developed CRS on Day 1
- Required hospitalization but not life-threatening
- Weight of 92 kg may have reduced CRS risk (HR=0.992 per kg)

---

## Variable Addition: How to Integrate New Variables

### Example: Adding Laboratory Values

If lab data becomes available (e.g., IL-6, ferritin):

```python
# 1. Add to variable list
new_variables = ['IL6_baseline', 'ferritin_baseline', 'CRP_baseline']

# 2. Process continuous lab values
df['IL6_log'] = np.log10(df['IL6_baseline'] + 1)  # log-transform
df['IL6_high'] = (df['IL6_baseline'] > 5).astype(int)  # binary threshold

# 3. Add to Cox model
features = features + ['IL6_log', 'ferritin_baseline', 'CRP_baseline']

# 4. Interpret
# Example: IL-6 > 5 pg/mL increases CRS risk by 45% (HR=1.45)
```

See [BIOMARKER_INTEGRATION.md](BIOMARKER_INTEGRATION.md) for detailed biomarker integration strategy.

---

## References

- FDA FAERS Data Dictionary: https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers
- PMDA JADER: https://www.pmda.go.jp/english/
- EudraVigilance: https://www.ema.europa.eu/en/human-regulatory/research-development/pharmacovigilance/eudravigilance

---

**Last Updated:** 2025-11-18  
**Version:** 1.0

