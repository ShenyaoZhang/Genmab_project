# Cancer Stage (DLBCL Stage) Integration Documentation

## Current Status: Structured Stage Data Not Available in FAERS

**DLBCL stage is NOT available as a structured variable in FAERS.**

FAERS does not have standardized cancer stage fields. We have implemented an interface to extract stage information from free-text fields, but this is imperfect.

---

## Current Implementation

### Stage Extraction Method

Currently, we attempt to extract cancer stage information from the `drug_indication` field using pattern matching:

- **Source Field**: `patient.drug.drugindication` (free-text field in FAERS)
- **Method**: Regular expression pattern matching
- **Patterns**: 
  - Stage I: "STAGE I", "STAGE 1"
  - Stage II: "STAGE II", "STAGE 2"
  - Stage III: "STAGE III", "STAGE 3"
  - Stage IV: "STAGE IV", "STAGE 4"

### Limitations

1. **Incomplete Coverage**: Many reports do not explicitly state cancer stage in the indication field
2. **Text Pattern Matching**: May miss variations in stage notation
3. **False Positives**: Pattern matching may occasionally match non-cancer stage mentions
4. **Missing Data**: A large proportion of cases will have missing stage information

### Current Usage

- Stage features (`cancer_stage_I`, `cancer_stage_II`, `cancer_stage_III`, `cancer_stage_IV`) are extracted in:
  - `03_preprocess_data.py` - General preprocessing
  - `12_crs_model_training.py` - CRS-specific feature engineering
  - `11_granular_crs_analysis.py` - Granular analysis

- In the CRS mortality model, **cancer_stage_III** was identified as the 7th most important feature, suggesting clinical relevance despite imperfect extraction.

---

## Future Integration: Structured Stage Data

### Pipeline Interface Design

Our pipeline is designed to seamlessly integrate structured cancer stage data when it becomes available:

### If Structured Stage Field Available (Numeric 1-4):

```python
# Example: Structured stage data becomes available
if 'cancer_stage' in df.columns and df['cancer_stage'].notna().any():
    # Use structured numeric stage directly
    df['cancer_stage_numeric'] = pd.to_numeric(df['cancer_stage'], errors='coerce')
    
    # Create derived features
    df['advanced_stage'] = ((df['cancer_stage_numeric'] >= 3) & 
                            (df['cancer_stage_numeric'] <= 4)).astype(int)
    df['early_stage'] = ((df['cancer_stage_numeric'] >= 1) & 
                         (df['cancer_stage_numeric'] <= 2)).astype(int)
    
    # Create binary stage indicators
    df['cancer_stage_I'] = (df['cancer_stage_numeric'] == 1).astype(int)
    df['cancer_stage_II'] = (df['cancer_stage_numeric'] == 2).astype(int)
    df['cancer_stage_III'] = (df['cancer_stage_numeric'] == 3).astype(int)
    df['cancer_stage_IV'] = (df['cancer_stage_numeric'] == 4).astype(int)
```

### Integration Points

The pipeline is ready to accept structured stage data at the following points:

1. **Data Extraction (`01_extract_data.py`)**: Can add stage extraction if available in FAERS API response
2. **Preprocessing (`03_preprocess_data.py`)**: Already checks for `cancer_stage` column and uses structured data if available
3. **CRS Feature Engineering (`12_crs_model_training.py`)**: Checks for structured `cancer_stage` field first, falls back to text extraction
4. **Granular Analysis (`11_granular_crs_analysis.py`)**: Uses stage features from preprocessing or extracts from text

### Expected Data Format

- **Field Name**: `cancer_stage` or `dlbcl_stage`
- **Type**: Numeric (1, 2, 3, 4) or categorical ("I", "II", "III", "IV")
- **Missing Values**: NaN or null for unknown stages

---

## Clinical Significance

Despite the limitations of current extraction:

- **Stage III** emerged as the 7th most important feature in the CRS mortality model
- Advanced stage (III-IV) is associated with higher CRS-related mortality
- The pipeline can leverage structured stage data immediately when available

---

## Summary

**Current State**: 
- Stage information extracted from free-text (imperfect)
- Features included in model (cancer_stage_I/II/III/IV)
- Stage III identified as important predictor

**Future Ready**: 
- Pipeline interface reserved for structured stage data
- Automatic fallback: structured data → text extraction → missing
- No code changes needed when structured data becomes available

