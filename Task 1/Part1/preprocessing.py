"""
Data Preprocessing Module
Handles continuous and ordinal variable preprocessing with clear documentation.

This module addresses feedback on:
- How continuous variables (weight, BMI, dose, lab values) are processed
- Normalization and bucketization strategies
- Clinical variable handling (disease stage)
- Clear examples for each preprocessing step
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PREPROCESSING DOCUMENTATION
# =============================================================================
"""
CONTINUOUS VARIABLE PREPROCESSING STRATEGIES
=============================================

This pipeline uses the following preprocessing approaches for different variable types:

┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Variable Type      │ Method              │ Example                                  │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Age (years)        │ Z-score             │ age_z = (age - mean) / std               │
│                    │                     │ Example: 72 → 0.85 (above average)       │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Weight (kg)        │ Z-score             │ weight_z = (weight - mean) / std         │
│                    │                     │ Example: 92 kg → 1.2 (above average)     │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ BMI (kg/m²)        │ Categorical buckets │ <25: Normal, 25-30: Overweight, >30:Obese│
│                    │ + Ordinal encoding  │ Example: 28.5 → "Overweight" → 1         │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Dose (mg)          │ Log transform +     │ dose_log = log(dose + 1)                 │
│                    │ Z-score             │ Example: 48mg → log(49) → z-score        │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Lab Values         │ Z-score or          │ IL-6: z-score (continuous)               │
│ (if available)     │ Clinical cutoffs    │ CRP: <10, 10-50, >50 (categorical)       │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Disease Stage      │ Ordinal encoding    │ I=1, II=2, III=3, IV=4                   │
│ (DLBCL)            │                     │ Example: "Stage IV" → 4                  │
├────────────────────┼─────────────────────┼──────────────────────────────────────────┤
│ Time to Event      │ Log transform       │ time_log = log(days + 1)                 │
│ (days)             │                     │ Example: 7 days → log(8) = 2.08          │
└─────────────────────────────────────────────────────────────────────────────────────┘

SHAP INTERPRETATION EXAMPLES
============================

Example SHAP interpretation for a safety physician:

"For Patient 203:
  - Weight of 92 kg REDUCED predicted mortality risk (SHAP value: -0.08)
  - Age of 72 years INCREASED predicted mortality risk (SHAP value: +0.15)
  - Steroid premedication REDUCED predicted CRS severity (SHAP value: -0.22)
  
Overall interpretation:
  • Positive SHAP values push the prediction HIGHER (toward worse outcome)
  • Negative SHAP values push the prediction LOWER (toward better outcome)
  • The magnitude indicates the strength of the effect"

"""


class ClinicalPreprocessor:
    """
    Preprocessor for clinical pharmacovigilance data.
    
    Handles:
    - Continuous variables (age, weight, dose)
    - Ordinal variables (disease stage)
    - Categorical variables (sex, country)
    - Missing value imputation
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.statistics = {}
        
    def preprocess_age(self, age: Union[float, pd.Series], 
                       method: str = 'zscore') -> Union[float, pd.Series]:
        """
        Preprocess age variable.
        
        Args:
            age: Age in years (single value or Series)
            method: 'zscore', 'minmax', or 'bucket'
        
        Returns:
            Preprocessed age value(s)
        
        Example:
            >>> preprocessor = ClinicalPreprocessor()
            >>> preprocessor.fit_age(df['age'])
            >>> preprocessor.preprocess_age(72)
            0.85  # 0.85 standard deviations above mean
        """
        if method == 'zscore':
            if 'age' not in self.scalers:
                raise ValueError("Call fit_age first")
            return (age - self.statistics['age_mean']) / self.statistics['age_std']
        
        elif method == 'minmax':
            return (age - self.statistics['age_min']) / (self.statistics['age_max'] - self.statistics['age_min'])
        
        elif method == 'bucket':
            if isinstance(age, pd.Series):
                return pd.cut(age, bins=[0, 50, 65, 75, 100], 
                             labels=['<50', '50-65', '65-75', '>75'])
            else:
                if age < 50:
                    return '<50'
                elif age < 65:
                    return '50-65'
                elif age < 75:
                    return '65-75'
                else:
                    return '>75'
        
        return age
    
    def fit_age(self, age_series: pd.Series):
        """Fit age scaler."""
        valid_ages = age_series.dropna()
        self.statistics['age_mean'] = valid_ages.mean()
        self.statistics['age_std'] = valid_ages.std()
        self.statistics['age_min'] = valid_ages.min()
        self.statistics['age_max'] = valid_ages.max()
        
    def preprocess_weight(self, weight: Union[float, pd.Series],
                          method: str = 'zscore') -> Union[float, pd.Series]:
        """
        Preprocess weight variable.
        
        Args:
            weight: Weight in kg
            method: 'zscore' or 'minmax'
        
        Returns:
            Preprocessed weight
        
        Example:
            >>> preprocessor.preprocess_weight(92)
            1.2  # 1.2 standard deviations above mean (heavier patient)
            
        SHAP Interpretation:
            "For Patient 203, a weight of 92 kg reduced predicted mortality 
            risk due to its SHAP value of -0.08"
        """
        if method == 'zscore':
            return (weight - self.statistics['weight_mean']) / self.statistics['weight_std']
        elif method == 'minmax':
            return (weight - self.statistics['weight_min']) / (self.statistics['weight_max'] - self.statistics['weight_min'])
        return weight
    
    def fit_weight(self, weight_series: pd.Series):
        """Fit weight scaler."""
        valid_weights = weight_series.dropna()
        self.statistics['weight_mean'] = valid_weights.mean()
        self.statistics['weight_std'] = valid_weights.std()
        self.statistics['weight_min'] = valid_weights.min()
        self.statistics['weight_max'] = valid_weights.max()
    
    def preprocess_bmi(self, bmi: Union[float, pd.Series]) -> Union[str, pd.Series]:
        """
        Preprocess BMI into clinical categories.
        
        Buckets:
            <18.5: Underweight
            18.5-25: Normal
            25-30: Overweight
            >30: Obese
        
        Args:
            bmi: BMI value(s)
        
        Returns:
            BMI category
        
        Example:
            >>> preprocessor.preprocess_bmi(28.5)
            'Overweight'  # Will be encoded as 2 in model
        """
        if isinstance(bmi, pd.Series):
            return pd.cut(bmi, 
                         bins=[0, 18.5, 25, 30, 100],
                         labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        else:
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
    
    def encode_bmi_category(self, bmi_category: str) -> int:
        """
        Encode BMI category as ordinal integer.
        
        Encoding:
            Underweight: 0
            Normal: 1
            Overweight: 2
            Obese: 3
        """
        encoding = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        return encoding.get(bmi_category, 1)  # Default to Normal
    
    def preprocess_dose(self, dose: Union[float, pd.Series],
                        method: str = 'log_zscore') -> Union[float, pd.Series]:
        """
        Preprocess dose variable.
        
        For epcoritamab, doses are typically:
        - Priming: 0.16 mg
        - Intermediate: 0.8 mg
        - Full: 24-48 mg
        
        Args:
            dose: Dose in mg
            method: 'log_zscore', 'zscore', or 'categorical'
        
        Returns:
            Preprocessed dose
        
        Example:
            >>> preprocessor.preprocess_dose(48, method='log_zscore')
            1.85  # Log-transformed and standardized
            
            >>> preprocessor.preprocess_dose(48, method='categorical')
            'Full'  # Categorical dose level
        """
        if method == 'log_zscore':
            log_dose = np.log(dose + 1)
            return (log_dose - self.statistics['dose_log_mean']) / self.statistics['dose_log_std']
        
        elif method == 'zscore':
            return (dose - self.statistics['dose_mean']) / self.statistics['dose_std']
        
        elif method == 'categorical':
            if isinstance(dose, pd.Series):
                return pd.cut(dose,
                             bins=[0, 0.5, 5, 100],
                             labels=['Priming', 'Intermediate', 'Full'])
            else:
                if dose <= 0.5:
                    return 'Priming'
                elif dose <= 5:
                    return 'Intermediate'
                else:
                    return 'Full'
        
        return dose
    
    def fit_dose(self, dose_series: pd.Series):
        """Fit dose scaler."""
        valid_doses = dose_series.dropna()
        self.statistics['dose_mean'] = valid_doses.mean()
        self.statistics['dose_std'] = valid_doses.std()
        self.statistics['dose_log_mean'] = np.log(valid_doses + 1).mean()
        self.statistics['dose_log_std'] = np.log(valid_doses + 1).std()
    
    def preprocess_disease_stage(self, stage: Union[str, pd.Series]) -> Union[int, pd.Series]:
        """
        Preprocess disease stage (DLBCL) as ordinal variable.
        
        NOTE: Disease stage is NOT directly available in FAERS/JADER/EV.
        This method shows how it WOULD be integrated if available.
        
        Encoding:
            Stage I: 1
            Stage II: 2
            Stage III: 3
            Stage IV: 4
        
        Args:
            stage: Disease stage string (e.g., "Stage IV", "IV", "4")
        
        Returns:
            Ordinal integer encoding
        
        Example:
            >>> preprocessor.preprocess_disease_stage("Stage IV")
            4
            >>> preprocessor.preprocess_disease_stage("II")
            2
        
        Integration Example:
            If disease stage data becomes available (e.g., from linked EHR data),
            it would be added to the feature matrix as:
            
            df['stage_ordinal'] = df['disease_stage'].apply(preprocess_disease_stage)
            
            This allows the model to capture the ordinal relationship where
            Stage IV > Stage III > Stage II > Stage I in terms of severity.
        """
        stage_map = {
            'I': 1, '1': 1, 'STAGE I': 1, 'STAGE 1': 1,
            'II': 2, '2': 2, 'STAGE II': 2, 'STAGE 2': 2,
            'III': 3, '3': 3, 'STAGE III': 3, 'STAGE 3': 3,
            'IV': 4, '4': 4, 'STAGE IV': 4, 'STAGE 4': 4,
        }
        
        if isinstance(stage, pd.Series):
            return stage.str.upper().map(stage_map).fillna(2)  # Default to Stage II
        else:
            return stage_map.get(str(stage).upper(), 2)
    
    def preprocess_lab_values(self, value: float, lab_type: str,
                              method: str = 'zscore') -> Union[float, str]:
        """
        Preprocess laboratory values.
        
        NOTE: Lab values are NOT available in standard pharmacovigilance data.
        This method shows how they WOULD be processed if integrated from
        clinical trial data or linked EHR data.
        
        Supported lab types:
            - IL-6: Interleukin-6 (key CRS biomarker)
            - CRP: C-reactive protein
            - Ferritin: Iron storage protein
            - LDH: Lactate dehydrogenase
        
        Args:
            value: Lab value
            lab_type: Type of lab test
            method: 'zscore' or 'clinical_cutoff'
        
        Returns:
            Preprocessed value
        
        Example:
            >>> preprocessor.preprocess_lab_values(150, 'IL-6', method='zscore')
            2.3  # 2.3 SD above mean (elevated)
            
            >>> preprocessor.preprocess_lab_values(150, 'IL-6', method='clinical_cutoff')
            'Elevated'  # Above clinical threshold
        """
        # Clinical reference ranges (would be customized per institution)
        reference_ranges = {
            'IL-6': {'mean': 5.0, 'std': 10.0, 'upper_normal': 7.0},
            'CRP': {'mean': 3.0, 'std': 5.0, 'upper_normal': 10.0},
            'Ferritin': {'mean': 150, 'std': 100, 'upper_normal': 300},
            'LDH': {'mean': 200, 'std': 50, 'upper_normal': 250},
        }
        
        if lab_type not in reference_ranges:
            return value
        
        ref = reference_ranges[lab_type]
        
        if method == 'zscore':
            return (value - ref['mean']) / ref['std']
        
        elif method == 'clinical_cutoff':
            if value <= ref['upper_normal']:
                return 'Normal'
            elif value <= ref['upper_normal'] * 2:
                return 'Elevated'
            else:
                return 'High'
        
        return value
    
    def fit(self, df: pd.DataFrame) -> 'ClinicalPreprocessor':
        """
        Fit all scalers on the dataset.
        
        Args:
            df: DataFrame with raw data
        
        Returns:
            Self (fitted preprocessor)
        """
        if 'age' in df.columns:
            self.fit_age(df['age'])
        
        if 'weight' in df.columns:
            self.fit_weight(df['weight'])
        
        if 'max_dose_mg' in df.columns:
            self.fit_dose(df['max_dose_mg'])
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform all continuous variables.
        
        Args:
            df: DataFrame with raw data
        
        Returns:
            DataFrame with preprocessed features
        """
        df_processed = df.copy()
        
        # Age
        if 'age' in df.columns and 'age_mean' in self.statistics:
            df_processed['age_zscore'] = self.preprocess_age(df['age'], method='zscore')
            df_processed['age_group'] = self.preprocess_age(df['age'], method='bucket')
        
        # Weight
        if 'weight' in df.columns and 'weight_mean' in self.statistics:
            df_processed['weight_zscore'] = self.preprocess_weight(df['weight'], method='zscore')
        
        # Dose
        if 'max_dose_mg' in df.columns and 'dose_log_mean' in self.statistics:
            df_processed['dose_log_zscore'] = self.preprocess_dose(df['max_dose_mg'], method='log_zscore')
            df_processed['dose_category'] = self.preprocess_dose(df['max_dose_mg'], method='categorical')
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


def generate_preprocessing_documentation() -> str:
    """Generate documentation for preprocessing steps."""
    
    doc = """
================================================================================
CONTINUOUS AND ORDINAL VARIABLE PREPROCESSING DOCUMENTATION
================================================================================

This document describes exactly how continuous and ordinal variables are 
processed in the CRS risk analysis pipeline.

--------------------------------------------------------------------------------
1. AGE PREPROCESSING
--------------------------------------------------------------------------------

Method: Z-score normalization

Formula: age_z = (age - mean_age) / std_age

Example:
    Raw age: 72 years
    Dataset mean: 65.2 years
    Dataset std: 12.4 years
    Normalized: (72 - 65.2) / 12.4 = 0.55
    
    Interpretation: Patient is 0.55 standard deviations older than average

Code:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['age_normalized'] = scaler.fit_transform(df[['age']])

--------------------------------------------------------------------------------
2. WEIGHT PREPROCESSING
--------------------------------------------------------------------------------

Method: Z-score normalization

Formula: weight_z = (weight - mean_weight) / std_weight

Example:
    Raw weight: 92 kg
    Dataset mean: 75.3 kg
    Dataset std: 14.2 kg
    Normalized: (92 - 75.3) / 14.2 = 1.18
    
    Interpretation: Patient is 1.18 SD heavier than average
    
SHAP Example:
    "For Patient 203, a weight of 92 kg reduced predicted mortality risk 
    due to its SHAP value of -0.08"
    
    This means: Despite being heavier, this patient's higher weight was 
    associated with BETTER outcomes (possibly due to better baseline health)

--------------------------------------------------------------------------------
3. BMI PREPROCESSING
--------------------------------------------------------------------------------

Method: Clinical category bucketing + ordinal encoding

Buckets:
    BMI < 18.5      → Underweight  → 0
    18.5 ≤ BMI < 25 → Normal       → 1
    25 ≤ BMI < 30   → Overweight   → 2
    BMI ≥ 30        → Obese        → 3

Example:
    Raw BMI: 28.5 kg/m²
    Category: Overweight
    Encoded: 2

Code:
    def categorize_bmi(bmi):
        if bmi < 18.5: return 0  # Underweight
        elif bmi < 25: return 1  # Normal
        elif bmi < 30: return 2  # Overweight
        else: return 3           # Obese
    
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)

--------------------------------------------------------------------------------
4. DOSE PREPROCESSING
--------------------------------------------------------------------------------

Method: Log transformation followed by Z-score normalization

Rationale: Drug doses often span orders of magnitude (0.16mg to 48mg for 
epcoritamab). Log transformation makes the distribution more normal.

Formula: 
    dose_log = log(dose + 1)
    dose_normalized = (dose_log - mean_log) / std_log

Example:
    Raw dose: 48 mg
    Log transform: log(49) = 3.89
    Dataset log mean: 2.5
    Dataset log std: 1.2
    Normalized: (3.89 - 2.5) / 1.2 = 1.16
    
    Interpretation: This is a high dose (1.16 SD above mean on log scale)

Alternative: Categorical encoding
    Priming dose (≤0.5 mg)     → 'Priming'
    Intermediate (0.5-5 mg)    → 'Intermediate'  
    Full dose (>5 mg)          → 'Full'

--------------------------------------------------------------------------------
5. DISEASE STAGE PREPROCESSING (When Available)
--------------------------------------------------------------------------------

NOTE: Disease stage (e.g., DLBCL Stage I-IV) is NOT directly available in 
FAERS, JADER, or EudraVigilance. However, it may appear in narrative text
or could be integrated from linked clinical data.

Method: Ordinal encoding

Mapping:
    Stage I   → 1
    Stage II  → 2
    Stage III → 3
    Stage IV  → 4

Example:
    Raw value: "DLBCL Stage IV"
    Encoded: 4
    
    Interpretation: The model treats this as an ordinal variable where 
    Stage IV > Stage III > Stage II > Stage I in terms of severity

Integration Example:
    If stage data becomes available from linked EHR or clinical trial data:
    
    df['stage_ordinal'] = df['disease_stage'].map({
        'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4
    })

--------------------------------------------------------------------------------
6. LAB VALUES PREPROCESSING (Future Integration)
--------------------------------------------------------------------------------

If biomarker data (e.g., IL-6, CRP, ferritin) becomes available, they would 
be processed as follows:

Method A: Z-score normalization
    IL-6_z = (IL-6 - population_mean) / population_std

Method B: Clinical cutoff categorization
    IL-6 ≤ 7 pg/mL      → Normal
    7 < IL-6 ≤ 14       → Elevated  
    IL-6 > 14           → High

Example (IL-6):
    Raw value: 150 pg/mL
    Normal range: <7 pg/mL
    
    Z-score method: (150 - 5) / 10 = 14.5 (very elevated)
    Cutoff method: 'High'

See biomarker_integration.py for detailed biomarker integration documentation.

--------------------------------------------------------------------------------
7. TIME-TO-EVENT PREPROCESSING
--------------------------------------------------------------------------------

Method: Log transformation

Rationale: Time intervals are often right-skewed (many short intervals, 
few long ones). Log transformation normalizes the distribution.

Formula: time_log = log(days + 1)

Example:
    CRS onset: 7 days after first dose
    Log transform: log(8) = 2.08
    
    Interpretation: Allows comparison across different time scales

================================================================================
"""
    return doc


def main():
    """Demonstrate preprocessing steps."""
    
    print(generate_preprocessing_documentation())
    
    # Example with sample data
    print("\n" + "="*70)
    print("PREPROCESSING DEMONSTRATION")
    print("="*70)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'patient_id': [201, 202, 203, 204, 205],
        'age': [65, 72, 58, 81, 45],
        'weight': [70, 92, 65, 58, 85],
        'max_dose_mg': [24, 48, 0.8, 24, 48],
    })
    
    print("\nRaw Data:")
    print(sample_data.to_string(index=False))
    
    # Apply preprocessing
    preprocessor = ClinicalPreprocessor()
    processed_data = preprocessor.fit_transform(sample_data)
    
    print("\nProcessed Data:")
    cols_to_show = ['patient_id', 'age', 'age_zscore', 'weight', 'weight_zscore', 
                    'max_dose_mg', 'dose_log_zscore', 'dose_category']
    available_cols = [c for c in cols_to_show if c in processed_data.columns]
    print(processed_data[available_cols].round(2).to_string(index=False))
    
    # Show SHAP interpretation example
    print("\n" + "="*70)
    print("SHAP INTERPRETATION EXAMPLE")
    print("="*70)
    print("""
For Patient 203:
  Feature              │ Raw Value │ SHAP Value │ Effect on Prediction
  ─────────────────────┼───────────┼────────────┼─────────────────────────
  Weight               │ 92 kg     │ -0.08      │ ↓ Reduces mortality risk
  Age                  │ 72 years  │ +0.15      │ ↑ Increases mortality risk  
  Steroid premedication│ Yes       │ -0.22      │ ↓ Reduces CRS severity
  Dose (full, 48mg)    │ 48 mg     │ +0.18      │ ↑ Increases CRS risk
  
Reading SHAP values:
  • Positive bar (▶▶▶) = Feature pushes risk UPWARD (bad)
  • Negative bar (◀◀◀) = Feature pushes risk DOWNWARD (good)
  • Bar length = Magnitude of effect
  
Example: "The patient's IL-6 level of 150 pg/mL increased the predicted 
CRS risk (SHAP: +0.35), indicating this elevated cytokine level is a 
strong risk factor for this patient."
""")


if __name__ == "__main__":
    main()

