#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 10: Prepare CRS Dataset for Analysis
==========================================

Parameterized function to prepare CRS dataset with feature engineering:
- Cancer stage interface reservation
- Continuous variable processing (age, weight, BMI, num_drugs)
- Missingness summary generation
- Export prepared dataset for downstream analysis

This script provides the data preparation step for CRS-specific analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def prepare_crs_dataset(drug_name="epcoritamab",
                        ae_keywords=None,
                        full_faers_path="main_data.csv",
                        output_path="crs_dataset_prepared.csv"):
    """
    Prepare CRS dataset with feature engineering.
    
    Parameters:
    -----------
    drug_name : str, default="epcoritamab"
        Target drug name (case-insensitive)
    ae_keywords : list, optional
        List of keywords to identify adverse event (default: CRS keywords)
        Example: ["CYTOKINE RELEASE SYNDROME", "CYTOKINE RELEASE", "CYTOKINE STORM"]
    full_faers_path : str, default="main_data.csv"
        Path to full FAERS dataset
    output_path : str, default="crs_dataset_prepared.csv"
        Path to save prepared dataset
    
    Returns:
    --------
    pd.DataFrame or None
        Prepared CRS dataset, or None if preparation fails
    """
    DEFAULT_CRS_KEYWORDS = ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM']
    
    if ae_keywords is None:
        ae_keywords = DEFAULT_CRS_KEYWORDS
    
    print("=" * 80)
    print(f"Preparing CRS Dataset: {drug_name} → {ae_keywords[0] if ae_keywords else 'Adverse Event'}")
    print("=" * 80)
    print()
    
    # Load data
    if not Path(full_faers_path).exists():
        print(f"❌ Data file not found: {full_faers_path}")
        return None
    
    print(f"📂 Loading data from {full_faers_path}...")
    df = pd.read_csv(full_faers_path)
    print(f"✅ Loaded {len(df)} records")
    print()
    
    # Step 1: Filter to target drug
    print("🔹 Step 1: Filtering to target drug...")
    drug_mask = df['target_drug'].str.contains(drug_name, case=False, na=False)
    drug_df = df[drug_mask].copy()
    
    if len(drug_df) == 0:
        print(f"❌ No {drug_name} records found.")
        return None
    
    print(f"   Found {len(drug_df)} records for {drug_name}")
    print()
    
    # Step 2: Identify adverse event cases
    print("🔹 Step 2: Identifying adverse event cases...")
    reactions_upper = drug_df['reactions'].fillna('').str.upper()
    ae_mask = pd.Series(False, index=drug_df.index)
    
    for keyword in ae_keywords:
        mask = reactions_upper.str.contains(keyword.upper(), na=False, regex=False)
        ae_mask |= mask
    
    drug_df['has_ae'] = ae_mask.astype(int)
    ae_df = drug_df[ae_mask].copy()
    
    print(f"   Found {len(ae_df)} records with target adverse event ({len(ae_df)/len(drug_df)*100:.1f}%)")
    print()
    
    # Step 3: Feature engineering with continuous variable processing
    print("🔹 Step 3: Feature engineering...")
    feature_df = apply_feature_engineering(ae_df.copy())
    
    # Step 4: Generate missingness summary
    print("🔹 Step 4: Generating missingness summary...")
    missingness_summary = generate_missingness_summary(feature_df)
    
    # Save missingness summary
    missingness_file = output_path.replace('.csv', '_missingness_summary.csv')
    missingness_summary.to_csv(missingness_file, index=True)
    print(f"✅ Saved: {missingness_file}")
    print()
    
    # Step 5: Save prepared dataset
    print("🔹 Step 5: Saving prepared dataset...")
    feature_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(feature_df)} records)")
    print()
    
    return feature_df

def apply_feature_engineering(df):
    """
    Apply comprehensive feature engineering with continuous variable processing.
    
    CONTINUOUS VARIABLE PROCESSING LOGIC:
    - Age (age_years): Used both as continuous and binary indicators (age_gt_65, age_gt_70, age_gt_75)
    - Weight (patientweight): Kept as continuous in original units (kg), no scaling needed for tree models
      Missing values filled with median, weight_missing flag created
    - BMI: Calculated from weight, kept as continuous feature
      Binary buckets created: <18.5 underweight, 18.5-25 normal, 25-30 overweight, >30 obese
    - Number of drugs (num_drugs): Used as continuous count
      Binary indicators created: polypharmacy, high_polypharmacy, etc.
    - Number of reactions (num_reactions): Used as continuous count
      Binary flags: multiple_reactions, many_reactions
    """
    feature_df = df.copy()
    
    # 1. Age features (continuous + binary indicators)
    if 'age_years' in feature_df.columns:
        age = pd.to_numeric(feature_df['age_years'], errors='coerce')
        feature_df['age_missing'] = age.isna().astype(int)
        feature_df['age_years'] = age.fillna(age.median())
        
        # Binary indicators for high-risk thresholds
        feature_df['age_gt_65'] = (feature_df['age_years'] > 65).astype(int)
        feature_df['age_gt_70'] = (feature_df['age_years'] > 70).astype(int)
        feature_df['age_gt_75'] = (feature_df['age_years'] > 75).astype(int)
    
    # 2. Weight normalization and BMI features
    if 'patientweight' in feature_df.columns:
        # Weight: Keep as continuous in original units (kg), no scaling for tree models
        weight = pd.to_numeric(feature_df['patientweight'], errors='coerce')
        feature_df['weight_missing'] = weight.isna().astype(int)
        median_weight = weight.median()
        feature_df['patientweight'] = weight.fillna(median_weight)
        
        # BMI: Calculate from weight (assuming default height if not available)
        height_default = 1.65  # meters
        feature_df['bmi'] = feature_df['patientweight'] / (height_default ** 2)
        
        # BMI buckets: Binary indicators based on thresholds
        feature_df['bmi_underweight'] = (feature_df['bmi'] < 18.5).astype(int)
        feature_df['bmi_normal'] = ((feature_df['bmi'] >= 18.5) & (feature_df['bmi'] < 25)).astype(int)
        feature_df['bmi_overweight'] = ((feature_df['bmi'] >= 25) & (feature_df['bmi'] < 30)).astype(int)
        feature_df['bmi_obese'] = (feature_df['bmi'] >= 30).astype(int)
    
    # 3. Cancer stage interface reservation
    # NOTE: DLBCL stage is NOT available as a structured variable in FAERS.
    # We reserve an interface for future structured stage data.
    if 'cancer_stage' not in feature_df.columns:
        # Reserve placeholder for structured stage data (if available in future)
        feature_df['cancer_stage'] = np.nan
    
    # If structured cancer_stage field exists (numeric 1-4), use it directly
    if 'cancer_stage' in feature_df.columns:
        stage_numeric = pd.to_numeric(feature_df['cancer_stage'], errors='coerce')
        if not stage_numeric.isna().all():
            feature_df['cancer_stage_numeric'] = stage_numeric.fillna(0)
            # Create advanced_stage indicator (Stage III/IV)
            feature_df['advanced_stage'] = ((stage_numeric >= 3) & (stage_numeric <= 4)).astype(int)
    
    # Attempt extraction from free-text drug_indication (imperfect)
    if 'drug_indication' in feature_df.columns:
        indication_upper = feature_df['drug_indication'].fillna('').str.upper()
        
        # Initialize stage binary features
        feature_df['cancer_stage_I'] = 0
        feature_df['cancer_stage_II'] = 0
        feature_df['cancer_stage_III'] = 0
        feature_df['cancer_stage_IV'] = 0
        
        # Extract from text patterns (imperfect)
        feature_df['cancer_stage_I'] = indication_upper.str.contains(r'STAGE\s+[I1](\s|$|,|\|)', regex=True, na=False).astype(int)
        feature_df['cancer_stage_II'] = indication_upper.str.contains(r'STAGE\s+[I2]{2}(\s|$|,|\|)', regex=True, na=False).astype(int)
        feature_df['cancer_stage_III'] = indication_upper.str.contains(r'STAGE\s+[I3]{3}(\s|$|,|\|)', regex=True, na=False).astype(int)
        feature_df['cancer_stage_IV'] = indication_upper.str.contains(r'STAGE\s+[IV4](\s|$|,|\|)', regex=True, na=False).astype(int)
    
    # 4. Polypharmacy (continuous count + binary indicators)
    if 'num_drugs' in feature_df.columns:
        num_drugs = pd.to_numeric(feature_df['num_drugs'], errors='coerce').fillna(1)
        feature_df['num_drugs'] = num_drugs
        
        # Binary indicators
        feature_df['polypharmacy'] = (feature_df['num_drugs'] > 1).astype(int)
        feature_df['high_polypharmacy'] = (feature_df['num_drugs'] > 5).astype(int)
        feature_df['moderate_polypharmacy'] = ((feature_df['num_drugs'] >= 2) & (feature_df['num_drugs'] <= 5)).astype(int)
        feature_df['very_high_polypharmacy'] = (feature_df['num_drugs'] > 10).astype(int)
    
    # 5. Number of reactions (continuous count + binary flags)
    if 'num_reactions' in feature_df.columns:
        num_reactions = pd.to_numeric(feature_df['num_reactions'], errors='coerce').fillna(1)
        feature_df['num_reactions'] = num_reactions
        
        # Binary flags
        feature_df['multiple_reactions'] = (feature_df['num_reactions'] > 1).astype(int)
        feature_df['many_reactions'] = (feature_df['num_reactions'] > 3).astype(int)
    
    return feature_df

def generate_missingness_summary(df):
    """
    Generate missingness summary for key model features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    pd.DataFrame
        Missingness summary with column name, missing count, missing percentage
    """
    # Key features to check for missingness
    key_features = [
        'age_years', 'age_missing', 'patientweight', 'weight_missing', 'bmi',
        'bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese',
        'num_drugs', 'polypharmacy', 'high_polypharmacy',
        'num_reactions', 'multiple_reactions',
        'cancer_stage', 'cancer_stage_I', 'cancer_stage_II', 'cancer_stage_III', 'cancer_stage_IV',
        'comorbidity_diabetes', 'comorbidity_hypertension', 'comorbidity_cardiac',
        'has_steroid', 'has_antibiotic', 'has_chemo'
    ]
    
    # Calculate missingness for available features
    missing_data = []
    for col in key_features:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            missing_data.append({
                'Feature': col,
                'Missing_Count': int(missing_count),
                'Missing_Percentage': round(missing_pct, 2),
                'Complete_Count': int(len(df) - missing_count),
                'Complete_Percentage': round(100 - missing_pct, 2)
            })
    
    missingness_df = pd.DataFrame(missing_data)
    missingness_df = missingness_df.sort_values('Missing_Percentage', ascending=False)
    
    return missingness_df

def main():
    """Main execution function (backward compatible)."""
    # Use default parameters
    result = prepare_crs_dataset()
    
    if result is not None:
        print("=" * 80)
        print("✅ CRS Dataset Preparation Complete!")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  - crs_dataset_prepared.csv")
        print(f"  - crs_dataset_prepared_missingness_summary.csv")
    else:
        print("\n❌ Dataset preparation failed.")

if __name__ == '__main__':
    main()

