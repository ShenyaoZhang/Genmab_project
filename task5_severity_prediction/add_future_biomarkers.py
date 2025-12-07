#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conceptual Module: Future Biomarker Integration
===============================================

This module demonstrates how the pipeline can be extended to include
biomarkers such as IL-6, CRP, ferritin, etc.

This is a conceptual module for slides - showing pipeline extensibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def add_biomarkers(df, biomarker_dict):
    """
    Add biomarker data to the dataset.
    
    This function demonstrates how biomarkers can be integrated into the pipeline.
    In production, biomarker data would come from lab results, electronic health records,
    or other clinical data sources.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with patient data
    biomarker_dict : dict
        Dictionary mapping patient IDs (or indices) to biomarker values
        Format: {patient_id: {biomarker_name: value, ...}}
        Or: {biomarker_name: pd.Series or list of values}
    
    Returns:
    --------
    pd.DataFrame : DataFrame with biomarker columns added
    
    Example:
    --------
    biomarker_data = {
        'IL6': [10.5, 25.3, 8.2, ...],  # IL-6 levels
        'CRP': [5.2, 15.8, 3.1, ...],   # C-reactive protein
        'ferritin': [150, 300, 120, ...]  # Ferritin levels
    }
    df = add_biomarkers(df, biomarker_data)
    """
    df = df.copy()
    
    # Handle different input formats
    if isinstance(biomarker_dict, dict):
        # Check if it's a dict of Series/lists (biomarker_name -> values)
        if all(isinstance(v, (list, pd.Series, np.ndarray)) for v in biomarker_dict.values()):
            # Format: {biomarker_name: [values]}
            for biomarker_name, values in biomarker_dict.items():
                if len(values) == len(df):
                    df[biomarker_name] = values
                else:
                    print(f"⚠️  Warning: {biomarker_name} length mismatch ({len(values)} vs {len(df)})")
                    # Pad or truncate
                    if len(values) < len(df):
                        # Pad with NaN
                        values_extended = list(values) + [np.nan] * (len(df) - len(values))
                        df[biomarker_name] = values_extended
                    else:
                        # Truncate
                        df[biomarker_name] = values[:len(df)]
        
        # Check if it's a dict of dicts (patient_id -> {biomarker: value})
        elif all(isinstance(v, dict) for v in biomarker_dict.values()):
            # Format: {patient_id: {biomarker_name: value}}
            # Need to match by patient ID
            if 'safetyreportid' in df.columns:
                for biomarker_name in set(b for b_dict in biomarker_dict.values() for b in b_dict.keys()):
                    df[biomarker_name] = df['safetyreportid'].map(
                        lambda x: biomarker_dict.get(x, {}).get(biomarker_name, np.nan)
                    )
            else:
                print("⚠️  Warning: Cannot match biomarkers - 'safetyreportid' column not found")
    
    return df


def create_biomarker_features(df, biomarker_cols=None):
    """
    Create derived features from biomarker values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with biomarker columns
    biomarker_cols : list, optional
        List of biomarker column names. If None, auto-detect.
    
    Returns:
    --------
    pd.DataFrame : DataFrame with biomarker features added
    """
    df = df.copy()
    
    # Common biomarkers for CRS and cytokine-related conditions
    if biomarker_cols is None:
        biomarker_cols = ['IL6', 'CRP', 'ferritin', 'd_dimer', 'procalcitonin']
    
    # Extract only columns that exist
    existing_biomarkers = [col for col in biomarker_cols if col in df.columns]
    
    if not existing_biomarkers:
        print("⚠️  No biomarker columns found. Returning original dataframe.")
        return df
    
    print(f"🔬 Creating features from {len(existing_biomarkers)} biomarkers...")
    
    # Example: IL-6 features
    if 'IL6' in df.columns:
        # High IL-6 threshold (e.g., >40 pg/mL for severe CRS)
        df['IL6_high'] = (pd.to_numeric(df['IL6'], errors='coerce') > 40).astype(int)
        df['IL6_elevated'] = (pd.to_numeric(df['IL6'], errors='coerce') > 10).astype(int)
    
    # Example: CRP features
    if 'CRP' in df.columns:
        # High CRP threshold (e.g., >10 mg/L)
        df['CRP_high'] = (pd.to_numeric(df['CRP'], errors='coerce') > 10).astype(int)
        df['CRP_elevated'] = (pd.to_numeric(df['CRP'], errors='coerce') > 3).astype(int)
    
    # Example: Ferritin features
    if 'ferritin' in df.columns:
        # High ferritin threshold (e.g., >500 ng/mL)
        df['ferritin_high'] = (pd.to_numeric(df['ferritin'], errors='coerce') > 500).astype(int)
    
    # Combined biomarker score
    biomarker_flags = [col for col in df.columns if any(b in col for b in existing_biomarkers) and col.endswith('_high')]
    if biomarker_flags:
        df['biomarker_severity_score'] = df[biomarker_flags].sum(axis=1)
    
    print(f"✅ Created {len([c for c in df.columns if any(b in c for b in existing_biomarkers + ['biomarker'])])} biomarker features")
    
    return df


# Example usage for slides
if __name__ == '__main__':
    print("=" * 80)
    print("Conceptual Module: Future Biomarker Integration")
    print("=" * 80)
    print()
    
    # Create sample dataframe
    sample_df = pd.DataFrame({
        'safetyreportid': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'patientage': [65, 72, 58, 70, 68],
        'num_drugs': [3, 5, 2, 4, 3]
    })
    
    # Example biomarker data
    biomarker_data = {
        'IL6': [45.2, 12.5, 38.9, 28.3, 15.7],      # IL-6 levels (pg/mL)
        'CRP': [15.8, 8.2, 12.5, 9.1, 6.3],         # C-reactive protein (mg/L)
        'ferritin': [520, 180, 450, 320, 210]       # Ferritin (ng/mL)
    }
    
    print("📊 Original DataFrame:")
    print(sample_df)
    print()
    
    # Add biomarkers
    df_with_biomarkers = add_biomarkers(sample_df, biomarker_data)
    print("📊 DataFrame with Biomarkers:")
    print(df_with_biomarkers)
    print()
    
    # Create biomarker features
    df_with_features = create_biomarker_features(df_with_biomarkers)
    print("📊 DataFrame with Biomarker Features:")
    print(df_with_features[['safetyreportid', 'IL6', 'IL6_high', 'CRP', 'CRP_high', 
                             'ferritin', 'ferritin_high', 'biomarker_severity_score']])
    print()
    
    print("=" * 80)
    print("✅ Biomarker integration module ready")
    print("=" * 80)
    print()
    print("💡 This module demonstrates pipeline extensibility:")
    print("   - Biomarkers can be added at any stage")
    print("   - Features can be derived from biomarker values")
    print("   - Pipeline remains flexible for future enhancements")

