#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI Mock Function: Signal Detection Example
==========================================

Simple function for UI mockup demonstration.
Returns signal detection results that can be visualized in slides.

Usage:
    result = check_signal(drug="epcoritamab", adverse_event="CRS")
    print(result)
"""

import pandas as pd
import os


def check_signal(drug, adverse_event, data_file='main_data.csv', threshold=10):
    """
    Check for unexpected rare signal in FAERS data.
    
    Parameters:
    -----------
    drug : str
        Drug name (e.g., "epcoritamab", "pembrolizumab")
    adverse_event : str
        Adverse event name (e.g., "CRS", "pneumonia")
    data_file : str
        Path to FAERS data file (default: 'main_data.csv')
    threshold : int
        Minimum number of occurrences to flag as signal (default: 10)
    
    Returns:
    --------
    str : Signal detection message
    """
    if not os.path.exists(data_file):
        return f"Data file not found: {data_file}"
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Filter for target drug
    drug_mask = df['target_drug'].str.contains(drug, case=False, na=False)
    drug_df = df[drug_mask]
    
    if len(drug_df) == 0:
        return f"No records found for drug: {drug}"
    
    # Identify adverse event cases
    ae_keywords = get_ae_keywords(adverse_event)
    
    if 'reactions' in drug_df.columns:
        reactions_upper = drug_df['reactions'].fillna('').str.upper()
        ae_mask = pd.Series(False, index=drug_df.index)
        
        for keyword in ae_keywords:
            mask = reactions_upper.str.contains(keyword.upper(), na=False)
            ae_mask |= mask
        
        ae_cases = drug_df[ae_mask]
        n_ae = len(ae_cases)
        n_total = len(drug_df)
        percentage = (n_ae / n_total * 100) if n_total > 0 else 0
    else:
        n_ae = 0
        n_total = len(drug_df)
        percentage = 0
    
    # Determine signal status
    if n_ae >= threshold:
        signal_status = "Unexpected rare signal detected"
        severity = "HIGH"
    elif n_ae > 0:
        signal_status = "Potential signal observed"
        severity = "MEDIUM"
    else:
        signal_status = "No signal detected"
        severity = "LOW"
    
    # Construct message
    message = (
        f"{signal_status}: {adverse_event} observed {n_ae} times in {n_total} FAERS reports "
        f"for {drug} ({percentage:.1f}%). "
        f"Signal severity: {severity}."
    )
    
    return message


def get_ae_keywords(adverse_event):
    """Map adverse event name to search keywords."""
    ae_keyword_map = {
        'CRS': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        'pneumonia': ['PNEUMONIA', 'PNEUMONITIS'],
        'cytokine release syndrome': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        # Add more as needed
    }
    
    # Try exact match
    if adverse_event.upper() in ae_keyword_map:
        return ae_keyword_map[adverse_event.upper()]
    
    # Try case-insensitive match
    for key, keywords in ae_keyword_map.items():
        if adverse_event.upper() == key.upper():
            return keywords
    
    # Default: use the AE name itself
    return [adverse_event.upper()]


# Example usage for slides
if __name__ == '__main__':
    print("=" * 80)
    print("UI Mock: Signal Detection Function")
    print("=" * 80)
    print()
    
    # Example 1: CRS in Epcoritamab
    result1 = check_signal(drug="epcoritamab", adverse_event="CRS")
    print("Example 1:")
    print(f"check_signal(drug='epcoritamab', adverse_event='CRS')")
    print(f"Result: {result1}")
    print()
    
    # Example 2: Pneumonia in Pembrolizumab
    result2 = check_signal(drug="pembrolizumab", adverse_event="pneumonia")
    print("Example 2:")
    print(f"check_signal(drug='pembrolizumab', adverse_event='pneumonia')")
    print(f"Result: {result2}")
    print()
    
    print("=" * 80)
    print("✅ UI mock function ready for slide demonstration")
    print("=" * 80)

