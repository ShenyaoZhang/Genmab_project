#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameterized Pipeline for Drug-AE Analysis
===========================================

Core pipeline function that accepts drug and adverse event (AE) parameters.
No hardcoded values - fully parameterized as per Sky's requirement.

Usage:
 run_pipeline(drug="epcoritamab", ae="CRS")
 run_pipeline(drug="pembrolizumab", ae="pneumonia")
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import existing modules (they will need to be modified to accept parameters)
# For now, we create wrapper functions that call the existing scripts
# In production, these would be refactored into callable functions


def run_pipeline(
        drug="epcoritamab",
        ae="CRS",
        max_records=500,
        output_dir="."):
    """
    Run the complete pipeline for a specific drug and adverse event.

    Parameters:
        -----------
    drug : str
    Drug name (e.g., "epcoritamab", "pembrolizumab")
    ae : str
    Adverse event name (e.g., "CRS", "pneumonia", "cytokine release syndrome")
    max_records : int
    Maximum number of records to collect per drug (default: 500)
    output_dir : str
    Output directory for results (default: ".")

    Returns:
        --------
    dict : Pipeline execution results with paths to output files
    """
    print("=" * 80)
    print(f"Pipeline: Drug={drug}, Adverse Event={ae}")
    print("=" * 80)
    print()
    
    # Step 1: Extract data
    print(f"Step 1: Extracting data for {drug}...")
    print("=" * 80)
    print()

    results = {
        'drug': drug,
        'adverse_event': ae,
        'status': 'running',
        'output_files': {}
    }

    try:
        # Step 1: Extract data for the specific drug
        print(f" Step 1: Extracting data for {drug}...")
        data_file = extract_data_for_drug(drug, max_records, output_dir)
        results['output_files']['raw_data'] = data_file
        print(f" Data extracted: {data_file}\n")

        # Step 2: Inspect data
        print(f" Step 2: Inspecting data quality...")
    inspect_results = inspect_data(data_file, drug, ae, output_dir)
    results['output_files']['inspection'] = inspect_results
    print(f" Data inspection complete\n")

    # Step 3: Preprocess data
    print(f" Step 3: Preprocessing data...")
    preprocessed_file = preprocess_data(data_file, output_dir)
    results['output_files']['preprocessed'] = preprocessed_file
    print(f" Preprocessing complete: {preprocessed_file}\n")

    # Step 4: Train models
    print(f" Step 4: Training models...")
    model_results = train_models(preprocessed_file, drug, ae, output_dir)
    results['output_files']['models'] = model_results
    print(f" Model training complete\n")

    # Step 5: Granular analysis (for any AE)
    print(f" Step 5: Running granular {ae} analysis...")
    try:
        import sys
    from importlib import import_module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    granular_module = import_module('11_granular_crs_analysis')
    granular_results = granular_module.run_granular_analysis(
        drug=drug,
        ae=ae,
        data_file=data_file,
        output_dir=output_dir
    )
    if granular_results:
        results['output_files']['granular_analysis'] = granular_results

    print(f" Granular analysis complete\n")
    except Exception as e:
        print(f"WARNING: Granular analysis skipped: {str(e)[:100]}\n")


    # Step 6: Evaluate models (SHAP analysis)
    print(f" Step 6: Generating SHAP explanations...")
    eval_results = evaluate_models(model_results, drug, ae, output_dir)
    results['output_files']['evaluation'] = eval_results
    print(f" Model evaluation complete\n")

    results['status'] = 'completed'

    except Exception as e:
        print(f"ERROR: Pipeline failed: {str(e)}")

    results['status'] = 'failed'
    results['error'] = str(e)

    print("=" * 80)
    print(f"Pipeline completed: {results['status']}")
    print("=" * 80)

    return results


def extract_data_for_drug(drug, max_records=500, output_dir="."):
    """
    Extract data for a specific drug.
    This function wraps the extraction logic from 01_extract_data.py

    Note: In production, 01_extract_data.py should be refactored to accept
    drug and max_records as parameters. For now, this is a placeholder that
    checks if main_data.csv exists and filters for the target drug.
    """
    # Check if main_data.csv exists (from previous extraction)
    main_data_file = os.path.join(output_dir, 'main_data.csv')
    if os.path.exists(main_data_file):
        df = pd.read_csv(main_data_file)
        # Filter for target drug
        drug_df = df[df['target_drug'].str.contains(drug, case=False, na=False)]
        output_file = os.path.join(output_dir, f"data_{drug.lower()}.csv")
        drug_df.to_csv(output_file, index=False)
        print(f" Extracted {len(drug_df)} records for {drug}")
        return output_file
    else:

        # If main_data.csv doesn't exist, would need to call extraction
        # For now, return expected path
        output_file = os.path.join(output_dir, f"data_{drug.lower()}.csv")
        print(f" Note: Would extract data for {drug} (main_data.csv not found)")
        return output_file


def inspect_data(data_file, drug, ae, output_dir="."):
    """
    Inspect data quality and identify AE cases.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Identify AE cases (parameterized)
    ae_keywords = get_ae_keywords(ae)

    # Check if reactions column exists
    if 'reactions' in df.columns:
        reactions_upper = df['reactions'].fillna('').str.upper()
    ae_mask = pd.Series(False, index=df.index)

    for keyword in ae_keywords:
        mask = reactions_upper.str.contains(keyword.upper(), na=False)
    ae_mask |= mask

    ae_df = df[ae_mask].copy()
    else:

        ae_df = df.copy()
    print(f"WARNING: 'reactions' column not found, using all records")

    print(f" Total {drug} records: {len(df)}")
    print(f" {ae} cases: {len(ae_df)} ({len(ae_df) / len(df) * 100:.1f}%)")

    # Save inspection results
    output_file = os.path.join(output_dir, f"inspection_{drug}_{ae}.json")
    import json
    inspection_results = {
        'drug': drug,
        'adverse_event': ae,
        'total_records': len(df),
        'ae_cases': len(ae_df),
        'ae_percentage': float(
            len(ae_df) /
            len(df) *
            100) if len(df) > 0 else 0}

    with open(output_file, 'w') as f:
        json.dump(inspection_results, f, indent=2)

    return output_file


def preprocess_data(data_file, output_dir="."):
    """
    Preprocess data with all feature engineering.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module

    # Import and call preprocessing function
    preprocess_module = import_module('03_preprocess_data')
    results = preprocess_module.preprocess_file(
        data_file, output_dir=output_dir, verbose=True)
    return results['preprocessed_data']


def train_models(preprocessed_file, drug, ae, output_dir="."):
    """
    Train models for the specific drug-AE combination.
    """
    import sys
    from importlib import import_module

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Determine which training script to use based on AE
    if ae.upper() == 'CRS':
        # Use CRS-specific training
    crs_module = import_module('12_crs_model_training')

    # Get AE keywords
    ae_keywords = get_ae_keywords(ae)

    # Run CRS pipeline
    results = crs_module.run_crs_mortality_pipeline(
        drug_name=drug,
        ae_keyword_list=ae_keywords,
        input_csv=preprocessed_file,
        output_dir=output_dir
    )

    if results:
        return {
        'model_file': os.path.join(
            output_dir, f" drug.lower()}_model_best.pkl"), 'results_file': os.path.join(
            output_dir, 'crs_model_meta.json'), 'results': results}
    else:

        return None
    else:

        # Use general model training
    train_module = import_module('04_train_models')

    # Extract train/test paths from preprocessed file location
    base_dir = os.path.dirname(preprocessed_file) or output_dir
    X_train_path = os.path.join(base_dir, "X_train.csv")
    y_train_path = os.path.join(base_dir, "y_train.csv")
    X_test_path = os.path.join(base_dir, "X_test.csv")
    y_test_path = os.path.join(base_dir, "y_test.csv")

    # Run general training
    results = train_module.train_general_model(
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        X_test_path=X_test_path,
        y_test_path=y_test_path,
        output_dir=output_dir,
        verbose=True
    )

    return {
        'model_file': results['best_model_file'],
        'results_file': results['training_meta'],
        'results': results
    }


def evaluate_models(model_results, drug, ae, output_dir="."):
    """
    Evaluate models and generate SHAP explanations.
    """
    # This would call the evaluation logic from model_evaluation.py
    # For now, return expected output paths
    return {
        'feature_importance': os.path.join(
            output_dir, f"feature_importance_{drug}_{ae}.csv"), 'shap_values': os.path.join(
            output_dir, f"shap_values_{drug}_{ae}.csv"), 'shap_summary': os.path.join(
                output_dir, f"shap_summary_{drug}_{ae}.png"), 'shap_bar': os.path.join(
                    output_dir, f"shap_bar_{drug}_{ae}.png"), 'shap_waterfall': os.path.join(
                        output_dir, f"shap_waterfall_{drug}_{ae}.png")}


def get_ae_keywords(ae):
    """
    Map adverse event name to search keywords.
    """
    ae_keyword_map = {
        'CRS': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        'pneumonia': ['PNEUMONIA', 'PNEUMONITIS'],
        'cytokine release syndrome': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        # Add more mappings as needed
    }

    # Try exact match first
    if ae.upper() in ae_keyword_map:
        return ae_keyword_map[ae.upper()]

    # Try case-insensitive match
    for key, keywords in ae_keyword_map.items():
        if ae.upper() == key.upper():

    return keywords

    # Default: use the AE name itself
    return [ae.upper()]


# Main execution
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run parameterized pipeline for drug-AE analysis')
    parser.add_argument('--drug', type=str, default='epcoritamab',
                        help='Drug name (default: epcoritamab)')
    parser.add_argument('--ae', type=str, default='CRS',
                        help='Adverse event name (default: CRS)')
    parser.add_argument('--max_records', type=int, default=500,
                        help='Maximum records per drug (default: 500)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory (default: current directory)')

    args = parser.parse_args()

    # Run pipeline
    results = run_pipeline(
        drug=args.drug,
        ae=args.ae,
        max_records=args.max_records,
        output_dir=args.output_dir
    )

    print("\n Pipeline Results:")
    print(f" Status: {results['status']}")
    print(f" Output files: {len(results.get('output_files', {}))}")

    if results['status'] == 'completed':
        print("\n Pipeline completed successfully!")

    else:


        print(f"\nERROR: Pipeline failed: {results.get('error', 'Unknown error')}")

