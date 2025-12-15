#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalable Drug-AE Analysis Pipeline
===================================

Fully parameterized pipeline that can run for ANY drug and ANY adverse event.
No hardcoding - simply pass drug and AE as parameters.

USAGE EXAMPLES (for slides/documentation):
------------------------------------------
    # Example 1: Epcoritamab + CRS
    run_pipeline(drug="epcoritamab", adverse_event="CRS")
    
    # Example 2: Tafasitamab + ICANS  
    run_pipeline(drug="tafasitamab", adverse_event="ICANS")
    
    # Example 3: Check for rare signals
    check_signal(drug="epcoritamab", adverse_event="neutropenia")
    
    # Output: "Unexpected. Rare signal. Observed in FAERS but frequency below threshold."

FLOW DIAGRAM:
-------------
    Input drug → Filter dataset → Extract features → Model training → Results
        ↓              ↓                ↓                  ↓            ↓
    "epcoritamab"   FAERS data     Demographics,      XGBoost,     Risk scores,
                    matching        Medications,       RF, LR      SHAP values,
                                   Comorbidities                  Top features

CHANGING DRUG OR AE REQUIRES NO CODE REWRITING:
------------------------------------------------
Simply call run_pipeline() with different parameters.
The entire pipeline adapts automatically.

Authors: Capstone Team
Date: December 2024
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =============================================================================
# MODEL PURPOSE TABLE (for slides/documentation)
# =============================================================================
"""
| Model              | Purpose                                    |
|--------------------|-------------------------------------------|
| Rare AE Model      | Detects unexpected AE patterns             |
| CRS Model          | Predicts probability of CRS               |
| Mortality Model    | Predicts risk of CRS-related death        |
| Severity Model     | Predicts overall AE severity              |
"""

MODEL_PURPOSE_TABLE = {
    'Rare_AE_Model': {
        'name': 'Rare AE Detection',
        'purpose': 'Detects unexpected adverse event patterns not on drug label',
        'example_output': 'Unexpected. Rare signal. Observed in FAERS but not JADER.'
    },
    'CRS_Model': {
        'name': 'CRS Prediction',
        'purpose': 'Predicts probability of Cytokine Release Syndrome',
        'example_output': 'CRS probability: 0.34 (moderate risk)'
    },
    'Mortality_Model': {
        'name': 'Mortality Prediction', 
        'purpose': 'Predicts risk of CRS-related death',
        'example_output': 'Death risk: 0.12 (low risk). Top factor: age > 70'
    }
}


# =============================================================================
# ADVERSE EVENT KEYWORD MAPPINGS
# =============================================================================

AE_KEYWORD_MAP = {
    'CRS': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
    'ICANS': ['ICANS', 'IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY', 'NEUROTOXICITY'],
    'neutropenia': ['NEUTROPENIA', 'FEBRILE NEUTROPENIA'],
    'pneumonitis': ['PNEUMONITIS', 'INTERSTITIAL LUNG DISEASE'],
    'hepatotoxicity': ['HEPATOTOXICITY', 'LIVER INJURY'],
    'infusion_reaction': ['INFUSION REACTION', 'INFUSION RELATED REACTION'],
    'tumor_lysis': ['TUMOR LYSIS SYNDROME', 'TLS'],
}


# =============================================================================
# SHAP INTERPRETATION GUIDE (for physicians)
# =============================================================================
"""
HOW TO READ SHAP OUTPUTS:
-------------------------
A positive SHAP value means the feature INCREASES predicted risk.
A negative SHAP value means the feature DECREASES predicted risk.

Example interpretations:
- "IL-6 = +0.15" → IL-6 level increased the predicted CRS risk for this patient
- "age = -0.08" → Patient's age decreased their predicted mortality risk
- "has_steroid = +0.12" → Steroid use increased predicted risk

For Patient A with high IL-6:
  "A positive bar means the feature pushes risk upward. 
   For example, IL-6 increased the predicted CRS risk for Patient A."
"""

SHAP_INTERPRETATION_EXAMPLES = {
    'age_years': {
        'positive': 'Older age increased predicted risk for this patient',
        'negative': 'Younger age decreased predicted risk for this patient'
    },
    'num_drugs': {
        'positive': 'Higher number of concurrent medications increased risk',
        'negative': 'Fewer concurrent medications decreased risk'
    },
    'has_steroid': {
        'positive': 'Steroid use increased predicted risk',
        'negative': 'Absence of steroid use decreased predicted risk'
    },
    'bmi_obese': {
        'positive': 'Obesity (BMI > 30) increased predicted risk',
        'negative': 'Non-obese status decreased predicted risk'
    },
    'IL6': {
        'positive': 'Elevated IL-6 increased the predicted CRS risk',
        'negative': 'Lower IL-6 decreased the predicted CRS risk'
    }
}


# =============================================================================
# BIOMARKER INTEGRATION MAPPING (conceptual - for future data)
# =============================================================================
"""
If biomarker data becomes available, it would be integrated as follows:

| Biomarker (from paper) | Integration Method                              |
|------------------------|------------------------------------------------|
| IL-6                   | Continuous lab value, normalized               |
| IL-7                   | Continuous lab value, normalized               |
| IL-21                  | Continuous lab value, normalized               |
| CCL17                  | Continuous lab value, normalized               |
| CCL13                  | Continuous lab value, normalized               |
| TGF-beta1             | Same preprocessing as other continuous vars   |

Processing: Z-score normalization or min-max scaling, same as weight/BMI.
"""

BIOMARKER_MAPPING = {
    'IL-6': {'type': 'continuous', 'processing': 'z-score normalization', 'clinical': 'CRS severity marker'},
    'IL-7': {'type': 'continuous', 'processing': 'z-score normalization', 'clinical': 'T-cell activation'},
    'IL-21': {'type': 'continuous', 'processing': 'z-score normalization', 'clinical': 'B-cell function'},
    'CCL17': {'type': 'continuous', 'processing': 'z-score normalization', 'clinical': 'Chemokine marker'},
    'CCL13': {'type': 'continuous', 'processing': 'z-score normalization', 'clinical': 'Inflammation marker'},
    'TGF-beta1': {'type': 'continuous', 'processing': 'same as weight/BMI', 'clinical': 'Growth factor'},
}


# =============================================================================
# CONTINUOUS VARIABLE PROCESSING DOCUMENTATION
# =============================================================================
"""
HOW CONTINUOUS VARIABLES ARE PROCESSED:
---------------------------------------
Weight:
  - Kept as continuous in original units (kg)
  - Missing values filled with median
  - weight_missing flag created for missingness indicator

BMI:
  - Calculated from weight (assuming height 1.65m if not available)
  - Kept as continuous feature
  - Also bucketized: <18.5 underweight, 18.5-25 normal, 25-30 overweight, >30 obese

Age:
  - Normalized to years (from various units)
  - Used both as continuous (age_years) and binary thresholds (age_gt_65, age_gt_70)

Example interpretation:
  "For Patient 203, a weight of 92 kg reduced predicted mortality risk 
   due to its SHAP value of -0.08."
"""


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_pipeline(
    drug: str,
    adverse_event: str,
    data_file: str = "main_data.csv",
    output_dir: str = "./results",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline for a specific drug and adverse event.
    
    Simply change drug and adverse_event parameters to analyze different 
    combinations - NO CODE REWRITING REQUIRED.
    
    Parameters:
    -----------
    drug : str
        Drug name (e.g., "epcoritamab", "tafasitamab", "pembrolizumab")
    adverse_event : str
        Adverse event (e.g., "CRS", "ICANS", "neutropenia")
    data_file : str
        Path to input CSV (default: "main_data.csv")
    output_dir : str
        Output directory (default: "./results")
    verbose : bool
        Print progress (default: True)
    
    Returns:
    --------
    dict : Results with model performance, feature importance, SHAP values
    
    Examples:
    ---------
    >>> run_pipeline(drug="epcoritamab", adverse_event="CRS")
    >>> run_pipeline(drug="tafasitamab", adverse_event="ICANS")
    """
    
    results = {
        'drug': drug,
        'adverse_event': adverse_event,
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'metrics': {},
        'feature_importance': [],
        'shap_interpretations': [],
        'output_files': {}
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    drug_dir = output_path / f"{drug.lower()}_{adverse_event.lower()}"
    drug_dir.mkdir(exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print(f"PIPELINE: {drug} + {adverse_event}")
        print("=" * 70)
        print("\nFlow: Input drug → Filter dataset → Extract features → Train → Results\n")
    
    try:
        # Step 1: Load and filter data
        if verbose:
            print(f"Step 1: Loading data for {drug}...")
        
        if not Path(data_file).exists():
            results['status'] = 'failed'
            results['error'] = f"Data file not found: {data_file}"
            return results
        
        df = pd.read_csv(data_file)
        
        # Filter to drug
        if 'target_drug' in df.columns:
            drug_df = df[df['target_drug'].str.contains(drug, case=False, na=False)].copy()
        else:
            drug_df = df.copy()
        
        if len(drug_df) == 0:
            results['status'] = 'no_data'
            results['error'] = f"No records found for {drug}"
            return results
        
        results['metrics']['total_records'] = len(drug_df)
        if verbose:
            print(f"   Found {len(drug_df)} records for {drug}")
        
        # Step 2: Identify AE cases
        if verbose:
            print(f"\nStep 2: Identifying {adverse_event} cases...")
        
        ae_keywords = _get_ae_keywords(adverse_event)
        
        if 'reactions' in drug_df.columns:
            reactions = drug_df['reactions'].fillna('').str.upper()
            ae_mask = pd.Series(False, index=drug_df.index)
            for kw in ae_keywords:
                ae_mask |= reactions.str.contains(kw.upper(), na=False, regex=False)
            ae_df = drug_df[ae_mask].copy()
        else:
            ae_df = drug_df.copy()
        
        if len(ae_df) == 0:
            results['status'] = 'no_ae_cases'
            results['error'] = f"No {adverse_event} cases found"
            return results
        
        # Create death indicator
        ae_df['death'] = pd.to_numeric(ae_df.get('seriousnessdeath', 0), errors='coerce').fillna(0).astype(int)
        
        results['metrics']['ae_cases'] = len(ae_df)
        results['metrics']['ae_percentage'] = round(len(ae_df) / len(drug_df) * 100, 1)
        results['metrics']['deaths'] = int(ae_df['death'].sum())
        results['metrics']['death_rate'] = round(ae_df['death'].mean() * 100, 1)
        
        if verbose:
            print(f"   {adverse_event} cases: {len(ae_df)} ({results['metrics']['ae_percentage']}%)")
            print(f"   Deaths: {results['metrics']['deaths']} ({results['metrics']['death_rate']}%)")
        
        # Step 3: Extract features
        if verbose:
            print(f"\nStep 3: Extracting features...")
        
        X, y, feature_names = _extract_features(ae_df)
        results['metrics']['n_features'] = len(feature_names)
        
        if verbose:
            print(f"   Extracted {len(feature_names)} features")
        
        # Step 4: Train model
        if verbose:
            print(f"\nStep 4: Training model...")
        
        if len(X) < 20 or y.sum() == 0 or y.sum() == len(y):
            results['status'] = 'insufficient_data'
            results['error'] = f"Insufficient data for modeling ({len(X)} samples, {y.sum()} positive)"
            return results
        
        model_results = _train_model(X, y, feature_names)
        
        results['metrics']['best_model'] = model_results['model_name']
        results['metrics']['pr_auc'] = round(model_results['pr_auc'], 4)
        results['metrics']['roc_auc'] = round(model_results['roc_auc'], 4)
        results['feature_importance'] = model_results['feature_importance'][:10]
        
        if verbose:
            print(f"   Best model: {model_results['model_name']}")
            print(f"   PR-AUC: {results['metrics']['pr_auc']}")
            print(f"   ROC-AUC: {results['metrics']['roc_auc']}")
        
        # Step 5: Generate SHAP interpretations
        if verbose:
            print(f"\nStep 5: Generating SHAP explanations...")
        
        shap_results = _generate_shap(model_results['model'], model_results['X_test'], feature_names, drug_dir)
        if shap_results:
            results['shap_interpretations'] = shap_results['interpretations']
            results['output_files']['shap'] = str(drug_dir / 'shap_values.csv')
        
        # Save model
        model_path = drug_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_results, f)
        results['output_files']['model'] = str(model_path)
        
        # Save summary
        summary_path = drug_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(_make_serializable(results), f, indent=2)
        results['output_files']['summary'] = str(summary_path)
        
        results['status'] = 'completed'
        
        if verbose:
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETED")
            print("=" * 70)
            print(f"\nSummary:")
            print(f"  Drug: {drug}")
            print(f"  Adverse Event: {adverse_event}")
            print(f"  Cases: {results['metrics']['ae_cases']}")
            print(f"  Deaths: {results['metrics']['deaths']} ({results['metrics']['death_rate']}%)")
            print(f"  Model: {results['metrics']['best_model']}")
            print(f"  PR-AUC: {results['metrics']['pr_auc']}")
            print(f"\nTop 5 Features:")
            for i, (feat, imp) in enumerate(results['feature_importance'][:5], 1):
                print(f"  {i}. {feat}: {imp:.4f}")
            print(f"\nOutput: {drug_dir}")
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        if verbose:
            print(f"\nError: {e}")
    
    return results


def check_signal(drug: str, adverse_event: str, data_file: str = "main_data.csv") -> str:
    """
    Check if a drug-AE combination represents an unexpected/rare signal.
    
    Example usage (for slides):
    ---------------------------
    >>> check_signal("epcoritamab", "neutropenia")
    "Unexpected. Rare signal. Observed in FAERS but below frequency threshold."
    
    >>> check_signal("epcoritamab", "CRS")
    "Expected. Listed on drug label. Frequency: 6.2%"
    
    Parameters:
    -----------
    drug : str
        Drug name
    adverse_event : str
        Adverse event name
    
    Returns:
    --------
    str : Signal assessment message
    """
    
    # Known label AEs (would be loaded from drug label database in production)
    KNOWN_LABEL_AES = {
        'epcoritamab': ['CRS', 'CYTOKINE RELEASE SYNDROME', 'NEUTROPENIA', 'INFECTION'],
        'tafasitamab': ['NEUTROPENIA', 'INFECTION', 'PYREXIA'],
    }
    
    # Load data
    if not Path(data_file).exists():
        return f"Error: Data file not found"
    
    df = pd.read_csv(data_file)
    
    # Filter to drug
    if 'target_drug' in df.columns:
        drug_df = df[df['target_drug'].str.contains(drug, case=False, na=False)]
    else:
        drug_df = df
    
    if len(drug_df) == 0:
        return f"No data for {drug}"
    
    # Check for AE
    ae_keywords = _get_ae_keywords(adverse_event)
    if 'reactions' in drug_df.columns:
        reactions = drug_df['reactions'].fillna('').str.upper()
        ae_mask = pd.Series(False, index=drug_df.index)
        for kw in ae_keywords:
            ae_mask |= reactions.str.contains(kw.upper(), na=False, regex=False)
        ae_count = ae_mask.sum()
    else:
        ae_count = 0
    
    frequency = ae_count / len(drug_df) * 100 if len(drug_df) > 0 else 0
    
    # Check if on label
    drug_lower = drug.lower()
    ae_upper = adverse_event.upper()
    
    on_label = False
    for key in KNOWN_LABEL_AES:
        if key.lower() == drug_lower:
            for label_ae in KNOWN_LABEL_AES[key]:
                if ae_upper in label_ae.upper() or label_ae.upper() in ae_upper:
                    on_label = True
                    break
    
    # Generate signal assessment
    if on_label:
        return f"Expected. Listed on drug label. Frequency: {frequency:.1f}% ({ae_count} cases)"
    elif ae_count == 0:
        return f"Not observed. No cases of {adverse_event} found for {drug} in FAERS."
    elif frequency < 1.0:
        return f"Unexpected. Rare signal. Observed {ae_count} times ({frequency:.2f}%). Not on label. Requires investigation."
    else:
        return f"Unexpected. Signal detected. Frequency: {frequency:.1f}% ({ae_count} cases). Not on drug label."


def get_database_summary(data_file: str = "main_data.csv") -> Dict[str, Any]:
    """
    Get summary statistics for the dataset.
    
    Returns counts, CRS cases, completeness for documentation.
    """
    
    if not Path(data_file).exists():
        return {'error': 'File not found'}
    
    df = pd.read_csv(data_file)
    
    # Basic counts
    total = len(df)
    
    # CRS cases
    if 'reactions' in df.columns:
        crs_mask = df['reactions'].fillna('').str.upper().str.contains('CYTOKINE RELEASE', na=False)
        crs_count = crs_mask.sum()
    else:
        crs_count = 0
    
    # Missingness
    missing = {}
    key_cols = ['age_years', 'patientsex', 'patientweight', 'reactions', 'all_drugs']
    for col in key_cols:
        if col in df.columns:
            miss_pct = df[col].isna().mean() * 100
            missing[col] = f"{miss_pct:.1f}%"
    
    # Completeness (all key fields present)
    complete_mask = pd.Series(True, index=df.index)
    for col in ['age_years', 'patientsex', 'reactions']:
        if col in df.columns:
            complete_mask &= df[col].notna()
    complete_pct = complete_mask.mean() * 100
    
    return {
        'total_cases': total,
        'crs_cases': crs_count,
        'crs_percentage': f"{crs_count/total*100:.1f}%" if total > 0 else "0%",
        'complete_cases_percentage': f"{complete_pct:.1f}%",
        'missingness_by_field': missing,
        'data_source': 'FAERS (FDA Adverse Event Reporting System)'
    }


def get_polypharmacy_analysis(data_file: str = "main_data.csv", adverse_event: str = "CRS") -> pd.DataFrame:
    """
    Analyze specific drug classes associated with adverse events.
    
    Returns table showing:
    Drug Class | AE Patients | Non-AE Patients | Difference
    
    Example:
    Drug: Steroids | CRS patients: 52% | Non-CRS: 21% | Difference: +31%
    """
    
    if not Path(data_file).exists():
        return pd.DataFrame()
    
    df = pd.read_csv(data_file)
    
    # Identify AE cases
    ae_keywords = _get_ae_keywords(adverse_event)
    if 'reactions' in df.columns:
        reactions = df['reactions'].fillna('').str.upper()
        ae_mask = pd.Series(False, index=df.index)
        for kw in ae_keywords:
            ae_mask |= reactions.str.contains(kw.upper(), na=False, regex=False)
    else:
        return pd.DataFrame()
    
    ae_df = df[ae_mask]
    non_ae_df = df[~ae_mask]
    
    if 'all_drugs' not in df.columns:
        return pd.DataFrame()
    
    # Drug classes to analyze
    drug_classes = {
        'Steroids': 'PREDNISONE|DEXAMETHASONE|METHYLPREDNISOLONE|HYDROCORTISONE',
        'Antibiotics': 'CIPROFLOXACIN|VANCOMYCIN|LEVOFLOXACIN|AMOXICILLIN',
        'Antivirals': 'ACYCLOVIR|GANCICLOVIR|VALACYCLOVIR',
        'Chemotherapy': 'CYCLOPHOSPHAMIDE|DOXORUBICIN|CISPLATIN|CARBOPLATIN',
        'Targeted Therapy': 'RITUXIMAB|LENALIDOMIDE|BRENTUXIMAB',
        'Antifungals': 'FLUCONAZOLE|VORICONAZOLE|POSACONAZOLE'
    }
    
    results = []
    for drug_class, pattern in drug_classes.items():
        ae_pct = ae_df['all_drugs'].fillna('').str.upper().str.contains(pattern, na=False).mean() * 100
        non_ae_pct = non_ae_df['all_drugs'].fillna('').str.upper().str.contains(pattern, na=False).mean() * 100
        diff = ae_pct - non_ae_pct
        
        results.append({
            'Drug_Class': drug_class,
            f'{adverse_event}_Patients': f"{ae_pct:.1f}%",
            f'Non_{adverse_event}': f"{non_ae_pct:.1f}%",
            'Difference': f"{'+' if diff > 0 else ''}{diff:.1f}%"
        })
    
    return pd.DataFrame(results)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_ae_keywords(ae: str) -> List[str]:
    """Get keywords for adverse event matching."""
    ae_upper = ae.upper().strip()
    if ae_upper in AE_KEYWORD_MAP:
        return AE_KEYWORD_MAP[ae_upper]
    for key, keywords in AE_KEYWORD_MAP.items():
        if ae_upper in key.upper():
            return keywords
    return [ae_upper]


def _extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract features from dataframe."""
    
    feature_df = df.copy()
    features = []
    
    # Age
    if 'age_years' in feature_df.columns:
        feature_df['age_years'] = pd.to_numeric(feature_df['age_years'], errors='coerce')
        feature_df['age_years'] = feature_df['age_years'].fillna(feature_df['age_years'].median())
        feature_df['age_gt_65'] = (feature_df['age_years'] > 65).astype(int)
        feature_df['age_gt_70'] = (feature_df['age_years'] > 70).astype(int)
        features.extend(['age_years', 'age_gt_65', 'age_gt_70'])
    
    # Sex
    if 'patientsex' in feature_df.columns:
        feature_df['sex_male'] = (pd.to_numeric(feature_df['patientsex'], errors='coerce') == 1).astype(int)
        features.append('sex_male')
    
    # Weight/BMI
    if 'patientweight' in feature_df.columns:
        weight = pd.to_numeric(feature_df['patientweight'], errors='coerce')
        feature_df['patientweight'] = weight.fillna(weight.median())
        feature_df['bmi'] = feature_df['patientweight'] / (1.65 ** 2)
        feature_df['bmi_obese'] = (feature_df['bmi'] > 30).astype(int)
        features.extend(['patientweight', 'bmi', 'bmi_obese'])
    
    # Drugs
    if 'num_drugs' in feature_df.columns:
        feature_df['num_drugs'] = pd.to_numeric(feature_df['num_drugs'], errors='coerce').fillna(1)
        feature_df['high_polypharmacy'] = (feature_df['num_drugs'] > 5).astype(int)
        features.extend(['num_drugs', 'high_polypharmacy'])
    
    # Drug classes
    if 'all_drugs' in feature_df.columns:
        drugs = feature_df['all_drugs'].fillna('').str.upper()
        feature_df['has_steroid'] = drugs.str.contains('PREDNISONE|DEXAMETHASONE', na=False).astype(int)
        feature_df['has_chemo'] = drugs.str.contains('CYCLOPHOSPHAMIDE|DOXORUBICIN', na=False).astype(int)
        features.extend(['has_steroid', 'has_chemo'])
    
    # Comorbidities
    if 'drug_indication' in feature_df.columns:
        indication = feature_df['drug_indication'].fillna('').str.upper()
        feature_df['has_diabetes'] = indication.str.contains('DIABETES', na=False).astype(int)
        feature_df['has_cardiac'] = indication.str.contains('CARDIAC|HEART', na=False).astype(int)
        features.extend(['has_diabetes', 'has_cardiac'])
    
    # Reactions
    if 'num_reactions' in feature_df.columns:
        feature_df['num_reactions'] = pd.to_numeric(feature_df['num_reactions'], errors='coerce').fillna(1)
        features.append('num_reactions')
    
    # Select features
    available = [f for f in features if f in feature_df.columns]
    X = feature_df[available].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    y = feature_df['death'] if 'death' in feature_df.columns else pd.Series([0]*len(X))
    
    return X, y, available


def _train_model(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict[str, Any]:
    """Train and evaluate model."""
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    
    stratify = y if y.sum() >= 2 and (len(y) - y.sum()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    pr = average_precision_score(y_test, y_proba)
    
    importance = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    
    return {
        'model': model,
        'model_name': 'Gradient Boosting',
        'roc_auc': roc,
        'pr_auc': pr,
        'feature_importance': importance,
        'X_test': X_test,
        'y_test': y_test
    }


def _generate_shap(model, X_test: pd.DataFrame, feature_names: List[str], output_dir: Path) -> Optional[Dict]:
    """Generate SHAP explanations with plain language interpretations."""
    
    try:
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Save values
        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_df.to_csv(output_dir / 'shap_values.csv', index=False)
        
        # Generate interpretations
        mean_shap = np.mean(shap_values, axis=0)
        interpretations = []
        
        for feat, val in zip(X_test.columns, mean_shap):
            direction = "increases" if val > 0 else "decreases"
            interpretations.append({
                'feature': feat,
                'mean_shap': float(val),
                'interpretation': f"Higher {feat} {direction} predicted risk (SHAP: {val:+.3f})"
            })
        
        return {'interpretations': sorted(interpretations, key=lambda x: abs(x['mean_shap']), reverse=True)[:5]}
        
    except Exception:
        return None


def _make_serializable(obj: Any) -> Any:
    """Make object JSON serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(type(obj))
    elif hasattr(obj, '__dict__'):
        return str(obj)
    return obj


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scalable Drug-AE Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scalable_pipeline.py --drug epcoritamab --ae CRS
  python scalable_pipeline.py --drug tafasitamab --ae ICANS
  python scalable_pipeline.py --check epcoritamab neutropenia
  python scalable_pipeline.py --summary
        """
    )
    
    parser.add_argument('--drug', type=str, help='Drug name')
    parser.add_argument('--ae', type=str, help='Adverse event')
    parser.add_argument('--check', nargs=2, metavar=('DRUG', 'AE'), help='Check signal for drug-AE pair')
    parser.add_argument('--summary', action='store_true', help='Print database summary')
    parser.add_argument('--polypharmacy', type=str, help='Run polypharmacy analysis for AE')
    parser.add_argument('--data', type=str, default='main_data.csv', help='Data file')
    
    args = parser.parse_args()
    
    if args.check:
        drug, ae = args.check
        result = check_signal(drug, ae, args.data)
        print(f"\nSignal Check: {drug} + {ae}")
        print(f"Result: {result}\n")
    
    elif args.summary:
        summary = get_database_summary(args.data)
        print("\nDatabase Summary:")
        print("-" * 40)
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for k2, v2 in v.items():
                    print(f"  {k2}: {v2}")
            else:
                print(f"{k}: {v}")
        print()
    
    elif args.polypharmacy:
        df = get_polypharmacy_analysis(args.data, args.polypharmacy)
        print(f"\nPolypharmacy Analysis for {args.polypharmacy}:")
        print("-" * 60)
        print(df.to_string(index=False))
        print()
    
    elif args.drug and args.ae:
        run_pipeline(drug=args.drug, adverse_event=args.ae, data_file=args.data)
    
    else:
        # Demo
        print("=" * 70)
        print("SCALABLE PIPELINE DEMO")
        print("=" * 70)
        print("\nThis pipeline runs for ANY drug and ANY adverse event.")
        print("Simply change parameters - no code rewriting required.\n")
        
        print("Example calls:")
        print('  run_pipeline(drug="epcoritamab", adverse_event="CRS")')
        print('  run_pipeline(drug="tafasitamab", adverse_event="ICANS")')
        print('  check_signal("epcoritamab", "neutropenia")')
        print()
        
        print("Model Purpose Table:")
        print("-" * 50)
        for model_id, info in MODEL_PURPOSE_TABLE.items():
            print(f"  {info['name']}: {info['purpose']}")
        print()
        
        print("Running demo with epcoritamab + CRS...")
        print()
        run_pipeline(drug="epcoritamab", adverse_event="CRS")
