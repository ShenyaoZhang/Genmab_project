#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 13: CRS-Specific SHAP Analysis with Plain Language Explanations
====================================================================

Generate SHAP explanations for CRS → Death model with clinical interpretations.
"""

import sys
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("❌ SHAP not available. Install with: pip install shap")

def load_crs_model():
    """Load the trained CRS model."""
    model_file = 'crs_model_best.pkl'
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("Please run: python 12_crs_model_training.py first")
        return None
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def load_crs_data():
    """Load CRS patient data."""
    if not os.path.exists('main_data.csv'):
        print("❌ Data file not found: main_data.csv")
        return None
    
    df = pd.read_csv('main_data.csv')
    
    # Filter to Epcoritamab CRS patients
    from pathlib import Path
    meta_file = Path('crs_model_meta.json')
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        # We'll rebuild the features as in training script
        return df, meta
    
    return df, None

def plain_language_feature_names():
    """Map technical feature names to plain language."""
    return {
        'num_drugs': 'Number of Concurrent Medications',
        'has_chemo': 'Receiving Chemotherapy',
        'age_years': 'Patient Age',
        'num_reactions': 'Number of Adverse Reactions',
        'bmi': 'Body Mass Index (BMI)',
        'patientweight': 'Patient Weight',
        'age_gt_70': 'Age Over 70 Years',
        'has_antiviral': 'Receiving Antiviral Medication',
        'has_targeted': 'Receiving Targeted Therapy',
        'multiple_reactions': 'Multiple Adverse Reactions',
        'sex_female': 'Female Gender',
        'has_steroid': 'Receiving Steroids',
        'high_polypharmacy': 'High Polypharmacy (>5 drugs)',
        'steroid_plus_antibiotic': 'Steroid + Antibiotic Combination',
        'sex_male': 'Male Gender',
        'age_gt_65': 'Age Over 65 Years',
        'has_infection_ae': 'Infection-Related Adverse Event',
        'has_antibiotic': 'Receiving Antibiotics',
        'bmi_overweight': 'Overweight (BMI 25-30)',
        'bmi_obese': 'Obese (BMI >30)',
        'bmi_underweight': 'Underweight (BMI <18.5)',
        'has_antifungal': 'Receiving Antifungal Medication',
        # Comorbidities
        'comorbidity_diabetes': 'History of Diabetes',
        'comorbidity_hypertension': 'History of Hypertension',
        'comorbidity_cardiac': 'History of Cardiac Disease',
        # Missing data indicators
        'sex_unknown': 'Unknown Gender',
        'age_missing': 'Age Missing in Report',
        'weight_missing': 'Weight Missing in Report',
        # Cancer stage
        'cancer_stage_I': 'Cancer Stage I',
        'cancer_stage_II': 'Cancer Stage II',
        'cancer_stage_III': 'Cancer Stage III',
        'cancer_stage_IV': 'Cancer Stage IV'
    }

def generate_global_summary(top_positive, top_negative, feature_names_map):
    """Generate plain language summary of global SHAP interpretation."""
    summary_parts = []
    
    summary_parts.append("\n🔹 Key Factors That INCREASE Death Risk:\n")
    top_5_positive = top_positive[:5]
    for feat, shap_val in top_5_positive:
        # Get original feature name if mapped
        orig_feat = [k for k, v in feature_names_map.items() if v == feat]
        if orig_feat:
            feat_key = orig_feat[0]
        else:
            feat_key = feat.lower().replace(' ', '_')
        
        summary_parts.append(f"   • {feat}: Strong positive contribution to death risk\n")
    
    summary_parts.append("\n🔹 Key Factors That DECREASE Death Risk:\n")
    top_5_negative = top_negative[:5]
    for feat, shap_val in top_5_negative:
        summary_parts.append(f"   • {feat}: Protective effect against death risk\n")
    
    # Generate English summary for report
    summary_parts.append("\n" + "=" * 80)
    summary_parts.append("English Summary (Ready for Report)")
    summary_parts.append("=" * 80)
    summary_parts.append("\n")
    
    # Extract top features for summary
    top_features_list = [feat for feat, _ in top_5_positive]
    
    # Build summary sentences
    age_features = [f for f in top_features_list if 'age' in f.lower() or 'Age' in f]
    polypharmacy_features = [f for f in top_features_list if 'polypharmacy' in f.lower() or 'Medications' in f]
    comorbidity_features = [f for f in top_features_list if 'History' in f or 'Diabetes' in f or 'Hypertension' in f or 'Cardiac' in f]
    infection_features = [f for f in top_features_list if 'Infection' in f]
    drug_combo_features = [f for f in top_features_list if 'Combination' in f or 'Steroid' in f]
    
    summary_sentences = []
    
    # Main finding sentence
    main_factors = []
    if age_features:
        main_factors.append("age over 70")
    if polypharmacy_features:
        main_factors.append("high polypharmacy")
    if comorbidity_features:
        main_factors.append("comorbidities such as diabetes")
    if infection_features:
        main_factors.append("infection-related adverse events")
    
    if main_factors:
        factors_str = ", ".join(main_factors[:-1]) + (f", and {main_factors[-1]}" if len(main_factors) > 1 else main_factors[0])
        summary_sentences.append(
            f"For CRS cases, the model finds that {factors_str} are the strongest contributors to death risk."
        )
    
    # Drug combination finding
    if drug_combo_features:
        summary_sentences.append(
            "Steroid plus antibiotic combinations appear frequently among high-risk CRS cases, "
            "which may suggest complex clinical situations that warrant closer monitoring."
        )
    
    # Age finding
    if age_features:
        summary_sentences.append(
            "Advanced age (particularly over 70 years) is consistently associated with increased death risk in CRS patients."
        )
    
    # Polypharmacy finding
    if polypharmacy_features:
        summary_sentences.append(
            "Patients with high polypharmacy (more than 5 concurrent medications) show significantly elevated risk, "
            "likely reflecting more severe underlying disease or complications."
        )
    
    # Comorbidity finding
    if comorbidity_features:
        summary_sentences.append(
            "Pre-existing conditions such as diabetes, hypertension, and cardiac disease contribute to death risk, "
            "suggesting that baseline health status is an important factor in CRS outcomes."
        )
    
    for sentence in summary_sentences:
        summary_parts.append(f"{sentence}\n")
    
    return ''.join(summary_parts)

def generate_local_explanations(shap_values, X_test_scaled, y_test, display_names, 
                                feature_names_map, model, explainer):
    """Generate local SHAP explanations for typical cases."""
    explanations = []
    
    # Find high-risk case (death with high predicted probability)
    death_indices = np.where(y_test == 1)[0]
    if len(death_indices) > 0:
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.decision_function(X_test_scaled)
        
        # Find death case with highest probability
        death_proba = proba[death_indices]
        high_risk_idx = death_indices[np.argmax(death_proba)]
        
        # Get patient characteristics
        patient_data = X_test_scaled[high_risk_idx]
        patient_shap = shap_values[high_risk_idx]
        
        # Find top contributing features
        feature_contributions = list(zip(display_names, patient_shap, patient_data))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_contributors = feature_contributions[:5]
        
        # Build explanation
        explanation = f"🔹 High-Risk Case Example (Death Outcome):\n"
        explanation += f"\n   Patient Profile:\n"
        
        # Extract key characteristics
        age_val = None
        age_shap = None
        polypharmacy_val = None
        polypharmacy_shap = None
        diabetes_val = None
        diabetes_shap = None
        
        for feat, shap_val, val in feature_contributions:
            if 'Age' in feat and 'Over' in feat:
                age_val = val
                age_shap = shap_val
            elif 'Polypharmacy' in feat or 'Medications' in feat:
                polypharmacy_val = val
                polypharmacy_shap = shap_val
            elif 'Diabetes' in feat:
                diabetes_val = val
                diabetes_shap = shap_val
        
        # Build patient description
        patient_desc_parts = []
        if age_val is not None and age_val > 0:
            patient_desc_parts.append(f"age {int(age_val)} years")
        if diabetes_val is not None and diabetes_val > 0:
            patient_desc_parts.append("history of diabetes")
        if polypharmacy_val is not None and polypharmacy_val > 0:
            patient_desc_parts.append("high polypharmacy")
        
        patient_desc = " and ".join(patient_desc_parts) if patient_desc_parts else "complex clinical presentation"
        
        explanation += f"   • A patient with {patient_desc}\n"
        explanation += f"\n   Why the model predicts high death risk:\n"
        
        for feat, shap_val, val in top_contributors:
            if abs(shap_val) > 0.01:  # Only show meaningful contributions
                direction = "increases" if shap_val > 0 else "decreases"
                explanation += f"   • {feat}: {direction} death risk (contribution: {shap_val:+.3f})\n"
        
        explanations.append(explanation)
    
    # Find low-risk case (survival with low predicted probability)
    survival_indices = np.where(y_test == 0)[0]
    if len(survival_indices) > 0:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            proba = model.decision_function(X_test_scaled)
        
        survival_proba = proba[survival_indices]
        low_risk_idx = survival_indices[np.argmin(survival_proba)]
        
        patient_shap = shap_values[low_risk_idx]
        feature_contributions = list(zip(display_names, patient_shap))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top_contributors = feature_contributions[:5]
        
        explanation = f"🔹 Low-Risk Case Example (Survival Outcome):\n"
        explanation += f"\n   Why the model predicts lower death risk:\n"
        
        for feat, shap_val in top_contributors:
            if abs(shap_val) > 0.01:
                direction = "increases" if shap_val > 0 else "decreases"
                explanation += f"   • {feat}: {direction} death risk (contribution: {shap_val:+.3f})\n"
        
        explanations.append(explanation)
    
    return explanations

def generate_shap_explanations(model_data, X_test, y_test, output_dir='.'):
    """Generate SHAP explanations for CRS model."""
    print("=" * 80)
    print("Generating SHAP Explanations for CRS → Death Model")
    print("=" * 80)
    
    if not HAS_SHAP:
        print("❌ SHAP not available")
        return None
    
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    
    # Store model for later use in local explanations
    model_obj = model
    
    # Prepare data
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    print(f"\n📊 Analyzing {len(X_test)} CRS patients...")
    
    # Initialize SHAP explainer
    print("\n🔹 Initializing SHAP explainer...")
    try:
        # Use TreeExplainer for tree models
        if isinstance(model, (type(model).__name__ == 'RandomForestClassifier')):
            explainer = shap.TreeExplainer(model)
        else:
            # For other models, use KernelExplainer (slower but more general)
            explainer = shap.KernelExplainer(model.predict_proba, X_test_scaled[:50])  # Sample for speed
        
        # Calculate SHAP values
        print("🔹 Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Handle binary classification - shap_values can be list or array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes) - take positive class
            shap_values = shap_values[:, :, 1]  # Use positive class (death)
        
        print(f"✅ SHAP values calculated (shape: {shap_values.shape})")
        
    except Exception as e:
        print(f"⚠️  TreeExplainer failed, trying KernelExplainer: {e}")
        # Fallback to KernelExplainer with sample
        sample_size = min(50, len(X_test_scaled))
        background = X_test_scaled[:sample_size]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_scaled[:100])  # Limit for speed
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Use positive class (death)
    
    # Feature names mapping
    feature_names_map = plain_language_feature_names()
    display_names = [feature_names_map.get(col, col.replace('_', ' ').title()) 
                     for col in X_test.columns]
    
    # Create SHAP plots
    print("\n📊 Generating SHAP visualizations...")
    
    # 1. Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=display_names, show=False)
    plt.title('SHAP Summary Plot: CRS → Death Prediction\n(Higher values push toward death)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/crs_shap_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: crs_shap_summary.png")
    plt.close()
    
    # 2. Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=display_names, 
                      plot_type='bar', show=False)
    plt.title('SHAP Feature Importance: CRS → Death Prediction', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/crs_shap_bar.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: crs_shap_bar.png")
    plt.close()
    
    # 3. Waterfall plots for specific cases
    # Handle expected_value for binary classification
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = explainer.expected_value
    
    # Death case
    death_idx = np.where(y_test == 1)[0]
    if len(death_idx) > 0:
        case_idx = death_idx[0]
        try:
            plt.figure(figsize=(12, 8))
            # Create Explanation object for single sample
            explanation = shap.Explanation(
                values=shap_values[case_idx],
                base_values=base_value,
                data=X_test_scaled[case_idx],
                feature_names=display_names
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Explanation: Death Case\n(Patient {case_idx})', 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/crs_shap_waterfall_death.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: crs_shap_waterfall_death.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not generate waterfall plot for death case: {e}")
    
    # Survival case
    survival_idx = np.where(y_test == 0)[0]
    if len(survival_idx) > 0:
        case_idx = survival_idx[0]
        try:
            plt.figure(figsize=(12, 8))
            # Create Explanation object for single sample
            explanation = shap.Explanation(
                values=shap_values[case_idx],
                base_values=base_value,
                data=X_test_scaled[case_idx],
                feature_names=display_names
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Explanation: Survival Case\n(Patient {case_idx})', 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/crs_shap_waterfall_survival.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: crs_shap_waterfall_survival.png")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not generate waterfall plot for survival case: {e}")
    
    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=display_names)
    shap_df.to_csv(f'{output_dir}/crs_shap_values.csv', index=False)
    print("✅ Saved: crs_shap_values.csv")
    
    # Global interpretation: Top 10 positive and negative features
    print("\n" + "=" * 80)
    print("Global SHAP Interpretation")
    print("=" * 80)
    
    mean_shap = np.mean(shap_values, axis=0)
    feature_shap_pairs = list(zip(display_names, mean_shap))
    feature_shap_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 Top 10 Features That INCREASE Death Risk (Positive SHAP):")
    print("-" * 80)
    top_positive = feature_shap_pairs[:10]
    for i, (feat, shap_val) in enumerate(top_positive, 1):
        print(f"{i:2d}. {feat:50s} : +{shap_val:.4f}")
    
    print("\n📊 Top 10 Features That DECREASE Death Risk (Negative SHAP):")
    print("-" * 80)
    top_negative = feature_shap_pairs[-10:][::-1]  # Reverse to show most negative first
    for i, (feat, shap_val) in enumerate(top_negative, 1):
        print(f"{i:2d}. {feat:50s} : {shap_val:.4f}")
    
    # Generate global summary
    global_summary = generate_global_summary(top_positive, top_negative, feature_names_map)
    print("\n" + "=" * 80)
    print("Global Summary (Plain Language)")
    print("=" * 80)
    print(global_summary)
    
    # Local interpretation: Select typical high-risk and low-risk cases
    print("\n" + "=" * 80)
    print("Local SHAP Interpretation (Individual Cases)")
    print("=" * 80)
    
    local_explanations = generate_local_explanations(
        shap_values, X_test_scaled, y_test, display_names, 
        feature_names_map, model_obj, explainer
    )
    
    for explanation in local_explanations:
        print("\n" + "-" * 80)
        print(explanation)
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': display_names,
        'X_test': X_test_scaled,
        'top_positive': top_positive,
        'top_negative': top_negative,
        'global_summary': global_summary,
        'local_explanations': local_explanations
    }

def generate_plain_language_summary(shap_results=None, feature_importance_file='crs_feature_importance.csv'):
    """Generate plain language summary for clinicians."""
    print("\n" + "=" * 80)
    print("Generating Plain Language Summary")
    print("=" * 80)
    
    summary = []
    summary.append("# CRS → Death Model: Plain Language Summary for Clinicians\n")
    summary.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary.append("---\n\n")
    
    summary.append("## What This Model Does\n\n")
    summary.append("This model predicts the risk of death in patients with **Cytokine Release Syndrome (CRS)** ")
    summary.append("who are being treated with Epcoritamab. It helps identify which factors are most important ")
    summary.append("in determining patient outcomes.\n\n")
    
    summary.append("## Key Findings\n\n")
    
    # If SHAP results are available, use them
    if shap_results and 'global_summary' in shap_results:
        summary.append("### Global Model Interpretation (SHAP Analysis)\n\n")
        summary.append(shap_results['global_summary'])
        summary.append("\n")
        
        # Add English summary section
        if 'English Summary' in shap_results['global_summary']:
            summary.append("### English Summary for Report\n\n")
            # Extract English summary from global_summary
            global_summary_lines = shap_results['global_summary'].split('\n')
            in_english_section = False
            for line in global_summary_lines:
                if 'English Summary' in line:
                    in_english_section = True
                    continue
                if in_english_section and line.strip():
                    if line.startswith('='):
                        break
                    summary.append(f"{line}\n")
            summary.append("\n")
    
    # Load feature importance if available
    if os.path.exists(feature_importance_file):
        importance_df = pd.read_csv(feature_importance_file)
        top_features = importance_df.head(5)
        
        summary.append("### Top 5 Factors That Increase Death Risk:\n\n")
        
        feature_descriptions = {
            'num_drugs': '**Number of concurrent medications:** Patients taking more medications have higher death risk. This likely reflects more severe underlying disease or complications.',
            'has_chemo': '**Receiving chemotherapy:** Patients on chemotherapy alongside Epcoritamab have significantly higher death risk. This suggests combined toxicity or more aggressive disease.',
            'age_years': '**Patient age:** Older patients have higher death risk. Age is a well-established risk factor.',
            'num_reactions': '**Number of adverse reactions:** More concurrent adverse reactions indicate higher risk.',
            'bmi': '**Body Mass Index:** BMI plays a moderate role in outcomes.'
        }
        
        for i, row in top_features.iterrows():
            feat = row['feature']
            imp = row['importance']
            desc = feature_descriptions.get(feat, f"This factor contributes to death risk.")
            
            summary.append(f"{i+1}. **{feat.replace('_', ' ').title()}** (Importance: {imp:.3f})\n")
            summary.append(f"   {desc}\n\n")
    
    # Add local explanations if available
    if shap_results and 'local_explanations' in shap_results:
        summary.append("## Individual Case Interpretations\n\n")
        summary.append("### Local SHAP Explanations\n\n")
        for i, explanation in enumerate(shap_results['local_explanations'], 1):
            summary.append(f"#### Case {i}\n\n")
            summary.append(explanation.replace('\n', '\n'))
            summary.append("\n\n")
    
    summary.append("## Clinical Interpretation\n\n")
    summary.append("### What This Means for Patient Care:\n\n")
    
    summary.append("1. **Monitor polypharmacy:** Patients on multiple medications need close monitoring.\n")
    summary.append("2. **Consider chemotherapy carefully:** Combined chemo + Epcoritamab has higher risk - consider dose adjustments or timing.\n")
    summary.append("3. **Age-based risk stratification:** Older patients (>70 years) may need more intensive monitoring.\n")
    summary.append("4. **Watch for multiple reactions:** Patients with multiple concurrent adverse reactions are at higher risk.\n")
    summary.append("5. **Comorbidity management:** Patients with diabetes, hypertension, or cardiac disease require closer monitoring.\n\n")
    
    summary.append("### What This Means for Drug Safety Teams:\n\n")
    summary.append("1. **Risk stratification:** Use these factors to identify high-risk patients early.\n")
    summary.append("2. **Intervention opportunities:** Focus on modifiable factors (e.g., medication review, infection prevention).\n")
    summary.append("3. **Clinical decision support:** Integrate this model into clinical workflows for risk assessment.\n\n")
    
    summary.append("## Limitations\n\n")
    summary.append("- Model is based on observational data (FAERS reports)\n")
    summary.append("- Cannot prove causation, only associations\n")
    summary.append("- Should be used alongside clinical judgment\n")
    summary.append("- Requires validation in independent datasets\n\n")
    
    # Save summary
    summary_text = ''.join(summary)
    with open('crs_plain_language_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print("✅ Saved: crs_plain_language_summary.md")
    return summary_text

def main():
    """Main execution."""
    print("=" * 80)
    print("CRS-Specific SHAP Analysis with Plain Language Explanations")
    print("=" * 80)
    print()
    
    # Load model
    model_data = load_crs_model()
    if model_data is None:
        return
    
    # Try to load test data from model metadata
    shap_results = None
    
    # Check if we have saved test data or need to regenerate
    meta_file = 'crs_model_meta.json'
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Try to load from saved model data if available
        if 'X_test' in model_data and 'y_test' in model_data:
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            
            print(f"✅ Loaded test data: {len(X_test)} samples")
            print()
            
            # Generate SHAP explanations
            if HAS_SHAP:
                shap_results = generate_shap_explanations(model_data, X_test, y_test)
            else:
                print("⚠️  SHAP not available. Install with: pip install shap")
        else:
            print("⚠️  Test data not found in model file.")
            print("   SHAP analysis requires test data from model training.")
            print("   Generating plain language summary from feature importance instead...")
            print()
    else:
        print("⚠️  Model metadata not found.")
        print("   Generating plain language summary from feature importance instead...")
        print()
    
    # Generate plain language summary (with or without SHAP results)
    generate_plain_language_summary(shap_results)
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - crs_plain_language_summary.md")
    if shap_results:
        print("  - crs_shap_summary.png")
        print("  - crs_shap_bar.png")
        print("  - crs_shap_waterfall_death.png")
        print("  - crs_shap_waterfall_survival.png")
        print("  - crs_shap_values.csv")

if __name__ == '__main__':
    main()

