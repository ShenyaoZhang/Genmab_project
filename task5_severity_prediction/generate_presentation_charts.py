#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Presentation Charts for CRS Analysis
==============================================

1. SHAP Summary Plot - Model interpretation (which factors matter most)
2. Granular CRS Risk Table/Bar Chart - Real data stratification (which groups have highest risk)
"""

import sys
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("❌ SHAP not available. Install with: pip install shap")
    sys.exit(1)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_crs_model_and_data():
    """Load CRS model and test data."""
    # Load model
    model_file = 'crs_model_best.pkl'
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("Please run: python 12_crs_model_training.py first")
        return None, None, None
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data.get('feature_names', None)
    X_test = model_data.get('X_test', None)
    y_test = model_data.get('y_test', None)
    
    return model, X_test, y_test, feature_names

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
        'comorbidity_diabetes': 'History of Diabetes',
        'comorbidity_hypertension': 'History of Hypertension',
        'comorbidity_cardiac': 'History of Cardiac Disease',
        'sex_unknown': 'Unknown Gender',
        'age_missing': 'Age Missing in Report',
        'weight_missing': 'Weight Missing in Report',
        'cancer_stage_I': 'Cancer Stage I',
        'cancer_stage_II': 'Cancer Stage II',
        'cancer_stage_III': 'Cancer Stage III',
        'cancer_stage_IV': 'Cancer Stage IV',
    }

def generate_shap_summary_plot(model, X_test, feature_names, output_file='crs_shap_summary_presentation.png'):
    """
    Generate SHAP Summary Plot for CRS → Death Model
    Shows which factors are most important according to the model.
    """
    print("\n" + "=" * 80)
    print("📊 Generating SHAP Summary Plot (Model Interpretation)")
    print("=" * 80)
    
    # Initialize SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        print("✅ Using TreeExplainer")
    except:
        try:
            # Sample background for KernelExplainer
            background = X_test.head(min(100, len(X_test)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            print("✅ Using KernelExplainer")
        except Exception as e:
            print(f"❌ Failed to initialize SHAP explainer: {e}")
            return False
    
    # Compute SHAP values
    print("🔹 Computing SHAP values...")
    try:
        shap_values = explainer.shap_values(X_test)
        
        # Handle binary classification (shap_values might be a list or array with shape (n_samples, n_features, 2))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (death)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Shape is (n_samples, n_features, 2) - use positive class (index 1)
            shap_values = shap_values[:, :, 1]
        
        shap_values = np.array(shap_values)
        print(f"✅ SHAP values computed: shape {shap_values.shape}")
    except Exception as e:
        print(f"❌ Failed to compute SHAP values: {e}")
        return False
    
    # Convert to DataFrame for easier handling
    if feature_names is not None:
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = pd.DataFrame(X_test)
    
    # Map feature names to plain language
    feature_map = plain_language_feature_names()
    if feature_names is not None:
        display_names = [feature_map.get(f, f) for f in feature_names]
    else:
        display_names = [feature_map.get(f, f) for f in X_test_df.columns]
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=X_test_df.values,
        feature_names=display_names
    )
    
    # Generate summary plot
    print("🔹 Generating summary plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    shap.summary_plot(
        shap_explanation,
        X_test_df,
        feature_names=display_names,
        max_display=20,  # Top 20 features
        show=False,
        plot_size=None
    )
    
    plt.title('SHAP Summary Plot: CRS → Death Model\n(Which Factors Matter Most?)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")
    return True

def load_crs_data_for_stratification():
    """Load CRS data for risk stratification analysis."""
    # Try to load from preprocessed data first (has more features)
    if os.path.exists('preprocessed_data.csv'):
        df = pd.read_csv('preprocessed_data.csv')
        print("✅ Loaded from preprocessed_data.csv (has more features)")
    elif os.path.exists('main_data.csv'):
        df = pd.read_csv('main_data.csv')
        print("✅ Loaded from main_data.csv")
    else:
        print("❌ Data file not found: main_data.csv or preprocessed_data.csv")
        return None
    
    # Filter to Epcoritamab patients
    epcor_df = df[df['target_drug'].str.contains('Epcoritamab', case=False, na=False)].copy()
    
    if len(epcor_df) == 0:
        print("❌ No Epcoritamab patients found")
        return None
    
    # Identify CRS cases
    CRS_KEYWORDS = ['CYTOKINE RELEASE', 'CRS', 'CYTOKINE RELEASE SYNDROME']
    if 'reactions' in epcor_df.columns:
        reactions_upper = epcor_df['reactions'].fillna('').str.upper()
        crs_mask = pd.Series(False, index=epcor_df.index)
        for keyword in CRS_KEYWORDS:
            crs_mask |= reactions_upper.str.contains(keyword, na=False)
        epcor_df['has_crs'] = crs_mask.astype(int)
    else:
        epcor_df['has_crs'] = 0
    
    # Filter to CRS patients
    crs_df = epcor_df[epcor_df['has_crs'] == 1].copy()
    
    if len(crs_df) == 0:
        print("❌ No CRS patients found")
        return None
    
    # Add death flag
    crs_df['death'] = pd.to_numeric(crs_df['seriousnessdeath'], errors='coerce').fillna(0).astype(int)
    
    # Ensure age_years exists (calculate if needed)
    if 'age_years' not in crs_df.columns and 'patientonsetage' in crs_df.columns:
        crs_df['age_years'] = pd.to_numeric(crs_df['patientonsetage'], errors='coerce')
    
    return crs_df

def generate_granular_risk_chart(crs_df, output_file='crs_granular_risk_stratification.png'):
    """
    Generate Granular CRS Risk Stratification Chart
    Shows real data: which stratified groups have highest death risk.
    """
    print("\n" + "=" * 80)
    print("📊 Generating Granular CRS Risk Stratification Chart")
    print("=" * 80)
    
    if crs_df is None or len(crs_df) == 0:
        print("❌ No CRS data available")
        return False
    
    # Prepare stratification data
    stratifications = []
    
    # 1. Age stratification
    if 'age_years' in crs_df.columns:
        age = pd.to_numeric(crs_df['age_years'], errors='coerce')
        age_bins = [0, 50, 65, 75, 100]
        age_labels = ['<50', '50-65', '65-75', '75+']
        crs_df['age_group'] = pd.cut(age, bins=age_bins, labels=age_labels, include_lowest=True)
        
        age_stats = crs_df.groupby('age_group')['death'].agg(['mean', 'count']).reset_index()
        age_stats['category'] = 'Age Group'
        age_stats['subcategory'] = age_stats['age_group'].astype(str)
        age_stats['death_rate'] = age_stats['mean'].fillna(0)
        age_stats['n_total'] = age_stats['count'].fillna(0).astype(int)
        age_stats['n_death'] = (age_stats['death_rate'] * age_stats['n_total']).fillna(0).astype(int)
        stratifications.append(age_stats[['category', 'subcategory', 'death_rate', 'n_total', 'n_death']])
    
    # 2. Cancer Stage stratification
    stage_cols = ['cancer_stage_I', 'cancer_stage_II', 'cancer_stage_III', 'cancer_stage_IV']
    available_stages = [col for col in stage_cols if col in crs_df.columns]
    if available_stages:
        stage_stats = []
        for stage_col in available_stages:
            stage_name = stage_col.replace('cancer_stage_', 'Stage ')
            stage_df = crs_df[crs_df[stage_col] == 1]
            if len(stage_df) > 0:
                death_rate = stage_df['death'].mean()
                n_total = len(stage_df)
                n_death = int(death_rate * n_total)
                stage_stats.append({
                    'category': 'Cancer Stage',
                    'subcategory': stage_name,
                    'death_rate': death_rate,
                    'n_total': n_total,
                    'n_death': n_death
                })
        if stage_stats:
            stratifications.append(pd.DataFrame(stage_stats))
    
    # 3. BMI stratification
    bmi_categories = {
        'bmi_underweight': 'Underweight\n(BMI<18.5)',
        'bmi_normal': 'Normal\n(18.5-25)',
        'bmi_overweight': 'Overweight\n(25-30)',
        'bmi_obese': 'Obese\n(BMI>30)'
    }
    bmi_stats = []
    for bmi_col, bmi_label in bmi_categories.items():
        if bmi_col in crs_df.columns:
            bmi_df = crs_df[crs_df[bmi_col] == 1]
            if len(bmi_df) > 0:
                death_rate = bmi_df['death'].mean()
                n_total = len(bmi_df)
                n_death = int(death_rate * n_total)
                bmi_stats.append({
                    'category': 'BMI Category',
                    'subcategory': bmi_label,
                    'death_rate': death_rate,
                    'n_total': n_total,
                    'n_death': n_death
                })
    if bmi_stats:
        stratifications.append(pd.DataFrame(bmi_stats))
    
    # 4. Polypharmacy stratification
    if 'num_drugs' in crs_df.columns:
        crs_df['polypharmacy_group'] = pd.cut(
            pd.to_numeric(crs_df['num_drugs'], errors='coerce'),
            bins=[0, 1, 5, 10, 100],
            labels=['Low (≤1)', 'Moderate (2-5)', 'High (6-10)', 'Very High (>10)'],
            include_lowest=True
        )
        poly_stats = crs_df.groupby('polypharmacy_group')['death'].agg(['mean', 'count']).reset_index()
        poly_stats['category'] = 'Polypharmacy'
        poly_stats['subcategory'] = poly_stats['polypharmacy_group'].astype(str)
        poly_stats['death_rate'] = poly_stats['mean'].fillna(0)
        poly_stats['n_total'] = poly_stats['count'].fillna(0).astype(int)
        poly_stats['n_death'] = (poly_stats['death_rate'] * poly_stats['n_total']).fillna(0).astype(int)
        stratifications.append(poly_stats[['category', 'subcategory', 'death_rate', 'n_total', 'n_death']])
    
    # Combine all stratifications
    if not stratifications:
        print("❌ No stratification data available")
        return False
    
    combined_df = pd.concat(stratifications, ignore_index=True)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by category
    categories = combined_df['category'].unique()
    colors = sns.color_palette("husl", len(categories))
    category_colors = dict(zip(categories, colors))
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    bar_positions = []
    bar_values = []
    bar_colors = []
    bar_labels = []
    
    for cat in categories:
        cat_df = combined_df[combined_df['category'] == cat].sort_values('death_rate', ascending=False)
        n_bars = len(cat_df)
        
        for i, row in cat_df.iterrows():
            bar_positions.append(x_pos)
            bar_values.append(row['death_rate'] * 100)  # Convert to percentage
            bar_colors.append(category_colors[cat])
            bar_labels.append(f"{row['subcategory']}\n(n={int(row['n_total'])})")
            x_pos += 1
        
        # Add spacing between categories
        x_pos += 0.5
    
    # Create bars
    bars = ax.bar(bar_positions, bar_values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for pos, val, label in zip(bar_positions, bar_values, bar_labels):
        ax.text(pos, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize axes
    ax.set_ylabel('Death Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Stratified Groups', fontsize=14, fontweight='bold')
    ax.set_title('Granular CRS Risk Stratification\n(Real Data: Which Groups Have Highest Risk?)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, max(bar_values) * 1.15])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels([label.split('\n')[0] for label in bar_labels], rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=category_colors[cat], label=cat) for cat in categories]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")
    
    # Print summary table
    print("\n📊 Risk Stratification Summary:")
    print("=" * 80)
    print(combined_df.to_string(index=False))
    print()
    
    return True

def main():
    """Main execution."""
    print("=" * 80)
    print("Generating Presentation Charts for CRS Analysis")
    print("=" * 80)
    
    # Chart 1: SHAP Summary Plot
    print("\n📊 Chart 1: SHAP Summary Plot (Model Interpretation)")
    model, X_test, y_test, feature_names = load_crs_model_and_data()
    if model is not None and X_test is not None:
        generate_shap_summary_plot(model, X_test, feature_names)
    else:
        print("⚠️  Skipping SHAP plot (model/data not available)")
    
    # Chart 2: Granular Risk Stratification
    print("\n📊 Chart 2: Granular CRS Risk Stratification (Real Data)")
    crs_df = load_crs_data_for_stratification()
    if crs_df is not None:
        generate_granular_risk_chart(crs_df)
    else:
        print("⚠️  Skipping risk stratification (data not available)")
    
    print("\n" + "=" * 80)
    print("✅ Chart Generation Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. crs_shap_summary_presentation.png - Model interpretation")
    print("  2. crs_granular_risk_stratification.png - Real data risk stratification")
    print()

if __name__ == '__main__':
    main()

