"""
Task 11: Granular CRS → Death Analysis
========================================

This script performs FINE-GRAINED feature engineering and analysis:
    1. Age stratification with specific cutoffs (e.g., >65 years)

2. BMI calculation and obesity analysis (BMI>30)
3. Specific drug combination analysis
4. Comorbidity analysis (diabetes, infections, etc.)
5. Detailed stratified reporting with specific percentages
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
except ImportError as e:
    print(f"ERROR: Missing required library: {e}")
    exit(1)

# Default constants (for backward compatibility)
DEFAULT_DATA_FILE = 'main_data.csv'
DEFAULT_TARGET_DRUG = 'Epcoritamab'
DEFAULT_CRS_KEYWORDS = [
    'CYTOKINE RELEASE SYNDROME',
    'CYTOKINE RELEASE',
     'CYTOKINE STORM']


def get_ae_keywords(ae):
    """
    Map adverse event name to search keywords.

    Parameters:
        -----------

    ae : str
    Adverse event name (e.g., "CRS", "pneumonia")

    Returns:
        --------

    list : List of keywords to search for in reactions field
    """
    ae_keyword_map = {
        'CRS': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        'pneumonia': ['PNEUMONIA', 'PNEUMONITIS'],
        'cytokine release syndrome': ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM'],
        'ICANS': ['ICANS', 'IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY', 'NEUROTOXICITY'],
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


# Common comorbidities to extract from drug_indication, reactions, and
# medications
COMORBIDITY_KEYWORDS = {
    'diabetes': ['DIABETES', 'DIABETIC', 'HYPERGLYCEMIA', 'INSULIN', 'METFORMIN', 'GLIPIZIDE', 'GLYBURIDE'],
    'hypertension': ['HYPERTENSION', 'HYPERTENSIVE', 'HIGH BLOOD PRESSURE', 'LISINOPRIL', 'AMLODIPINE', 'LOSARTAN', 'VALSARTAN'],
    'infection': ['INFECTION', 'SEPSIS', 'PNEUMONIA', 'BACTERIA', 'VIRAL', 'FUNGAL', 'BACTEREMIA', 'ASPERGILLOSIS'],
    'cardiac': ['CARDIAC', 'HEART', 'CARDIOMYOPATHY', 'ARRHYTHMIA', 'ATRIAL FIBRILLATION', 'CARDIAC DISORDER'],
    'renal': ['RENAL', 'KIDNEY', 'NEPHROPATHY', 'RENAL FAILURE', 'KIDNEY DISEASE'],
    'liver': ['HEPATIC', 'LIVER', 'HEPATITIS', 'HEPATIC FAILURE'],
    'cancer': ['CANCER', 'MALIGNANT', 'NEOPLASM', 'TUMOR', 'LYMPHOMA', 'LEUKEMIA']
}

# Common drug categories for combination analysis
DRUG_CATEGORIES = {
    'steroids': ['PREDNISONE', 'PREDNISOLONE', 'DEXAMETHASONE', 'METHYLPREDNISOLONE', 'CORTICOSTEROID'],
    'immunosuppressants': ['TACROLIMUS', 'CYCLOSPORINE', 'MYCOPHENOLATE', 'AZATHIOPRINE'],
    'antibiotics': ['CIPROFLOXACIN', 'LEVOFLOXACIN', 'VANCOMYCIN', 'CEFTAZIDIME', 'PIPERACILLIN'],
    'antifungals': ['FLUCONAZOLE', 'VORICONAZOLE', 'AMPHOTERICIN'],
    'antivirals': ['ACYCLOVIR', 'GANCICLOVIR', 'OSELTAMIVIR'],
    'chemo': ['CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE', 'CISPLATIN'],
    'targeted_therapy': ['RITUXIMAB', 'BRENTUXIMAB', 'OBINUTUZUMAB']
}


def calculate_bmi(weight_kg, height_m=None):
    """
    Calculate BMI. If height not available, use average height by age/sex.
    Default: assume average height for adults.
    """
    if pd.isna(weight_kg) or weight_kg <= 0:
        return np.nan

    # If height not available, use average height assumptions
    if height_m is None or pd.isna(height_m) or height_m <= 0:
        # Assume average adult height: 1.7m for males, 1.6m for females
        # For simplicity, use 1.65m as default
    height_m = 1.65

    if height_m > 0:
        bmi = weight_kg / (height_m ** 2)

    return round(bmi, 1)
    return np.nan


def extract_comorbidities(
    drug_indication_str,
    reactions_str=None,
     all_drugs_str=None):
         """
    Extract comorbidities from multiple sources:
        - drug_indication field

    - reactions field (adverse events can indicate underlying conditions)
    - all_drugs field (medications can indicate comorbidities)
    Returns a dictionary of comorbidity flags.
    """
    comorbidities = {comorb: 0 for comorb in COMORBIDITY_KEYWORDS.keys()}

    # Combine all text sources
    all_text = []

    if not pd.isna(drug_indication_str) and drug_indication_str != '':
        all_text.append(str(drug_indication_str).upper())

    if reactions_str is not None and not pd.isna(
        reactions_str) and reactions_str != '':
            all_text.append(str(reactions_str).upper())

    if all_drugs_str is not None and not pd.isna(
        all_drugs_str) and all_drugs_str != '':
            all_text.append(str(all_drugs_str).upper())

    combined_text = '|'.join(all_text)

    if combined_text == '':
        return comorbidities

    # Check for each comorbidity
    for comorb_name, keywords in COMORBIDITY_KEYWORDS.items():
        found = any(keyword in combined_text for keyword in keywords)

    comorbidities[comorb_name] = 1 if found else 0

    return comorbidities


def extract_drug_categories(all_drugs_str):
    """
    Extract drug categories from all_drugs field.
    Returns a dictionary of drug category flags.
    """
    if pd.isna(all_drugs_str) or all_drugs_str == '':
        return {cat: 0 for cat in DRUG_CATEGORIES.keys()}

    drugs_upper = str(all_drugs_str).upper()
    categories = {}

    for cat_name, keywords in DRUG_CATEGORIES.items():
        found = any(keyword in drugs_upper for keyword in keywords)

    categories[cat_name] = 1 if found else 0

    return categories


def identify_ae_cases(df, target_drug, ae_keywords):
    """
    Identify adverse event cases in patients taking a specific drug.

    Parameters:
        -----------

    df : pd.DataFrame
    Input dataframe with FAERS data
    target_drug : str
    Target drug name (e.g., "Epcoritamab")
    ae_keywords : list
    List of keywords to search for in reactions field

    Returns:
        --------

    pd.DataFrame or None
    DataFrame with AE cases identified, or None if no records found
    """
    print("=" * 70)
    print(
    f"Step 1: Identifying ae_keywords[0] if ae_keywords else 'AE'} Cases")
    print("=" * 70)

    # Filter to target drug patients
    drug_mask = df['target_drug'].str.contains(
        target_drug, case=False, na=False)
    drug_df = df[drug_mask].copy()

    if len(drug_df) == 0:
        print(f"ERROR: No {target_drug} records found.")

    return None

    print(f" Total {target_drug} records: {len(drug_df)}")

    # Identify AE cases
    reactions_upper = drug_df['reactions'].fillna('').str.upper()
    ae_mask = pd.Series(False, index=drug_df.index)

    for keyword in ae_keywords:
        mask = reactions_upper.str.contains(
    keyword.upper(), na=False, regex=False)

    ae_mask |= mask

    drug_df['has_ae'] = ae_mask.astype(int)  # Generic name for any AE
    drug_df['has_crs'] = ae_mask.astype(int)  # Keep for backward compatibility

    # Outcome: death
    drug_df['death'] = pd.to_numeric(
    drug_df['seriousnessdeath'],
     errors='coerce').fillna(0).astype(int)

    n_ae = ae_mask.sum()
    n_ae_death = drug_df[ae_mask & (drug_df['death'] == 1)].shape[0]

    ae_name = ae_keywords[0] if ae_keywords else 'AE'
    print(
    f" Patients with {ae_name}: {n_ae}{ n_ae /
        len(drug_df) *
         100:.1f}%)")
    print(
        f" Deaths in {ae_name} patients: {n_ae_death}{ n_ae_death /
            n_ae *
            100:.1f}% of {ae_name} patients)" if n_ae > 0 else f" No {ae_name} patients")

    return drug_df

# Backward compatibility alias


def identify_crs_cases(df, target_drug=None, ae_keywords=None):
    """Backward compatibility wrapper for identify_ae_cases."""
    if target_drug is None:
        target_drug = DEFAULT_TARGET_DRUG

    if ae_keywords is None:
        ae_keywords = DEFAULT_CRS_KEYWORDS

    return identify_ae_cases(df, target_drug, ae_keywords)


def granular_feature_engineering(df):
    """
    Perform granular feature engineering:
        - Age stratification with specific cutoffs

    - BMI calculation and obesity flags
    - Comorbidity extraction
    - Drug category extraction
    - Specific drug combinations
    """
    print("\n" + "=" * 70)
    print("Step 2: Granular Feature Engineering")
    print("=" * 70)

    feature_df = df.copy()

    # 1. AGE STRATIFICATION with specific cutoffs
    print("\n Age Stratification...")
    if 'age_years' in feature_df.columns:
        age = pd.to_numeric(feature_df['age_years'], errors='coerce')

    # Specific cutoffs as requested
    feature_df['age_gt_65'] = (age > 65).astype(int)
    feature_df['age_gt_70'] = (age > 70).astype(int)
    feature_df['age_gt_75'] = (age > 75).astype(int)
    feature_df['age_50_65'] = ((age >= 50) & (age <= 65)).astype(int)
    feature_df['age_65_75'] = ((age > 65) & (age <= 75)).astype(int)
    feature_df['age_lt_50'] = (age < 50).astype(int)

    print(f" Age >65: {feature_df['age_gt_65'].sum()} patients")
    print(f" Age >70: {feature_df['age_gt_70'].sum()} patients")
    print(f" Age >75: {feature_df['age_gt_75'].sum()} patients")

    # 2. BMI CALCULATION and obesity
    print("\n BMI Calculation...")
    if 'patientweight' in feature_df.columns:
        weight = pd.to_numeric(feature_df['patientweight'], errors='coerce')

    # Calculate BMI (assuming average height if not available)
    feature_df['bmi'] = weight.apply(lambda w: calculate_bmi(w))

    # Obesity flags
    feature_df['bmi_obese'] = (feature_df['bmi'] > 30).astype(int)
    feature_df['bmi_overweight'] = (
    (feature_df['bmi'] >= 25) & (
        feature_df['bmi'] <= 30)).astype(int)
    feature_df['bmi_normal'] = (
    (feature_df['bmi'] >= 18.5) & (
        feature_df['bmi'] < 25)).astype(int)
    feature_df['bmi_underweight'] = (feature_df['bmi'] < 18.5).astype(int)
    feature_df['bmi_missing'] = feature_df['bmi'].isna().astype(int)

    n_obese = feature_df['bmi_obese'].sum()
    n_with_bmi = feature_df['bmi'].notna().sum()
    print(f" Patients with BMI data: {n_with_bmi}")
    print(
        f" Obese (BMI>30): {n_obese} patients ( n_obese /
            n_with_bmi *
            100:.1f}%)" if n_with_bmi > 0 else " No BMI data")

    # 3. COMORBIDITY EXTRACTION (from multiple sources)
    print("\n Comorbidity Extraction...")
    if 'drug_indication' in feature_df.columns:
        reactions_col = feature_df['reactions'] if 'reactions' in feature_df.columns else None

    drugs_col = feature_df['all_drugs'] if 'all_drugs' in feature_df.columns else None

    comorb_dicts = feature_df.apply(
        lambda row: extract_comorbidities(
            row['drug_indication'],
            reactions_col[row.name] if reactions_col is not None else None,
            drugs_col[row.name] if drugs_col is not None else None
        ),
        axis=1
    )

    for comorb in COMORBIDITY_KEYWORDS.keys():
        feature_df[f'comorbidity_{comorb}'] = [d[comorb] for d in comorb_dicts]

    n_comorb = feature_df[f'comorbidity_{comorb}'].sum()
    print(f" {comorb.capitalize()}: {n_comorb} patients ({n_comorb /
    len(feature_df) *
     100:.1f}%)")

    # 4. DRUG CATEGORY EXTRACTION
    print("\n Drug Category Extraction...")
    if 'all_drugs' in feature_df.columns:
        drug_cat_dicts = feature_df['all_drugs'].apply(extract_drug_categories)

    for cat in DRUG_CATEGORIES.keys():
        feature_df[f'drug_category_{cat}'] = [d[cat] for d in drug_cat_dicts]

    n_cat = feature_df[f'drug_category_{cat}'].sum()
    print(f" {cat}: {n_cat} patients ({n_cat / len(feature_df) * 100:.1f}%)")

    # 5. SPECIFIC DRUG COMBINATIONS
    print("\n Specific Drug Combination Analysis...")
    if 'all_drugs' in feature_df.columns:
        # Common combinations with Epcoritamab
    feature_df['has_steroid'] = feature_df['drug_category_steroids']
    feature_df['has_antibiotic'] = feature_df['drug_category_antibiotics']
    feature_df['has_antifungal'] = feature_df['drug_category_antifungals']
    feature_df['has_antiviral'] = feature_df['drug_category_antivirals']

    # Combination flags
    feature_df['steroid_plus_antibiotic'] = (
        (feature_df['has_steroid'] == 1) & (feature_df['has_antibiotic'] == 1)
    ).astype(int)

    feature_df['steroid_plus_antifungal'] = (
        (feature_df['has_steroid'] == 1) & (feature_df['has_antifungal'] == 1)
    ).astype(int)

    print(
        f" Steroid + Antibiotic: {feature_df['steroid_plus_antibiotic'].sum()} patients")
    print(
        f" Steroid + Antifungal: {feature_df['steroid_plus_antifungal'].sum()} patients")

    # 6. INFECTION-RELATED ADVERSE EVENTS
    print("\n Infection-Related AE Analysis...")
    if 'reactions' in feature_df.columns:
        reactions_upper = feature_df['reactions'].fillna('').str.upper()

    infection_ae_keywords = [
        'INFECTION', 'SEPSIS', 'PNEUMONIA', 'BACTEREMIA',
        'FUNGAL', 'ASPERGILLOSIS', 'URINARY TRACT INFECTION'
    ]
    feature_df['has_infection_ae'] = pd.Series(False, index=feature_df.index)
    for keyword in infection_ae_keywords:
        feature_df['has_infection_ae'] |= reactions_upper.str.contains(
            keyword, na=False)

    feature_df['has_infection_ae'] = feature_df['has_infection_ae'].astype(int)
    n_infection_ae = feature_df['has_infection_ae'].sum()
    print(
    f" Infection-related AE: {n_infection_ae} patients ( n_infection_ae /
        len(feature_df) *
         100:.1f}%)")

    # 7. CANCER STAGE EXTRACTION (Nicole's requirement - from drug_indication)
    print("\n Cancer Stage Extraction (from drug_indication field)...")
    if 'drug_indication' in feature_df.columns:
        indication_upper = feature_df['drug_indication'].fillna('').str.upper()

    # Extract cancer stage (Stage I, II, III, IV) from free-text
    # NOTE: This is text pattern matching and may miss some cases
    # If stage features already exist from preprocessing, use those
    if 'cancer_stage_I' not in feature_df.columns:
        # Initialize all stages as 0
    feature_df['cancer_stage_I'] = 0
    feature_df['cancer_stage_II'] = 0
    feature_df['cancer_stage_III'] = 0
    feature_df['cancer_stage_IV'] = 0

    # Stage I/1
    stage_i_pattern = r'STAGE\s+[I1](\s|$|,|\|)'
    feature_df['cancer_stage_I'] = indication_upper.str.contains(
        stage_i_pattern, regex=True, na=False).astype(int)

    # Stage II/2
    stage_ii_pattern = r'STAGE\s+[I2]{2}{\s|$|,|\|)'
    feature_df['cancer_stage_II'] = indication_upper.str.contains(
        stage_ii_pattern, regex=True, na=False).astype(int)
    feature_df['cancer_stage_II'] = (feature_df['cancer_stage_II'] |
                                     indication_upper.str.contains(r'STAGE\s+2(\s|$|,|\|)', regex=True, na=False)).astype(int)

    # Stage III/3
    stage_iii_pattern = r'STAGE\s+[I3]{3}{\s|$|,|\|)'
    feature_df['cancer_stage_III'] = indication_upper.str.contains(
        stage_iii_pattern, regex=True, na=False).astype(int)
    feature_df['cancer_stage_III'] = (feature_df['cancer_stage_III'] |
                                      indication_upper.str.contains(r'STAGE\s+3(\s|$|,|\|)', regex=True, na=False)).astype(int)

    # Stage IV/4
    stage_iv_pattern = r'STAGE\s+[IV4](\s|$|,|\|)'
    feature_df['cancer_stage_IV'] = indication_upper.str.contains(
        stage_iv_pattern, regex=True, na=False).astype(int)
    feature_df['cancer_stage_IV'] = (feature_df['cancer_stage_IV'] |
                                     indication_upper.str.contains(r'STAGE\s+4(\s|$|,|\|)', regex=True, na=False)).astype(int)

    # Print statistics
    total_with_stage = (feature_df['cancer_stage_I'] + feature_df['cancer_stage_II'] +
                        feature_df['cancer_stage_III'] + feature_df['cancer_stage_IV']).sum()
    print(
    f" Patients with stage information: {total_with_stage}{ total_with_stage /
        len(feature_df) *
         100:.1f}%)")
    print(f" Stage I: {feature_df['cancer_stage_I'].sum()} patients")
    print(f" Stage II: {feature_df['cancer_stage_II'].sum()} patients")
    print(f" Stage III: {feature_df['cancer_stage_III'].sum()} patients")
    print(f" Stage IV: {feature_df['cancer_stage_IV'].sum()} patients")

    print("\n Granular feature engineering complete")
    return feature_df


def stratified_analysis(df):
    """
    Perform stratified analysis with specific percentages.
    Returns detailed findings for report.
    """
    print("\n" + "=" * 70)
    print("Step 3: Stratified Analysis with Specific Cutoffs")
    print("=" * 70)

    findings = []

    # Filter to AE patients only (support both has_ae and has_crs for backward
    # compatibility)
    ae_flag = 'has_ae' if 'has_ae' in df.columns else 'has_crs'
    crs_df = df[df[ae_flag] == 1].copy()

    if len(crs_df) == 0:
        print("WARNING: No AE patients found for stratified analysis")

    return findings

    print(f"\n Analyzing {len(crs_df)} CRS patients...")

    # 1. AGE STRATIFICATION
    print("\n Age Stratification Analysis:")
    if 'age_years' in crs_df.columns:
        age = pd.to_numeric(crs_df['age_years'], errors='coerce')

    # Age >65 cutoff
    age_gt_65 = (age > 65)
    if age_gt_65.sum() > 0:
        death_rate_gt_65 = crs_df.loc[age_gt_65, 'death'].mean()

    n_gt_65 = age_gt_65.sum()
    n_death_gt_65 = crs_df.loc[age_gt_65, 'death'].sum()
    print(
        f" Age >65: {n_death_gt_65}/{n_gt_65} deaths ({death_rate_gt_65 * 100:.1f}%)")
    findings.append({
        'variable': 'Age',
        'category': '>65 years',
        'death_rate': float(death_rate_gt_65),
        'n_deaths': int(n_death_gt_65),
        'n_total': int(n_gt_65)
    })

    # Age <=65
    age_le_65 = (age <= 65)
    if age_le_65.sum() > 0:
        death_rate_le_65 = crs_df.loc[age_le_65, 'death'].mean()

    n_le_65 = age_le_65.sum()
    n_death_le_65 = crs_df.loc[age_le_65, 'death'].sum()
    print(
        f" Age ≤65: {n_death_le_65}/{n_le_65} deaths ({death_rate_le_65 * 100:.1f}%)")
    findings.append({
        'variable': 'Age',
        'category': '≤65 years',
        'death_rate': float(death_rate_le_65),
        'n_deaths': int(n_death_le_65),
        'n_total': int(n_le_65)
    })

    # 2. BMI/OBESITY ANALYSIS
    print("\n BMI/Obesity Analysis:")
    if 'bmi_obese' in crs_df.columns:
        obese_crs = crs_df[crs_df['bmi_obese'] == 1]

    non_obese_crs = crs_df[crs_df['bmi_obese'] == 0]

    if len(obese_crs) > 0:
        death_rate_obese = obese_crs['death'].mean()

    print(
    f" Obese (BMI>30): obese_crs['death'].sum()}/{len(obese_crs)}deaths ({
                death_rate_obese * 100:.1f}%)")
    findings.append({
        'variable': 'BMI',
        'category': 'Obese (BMI>30)',
        'death_rate': float(death_rate_obese),
        'n_deaths': int(obese_crs['death'].sum()),
        'n_total': int(len(obese_crs))
    })

    if len(non_obese_crs) > 5:  # Only if enough samples
    death_rate_non_obese = non_obese_crs['death'].mean()
    print(
    f" Non-obese: non_obese_crs['death'].sum()}/{len(non_obese_crs)}deaths ({
                death_rate_non_obese * 100:.1f}%)")
    findings.append({
        'variable': 'BMI',
        'category': 'Non-obese',
        'death_rate': float(death_rate_non_obese),
        'n_deaths': int(non_obese_crs['death'].sum()),
        'n_total': int(len(non_obese_crs))
    })

    # 3. COMORBIDITY ANALYSIS
    print("\n Comorbidity Analysis:")
    for comorb in COMORBIDITY_KEYWORDS.keys():
        comorb_col = f'comorbidity_{comorb}'

    if comorb_col in crs_df.columns:
        with_comorb = crs_df[crs_df[comorb_col] == 1]

    if len(with_comorb) > 0:
        death_rate = with_comorb['death'].mean()

    print(
    f" comorb.capitalize()}: {with_comorb['death'].sum()}/{len(with_comorb)} deaths ({
                    death_rate * 100:.1f}%)")
    findings.append({
        'variable': 'Comorbidity',
        'category': comorb.capitalize(),
        'death_rate': float(death_rate),
        'n_deaths': int(with_comorb['death'].sum()),
        'n_total': int(len(with_comorb))
    })

    # 4. DRUG COMBINATION ANALYSIS
    print("\n Drug Combination Analysis:")

    # Steroid + Antibiotic
    if 'steroid_plus_antibiotic' in crs_df.columns:
        combo = crs_df[crs_df['steroid_plus_antibiotic'] == 1]

    if len(combo) > 0:
        death_rate = combo['death'].mean()

    print(
        f" Steroid + Antibiotic: {combo['death'].sum()}/{len(combo)} deaths ({death_rate * 100:.1f}%)")
    findings.append({
        'variable': 'Drug Combination',
        'category': 'Steroid + Antibiotic',
        'death_rate': float(death_rate),
        'n_deaths': int(combo['death'].sum()),
        'n_total': int(len(combo))
    })

    # 5. INFECTION AE + COMORBIDITY
    print("\n Infection AE + Comorbidity Analysis:")
    if 'has_infection_ae' in crs_df.columns and 'comorbidity_diabetes' in crs_df.columns:
        infection_diabetes = crs_df[(crs_df['has_infection_ae'] == 1) & (
            crs_df['comorbidity_diabetes'] == 1)]

    if len(infection_diabetes) > 0:
        death_rate = infection_diabetes['death'].mean()

    print(
        f" Infection AE + Diabetes: {infection_diabetes['death'].sum()}/{len(infection_diabetes)} deaths ({death_rate * 100:.1f}%)")
    findings.append({
        'variable': 'Infection AE + Comorbidity',
        'category': 'Infection AE + Diabetes',
        'death_rate': float(death_rate),
        'n_deaths': int(infection_diabetes['death'].sum()),
        'n_total': int(len(infection_diabetes))
    })

    # 6. CANCER STAGE STRATIFICATION (Nicole's requirement)
    print("\n Cancer Stage Stratification Analysis:")

    # Check if stage features exist (from 03_preprocess_data.py or
    # 12_crs_model_training.py)
    stage_features = [
    'cancer_stage_I',
    'cancer_stage_II',
    'cancer_stage_III',
     'cancer_stage_IV']
    has_stage_features = any(col in crs_df.columns for col in stage_features)

    if has_stage_features:
        # Create early vs advanced stage indicators
    stage_i_col = 'cancer_stage_I' if 'cancer_stage_I' in crs_df.columns else None
    stage_ii_col = 'cancer_stage_II' if 'cancer_stage_II' in crs_df.columns else None
    stage_iii_col = 'cancer_stage_III' if 'cancer_stage_III' in crs_df.columns else None
    stage_iv_col = 'cancer_stage_IV' if 'cancer_stage_IV' in crs_df.columns else None

    # Early stage (I-II)
    early_stage_mask = pd.Series(False, index=crs_df.index)
    if stage_i_col:
        early_stage_mask |= (crs_df[stage_i_col] == 1)

    if stage_ii_col:
        early_stage_mask |= (crs_df[stage_ii_col] == 1)

    # Advanced stage (III-IV)
    advanced_stage_mask = pd.Series(False, index=crs_df.index)
    if stage_iii_col:
        advanced_stage_mask |= (crs_df[stage_iii_col] == 1)

    if stage_iv_col:
        advanced_stage_mask |= (crs_df[stage_iv_col] == 1)

    # Analyze early stage
    if early_stage_mask.sum() > 0:
        early_crs = crs_df[early_stage_mask]

    death_rate_early = early_crs['death'].mean()
    n_early = len(early_crs)
    n_death_early = early_crs['death'].sum()
    print(
        f" Stage I-II (Early): {n_death_early}/{n_early} deaths ({death_rate_early * 100:.1f}%)")
    findings.append({
        'variable': 'Cancer Stage',
        'category': 'Stage I-II (Early)',
        'death_rate': float(death_rate_early),
        'n_deaths': int(n_death_early),
        'n_total': int(n_early)
    })

    # Analyze advanced stage
    if advanced_stage_mask.sum() > 0:
        advanced_crs = crs_df[advanced_stage_mask]

    death_rate_advanced = advanced_crs['death'].mean()
    n_advanced = len(advanced_crs)
    n_death_advanced = advanced_crs['death'].sum()
    print(
    f" Stage III-IV (Advanced): {n_death_advanced}/{n_advanced} deaths ( death_rate_advanced * 100:.1f}%)")
    findings.append({
        'variable': 'Cancer Stage',
        'category': 'Stage III-IV (Advanced)',
        'death_rate': float(death_rate_advanced),
        'n_deaths': int(n_death_advanced),
        'n_total': int(n_advanced)
    })

    # Also analyze individual stages if enough samples
    for stage_num, stage_col in [('I', stage_i_col), ('II', stage_ii_col),
                                 ('III', stage_iii_col), ('IV', stage_iv_col)]:
                                     if stage_col and stage_col in crs_df.columns:

        stage_mask = (crs_df[stage_col] == 1)

    if stage_mask.sum() > 0:
        stage_crs = crs_df[stage_mask]

    death_rate = stage_crs['death'].mean()
    n_stage = len(stage_crs)
    n_death_stage = stage_crs['death'].sum()
    if n_stage >= 3:  # Only report if at least 3 patients
    print(f" Stage {stage_num}: {n_death_stage}/{n_stage} deaths ({death_rate * 100:.1f}%)")
    else:

        print(" WARNING: Cancer stage features not found in dataset")

    print(" Note: Stage information may need to be extracted from drug_indication field")

    return findings


def generate_summary_tables(df):
    """
    Generate clear summary tables with groupby + agg for key variables.
    Returns tables as DataFrames and prints formatted output.
    """
    print("\n" + "=" * 70)
    print("Step 3.5: Summary Tables (Clear Cutoffs & Numbers)")
    print("=" * 70)

    # Filter to AE patients only (support both has_ae and has_crs)
    ae_flag = 'has_ae' if 'has_ae' in df.columns else 'has_crs'
    crs_df = df[df[ae_flag] == 1].copy()

    if len(crs_df) == 0:
        print("WARNING: No AE patients found")

    return {}

    tables = {}

    # 1. AGE STRATIFICATION TABLE (<50, 50-65, 65-75, >75)
    print("\n Table 1: Age Stratification - CRS → Death Rates")
    print("-" * 70)
    if 'age_years' in crs_df.columns:
        age = pd.to_numeric(crs_df['age_years'], errors='coerce')

    crs_df['age_group_detailed'] = pd.cut(
        age,
        bins=[0, 50, 65, 75, 120],
        labels=['<50', '50-65', '65-75', '>75'],
        include_lowest=True
    )

    age_table = crs_df.groupby('age_group_detailed').agg({
        'death': ['count', 'sum', 'mean']
    }).round(3)
    age_table.columns = ['N_Total', 'N_Deaths', 'Death_Rate']
    age_table['Death_Rate_Pct'] = (age_table['Death_Rate'] * 100).round(1)

    print(age_table.to_string())
    print()
    tables['age_stratification'] = age_table

    # 2. BMI STRATIFICATION TABLE (Normal, Overweight, Obese, Underweight)
    print(" Table 2: BMI Stratification - CRS → Death Rates")
    print("-" * 70)
    bmi_categories = []
    bmi_data = []

    if 'bmi_normal' in crs_df.columns:
        normal = crs_df[crs_df['bmi_normal'] == 1]

    if len(normal) > 0:
        bmi_categories.append('Normal (18.5-25)')

    bmi_data.append({
        'Category': 'Normal (18.5-25)',
        'N_Total': len(normal),
        'N_Deaths': int(normal['death'].sum()),
        'Death_Rate': normal['death'].mean()
    })

    if 'bmi_overweight' in crs_df.columns:
        overweight = crs_df[crs_df['bmi_overweight'] == 1]

    if len(overweight) > 0:
        bmi_categories.append('Overweight (25-30)')

    bmi_data.append({
        'Category': 'Overweight (25-30)',
        'N_Total': len(overweight),
        'N_Deaths': int(overweight['death'].sum()),
        'Death_Rate': overweight['death'].mean()
    })

    if 'bmi_obese' in crs_df.columns:
        obese = crs_df[crs_df['bmi_obese'] == 1]

    if len(obese) > 0:
        bmi_categories.append('Obese (>30)')

    bmi_data.append({
        'Category': 'Obese (>30)',
        'N_Total': len(obese),
        'N_Deaths': int(obese['death'].sum()),
        'Death_Rate': obese['death'].mean()
    })

    if 'bmi_underweight' in crs_df.columns:
        underweight = crs_df[crs_df['bmi_underweight'] == 1]

    if len(underweight) > 0:
        bmi_categories.append('Underweight (<18.5)')

    bmi_data.append({
        'Category': 'Underweight (<18.5)',
        'N_Total': len(underweight),
        'N_Deaths': int(underweight['death'].sum()),
        'Death_Rate': underweight['death'].mean()
    })

    if bmi_data:
        bmi_table = pd.DataFrame(bmi_data)

    bmi_table['Death_Rate_Pct'] = (bmi_table['Death_Rate'] * 100).round(1)
    bmi_table = bmi_table[['Category', 'N_Total', 'N_Deaths', 'Death_Rate_Pct']]
    print(bmi_table.to_string(index=False))
    print()
    tables['bmi_stratification'] = bmi_table

    # 3. POLYPHARMACY COMPARISON TABLE
    print(" Table 3: Polypharmacy Comparison - CRS → Death Rates")
    print("-" * 70)
    if 'num_drugs' in crs_df.columns:
        crs_df['polypharmacy_group'] = pd.cut(

        crs_df['num_drugs'],
        bins=[0, 1, 5, 100],
        labels=['Low (≤1)', 'Moderate (2-5)', 'High (>5)'],
        include_lowest=True
    )

    poly_table = crs_df.groupby('polypharmacy_group').agg({
        'death': ['count', 'sum', 'mean']
    }).round(3)
    poly_table.columns = ['N_Total', 'N_Deaths', 'Death_Rate']
    poly_table['Death_Rate_Pct'] = (poly_table['Death_Rate'] * 100).round(1)

    print(poly_table.to_string())
    print()
    tables['polypharmacy'] = poly_table

    # 4. DRUG COMBINATION COMPARISON (Steroid + Antibiotic vs No Combination)
    print(" Table 4: Drug Combination Comparison - CRS → Death Rates")
    print("-" * 70)
    if 'steroid_plus_antibiotic' in crs_df.columns:
        combo_data = []


    # With combination
    with_combo = crs_df[crs_df['steroid_plus_antibiotic'] == 1]
    if len(with_combo) > 0:
        combo_data.append({

        'Group': 'Steroid + Antibiotic',
        'N_Total': len(with_combo),
        'N_Deaths': int(with_combo['death'].sum()),
        'Death_Rate': with_combo['death'].mean()
    })

    # Without combination
    without_combo = crs_df[crs_df['steroid_plus_antibiotic'] == 0]
    if len(without_combo) > 0:
        combo_data.append({

        'Group': 'No Steroid+Antibiotic',
        'N_Total': len(without_combo),
        'N_Deaths': int(without_combo['death'].sum()),
        'Death_Rate': without_combo['death'].mean()
    })

    if combo_data:
        combo_table = pd.DataFrame(combo_data)

    combo_table['Death_Rate_Pct'] = (combo_table['Death_Rate'] * 100).round(1)
    combo_table = combo_table[['Group', 'N_Total', 'N_Deaths', 'Death_Rate_Pct']]
    print(combo_table.to_string(index=False))
    print()
    tables['drug_combination'] = combo_table

    # 5. COMORBIDITY COMPARISON (Diabetes vs No Diabetes)
    print(" Table 5: Comorbidity Comparison - CRS → Death Rates")
    print("-" * 70)
    comorb_data = []

    if 'comorbidity_diabetes' in crs_df.columns:
        with_diabetes = crs_df[crs_df['comorbidity_diabetes'] == 1]

    without_diabetes = crs_df[crs_df['comorbidity_diabetes'] == 0]

    if len(with_diabetes) > 0:
        comorb_data.append({

        'Comorbidity': 'Diabetes',
        'Status': 'With',
        'N_Total': len(with_diabetes),
        'N_Deaths': int(with_diabetes['death'].sum()),
        'Death_Rate': with_diabetes['death'].mean()
    })

    if len(without_diabetes) > 0:
        comorb_data.append({

        'Comorbidity': 'Diabetes',
        'Status': 'Without',
        'N_Total': len(without_diabetes),
        'N_Deaths': int(without_diabetes['death'].sum()),
        'Death_Rate': without_diabetes['death'].mean()
    })

    if comorb_data:
        comorb_table = pd.DataFrame(comorb_data)

    comorb_table['Death_Rate_Pct'] = (comorb_table['Death_Rate'] * 100).round(1)
    comorb_table = comorb_table[['Comorbidity', 'Status', 'N_Total', 'N_Deaths', 'Death_Rate_Pct']]
    print(comorb_table.to_string(index=False))
    print()
    tables['comorbidity'] = comorb_table

    # 6. INFECTION AE COMPARISON (With vs Without)
    print(" Table 6: Infection AE Comparison - CRS → Death Rates")
    print("-" * 70)
    if 'has_infection_ae' in crs_df.columns:
        infection_data = []


    with_infection = crs_df[crs_df['has_infection_ae'] == 1]
    without_infection = crs_df[crs_df['has_infection_ae'] == 0]

    if len(with_infection) > 0:
        infection_data.append({

        'Group': 'With Infection AE',
        'N_Total': len(with_infection),
        'N_Deaths': int(with_infection['death'].sum()),
        'Death_Rate': with_infection['death'].mean()
    })

    if len(without_infection) > 0:
        infection_data.append({

        'Group': 'Without Infection AE',
        'N_Total': len(without_infection),
        'N_Deaths': int(without_infection['death'].sum()),
        'Death_Rate': without_infection['death'].mean()
    })

    if infection_data:
        infection_table = pd.DataFrame(infection_data)

    infection_table['Death_Rate_Pct'] = (infection_table['Death_Rate'] * 100).round(1)
    infection_table = infection_table[['Group', 'N_Total', 'N_Deaths', 'Death_Rate_Pct']]
    print(infection_table.to_string(index=False))
    print()
    tables['infection_ae'] = infection_table

    return tables


def generate_missingness_summary(df):
    """
    Generate missingness summary for key model features in CRS dataset.

    Parameters:
        -----------

    df : pd.DataFrame
    CRS patient dataframe

    Returns:
        --------

    pd.DataFrame
    Missingness summary with feature name, missing count, missing percentage
    """
    key_features = [
        'age_years', 'age_missing', 'patientweight', 'weight_missing', 'bmi',
        'bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obese',
        'num_drugs', 'polypharmacy', 'high_polypharmacy',
        'num_reactions', 'multiple_reactions',
        'cancer_stage_I', 'cancer_stage_II', 'cancer_stage_III', 'cancer_stage_IV',
        'comorbidity_diabetes', 'comorbidity_hypertension', 'comorbidity_cardiac',
        'has_steroid', 'has_antibiotic', 'has_chemo'
    ]

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


def summarize_drug_usage(df_crs):
    """
    Summarize drug usage patterns in CRS-death vs CRS-survival cases.
    Returns a table and plain language conclusions for slides.

    Parameters:
        -----------

    df_crs : pd.DataFrame
    CRS patient dataframe with drug flags and death outcome

    Returns:
        --------

    tuple: (summary_table DataFrame, plain_language_conclusions list)
    """
    print("\n" + "=" * 70)
    print(" Drug Usage Summary: CRS-Death vs CRS-Survival")
    print("=" * 70)
    print()

    summary_data = []

    # Define drug categories to analyze
    drug_categories = {
        'Steroids': 'has_steroid',
        'Chemo': 'has_chemo',
        'Antibiotic': 'has_antibiotic',
        'Antiviral': 'has_antiviral',
        'Targeted Therapy': 'has_targeted',
        'Antifungal': 'has_antifungal'
    }

    # Analyze each drug category
    for category_name, col_name in drug_categories.items():
        if col_name in df_crs.columns:

        # CRS-death cases
    death_cases = df_crs[df_crs['death'] == 1]
    survival_cases = df_crs[df_crs['death'] == 0]

    if len(death_cases) > 0 and len(survival_cases) > 0:
        death_usage_pct = (death_cases[col_name].sum() / len(death_cases)) * 100

    survival_usage_pct = (survival_cases[col_name].sum() / len(survival_cases)) * 100
    difference = death_usage_pct - survival_usage_pct

    summary_data.append({
        'Drug Category': category_name,
        'CRS-Death %': f"{death_usage_pct:.1f}%",
        'CRS-Survival %': f"{survival_usage_pct:.1f}%",
        'Difference': f"{difference:+.1f}%"
    })

    # Special: Steroid + Antibiotic combination
    if 'steroid_plus_antibiotic' in df_crs.columns:
        combo_death = df_crs[(df_crs['death'] == 1) & (df_crs['steroid_plus_antibiotic'] == 1)]

    combo_all = df_crs[df_crs['steroid_plus_antibiotic'] == 1]

    if len(combo_all) > 0:
        combo_death_rate = (len(combo_death) / len(combo_all)) * 100


    summary_data.append({
        'Drug Category': 'Steroid + Antibiotic',
        'CRS-Death %': f"{combo_death_rate:.1f}%",
        'CRS-Survival %': 'N/A',
        'Difference': f"{len(combo_death)}/{len(combo_all)} patients"
    })

    # Create summary table
    summary_table = pd.DataFrame(summary_data)

    if len(summary_table) > 0:
        print(summary_table.to_string(index=False))

    print()

    # Save to CSV
    summary_table.to_csv('crs_drug_usage_summary.csv', index=False)
    print(" Saved: crs_drug_usage_summary.csv")
    print()

    # Generate plain language conclusions
    conclusions = []

    # Conclusion 1: Steroid usage comparison
    if 'Steroids' in summary_table['Drug Category'].values:
        steroids_row = summary_table[summary_table['Drug Category'] == 'Steroids'].iloc[0]

    death_pct = float(steroids_row['CRS-Death %'].replace('%', ''))
    survival_pct = float(steroids_row['CRS-Survival %'].replace('%', ''))

    if death_pct > survival_pct:
        conclusions.append(

        f"For CRS patients on epcoritamab, steroid use is much more common in fatal cases "
        f"than in survivors ({death_pct:.0f}% vs {survival_pct:.0f}%)."
    )

    # Conclusion 2: Steroid + Antibiotic combination
    if 'Steroid + Antibiotic' in summary_table['Drug Category'].values:
        combo_row = summary_table[summary_table['Drug Category'] == 'Steroid + Antibiotic'].iloc[0]

    death_info = combo_row['Difference']  # Format: "29/31 patients"

    if '/' in death_info:
        parts = death_info.split('/')

    n_deaths = int(parts[0])
    n_total = int(parts[1].split()[0])
    death_rate = (n_deaths / n_total) * 100 if n_total > 0 else 0

    conclusions.append(
        f"The combination of steroids and antibiotics is associated with a very high observed "
        f"death rate in CRS cases ({death_rate:.1f} percent, {n_deaths} of {n_total} patients)."
    )

    return summary_table, conclusions

    return None, []


def generate_plain_language_summary(tables, findings, crs_df, drug_conclusions=None):
    """
    Generate plain language summary statements for slides and reports.
    Based on summary tables and findings.

    Parameters:
        -----------

    tables : dict
    Summary tables from generate_summary_tables()
    findings : list
    Findings from stratified_analysis()
    crs_df : pd.DataFrame
    CRS patient dataframe
    drug_conclusions : list, optional
    Plain language conclusions about drug usage (from summarize_drug_usage())
    """
    print("\n" + "=" * 70)
    print("Step 7: Plain Language Summary (Ready for Slides/Reports)")
    print("=" * 70)
    print()

    summary_statements = []

    # Overall CRS statistics
    n_crs = len(crs_df)
    n_death = crs_df['death'].sum()
    overall_death_rate = (n_death / n_crs) * 100 if n_crs > 0 else 0

    summary_statements.append("=" * 70)
    summary_statements.append("PLAIN LANGUAGE SUMMARY FOR PRESENTATION")
    summary_statements.append("=" * 70)
    summary_statements.append("")

    # Add drug usage conclusions at the beginning (key findings)
    if drug_conclusions and len(drug_conclusions) > 0:
        summary_statements.append(" Key Drug Usage Findings:")

    summary_statements.append("")
    for conclusion in drug_conclusions:
        summary_statements.append(f" • {conclusion}")

    summary_statements.append("")
    summary_statements.append("-" * 70)
    summary_statements.append("")

    # Age findings
    if 'age_stratification' in tables:
        age_table = tables['age_stratification']

    summary_statements.append(" Age Stratification Findings:")
    summary_statements.append("")

    for age_group in age_table.index:
        if pd.notna(age_group):

        n_total = int(age_table.loc[age_group, 'N_Total'])

    n_deaths = int(age_table.loc[age_group, 'N_Deaths'])
    death_rate = age_table.loc[age_group, 'Death_Rate_Pct']

    summary_statements.append(
        f" • Age {age_group} CRS patients: {death_rate:.1f}% death rate "
        f"({n_deaths}/{n_total} patients)"
    )

    # Compare extremes
    if '<50' in age_table.index and '>75' in age_table.index:
        young_rate = age_table.loc['<50', 'Death_Rate_Pct']

    old_rate = age_table.loc['>75', 'Death_Rate_Pct']
    ratio = old_rate / young_rate if young_rate > 0 else 0
    summary_statements.append("")
    summary_statements.append(
        f" → Patients over 75 years have {ratio:.1f}x higher death risk "
        f"than patients under 50 years ({old_rate:.1f}% vs {young_rate:.1f}%)"
    )
    summary_statements.append("")

    # BMI findings
    if 'bmi_stratification' in tables:
        bmi_table = tables['bmi_stratification']

    summary_statements.append(" BMI Stratification Findings:")
    summary_statements.append("")

    for _, row in bmi_table.iterrows():
        category = row['Category']

    n_total = int(row['N_Total'])
    n_deaths = int(row['N_Deaths'])
    death_rate = row['Death_Rate_Pct']

    summary_statements.append(
        f" • {category} CRS patients: {death_rate:.1f}% death rate "
        f"({n_deaths}/{n_total} patients)"
    )

    # Compare obese vs underweight
    obese_row = bmi_table[bmi_table['Category'].str.contains('Obese', na=False)]
    underweight_row = bmi_table[bmi_table['Category'].str.contains('Underweight', na=False)]

    if len(obese_row) > 0 and len(underweight_row) > 0:
        obese_rate = obese_row.iloc[0]['Death_Rate_Pct']

    underweight_rate = underweight_row.iloc[0]['Death_Rate_Pct']
    summary_statements.append("")
    summary_statements.append(
        f" → Obese patients: {obese_rate:.1f}% death rate; "
        f"Underweight patients: {underweight_rate:.1f}% death rate"
    )
    summary_statements.append("")

    # Polypharmacy findings
    if 'polypharmacy' in tables:
        poly_table = tables['polypharmacy']

    summary_statements.append(" Polypharmacy Findings:")
    summary_statements.append("")

    high_poly = poly_table[poly_table.index.str.contains('High', na=False)]
    low_poly = poly_table[poly_table.index.str.contains('Low', na=False)]

    if len(high_poly) > 0 and len(low_poly) > 0:
        high_rate = high_poly.iloc[0]['Death_Rate_Pct']

    low_rate = low_poly.iloc[0]['Death_Rate_Pct']
    high_n = int(high_poly.iloc[0]['N_Total'])
    low_n = int(low_poly.iloc[0]['N_Total'])
    high_deaths = int(high_poly.iloc[0]['N_Deaths'])
    low_deaths = int(low_poly.iloc[0]['N_Deaths'])

    ratio = high_rate / low_rate if low_rate > 0 else 0

    summary_statements.append(
        f" • High polypharmacy (>5 drugs): {high_rate:.1f}% death rate "
        f"({high_deaths}/{high_n} patients)"
    )
    summary_statements.append(
        f" • Low polypharmacy (≤1 drug): {low_rate:.1f}% death rate "
        f"({low_deaths}/{low_n} patients)"
    )
    summary_statements.append("")
    summary_statements.append(
        f" → High polypharmacy patients have {ratio:.1f}x higher death risk "
        f"than low polypharmacy patients"
    )
    summary_statements.append("")

    # Drug combination findings
    if 'drug_combination' in tables:
        combo_table = tables['drug_combination']

    summary_statements.append(" Drug Combination Findings:")
    summary_statements.append("")

    with_combo = combo_table[combo_table['Group'].str.contains('Steroid.*Antibiotic', na=False, regex=True)]
    without_combo = combo_table[combo_table['Group'].str.contains('No', na=False)]

    if len(with_combo) > 0 and len(without_combo) > 0:
        combo_rate = with_combo.iloc[0]['Death_Rate_Pct']

    no_combo_rate = without_combo.iloc[0]['Death_Rate_Pct']
    combo_n = int(with_combo.iloc[0]['N_Total'])
    no_combo_n = int(without_combo.iloc[0]['N_Total'])
    combo_deaths = int(with_combo.iloc[0]['N_Deaths'])
    no_combo_deaths = int(without_combo.iloc[0]['N_Deaths'])

    ratio = combo_rate / no_combo_rate if no_combo_rate > 0 else 0

    summary_statements.append(
        f" • CRS patients on Steroid + Antibiotic: {combo_rate:.1f}% death rate "
        f"({combo_deaths}/{combo_n} patients)"
    )
    summary_statements.append(
        f" • CRS patients without this combination: {no_combo_rate:.1f}% death rate "
        f"({no_combo_deaths}/{no_combo_n} patients)"
    )
    summary_statements.append("")
    summary_statements.append(
        f" → Steroid + Antibiotic combination is associated with {ratio:.1f}x higher "
        f"death risk compared to patients without this combination"
    )
    summary_statements.append("")

    # Comorbidity findings
    if 'comorbidity' in tables:
        comorb_table = tables['comorbidity']

    summary_statements.append(" Comorbidity Findings:")
    summary_statements.append("")

    with_comorb = comorb_table[comorb_table['Status'] == 'With']
    without_comorb = comorb_table[comorb_table['Status'] == 'Without']

    if len(with_comorb) > 0 and len(without_comorb) > 0:
        with_rate = with_comorb.iloc[0]['Death_Rate_Pct']

    without_rate = without_comorb.iloc[0]['Death_Rate_Pct']
    with_n = int(with_comorb.iloc[0]['N_Total'])
    without_n = int(without_comorb.iloc[0]['N_Total'])
    with_deaths = int(with_comorb.iloc[0]['N_Deaths'])
    without_deaths = int(without_comorb.iloc[0]['N_Deaths'])

    ratio = with_rate / without_rate if without_rate > 0 else 0
    comorb_name = with_comorb.iloc[0]['Comorbidity']

    summary_statements.append(
        f" • CRS patients with {comorb_name}: {with_rate:.1f}% death rate "
        f"({with_deaths}/{with_n} patients)"
    )
    summary_statements.append(
        f" • CRS patients without {comorb_name}: {without_rate:.1f}% death rate "
        f"({without_deaths}/{without_n} patients)"
    )
    summary_statements.append("")
    summary_statements.append(
        f" → {comorb_name} is associated with {ratio:.1f}x higher death risk "
        f"in CRS patients"
    )
    summary_statements.append("")

    # Infection AE findings
    if 'infection_ae' in tables:
        infection_table = tables['infection_ae']

    summary_statements.append(" Infection AE Findings:")
    summary_statements.append("")

    with_inf = infection_table[infection_table['Group'].str.contains('With', na=False)]
    without_inf = infection_table[infection_table['Group'].str.contains('Without', na=False)]

    if len(with_inf) > 0 and len(without_inf) > 0:
        inf_rate = with_inf.iloc[0]['Death_Rate_Pct']

    no_inf_rate = without_inf.iloc[0]['Death_Rate_Pct']
    inf_n = int(with_inf.iloc[0]['N_Total'])
    no_inf_n = int(without_inf.iloc[0]['N_Total'])
    inf_deaths = int(with_inf.iloc[0]['N_Deaths'])
    no_inf_deaths = int(without_inf.iloc[0]['N_Deaths'])

    ratio = inf_rate / no_inf_rate if no_inf_rate > 0 else 0

    summary_statements.append(
        f" • CRS patients with infection-related AE: {inf_rate:.1f}% death rate "
        f"({inf_deaths}/{inf_n} patients)"
    )
    summary_statements.append(
        f" • CRS patients without infection AE: {no_inf_rate:.1f}% death rate "
        f"({no_inf_deaths}/{no_inf_n} patients)"
    )
    summary_statements.append("")
    summary_statements.append(
        f" → Infection-related adverse events are associated with {ratio:.1f}x higher "
        f"death risk in CRS patients"
    )
    summary_statements.append("")

    # Combined risk factors
    summary_statements.append("=" * 70)
    summary_statements.append("KEY COMBINED RISK FACTORS")
    summary_statements.append("=" * 70)
    summary_statements.append("")

    # Example: Age + Polypharmacy
    if 'age_stratification' in tables and 'polypharmacy' in tables:
        age_table = tables['age_stratification']

    poly_table = tables['polypharmacy']

    if '>75' in age_table.index:
        old_rate = age_table.loc['>75', 'Death_Rate_Pct']

    high_poly = poly_table[poly_table.index.str.contains('High', na=False)]
    if len(high_poly) > 0:
        high_poly_rate = high_poly.iloc[0]['Death_Rate_Pct']

    combined_risk = (old_rate + high_poly_rate) / 2
    risk_multiplier = combined_risk / overall_death_rate if overall_death_rate > 0 else 0

    summary_statements.append(
        f" • CRS patients aged >75 years with high polypharmacy (>5 drugs) "
        f"have approximately {risk_multiplier:.1f}x higher death risk "
        f"compared to the overall CRS population ({overall_death_rate:.1f}%)"
    )
    summary_statements.append("")

    # Print all statements
    summary_text = '\n'.join(summary_statements)
    print(summary_text)

    # Save to file
    with open('crs_plain_language_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)


    print(f"\n Saved plain language summary: crs_plain_language_summary.txt")
    print(" → This file can be directly copied to slides and reports!")

    return summary_text


def analyze_top_drug_combinations(df, top_n=10):
    """
    Analyze most common drug combinations in death vs survival.
    """
    print("\n" + "=" * 70)
    print("Step 4: Top Drug Combination Analysis")
    print("=" * 70)

    # Filter to AE patients only (support both has_ae and has_crs)
    ae_flag = 'has_ae' if 'has_ae' in df.columns else 'has_crs'
    crs_df = df[df[ae_flag] == 1].copy()

    if len(crs_df) == 0 or 'all_drugs' not in crs_df.columns:
        print("WARNING: Insufficient data for drug combination analysis")

    return []

    # Extract drug combinations (pairs)
    combinations_death = []
    combinations_survival = []

    for idx, row in crs_df.iterrows():
        drugs_str = str(row['all_drugs']) if not pd.isna(row['all_drugs']) else ''

    if drugs_str == '':
        continue


    drugs = [d.strip().upper() for d in drugs_str.split('|') if d.strip()]
    drugs = [d for d in drugs if d != 'EPCORITAMAB']  # Remove target drug

    # Get top drugs (excluding Epcoritamab)
    if len(drugs) >= 2:
        # Create pairs
    for i, drug1 in enumerate(drugs[:5]):  # Limit to top 5 per patient
    for drug2 in drugs[i + 1:5]:
        combo = tuple(sorted([drug1, drug2]))

    if row['death'] == 1:
        combinations_death.append(combo)

    else:


        combinations_survival.append(combo)


    # Count combinations
    death_counts = Counter(combinations_death)
    survival_counts = Counter(combinations_survival)

    # Find combinations more common in death
    all_combos = set(list(death_counts.keys()) + list(survival_counts.keys()))

    combo_analysis = []
    for combo in all_combos:
        n_death = death_counts.get(combo, 0)

    n_survival = survival_counts.get(combo, 0)
    n_total = n_death + n_survival

    if n_total >= 3:  # At least 3 occurrences
    death_rate = n_death / n_total
    combo_analysis.append({
        'drug1': combo[0],
        'drug2': combo[1],
        'n_total': n_total,
        'n_death': n_death,
        'n_survival': n_survival,
        'death_rate': death_rate
    })

    # Sort by death rate
    combo_analysis.sort(key=lambda x: x['death_rate'], reverse=True)

    print(f"\n Top {min(top_n, len(combo_analysis))} Drug Combinations by Death Rate:")
    for i, combo in enumerate(combo_analysis[:top_n], 1):
        print(f" {i}. {combo['drug1'][:30]} + {combo['drug2'][:30]}")

    print(f" Death rate: {combo['death_rate'] * 100:.1f}% ({combo['n_death']}/{combo['n_total']})")

    return combo_analysis[:top_n]


def generate_granular_report(findings, combo_analysis, output_file='granular_crs_report.md'):
    """
    Generate detailed report with specific percentages and cutoffs.
    """
    print("\n" + "=" * 70)
    print("Step 5: Generating Granular Report")
    print("=" * 70)

    report = []
    report.append("# Granular CRS → Death Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")

    report.append("## Key Findings with Specific Cutoffs\n")

    # Age findings
    age_findings = [f for f in findings if f['variable'] == 'Age']
    if age_findings:
        report.append("### 1. Age Stratification\n")

    for finding in age_findings:
        report.append(

        f"- **Age {finding['category']} CRS patients:** {finding['death_rate'] * 100:.1f}% death rate ({finding['n_deaths']}/{finding['n_total']} patients)")
    report.append("\n**Interpretation:** " +
                  f"Age >65 years: {age_findings[0]['death_rate'] * 100:.1f}% vs ≤65 years: {age_findings[1]['death_rate'] * 100:.1f}% " +
                  "in CRS patients.\n")

    # BMI findings
    bmi_findings = [f for f in findings if f['variable'] == 'BMI']
    if bmi_findings:
        report.append("### 2. BMI/Obesity Analysis\n")

    for finding in bmi_findings:
        report.append(

        f"- **{finding['category']} CRS patients:** {finding['death_rate'] * 100:.1f}% death rate ({finding['n_deaths']}/{finding['n_total']} patients)")
    report.append("\n**Interpretation:** " +
                  f"Obese patients (BMI>30) with CRS have a {bmi_findings[0]['death_rate'] * 100:.1f}% death rate.\n")

    # Comorbidity findings
    comorb_findings = [f for f in findings if f['variable'] == 'Comorbidity']
    if comorb_findings:
        report.append("### 3. Comorbidity Analysis\n")

    for finding in comorb_findings:
        report.append(

        f"- **CRS patients with {finding['category']}:** {finding['death_rate'] * 100:.1f}% death rate ({finding['n_deaths']}/{finding['n_total']} patients)")
    report.append("\n")

    # Drug combination findings
    combo_findings = [f for f in findings if 'Drug Combination' in f['variable']]
    if combo_findings:
        report.append("### 4. Drug Combination Analysis\n")

    for finding in combo_findings:
        report.append(

        f"- **CRS patients on {finding['category']}:** {finding['death_rate'] * 100:.1f}% death rate ({finding['n_deaths']}/{finding['n_total']} patients)")
    report.append("\n")

    # Infection + Comorbidity
    infection_findings = [f for f in findings if 'Infection' in f['variable']]
    if infection_findings:
        report.append("### 5. Infection AE + Comorbidity\n")

    for finding in infection_findings:
        report.append(

        f"- **{finding['category']} CRS patients:** {finding['death_rate'] * 100:.1f}% death rate ({finding['n_deaths']}/{finding['n_total']} patients)")
    report.append("\n**Interpretation:** " +
                  f"CRS patients with infection-related adverse events and diabetes have a {infection_findings[0]['death_rate'] * 100:.1f}% death rate.\n")

    # Top drug combinations
    if combo_analysis:
        report.append("### 6. Top Drug Combinations Associated with Death\n")

    report.append("| Rank | Drug 1 | Drug 2 | Death Rate | N Deaths / N Total |\n")
    report.append("|------|--------|--------|------------|-------------------|\n")
    for i, combo in enumerate(combo_analysis[:10], 1):
        drug1_short = combo['drug1'][:30] if len(combo['drug1']) <= 30 else combo['drug1'][:27] + '...'

    drug2_short = combo['drug2'][:30] if len(combo['drug2']) <= 30 else combo['drug2'][:27] + '...'
    report.append(
        f"| {i} | {drug1_short} | {drug2_short} | {combo['death_rate'] * 100:.1f}% | {combo['n_death']}/{combo['n_total']} |\n")
    report.append("\n")

    # Summary statements
    report.append("## Key Statements for Presentation\n\n")

    if age_findings and len(age_findings) >= 2:
        gt_65 = next((f for f in age_findings if '>' in f['category']), None)

    le_65 = next((f for f in age_findings if '≤' in f['category'] or '<=' in f['category']), None)
    if gt_65 and le_65:
        report.append(

        f"**Finding 1:** We found that CRS patients aged >65 years had a {gt_65['death_rate'] * 100:.1f}% death rate, ")
    report.append(f"compared to {le_65['death_rate'] * 100:.1f}% in patients aged ≤65 years.\n\n")

    if combo_findings:
        for finding in combo_findings:

        report.append(

        f"**Finding 2:** CRS patients using finding['category']} had a {100:.1f}% death rate ")
    report.append(f"({finding['n_deaths']}/{finding['n_total']} patients).\n\n")

    if infection_findings:
        report.append(f"**Finding 3:** CRS patients with infection-related adverse events and diabetes had a ")

    report.append(
        f"{infection_findings[0]['death_rate'] * 100:.1f}% death rate ({infection_findings[0]['n_deaths']}/{infection_findings[0]['n_total']} patients).\n\n")

    # Save report
    report_text = ''.join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)


    print(f" Saved report: {output_file}")
    return report_text


def visualize_granular_findings(df, findings, output_dir='.'):
    """Create visualizations for granular findings."""
    print("\n" + "=" * 70)
    print("Step 6: Generating Visualizations")
    print("=" * 70)

    fig_dir = Path(output_dir)
    fig_dir.mkdir(exist_ok=True)

    # Filter to AE patients only (support both has_ae and has_crs)
    ae_flag = 'has_ae' if 'has_ae' in df.columns else 'has_crs'
    crs_df = df[df[ae_flag] == 1].copy()

    if len(crs_df) == 0:
        print("WARNING: No AE patients for visualization")

    return

    # 1. Age stratification plot
    if 'age_years' in crs_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))


    age = pd.to_numeric(crs_df['age_years'], errors='coerce')
    death = crs_df['death']

    # Age groups
    age_bins = [0, 50, 65, 75, 100]
    age_labels = ['<50', '50-65', '65-75', '75+']
    crs_df['age_group'] = pd.cut(age, bins=age_bins, labels=age_labels, include_lowest=True)

    # Calculate death rates by age group
    age_death_rates = crs_df.groupby('age_group')['death'].agg(['mean', 'count'])

    bars = ax.bar(age_death_rates.index, age_death_rates['mean'], color=['#3498db', '#9b59b6', '#e74c3c', '#c0392b'])
    ax.set_ylabel('Death Rate', fontsize=12)
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_title('Death Rate by Age Group in CRS Patients', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])

    for i, (rate, count) in enumerate(zip(age_death_rates['mean'], age_death_rates['count'])):
        ax.text(i, rate + 0.05, f'{rate * 100:.1f}%\n(n={int(count)})', ha='center', fontsize=10)


    plt.tight_layout()
    plt.savefig(fig_dir / 'crs_age_stratification.png', dpi=300, bbox_inches='tight')
    print(" Saved: crs_age_stratification.png")
    plt.close()

    # 2. BMI stratification plot
    if 'bmi_obese' in crs_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))


    bmi_groups = []
    death_rates = []
    counts = []

    if crs_df['bmi_obese'].sum() > 0:
        bmi_groups.append('Obese\n(BMI>30)')

    death_rates.append(crs_df[crs_df['bmi_obese'] == 1]['death'].mean())
    counts.append(crs_df['bmi_obese'].sum())

    if crs_df['bmi_overweight'].sum() > 0:
        bmi_groups.append('Overweight\n(25-30)')

    death_rates.append(crs_df[crs_df['bmi_overweight'] == 1]['death'].mean())
    counts.append(crs_df['bmi_overweight'].sum())

    if crs_df['bmi_normal'].sum() > 0:
        bmi_groups.append('Normal\n(18.5-25)')

    death_rates.append(crs_df[crs_df['bmi_normal'] == 1]['death'].mean())
    counts.append(crs_df['bmi_normal'].sum())

    if bmi_groups:
        bars = ax.bar(bmi_groups, death_rates, color=['#e74c3c', '#f39c12', '#2ecc71'])

    ax.set_ylabel('Death Rate', fontsize=12)
    ax.set_title('Death Rate by BMI Category in CRS Patients', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(death_rates) * 1.3 if max(death_rates) > 0 else 1])

    for i, (rate, count) in enumerate(zip(death_rates, counts)):
        ax.text(i, rate + max(death_rates) * 0.05, f'{rate * 100:.1f}%\n(n={count})', ha='center', fontsize=10)


    plt.tight_layout()
    plt.savefig(fig_dir / 'crs_bmi_stratification.png', dpi=300, bbox_inches='tight')
    print(" Saved: crs_bmi_stratification.png")
    plt.close()

    # 3. Comorbidity comparison
    comorb_findings = [f for f in findings if f['variable'] == 'Comorbidity']
    if comorb_findings:
        fig, ax = plt.subplots(figsize=(10, 6))


    comorbs = [f['category'] for f in comorb_findings]
    rates = [f['death_rate'] for f in comorb_findings]
    counts = [f['n_total'] for f in comorb_findings]

    bars = ax.barh(comorbs, rates, color='#3498db')
    ax.set_xlabel('Death Rate', fontsize=12)
    ax.set_title('Death Rate by Comorbidity in CRS Patients', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    for i, (rate, count) in enumerate(zip(rates, counts)):
        ax.text(rate + 0.02, i, f'{rate * 100:.1f}% (n={count})', va='center', fontsize=10)


    plt.tight_layout()
    plt.savefig(fig_dir / 'crs_comorbidity_comparison.png', dpi=300, bbox_inches='tight')
    print(" Saved: crs_comorbidity_comparison.png")
    plt.close()

    print("\n Visualization complete")


def run_granular_analysis(drug='Epcoritamab', ae='CRS', data_file='main_data.csv', output_dir='.'):
    """
    Run granular analysis for a specific drug and adverse event.

    Parameters:
        -----------

    drug : str
    Target drug name (default: 'Epcoritamab')
    ae : str
    Adverse event name (default: 'CRS')
    data_file : str
    Path to input CSV file (default: 'main_data.csv')
    output_dir : str
    Output directory for results (default: '.')

    Returns:
        --------

    dict : Analysis results and output file paths
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get AE keywords
    ae_keywords = get_ae_keywords(ae)

    print("=" * 70)
    print(f"Granular {ae} → Death Analysis: {drug}")
    print("=" * 70)
    print()

    # Load data
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")

    return None

    print(f" Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f" Loaded {len(df)} records")
    print()

    # Step 1: Identify AE cases
    epcor_df = identify_ae_cases(df, drug, ae_keywords)
    if epcor_df is None or len(epcor_df) == 0:
        print(f"\nERROR: No {drug} patients with {ae} found. Analysis cannot proceed.")

    return None

    # Continue with the rest of the analysis...
    # (The rest of the main() function logic goes here)
    # For now, we'll call the original main() logic but with parameters

    # Filter to AE cases only
    crs_df = epcor_df[epcor_df['has_ae'] == 1].copy()

    if len(crs_df) == 0:
        print(f"\nERROR: No {ae} cases found. Analysis cannot proceed.")

    return None

    print(f"\n {ae} cases for analysis: {len(crs_df)}")
    print()

    # Step 2: Granular feature engineering
    crs_df = granular_feature_engineering(crs_df)

    # Step 3: Stratified analysis
    findings = stratified_analysis(crs_df)

    # Step 4: Generate summary tables
    tables = generate_summary_tables(crs_df)

    # Step 5: Missingness summary
    missingness_summary = generate_missingness_summary(crs_df)
    missingness_file = output_path / 'crs_missingness_summary.csv'
    missingness_summary.to_csv(missingness_file, index=False)
    print(f" Saved: {missingness_file}")

    # Step 6: Drug usage summary
    drug_conclusions = summarize_drug_usage(crs_df)

    # Step 7: Top drug combinations
    combo_analysis = analyze_top_drug_combinations(crs_df, top_n=10)

    # Step 8: Generate plain language summary
    summary_text = generate_plain_language_summary(tables, findings, crs_df, drug_conclusions)
    summary_file = output_path / 'crs_plain_language_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f" Saved plain language summary: {summary_file}")

    # Step 9: Generate report
    report_file = output_path / 'granular_crs_report.md'
    generate_granular_report(findings, combo_analysis, output_file=str(report_file))

    # Step 10: Visualizations
    visualize_granular_findings(crs_df, findings, output_dir=str(output_path))

    # Save metadata
    meta = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'drug': drug,
        'adverse_event': ae,
        'data_file': data_file,
        'n_total': len(epcor_df),
        'n_ae': int(epcor_df['has_ae'].sum()),
        'findings': findings,
        'top_combinations': combo_analysis[:5]  # Top 5 only
    }

    meta_file = output_path / 'granular_crs_meta.json'
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


    print("\n" + "=" * 70)
    print(f" Granular {ae} Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f" - {report_file}")
    print(f" - {summary_file} ⭐ (Ready for slides!)")
    print(f" - {missingness_file}")
    print(f" - {meta_file}")

    return {
        'report': str(report_file),
        'summary': str(summary_file),
        'missingness': str(missingness_file),
        'metadata': str(meta_file)
    }


def main(drug=None, ae=None, data_file=None, output_dir='.'):
    """
    Main execution function with parameterized support.

    Parameters:
        -----------

    drug : str, optional
    Target drug name (default: DEFAULT_TARGET_DRUG)
    ae : str, optional
    Adverse event name (default: 'CRS')
    data_file : str, optional
    Input data file path (default: DEFAULT_DATA_FILE)
    output_dir : str, optional
    Output directory (default: '.')
    """
    # Use defaults if not provided (backward compatibility)
    if drug is None:
        drug = DEFAULT_TARGET_DRUG

    if ae is None:
        ae = 'CRS'

    if data_file is None:
        data_file = DEFAULT_DATA_FILE


    # Get AE keywords
    ae_keywords = get_ae_keywords(ae)

    print("=" * 70)
    print(f"Granular {ae} → Death Analysis: {drug}")
    print("=" * 70)
    print()

    # Load data
    if not Path(data_file).exists():
        print(f"ERROR: Data file not found: {data_file}")

    return

    print(f" Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f" Loaded {len(df)} records")
    print()

    # Step 1: Identify AE cases (using provided parameters)
    epcor_df = identify_ae_cases(df, drug, ae_keywords)
    if epcor_df is None or len(epcor_df) == 0:
        print(f"\nERROR: No {drug} patients found. Analysis cannot proceed.")

    return

    # Step 2: Granular feature engineering
    feature_df = granular_feature_engineering(epcor_df)

    # Step 3: Stratified analysis
    findings = stratified_analysis(feature_df)

    # Step 3.5: Generate summary tables
    tables = generate_summary_tables(feature_df)

    # Step 4: Drug combination analysis
    combo_analysis = analyze_top_drug_combinations(feature_df, top_n=10)

    # Step 4.5: Drug usage summary (new - for slides)
    # Use 'has_ae' instead of 'has_crs' for generic AE support
    ae_flag_col = 'has_crs' if 'has_crs' in feature_df.columns else 'has_ae'
    if ae_flag_col not in feature_df.columns:
        # Create has_ae flag if it doesn't exist
    feature_df['has_ae'] = 1  # All rows in feature_df are AE cases
    crs_df = feature_df[feature_df[ae_flag_col] == 1].copy()
    drug_summary_table, drug_conclusions = summarize_drug_usage(crs_df)

    # Step 5: Generate report
    report_file = os.path.join(output_dir, 'granular_crs_report.md')
    generate_granular_report(findings, combo_analysis, output_file=report_file)

    # Step 6: Visualizations
    visualize_granular_findings(feature_df, findings, output_dir=output_dir)

    # Step 6.5: Generate missingness summary
    if len(crs_df) > 0:
        print("\n" + "=" * 70)

    print("Step 6.5: Missingness Summary")
    print("=" * 70)
    print()

    missingness_summary = generate_missingness_summary(crs_df)
    missingness_file = os.path.join(output_dir, 'crs_missingness_summary.csv')
    missingness_summary.to_csv(missingness_file, index=True)
    print(f" Saved: {missingness_file}")
    print()

    # Print key missingness statistics
    print(" Key Missingness Statistics:")
    print("-" * 70)
    key_features = ['age_years', 'patientweight', 'bmi', 'num_drugs', 'num_reactions']
    for feat in key_features:
        if feat in missingness_summary['Feature'].values:

        row = missingness_summary[missingness_summary['Feature'] == feat].iloc[0]

    missing_pct = row['Missing_Percentage']
    complete_pct = row['Complete_Percentage']
    print(f" {feat}: {complete_pct:.1f}% complete, {missing_pct:.1f}% missing")
    print()

    # Step 7: Plain language summary (includes drug conclusions)
    generate_plain_language_summary(tables, findings, crs_df, drug_conclusions)

    # Save metadata
    meta = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'drug': drug,
        'adverse_event': ae,
        'n_total': len(epcor_df),
        'n_ae': int(epcor_df[ae_flag_col].sum() if ae_flag_col in epcor_df.columns else len(epcor_df)),
        'findings': findings,
        'top_combinations': combo_analysis[:5]  # Top 5 only
    }

    meta_file = os.path.join(output_dir, 'granular_crs_meta.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


    print("\n" + "=" * 70)
    print(f" Granular {ae} Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f" - {report_file}")
    print(f" - {os.path.join(output_dir, 'crs_plain_language_summary.txt')} ⭐ (Ready for slides!)")
    print(f" - {os.path.join(output_dir, 'crs_age_stratification.png')}")
    print(f" - {os.path.join(output_dir, 'crs_bmi_stratification.png')}{if BMI data available)")
    print(f" - {os.path.join(output_dir, 'crs_comorbidity_comparison.png')}")
    print(f" - {meta_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run granular adverse event analysis')
    parser.add_argument('--drug', type=str, default=None,
                        help=f'Target drug name (default: {DEFAULT_TARGET_DRUG})')
    parser.add_argument('--ae', type=str, default=None,
                        help='Adverse event name (default: CRS)')
    parser.add_argument('--data_file', type=str, default=None,
                        help=f'Input data file (default: {DEFAULT_DATA_FILE})')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory (default: current directory)')

    args = parser.parse_args()

    # Run analysis with parameters (use main() for backward compatibility)
    main(
        drug=args.drug,
        ae=args.ae,
        data_file=args.data_file,
        output_dir=args.output_dir
    )
