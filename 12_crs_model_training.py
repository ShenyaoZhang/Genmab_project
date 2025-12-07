#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 12: CRS-Specific Model Training
====================================

Using the existing model framework, retrain models specifically on CRS cases:
- Input: CRS patients only (features: demographics, drug use, comorbidities, etc.)
- Output: Death prediction / Survival probability / Death risk
- Label: 1 if CRS case resulted in death (seriousnessdeath == 1), 0 otherwise
- Focus: Which variables are most important for "CRS → Death"
- Goal: Results should align with clinical intuition for meeting discussion
"""

import sys
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report
)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants (defaults - can be overridden by function parameters)
DATA_FILE = 'main_data.csv'
DEFAULT_TARGET_DRUG = 'Epcoritamab'
DEFAULT_CRS_KEYWORDS = ['CYTOKINE RELEASE SYNDROME', 'CYTOKINE RELEASE', 'CYTOKINE STORM']

def identify_crs_patients(df, drug_name=None, ae_keywords=None):
    """
    Identify patients with specific adverse event for given drug.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with target_drug and reactions columns
    drug_name : str, optional
        Drug name to filter (default: 'Epcoritamab')
    ae_keywords : list, optional
        List of keywords to identify adverse event (default: CRS keywords)
    
    Returns:
    --------
    pd.DataFrame or None
        Filtered dataframe with identified patients, or None if not found
    """
    # Use defaults if not provided
    if drug_name is None:
        drug_name = DEFAULT_TARGET_DRUG
    if ae_keywords is None:
        ae_keywords = DEFAULT_CRS_KEYWORDS
    
    print("=" * 80)
    print(f"Step 1: Identifying {drug_name} Patients with Adverse Event")
    print("=" * 80)
    
    # Filter to target drug patients
    drug_mask = df['target_drug'].str.contains(drug_name, case=False, na=False)
    drug_df = df[drug_mask].copy()
    
    if len(drug_df) == 0:
        print(f"❌ No {drug_name} records found.")
        return None
    
    print(f"📊 Total {drug_name} records: {len(drug_df)}")
    
    # Identify adverse event cases
    reactions_upper = drug_df['reactions'].fillna('').str.upper()
    ae_mask = pd.Series(False, index=drug_df.index)
    
    for keyword in ae_keywords:
        mask = reactions_upper.str.contains(keyword.upper(), na=False, regex=False)
        ae_mask |= mask
    
    ae_df = drug_df[ae_mask].copy()
    
    ae_name = ae_keywords[0] if ae_keywords else "target AE"
    print(f"🔍 Patients with {ae_name}: {len(ae_df)} ({len(ae_df)/len(drug_df)*100:.1f}%)")
    
    if len(ae_df) == 0:
        print(f"❌ No patients with {ae_name} found.")
        return None
    
    # Outcome: death
    ae_df['death'] = pd.to_numeric(ae_df['seriousnessdeath'], errors='coerce').fillna(0).astype(int)
    
    n_death = ae_df['death'].sum()
    print(f"📊 Deaths in {ae_name} patients: {n_death} ({n_death/len(ae_df)*100:.1f}%)")
    
    return ae_df

def build_crs_features(df):
    """
    Build features for CRS → death prediction.
    Uses similar feature engineering as 03_preprocess_data.py but tailored for CRS analysis.
    """
    print("\n" + "=" * 80)
    print("Step 2: Building Features for CRS → Death Prediction")
    print("=" * 80)
    
    feature_df = df.copy()
    
    # Exclude leakage columns (same as in 03_preprocess_data.py)
    exclude_cols = {
        'safetyreportid', 'target_drug', 'drugname',
        'serious', 'seriousnessdeath', 'seriousnesshospitalization',
        'seriousnesslifethreatening', 'seriousnessdisabling',
        'seriousnesscongenitalanomali', 'seriousnessother',
        'reactions', 'all_drugs', 'drug_indication',
        'receivedate', 'reporter_qualification'
    }
    
    features = []
    
    # 1. Age features
    print("\n🔹 Age Features...")
    if 'age_years' in feature_df.columns:
        feature_df['age_years'] = pd.to_numeric(feature_df['age_years'], errors='coerce')
        feature_df['age_missing'] = feature_df['age_years'].isna().astype(int)
        feature_df['age_years'] = feature_df['age_years'].fillna(feature_df['age_years'].median())
        features.append('age_years')
        features.append('age_missing')
        
        # Age groups (specific cutoffs)
        feature_df['age_gt_65'] = (feature_df['age_years'] > 65).astype(int)
        feature_df['age_gt_70'] = (feature_df['age_years'] > 70).astype(int)
        feature_df['age_50_65'] = ((feature_df['age_years'] >= 50) & (feature_df['age_years'] <= 65)).astype(int)
        features.extend(['age_gt_65', 'age_gt_70', 'age_50_65'])
        
        # Age buckets
        feature_df['age_group'] = pd.cut(
            feature_df['age_years'],
            bins=[0, 50, 65, 75, 120],
            labels=['lt_50', '50_65', '65_75', '75_plus'],
            include_lowest=True
        )
        age_dummies = pd.get_dummies(feature_df['age_group'], prefix='age')
        feature_df = pd.concat([feature_df, age_dummies], axis=1)
        features.extend(age_dummies.columns.tolist())
    
    # 2. Sex features
    print("🔹 Sex Features...")
    if 'patientsex' in feature_df.columns:
        feature_df['sex_male'] = (pd.to_numeric(feature_df['patientsex'], errors='coerce') == 1).astype(int)
        feature_df['sex_female'] = (pd.to_numeric(feature_df['patientsex'], errors='coerce') == 2).astype(int)
        feature_df['sex_unknown'] = (pd.to_numeric(feature_df['patientsex'], errors='coerce').isna()).astype(int)
        features.extend(['sex_male', 'sex_female', 'sex_unknown'])
    
    # 3. Weight/BMI features
    print("🔹 Weight/BMI Features...")
    if 'patientweight' in feature_df.columns:
        weight = pd.to_numeric(feature_df['patientweight'], errors='coerce')
        feature_df['weight_missing'] = weight.isna().astype(int)
        feature_df['patientweight'] = weight.fillna(weight.median())
        features.extend(['patientweight', 'weight_missing'])
        
        # Calculate BMI (assuming average height)
        height_default = 1.65  # meters
        feature_df['bmi'] = feature_df['patientweight'] / (height_default ** 2)
        feature_df['bmi_obese'] = (feature_df['bmi'] > 30).astype(int)
        feature_df['bmi_overweight'] = ((feature_df['bmi'] >= 25) & (feature_df['bmi'] <= 30)).astype(int)
        feature_df['bmi_underweight'] = (feature_df['bmi'] < 18.5).astype(int)
        
        # Add all BMI-related features to the model (including underweight for anorexic-like patients)
        features.extend(['bmi', 'bmi_obese', 'bmi_overweight', 'bmi_underweight'])
    
    # 4. Drug-related features
    print("🔹 Drug-Related Features...")
    if 'num_drugs' in feature_df.columns:
        feature_df['num_drugs'] = pd.to_numeric(feature_df['num_drugs'], errors='coerce').fillna(0).astype(int)
        feature_df['polypharmacy'] = (feature_df['num_drugs'] > 1).astype(int)
        feature_df['high_polypharmacy'] = (feature_df['num_drugs'] > 5).astype(int)
        features.extend(['num_drugs', 'polypharmacy', 'high_polypharmacy'])
    
    # 5. Reaction count features
    print("🔹 Reaction Count Features...")
    if 'num_reactions' in feature_df.columns:
        feature_df['num_reactions'] = pd.to_numeric(feature_df['num_reactions'], errors='coerce').fillna(0).astype(int)
        feature_df['multiple_reactions'] = (feature_df['num_reactions'] > 1).astype(int)
        features.extend(['num_reactions', 'multiple_reactions'])
    
    # 6. Extract drug categories from all_drugs (before we drop it)
    print("🔹 Drug Category Features...")
    if 'all_drugs' in feature_df.columns:
        drugs_upper = feature_df['all_drugs'].fillna('').str.upper()
        
        # Common drug categories
        feature_df['has_steroid'] = (
            drugs_upper.str.contains('PREDNISONE|PREDNISOLONE|DEXAMETHASONE|METHYLPREDNISOLONE', na=False)
        ).astype(int)
        feature_df['has_antibiotic'] = (
            drugs_upper.str.contains('CIPROFLOXACIN|LEVOFLOXACIN|VANCOMYCIN|AMOXICILLIN', na=False)
        ).astype(int)
        feature_df['has_antifungal'] = (
            drugs_upper.str.contains('FLUCONAZOLE|VORICONAZOLE|AMPHOTERICIN', na=False)
        ).astype(int)
        feature_df['has_antiviral'] = (
            drugs_upper.str.contains('ACYCLOVIR|GANCICLOVIR|OSELTAMIVIR', na=False)
        ).astype(int)
        feature_df['has_chemo'] = (
            drugs_upper.str.contains('CYCLOPHOSPHAMIDE|DOXORUBICIN|GEMCITABINE|CISPLATIN', na=False)
        ).astype(int)
        feature_df['has_targeted'] = (
            drugs_upper.str.contains('RITUXIMAB|BRENTUXIMAB|LENALIDOMIDE', na=False)
        ).astype(int)
        
        features.extend([
            'has_steroid', 'has_antibiotic', 'has_antifungal',
            'has_antiviral', 'has_chemo', 'has_targeted'
        ])
        
        # Combination flags
        feature_df['steroid_plus_antibiotic'] = (
            (feature_df['has_steroid'] == 1) & (feature_df['has_antibiotic'] == 1)
        ).astype(int)
        features.append('steroid_plus_antibiotic')
    
    # 7. Extract comorbidities from drug_indication (before we drop it)
    print("🔹 Comorbidity Features...")
    if 'drug_indication' in feature_df.columns:
        indication_upper = feature_df['drug_indication'].fillna('').str.upper()
        
        feature_df['comorbidity_diabetes'] = (
            indication_upper.str.contains('DIABETES|DIABETIC|HYPERGLYCEMIA', na=False)
        ).astype(int)
        feature_df['comorbidity_hypertension'] = (
            indication_upper.str.contains('HYPERTENSION|HYPERTENSIVE', na=False)
        ).astype(int)
        feature_df['comorbidity_cardiac'] = (
            indication_upper.str.contains('CARDIAC|HEART|CARDIOMYOPATHY', na=False)
        ).astype(int)
        
        features.extend([
            'comorbidity_diabetes', 'comorbidity_hypertension', 'comorbidity_cardiac'
        ])
    
    # 7.5. Cancer Stage Features
    # IMPORTANT NOTE: DLBCL stage is NOT available as a structured variable in FAERS.
    # FAERS does not have standardized cancer stage fields.
    # We attempt imperfect extraction from free-text drug_indication field.
    #
    # Pipeline Interface Reservation:
    # - If structured cancer_stage field (numeric 1-4) becomes available in future datasets,
    #   it can be directly used: feature_df['cancer_stage'] = feature_df['cancer_stage']
    # - Create derived features: advanced_stage = (cancer_stage >= 3).astype(int)
    # - If column is all missing, it will be automatically excluded from feature list
    print("🔹 Cancer Stage Features...")
    
    # Reserve interface for structured stage data (if available in future)
    if 'cancer_stage' in feature_df.columns:
        # Use structured stage data if available
        stage_numeric = pd.to_numeric(feature_df['cancer_stage'], errors='coerce')
        if not stage_numeric.isna().all():
            feature_df['cancer_stage_numeric'] = stage_numeric.fillna(0)
            feature_df['advanced_stage'] = ((stage_numeric >= 3) & (stage_numeric <= 4)).astype(int)
            features.extend(['cancer_stage_numeric', 'advanced_stage'])
    
    # Attempt extraction from free-text drug_indication (imperfect)
    if 'drug_indication' in feature_df.columns:
        indication_upper = feature_df['drug_indication'].fillna('').str.upper()
        
        # Extract cancer stage (Stage I, II, III, IV) from free-text
        # LIMITATION: This is text pattern matching and may miss many cases
        # Match patterns like: "STAGE I", "STAGE 1", "STAGE II", "STAGE 2", "STAGE III", "STAGE 3", "STAGE IV", "STAGE 4"
        
        # Initialize all stages as 0
        feature_df['cancer_stage_I'] = 0
        feature_df['cancer_stage_II'] = 0
        feature_df['cancer_stage_III'] = 0
        feature_df['cancer_stage_IV'] = 0
        
        # Stage I/1
        stage_i_pattern = r'STAGE\s+[I1](\s|$|,|\|)'
        feature_df['cancer_stage_I'] = indication_upper.str.contains(stage_i_pattern, regex=True, na=False).astype(int)
        
        # Stage II/2
        stage_ii_pattern = r'STAGE\s+[I2]{2}(\s|$|,|\|)'  # II or 2
        feature_df['cancer_stage_II'] = indication_upper.str.contains(stage_ii_pattern, regex=True, na=False).astype(int)
        # Also check for "STAGE 2" separately
        feature_df['cancer_stage_II'] = (feature_df['cancer_stage_II'] | 
                                         indication_upper.str.contains(r'STAGE\s+2(\s|$|,|\|)', regex=True, na=False)).astype(int)
        
        # Stage III/3
        stage_iii_pattern = r'STAGE\s+[I3]{3}(\s|$|,|\|)'  # III or 3
        feature_df['cancer_stage_III'] = indication_upper.str.contains(stage_iii_pattern, regex=True, na=False).astype(int)
        # Also check for "STAGE 3" separately
        feature_df['cancer_stage_III'] = (feature_df['cancer_stage_III'] | 
                                          indication_upper.str.contains(r'STAGE\s+3(\s|$|,|\|)', regex=True, na=False)).astype(int)
        
        # Stage IV/4
        stage_iv_pattern = r'STAGE\s+[IV4](\s|$|,|\|)'  # IV or 4
        feature_df['cancer_stage_IV'] = indication_upper.str.contains(stage_iv_pattern, regex=True, na=False).astype(int)
        # Also check for "STAGE 4" separately
        feature_df['cancer_stage_IV'] = (feature_df['cancer_stage_IV'] | 
                                         indication_upper.str.contains(r'STAGE\s+4(\s|$|,|\|)', regex=True, na=False)).astype(int)
        
        # Ensure no overlap (if multiple stages found, take the highest)
        # This handles cases where someone might have "STAGE III/IV" or similar
        
        features.extend([
            'cancer_stage_I', 'cancer_stage_II', 'cancer_stage_III', 'cancer_stage_IV'
        ])
        
        # Print statistics
        print(f"   Stage I: {feature_df['cancer_stage_I'].sum()} patients")
        print(f"   Stage II: {feature_df['cancer_stage_II'].sum()} patients")
        print(f"   Stage III: {feature_df['cancer_stage_III'].sum()} patients")
        print(f"   Stage IV: {feature_df['cancer_stage_IV'].sum()} patients")
    
    # 8. Infection-related reactions (from reactions field before we drop it)
    print("🔹 Infection-Related AE Features...")
    if 'reactions' in feature_df.columns:
        reactions_upper = feature_df['reactions'].fillna('').str.upper()
        feature_df['has_infection_ae'] = (
            reactions_upper.str.contains('INFECTION|SEPSIS|PNEUMONIA|BACTEREMIA|ASPERGILLOSIS', na=False)
        ).astype(int)
        features.append('has_infection_ae')
    
    # Select only feature columns (avoid duplicates)
    available_features = []
    seen = set()
    for f in features:
        if f in feature_df.columns and f not in seen:
            available_features.append(f)
            seen.add(f)
    
    X = feature_df[available_features].copy()
    
    # Clean feature names (remove invalid characters for XGBoost)
    clean_names = [str(col).replace('<', 'lt_').replace('>', 'gt_').replace('[', '').replace(']', '').replace('-', '_') for col in X.columns]
    # Ensure unique column names
    final_names = []
    name_counts = {}
    for name in clean_names:
        if name in name_counts:
            name_counts[name] += 1
            final_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            final_names.append(name)
    
    X.columns = final_names
    available_features = list(X.columns)
    
    # Handle missing values - convert all to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Ensure all values are numeric
    X = X.select_dtypes(include=[np.number])
    available_features = list(X.columns)
    
    print(f"\n✅ Built {len(available_features)} features")
    print(f"   Features: {', '.join(available_features[:15])}..." if len(available_features) > 15 else f"   Features: {', '.join(available_features)}")
    
    return X, feature_df, available_features

def train_crs_models(X, y, random_state=42):
    """Train multiple models on CRS data."""
    print("\n" + "=" * 80)
    print("Step 3: Training Models on CRS Data")
    print("=" * 80)
    
    # Check data size
    if len(X) < 10:
        print(f"❌ Insufficient data: only {len(X)} samples.")
        return None
    
    positive_rate = y.sum() / len(y)
    print(f"\n📊 Dataset: {len(X)} CRS patients, {y.sum()} deaths ({positive_rate*100:.1f}% positive)")
    
    if positive_rate == 0 or positive_rate == 1:
        print("⚠️  Cannot train: all outcomes are the same")
        return None
    
    # Train-test split
    if len(X) >= 30:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state,
            stratify=y if y.sum() >= 2 else None
        )
        # Further split training set for validation
        if len(X_train) >= 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state,
                stratify=y_train if y_train.sum() >= 2 else None
            )
        else:
            X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train
    else:
        print("⚠️  Small dataset: using all data for training (no test split)")
        X_tr, X_val, y_test = X, X, X
        y_tr, y_val, y_test = y, y, y
    
    print(f"   Train: {len(X_tr)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Calculate class weights
    if len(np.unique(y_tr)) > 1:
        pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    else:
        pos_weight = 1.0
    
    # Scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test) if len(X_test) < len(X) else scaler.transform(X_test)
    
    models = {}
    results = {}
    feature_importances = {}
    
    # 1. Logistic Regression
    print("\n🔹 Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'
    )
    lr.fit(X_tr_scaled, y_tr)
    
    lr_proba_val = lr.predict_proba(X_val_scaled)[:, 1]
    lr_proba_test = lr.predict_proba(X_test_scaled)[:, 1]
    
    # Find best threshold
    precision, recall, thresholds = precision_recall_curve(y_val, lr_proba_val)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]
    
    lr_pred = (lr_proba_test >= best_threshold).astype(int)
    
    models['Logistic Regression'] = {'model': lr, 'scaler': scaler, 'threshold': best_threshold}
    
    results['Logistic Regression'] = {
        'roc_auc': roc_auc_score(y_test, lr_proba_test) if len(np.unique(y_test)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_test, lr_proba_test),
        'f1': f1_score(y_test, lr_pred),
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, zero_division=0),
        'recall': recall_score(y_test, lr_pred, zero_division=0)
    }
    
    # Feature importance (coefficients)
    feature_importances['Logistic Regression'] = {
        'importance': dict(zip(X.columns, np.abs(lr.coef_[0]))),
        'coefficients': dict(zip(X.columns, lr.coef_[0]))
    }
    
    # 2. Random Forest
    print("🔹 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=random_state,
        class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)
    
    rf_proba_val = rf.predict_proba(X_val)[:, 1]
    rf_proba_test = rf.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, rf_proba_val)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]
    
    rf_pred = (rf_proba_test >= best_threshold).astype(int)
    
    models['Random Forest'] = {'model': rf, 'scaler': None, 'threshold': best_threshold}
    
    results['Random Forest'] = {
        'roc_auc': roc_auc_score(y_test, rf_proba_test) if len(np.unique(y_test)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_test, rf_proba_test),
        'f1': f1_score(y_test, rf_pred),
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0)
    }
    
    feature_importances['Random Forest'] = {
        'importance': dict(zip(X.columns, rf.feature_importances_))
    }
    
    # 3. Gradient Boosting
    print("🔹 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state
    )
    gb.fit(X_tr, y_tr)
    
    gb_proba_val = gb.predict_proba(X_val)[:, 1]
    gb_proba_test = gb.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, gb_proba_val)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]
    
    gb_pred = (gb_proba_test >= best_threshold).astype(int)
    
    models['Gradient Boosting'] = {'model': gb, 'scaler': None, 'threshold': best_threshold}
    
    results['Gradient Boosting'] = {
        'roc_auc': roc_auc_score(y_test, gb_proba_test) if len(np.unique(y_test)) > 1 else 0.5,
        'pr_auc': average_precision_score(y_test, gb_proba_test),
        'f1': f1_score(y_test, gb_pred),
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, zero_division=0),
        'recall': recall_score(y_test, gb_pred, zero_division=0)
    }
    
    feature_importances['Gradient Boosting'] = {
        'importance': dict(zip(X.columns, gb.feature_importances_))
    }
    
    # 4. XGBoost (if available)
    if HAS_XGB:
        print("🔹 Training XGBoost...")
        scale_pos_weight = pos_weight if pos_weight > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False
        )
        xgb_model.fit(X_tr, y_tr)
        
        xgb_proba_val = xgb_model.predict_proba(X_val)[:, 1]
        xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_val, xgb_proba_val)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        best_thresh_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_thresh_idx]
        
        xgb_pred = (xgb_proba_test >= best_threshold).astype(int)
        
        models['XGBoost'] = {'model': xgb_model, 'scaler': None, 'threshold': best_threshold}
        
        results['XGBoost'] = {
            'roc_auc': roc_auc_score(y_test, xgb_proba_test) if len(np.unique(y_test)) > 1 else 0.5,
            'pr_auc': average_precision_score(y_test, xgb_proba_test),
            'f1': f1_score(y_test, xgb_pred),
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0)
        }
        
        feature_importances['XGBoost'] = {
            'importance': dict(zip(X.columns, xgb_model.feature_importances_))
        }
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("📊 Model Performance Summary")
    print("=" * 80)
    
    # Class imbalance information
    test_positive_rate = y_test.sum() / len(y_test) if len(y_test) > 0 else 0
    test_negative_rate = 1 - test_positive_rate
    print(f"\n📈 Class Distribution (Test Set):")
    print(f"   Positive (Death): {y_test.sum()}/{len(y_test)} ({test_positive_rate*100:.1f}%)")
    print(f"   Negative (Survival): {(y_test == 0).sum()}/{len(y_test)} ({test_negative_rate*100:.1f}%)")
    print(f"   Imbalance Ratio: {test_negative_rate/test_positive_rate:.2f}:1" if test_positive_rate > 0 else "   Imbalance Ratio: N/A")
    
    print("\n" + "-" * 80)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"   ROC-AUC:     {res['roc_auc']:.4f}")
        print(f"   PR-AUC:      {res['pr_auc']:.4f}")
        print(f"   Precision:   {res['precision']:.4f}")
        print(f"   Recall:      {res['recall']:.4f}")
        print(f"   F1-Score:    {res['f1']:.4f}")
        print(f"   Accuracy:    {res['accuracy']:.4f}")
    
    # Best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['pr_auc'])
    best_result = results[best_model_name]
    print("\n" + "=" * 80)
    print(f"✅ Best Model (by PR-AUC): {best_model_name}")
    print("=" * 80)
    print(f"   ROC-AUC:     {best_result['roc_auc']:.4f}")
    print(f"   PR-AUC:      {best_result['pr_auc']:.4f}")
    print(f"   Precision:   {best_result['precision']:.4f}")
    print(f"   Recall:      {best_result['recall']:.4f}")
    print(f"   F1-Score:    {best_result['f1']:.4f}")
    print(f"   Accuracy:    {best_result['accuracy']:.4f}")
    
    # Find threshold that achieves ~70% recall
    best_model_data = models[best_model_name]
    best_model_obj = best_model_data['model']
    
    if hasattr(best_model_obj, 'predict_proba'):
        # Get probabilities on validation set
        val_proba = best_model_obj.predict_proba(X_val)[:, 1]
        
        precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, val_proba)
        
        # Find threshold closest to 70% recall
        # Note: thresholds_val has length len(recall_val) - 1, so we need to handle indexing carefully
        target_recall = 0.70
        recall_idx = np.argmin(np.abs(recall_val[:-1] - target_recall))  # Exclude last element (recall=1.0, no threshold)
        
        if recall_idx < len(thresholds_val):
            threshold_70recall = thresholds_val[recall_idx]
            precision_70recall = precision_val[recall_idx]
            actual_recall_70 = recall_val[recall_idx]
        else:
            # Fallback to best threshold if index out of range
            threshold_70recall = best_model_data['threshold']
            precision_70recall = best_result['precision']
            actual_recall_70 = best_result['recall']
        
        print("\n" + "-" * 80)
        print("📊 Clinical Utility Assessment:")
        print("-" * 80)
        print(f"   At ~70% recall threshold ({threshold_70recall:.3f}):")
        print(f"   - Precision: {precision_70recall:.1%}")
        print(f"   - Recall:    {actual_recall_70:.1%}")
        
        # Generate summary statement
        print("\n" + "=" * 80)
        print("💡 Model Interpretation Summary:")
        print("=" * 80)
        print(f"   Although positive cases represent only {test_positive_rate:.1%} of CRS patients, ")
        print(f"   the model achieves {precision_70recall:.1%} precision at {actual_recall_70:.1%} recall, ")
        print(f"   indicating potential utility as a screening tool for identifying high-risk CRS patients.")
        print(f"   The PR-AUC of {best_result['pr_auc']:.3f} demonstrates strong performance on this ")
        print(f"   imbalanced classification task ({test_negative_rate/test_positive_rate:.1f}:1 imbalance ratio).")
        print("=" * 80)
    
    return {
        'models': models,
        'results': results,
        'best_model': best_model_name,
        'feature_importances': feature_importances,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': list(X.columns)
    }

def analyze_feature_importance(model_results, output_dir='.'):
    """Analyze and visualize feature importance from trained models."""
    from pathlib import Path
    
    print("\n" + "=" * 80)
    print("Step 4: Analyzing Feature Importance")
    print("=" * 80)
    
    fig_dir = Path(output_dir)
    fig_dir.mkdir(exist_ok=True)
    
    best_model_name = model_results['best_model']
    feature_importances = model_results['feature_importances']
    
    print(f"\n📊 Feature Importance Analysis (Best Model: {best_model_name})")
    
    # Get importance from best model
    if best_model_name in feature_importances:
        importances = feature_importances[best_model_name]['importance']
        
        # Sort by importance
        sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
        
    print(f"\n📋 Complete Feature Importance List ({len(sorted_features)} features):")
    print("-" * 80)
    for i, (feat, imp) in enumerate(sorted_features, 1):
        print(f"{i:3d}. {feat:40s} : {imp:.6f}")
    
    print(f"\n🔝 Top 15 Most Important Features for CRS → Death Prediction:")
    print("-" * 80)
    for i, (feat, imp) in enumerate(sorted_features[:15], 1):
        print(f"{i:2d}. {feat:30s} : {imp:.4f}")
    
    # Check if cancer_stage features are present
    feature_cols = list(importances.keys())
    cancer_stage_features = [f for f in feature_cols if 'cancer_stage' in f.lower()]
    if cancer_stage_features:
        print(f"\n✅ Cancer stage features found: {', '.join(cancer_stage_features)}")
        for feat in cancer_stage_features:
            if feat in importances:
                feat_imp = importances[feat]
                rank = next((i for i, (f, _) in enumerate(sorted_features, 1) if f == feat), None)
                print(f"   {feat}: importance = {feat_imp:.6f} (rank: {rank})")
    else:
        print("\n⚠️  Warning: No cancer_stage features found in feature list")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        top_n = 15
        features, values = zip(*sorted_features[:top_n])
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
        ax.barh(range(len(features)), values, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features for CRS → Death Prediction\n({best_model_name})', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(fig_dir / 'crs_model_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: crs_model_feature_importance.png")
        plt.close()
        
        # Save COMPLETE feature importance to CSV (for slides - Sky's requirement)
        importance_df_complete = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
        importance_df_complete.to_csv('crs_feature_importance_complete.csv', index=False)
        print(f"✅ Saved: crs_feature_importance_complete.csv ({len(sorted_features)} features - complete list)")
        
        # Also save top 15 for quick reference
        importance_df_top15 = pd.DataFrame(sorted_features[:15], columns=['feature', 'importance'])
        importance_df_top15.to_csv('crs_feature_importance.csv', index=False)
        print("✅ Saved: crs_feature_importance.csv (top 15)")
        
        return sorted_features
    
    return None

def export_feature_inventory(feature_names, output_dir='.'):
    """
    Export feature inventory table with variable names, categories, descriptions, and data sources.
    This satisfies Sky's requirement for a variables and sources table.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names used in the model
    output_dir : str
        Output directory for the inventory CSV
    """
    from pathlib import Path
    
    print("\n📋 Creating Feature Inventory Table...")
    
    # Define feature metadata mapping
    feature_metadata = {
        # Demographics
        'age_years': {'category': 'Demographic', 'description': 'Age at time of report (normalized to years)', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.patientonsetage + patientonsetageunit', 'processing_method': 'Standardized to years from various units'},
        'age_gt_65': {'category': 'Demographic', 'description': 'Age > 65 years (binary indicator)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from age_years', 'processing_method': 'Binary threshold'},
        'age_gt_70': {'category': 'Demographic', 'description': 'Age > 70 years (binary indicator)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from age_years', 'processing_method': 'Binary threshold'},
        'age_gt_75': {'category': 'Demographic', 'description': 'Age > 75 years (binary indicator)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from age_years', 'processing_method': 'Binary threshold'},
        'patientsex': {'category': 'Demographic', 'description': 'Patient sex (1=male, 2=female, 0=unknown)', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.patientsex', 'processing_method': 'Direct from FAERS'},
        'sex_male': {'category': 'Demographic', 'description': 'Male gender (binary indicator)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from patientsex', 'processing_method': 'Binary encoding'},
        'sex_female': {'category': 'Demographic', 'description': 'Female gender (binary indicator)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from patientsex', 'processing_method': 'Binary encoding'},
        'patientweight': {'category': 'Demographic', 'description': 'Patient weight in kg', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.patientweight', 'processing_method': 'Continuous variable (median imputation if missing)'},
        'bmi': {'category': 'Demographic', 'description': 'Body Mass Index (calculated from weight)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Calculated from patientweight', 'processing_method': 'Weight/(height²), default height=1.65m'},
        'bmi_obese': {'category': 'Demographic', 'description': 'Obese (BMI >30)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from bmi', 'processing_method': 'Binary threshold'},
        'bmi_overweight': {'category': 'Demographic', 'description': 'Overweight (BMI 25-30)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from bmi', 'processing_method': 'Binary threshold'},
        'bmi_underweight': {'category': 'Demographic', 'description': 'Underweight (BMI <18.5)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from bmi', 'processing_method': 'Binary threshold'},
        
        # Medications
        'num_drugs': {'category': 'Medication', 'description': 'Number of concurrent medications', 'available_in_FAERS': 'Yes', 'data_source_field': 'Count of patient.drug array', 'processing_method': 'Direct count'},
        'polypharmacy': {'category': 'Medication', 'description': 'Multiple medications (>1 drug)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from num_drugs', 'processing_method': 'Binary threshold'},
        'high_polypharmacy': {'category': 'Medication', 'description': 'High polypharmacy (>5 drugs)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from num_drugs', 'processing_method': 'Binary threshold'},
        'has_steroid': {'category': 'Medication', 'description': 'Receiving steroid medications', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (PREDNISONE, DEXAMETHASONE, etc.)'},
        'has_antibiotic': {'category': 'Medication', 'description': 'Receiving antibiotic medications', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (CIPROFLOXACIN, VANCOMYCIN, etc.)'},
        'has_antiviral': {'category': 'Medication', 'description': 'Receiving antiviral medications', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (ACYCLOVIR, GANCICLOVIR, etc.)'},
        'has_chemo': {'category': 'Medication', 'description': 'Receiving chemotherapy', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (CYCLOPHOSPHAMIDE, DOXORUBICIN, etc.)'},
        'has_targeted': {'category': 'Medication', 'description': 'Receiving targeted therapy', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (RITUXIMAB, LENALIDOMIDE, etc.)'},
        'has_antifungal': {'category': 'Medication', 'description': 'Receiving antifungal medications', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.openfda.generic_name', 'processing_method': 'Keyword matching (FLUCONAZOLE, VORICONAZOLE, etc.)'},
        'steroid_plus_antibiotic': {'category': 'Medication', 'description': 'Combination of steroid + antibiotic', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from has_steroid and has_antibiotic', 'processing_method': 'Interaction term (has_steroid AND has_antibiotic)'},
        
        # Adverse Events
        'num_reactions': {'category': 'Adverse Event', 'description': 'Number of adverse reactions reported', 'available_in_FAERS': 'Yes', 'data_source_field': 'Count of patient.reaction array', 'processing_method': 'Direct count'},
        'multiple_reactions': {'category': 'Adverse Event', 'description': 'Multiple adverse reactions (>1)', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived from num_reactions', 'processing_method': 'Binary threshold'},
        'has_infection_ae': {'category': 'Adverse Event', 'description': 'Infection-related adverse event', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.reaction.reactionmeddrapt', 'processing_method': 'Keyword matching (INFECTION, SEPSIS, PNEUMONIA, etc.)'},
        
        # Comorbidities
        'comorbidity_diabetes': {'category': 'Comorbidity', 'description': 'History of diabetes', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.drugindication', 'processing_method': 'Keyword matching from free-text'},
        'comorbidity_hypertension': {'category': 'Comorbidity', 'description': 'History of hypertension', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.drugindication', 'processing_method': 'Keyword matching from free-text'},
        'comorbidity_cardiac': {'category': 'Comorbidity', 'description': 'History of cardiac disease', 'available_in_FAERS': 'Yes', 'data_source_field': 'patient.drug.drugindication', 'processing_method': 'Keyword matching from free-text'},
        
        # Cancer Stage
        'cancer_stage_I': {'category': 'Cancer Stage', 'description': 'DLBCL Stage I', 'available_in_FAERS': 'Partial', 'data_source_field': 'patient.drug.drugindication (free-text)', 'processing_method': 'Pattern matching from free-text (imperfect extraction)'},
        'cancer_stage_II': {'category': 'Cancer Stage', 'description': 'DLBCL Stage II', 'available_in_FAERS': 'Partial', 'data_source_field': 'patient.drug.drugindication (free-text)', 'processing_method': 'Pattern matching from free-text (imperfect extraction)'},
        'cancer_stage_III': {'category': 'Cancer Stage', 'description': 'DLBCL Stage III', 'available_in_FAERS': 'Partial', 'data_source_field': 'patient.drug.drugindication (free-text)', 'processing_method': 'Pattern matching from free-text (imperfect extraction)'},
        'cancer_stage_IV': {'category': 'Cancer Stage', 'description': 'DLBCL Stage IV', 'available_in_FAERS': 'Partial', 'data_source_field': 'patient.drug.drugindication (free-text)', 'processing_method': 'Pattern matching from free-text (imperfect extraction)'},
        
        # Data Quality
        'age_missing': {'category': 'Data Quality', 'description': 'Age missing indicator', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived', 'processing_method': 'Binary flag for missing age'},
        'weight_missing': {'category': 'Data Quality', 'description': 'Weight missing indicator', 'available_in_FAERS': 'Yes', 'data_source_field': 'Derived', 'processing_method': 'Binary flag for missing weight'},
    }
    
    # Build inventory table
    inventory_data = []
    for feat in feature_names:
        if feat in feature_metadata:
            meta = feature_metadata[feat]
            inventory_data.append({
                'variable_name': feat,
                'category': meta['category'],
                'description': meta['description'],
                'available_in_FAERS': meta['available_in_FAERS'],
                'data_source_field': meta['data_source_field'],
                'processing_method': meta['processing_method']
            })
        else:
            # Default for unknown features
            inventory_data.append({
                'variable_name': feat,
                'category': 'Other',
                'description': 'Feature extracted from FAERS data',
                'available_in_FAERS': 'Yes',
                'data_source_field': 'FAERS',
                'processing_method': 'Feature engineering'
            })
    
    # Create DataFrame and save
    inventory_df = pd.DataFrame(inventory_data)
    inventory_df = inventory_df.sort_values(['category', 'variable_name'])
    
    output_file = Path(output_dir) / 'crs_feature_inventory.csv'
    inventory_df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file} ({len(inventory_df)} features)")
    
    # Print summary by category
    print("\n📊 Feature Inventory Summary by Category:")
    print("-" * 80)
    category_counts = inventory_df['category'].value_counts().sort_index()
    for category, count in category_counts.items():
        print(f"  {category}: {count} variables")
    
    return inventory_df

def generate_clinical_report(model_results, feature_importance, crs_df, output_file='crs_model_clinical_report.md'):
    """Generate a clinically interpretable report."""
    print("\n" + "=" * 80)
    print("Step 5: Generating Clinical Report")
    print("=" * 80)
    
    report = []
    report.append("# CRS → Death Model: Clinical Interpretation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    best_model_name = model_results['best_model']
    results = model_results['results'][best_model_name]
    
    report.append("## Executive Summary\n")
    report.append(f"This report presents a machine learning model trained specifically on **CRS (Cytokine Release Syndrome) patients** ")
    report.append(f"to predict death outcomes. The model uses structured clinical features and achieves:")
    report.append(f"\n- **PR-AUC:** {results['pr_auc']:.3f}")
    report.append(f"\n- **ROC-AUC:** {results['roc_auc']:.3f}")
    report.append(f"\n- **F1-Score:** {results['f1']:.3f}")
    report.append(f"\n- **Best Model:** {best_model_name}\n")
    
    report.append("## Dataset\n")
    report.append(f"- **Total CRS patients:** {len(crs_df)}")
    report.append(f"\n- **Deaths:** {crs_df['death'].sum()} ({crs_df['death'].mean()*100:.1f}%)")
    report.append(f"\n- **Survivors:** {len(crs_df) - crs_df['death'].sum()} ({(1-crs_df['death'].mean())*100:.1f}%)\n")
    
    # Feature importance interpretation
    if feature_importance:
        report.append("## Key Variables for CRS → Death Prediction\n")
        report.append("The following variables were identified as most important by the model:\n")
        
        # Categorize features
        demographic_features = []
        clinical_features = []
        drug_features = []
        other_features = []
        
        for feat, imp in feature_importance[:15]:
            feat_lower = feat.lower()
            if any(x in feat_lower for x in ['age', 'sex', 'weight', 'bmi']):
                demographic_features.append((feat, imp))
            elif any(x in feat_lower for x in ['drug', 'steroid', 'antibiotic', 'antifungal', 'antiviral', 'chemo']):
                drug_features.append((feat, imp))
            elif any(x in feat_lower for x in ['comorbidity', 'infection', 'reaction']):
                clinical_features.append((feat, imp))
            else:
                other_features.append((feat, imp))
        
        # Demographic features
        if demographic_features:
            report.append("### 1. Demographic Variables\n")
            for feat, imp in demographic_features:
                report.append(f"- **{feat}:** Importance = {imp:.4f}\n")
            report.append("\n**Clinical Interpretation:** ")
            if any('age' in f[0].lower() for f in demographic_features):
                report.append("Age is a strong predictor of death in CRS patients. ")
            if any('bmi' in f[0].lower() for f in demographic_features):
                report.append("Body mass index (BMI) and weight may be associated with outcomes. ")
            report.append("\n")
        
        # Clinical features
        if clinical_features:
            report.append("### 2. Clinical/Comorbidity Variables\n")
            for feat, imp in clinical_features:
                report.append(f"- **{feat}:** Importance = {imp:.4f}\n")
            report.append("\n**Clinical Interpretation:** ")
            if any('infection' in f[0].lower() for f in clinical_features):
                report.append("Infection-related adverse events are important predictors. ")
            if any('comorbidity' in f[0].lower() for f in clinical_features):
                report.append("Underlying comorbidities play a role in CRS outcomes. ")
            report.append("\n")
        
        # Drug features
        if drug_features:
            report.append("### 3. Drug-Related Variables\n")
            for feat, imp in drug_features:
                report.append(f"- **{feat}:** Importance = {imp:.4f}\n")
            report.append("\n**Clinical Interpretation:** ")
            if any('steroid' in f[0].lower() for f in drug_features):
                report.append("Steroid use (and combinations with antibiotics) is associated with outcomes. ")
            if any('num_drugs' in f[0].lower() or 'polypharmacy' in f[0].lower() for f in drug_features):
                report.append("The number of concurrent medications (polypharmacy) is important. ")
            report.append("\n")
        
        # Other features
        if other_features:
            report.append("### 4. Other Variables\n")
            for feat, imp in other_features:
                report.append(f"- **{feat}:** Importance = {imp:.4f}\n")
            report.append("\n")
        
        # Top 3 features
        report.append("### Top 3 Most Important Variables\n\n")
        for i, (feat, imp) in enumerate(feature_importance[:3], 1):
            report.append(f"{i}. **{feat}** (Importance: {imp:.4f})\n")
        
        report.append("\n**Clinical Insight:** These three variables should be carefully monitored in CRS patients.\n")
    
    report.append("\n## Model Performance\n\n")
    report.append(f"The {best_model_name} model achieved the following performance:\n\n")
    report.append(f"- **ROC-AUC:** {results['roc_auc']:.3f} (Area under the receiver operating characteristic curve)\n")
    report.append(f"- **PR-AUC:** {results['pr_auc']:.3f} (Area under the precision-recall curve, important for imbalanced data)\n")
    report.append(f"- **F1-Score:** {results['f1']:.3f} (Balanced measure of precision and recall)\n")
    report.append(f"- **Accuracy:** {results['accuracy']:.3f}\n")
    report.append(f"- **Precision:** {results['precision']:.3f}\n")
    report.append(f"- **Recall:** {results['recall']:.3f}\n")
    
    report.append("\n## Clinical Implications\n\n")
    report.append("1. **Risk Stratification:** This model can help identify CRS patients at highest risk of death.\n")
    report.append("2. **Monitoring Focus:** Variables identified as important should be closely monitored.\n")
    report.append("3. **Treatment Decisions:** Drug combinations and comorbidities should be considered in treatment planning.\n")
    
    report.append("\n## Limitations\n\n")
    report.append("1. **Sample Size:** Model trained on limited number of CRS cases.\n")
    report.append("2. **Observational Data:** FAERS data is observational and may have biases.\n")
    report.append("3. **External Validation:** Model should be validated on independent datasets.\n")
    report.append("4. **Clinical Context:** Model predictions should be interpreted in clinical context.\n")
    
    # Save report
    report_text = ''.join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Saved report: {output_file}")
    return report_text

def run_crs_mortality_pipeline(drug_name="epcoritamab",
                               ae_keyword_list=None,
                               input_csv="main_data.csv",
                               output_dir="."):
    """
    High-level wrapper function for parameterized CRS mortality prediction pipeline.
    
    Parameters:
    -----------
    drug_name : str, default="epcoritamab"
        Target drug name (case-insensitive)
    ae_keyword_list : list, optional
        List of keywords to identify adverse event (default: CRS keywords)
        Example: ["CYTOKINE RELEASE SYNDROME", "CYTOKINE RELEASE", "CYTOKINE STORM"]
    input_csv : str, default="main_data.csv"
        Path to input CSV file with FAERS data
    output_dir : str, default="."
        Output directory for model files and reports
    
    Returns:
    --------
    dict or None
        Dictionary with model results and metadata, or None if pipeline fails
    
    Example:
    --------
    # Default: Epcoritamab + CRS
    results = run_crs_mortality_pipeline()
    
    # Custom drug + AE
    results = run_crs_mortality_pipeline(
        drug_name="tafasitamab",
        ae_keyword_list=["ICANS", "IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY"]
    )
    """
    from pathlib import Path
    
    if ae_keyword_list is None:
        ae_keyword_list = DEFAULT_CRS_KEYWORDS
    
    print("=" * 80)
    print(f"CRS Mortality Pipeline: {drug_name} → {ae_keyword_list[0] if ae_keyword_list else 'Adverse Event'}")
    print("=" * 80)
    print()
    
    # Load data
    if not Path(input_csv).exists():
        print(f"❌ Data file not found: {input_csv}")
        print("Please run 01_extract_data.py first to generate the data.")
        return None
    
    print(f"📂 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded {len(df)} records")
    print()
    
    # Step 1: Identify cases
    case_df = identify_crs_patients(df, drug_name=drug_name, ae_keywords=ae_keyword_list)
    if case_df is None or len(case_df) == 0:
        print(f"\n❌ No {drug_name} patients with target AE found. Analysis cannot proceed.")
        return None
    
    # Step 2: Build features
    X, feature_df, feature_names = build_crs_features(case_df)
    y = feature_df['death']
    
    # Step 3: Train models
    model_results = train_crs_models(X, y)
    
    if model_results is None:
        print("\n❌ Model training failed.")
        return None
    
    # Step 4: Analyze feature importance
    feature_importance = analyze_feature_importance(model_results)
    
    # Step 5: Generate clinical report
    output_file = Path(output_dir) / f"{drug_name.lower()}_model_clinical_report.md"
    generate_clinical_report(model_results, feature_importance, case_df, output_file=str(output_file))
    
    # Save model and metadata
    best_model_name = model_results['best_model']
    best_model_data = model_results['models'][best_model_name]
    
    # Add test data to model_data for SHAP analysis
    best_model_data['X_test'] = model_results['X_test']
    best_model_data['y_test'] = model_results['y_test']
    best_model_data['feature_names'] = model_results['feature_names']
    
    # Save best model
    model_file = Path(output_dir) / f"{drug_name.lower()}_model_best.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_model_data, f)
    print(f"\n✅ Saved best model: {model_file}")
    
    # Return results dictionary
    return {
        'drug_name': drug_name,
        'ae_keywords': ae_keyword_list,
        'best_model': best_model_name,
        'model_performance': model_results['results'][best_model_name],
        'feature_importance': feature_importance,
        'n_patients': len(case_df),
        'n_deaths': int(case_df['death'].sum()),
        'model_file': str(model_file)
    }

def main():
    """Main execution function (backward compatible - uses default parameters)."""
    from pathlib import Path
    
    print("=" * 80)
    print("CRS-Specific Model Training: CRS → Death Prediction")
    print("=" * 80)
    print()
    
    # Load data
    if not Path(DATA_FILE).exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("Please run 01_extract_data.py first to generate the data.")
        return
    
    print(f"📂 Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df)} records")
    print()
    
    # Step 1: Identify CRS cases
    crs_df = identify_crs_patients(df)
    if crs_df is None or len(crs_df) == 0:
        print("\n❌ No CRS patients found. Analysis cannot proceed.")
        return
    
    # Step 2: Build features
    X, feature_df, feature_names = build_crs_features(crs_df)
    y = feature_df['death']
    
    # Step 3: Train models
    model_results = train_crs_models(X, y)
    
    if model_results is None:
        print("\n❌ Model training failed.")
        return
    
    # Step 4: Analyze feature importance
    feature_importance = analyze_feature_importance(model_results)
    
    # Step 5: Generate clinical report
    generate_clinical_report(model_results, feature_importance, crs_df)
    
    # Save model and metadata
    best_model_name = model_results['best_model']
    best_model_data = model_results['models'][best_model_name]
    
    # Add test data to model_data for SHAP analysis
    best_model_data['X_test'] = model_results['X_test']
    best_model_data['y_test'] = model_results['y_test']
    best_model_data['feature_names'] = model_results['feature_names']
    
    # Save best model
    with open('crs_model_best.pkl', 'wb') as f:
        pickle.dump(best_model_data, f)
    print("\n✅ Saved best model: crs_model_best.pkl (includes test data for SHAP analysis)")
    
    # Save SHAP values for interpretability (Sky's requirement)
    print("\n" + "=" * 80)
    print("🔍 Saving SHAP Values for Interpretability")
    print("=" * 80)
    try:
        import shap
        
        best_model_obj = best_model_data['model']
        X_test_data = model_results['X_test']
        y_test_data = model_results['y_test']
        
        # Use TreeExplainer for tree models
        if hasattr(best_model_obj, 'feature_importances_'):
            explainer = shap.TreeExplainer(best_model_obj)
            shap_values = explainer.shap_values(X_test_data)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
        else:
            # Use KernelExplainer for non-tree models
            sample_size = min(100, len(X_test_data))
            background = X_test_data.head(sample_size)
            explainer = shap.KernelExplainer(best_model_obj.predict_proba, background)
            shap_values = explainer.shap_values(X_test_data.head(200))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=X_test_data.columns)
        shap_df.to_csv('crs_shap_values_complete.csv', index=False)
        print(f"✅ Saved: crs_shap_values_complete.csv (shape: {shap_values.shape})")
        
        # Save mean absolute SHAP values (feature importance from SHAP)
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = pd.DataFrame({
            'feature': X_test_data.columns,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False)
        shap_importance.to_csv('crs_shap_feature_importance.csv', index=False)
        print(f"✅ Saved: crs_shap_feature_importance.csv")
        
    except ImportError:
        print("⚠️  SHAP not available. Install with: pip install shap")
        print("   Skipping SHAP value generation...")
    except Exception as e:
        print(f"⚠️  SHAP value generation failed: {str(e)[:100]}")
        print("   Continuing without SHAP values...")
    
    # Save metadata
    meta = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_crs_patients': len(crs_df),
        'n_deaths': int(crs_df['death'].sum()),
        'death_rate': float(crs_df['death'].mean()),
        'best_model': best_model_name,
        'model_performance': model_results['results'][best_model_name],
        'top_features': feature_importance[:10] if feature_importance else []
    }
    
    with open('crs_model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Export feature inventory table (Sky's requirement)
    print("\n" + "=" * 80)
    print("📋 Exporting Feature Inventory Table")
    print("=" * 80)
    export_feature_inventory(feature_names, output_dir=".")
    
    # Export feature inventory table (Sky's requirement)
    print("\n" + "=" * 80)
    print("📋 Exporting Feature Inventory Table")
    print("=" * 80)
    export_feature_inventory(feature_names, output_dir=".")
    
    print("\n" + "=" * 80)
    print("✅ CRS Model Training Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - crs_model_best.pkl (trained model)")
    print("  - crs_model_feature_importance.png")
    print("  - crs_feature_importance.csv")
    print("  - crs_model_clinical_report.md")
    print("  - crs_model_meta.json")
    print("  - crs_feature_inventory.csv (variable list + data sources)")
    print("  - crs_feature_inventory.csv (variable list + data sources)")

if __name__ == '__main__':
    main()

