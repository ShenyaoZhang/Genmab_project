#!/usr/bin/env python3
"""
Evaluate trained models on Epcoritamab data
This test performs:
    1. Extract Epcoritamab records as the test set
2. Run predictions using trained models
3. Assess performance on the target drug
"""
from sklearn.impute import SimpleImputer
import sys
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
print("=" * 80)
print("Task 5 - Epcoritamab Focused Evaluation")
print("=" * 80)
print()
# Step 1: load raw data and extract Epcoritamab
print(" Step 1: Extract Epcoritamab test data")
print("-" * 80)
DATA_FILE = "main_data.csv"
if not os.path.exists(DATA_FILE):
    print(f"ERROR: Error: data file not found {DATA_FILE}")
    print(" Please run: python 01_extract_data.py")
    sys.exit(1)
# Load raw data
# Extract Epcoritamab subset
    df_all = pd.read_csv(DATA_FILE)
    print(f" Loaded raw data: {len(df_all)} records")
    df_epc = df_all[df_all['target_drug'].str.lower() == 'epcoritamab'].copy()
    print(f" Extracted Epcoritamab data: {len(df_epc)} records")
    print()
    if len(df_epc) == 0:
        print("ERROR: Error: no Epcoritamab records found")
        sys.exit(1)
# Data overview
print(" Epcoritamab data overview:")
print(f" Total records: {len(df_epc)}")
# Severity distribution
death_count = pd.to_numeric(
    df_epc['seriousnessdeath'],
    errors='coerce').fillna(0).sum()
hosp_count = pd.to_numeric(
    df_epc['seriousnesshospitalization'],
    errors='coerce').fillna(0).sum()
life_count = pd.to_numeric(
    df_epc['seriousnesslifethreatening'],
    errors='coerce').fillna(0).sum()
print(f" Death cases: {int(death_count)} ({int(death_count) / len(df_epc) * 100:.1f}%)")
print(
    f" Hospitalisation cases: {int(hosp_count){len(df_epc)} * 
        100:.1f}%)")
print(
    f" Life-threatening cases: {int(life_count)}{life_count / len(df_epc) * 100:.1f}%)")
print()
# Step 2: preprocess Epcoritamab data (mirror training pipeline)
print(" Step 2: Preprocess Epcoritamab data")
print("-" * 80)
# Convert severity indicators to binary
severity_cols = ['serious', 'seriousnessdeath', 'seriousnesshospitalization',
                 'seriousnesslifethreatening', 'seriousnessdisabling',
                 'seriousnesscongenitalanomali', 'seriousnessother']
for col in severity_cols:
    if col in df_epc.columns:
        df_epc[col] = pd.to_numeric(df_epc[col], errors='coerce').fillna(0)
    df_epc[col] = (df_epc[col] > 0).astype(int)
# Transform patient attributes
if 'patientsex' in df_epc.columns:
    df_epc['patientsex'] = pd.to_numeric(
        df_epc['patientsex'],
        errors='coerce').fillna(0).astype(int)
    if 'patientonsetage' in df_epc.columns:
        df_epc['patientonsetage'] = pd.to_numeric(
            df_epc['patientonsetage'], errors='coerce')
    df_epc.loc[df_epc['patientonsetage'] > 120, 'patientonsetage'] = np.nan
    df_epc.loc[df_epc['patientonsetage'] < 0, 'patientonsetage'] = np.nan
if 'age_years' in df_epc.columns:
    df_epc['age_years'] = pd.to_numeric(df_epc['age_years'], errors='coerce')
    if 'patientweight' in df_epc.columns:
        df_epc['patientweight'] = pd.to_numeric(
            df_epc['patientweight'], errors='coerce')
if 'num_drugs' in df_epc.columns:
    df_epc['num_drugs'] = pd.to_numeric(
        df_epc['num_drugs'],
        errors='coerce').fillna(1).astype(int)
    if 'num_reactions' in df_epc.columns:
        df_epc['num_reactions'] = pd.to_numeric(
            df_epc['num_reactions'],
            errors='coerce').fillna(1).astype(int)
# Feature engineering (match training transformations)
# Age bucket features
if 'patientonsetage' in df_epc.columns:
    df_epc['age_group'] = pd.cut(df_epc['patientonsetage'],
                                 bins=[0, 18, 45, 65, 120],
                                 labels=['0-18', '19-45', '46-65', '66+'],
                                 include_lowest=True)
    age_dummies = pd.get_dummies(df_epc['age_group'], prefix='age')
    df_epc = pd.concat([df_epc, age_dummies], axis=1)
    df_epc['age_missing'] = df_epc['patientonsetage'].isna().astype(int)
# Gender features
    if 'patientsex' in df_epc.columns:
        df_epc['sex_male'] = (df_epc['patientsex'] == 1).astype(int)
    df_epc['sex_female'] = (df_epc['patientsex'] == 2).astype(int)
    df_epc['sex_unknown'] = (df_epc['patientsex'] == 0).astype(int)
# Polypharmacy features
if 'num_drugs' in df_epc.columns:
    df_epc['polypharmacy'] = (df_epc['num_drugs'] > 1).astype(int)
    df_epc['high_polypharmacy'] = (df_epc['num_drugs'] > 5).astype(int)
# Reaction count features
    if 'num_reactions' in df_epc.columns:
        df_epc['multiple_reactions'] = (
            df_epc['num_reactions'] > 1).astype(int)
    df_epc['many_reactions'] = (df_epc['num_reactions'] > 3).astype(int)
print(" Feature engineering complete")
print()
# Step 3: prepare feature matrix
print(" Step 3: Prepare test features")
print("-" * 80)
# Target variable
y_test_epc = df_epc['seriousnessdeath'].copy()
# Feature selection (align with training)
exclude_cols = [
    'safetyreportid', 'receivedate', 'target_drug', 'drugname', 'all_drugs',
    'drug_indication', 'reactions', 'patientonsetageunit', 'age_group',
    'reporter_qualification', 'seriousnessdeath', 'serious',
    'seriousnesshospitalization', 'seriousnesslifethreatening',
    'seriousnessdisabling', 'seriousnesscongenitalanomali', 'seriousnessother'
]
if not os.path.exists("X_train.csv"):
    print("ERROR: Error: missing reference feature file X_train.csv")
    print(" Please run: python 03_preprocess_data.py")
    sys.exit(1)
    reference_features = pd.read_csv("X_train.csv", nrows=0).columns.tolist()
# Ensure reference features exist
    missing_features = [
        col for col in reference_features if col not in df_epc.columns]
    for col in missing_features:
        df_epc[col] = 0
# Filter to reference feature order
feature_cols = [col for col in reference_features if col not in exclude_cols]
X_test_epc = df_epc[feature_cols].copy()
print(f"Test samples: {len(X_test_epc)}")
print(f"Feature count: {len(feature_cols)}")
print()
print(f"Target distribution:")
positive = y_test_epc.sum()
negative = len(y_test_epc) - positive
print(f" Death: {positive}{positive / len(y_test_epc) * 100:.1f}%)")
print(f" Survival: {negative}{negative / len(y_test_epc) * 100:.1f}%)")
print()
# Handle missing values (median imputation)
imputer = SimpleImputer(strategy='median')
X_test_epc_imputed = pd.DataFrame(
    imputer.fit_transform(X_test_epc),
    columns=X_test_epc.columns,
    index=X_test_epc.index
)
print(" Data preprocessing complete")
print()
# Step 4: model evaluation
print("=" * 80)
print(" Step 4: Model evaluation")
print("=" * 80)
print()
# Locate trained models
model_files = {
    'Gradient Boosting': 'trained_model_gradient_boosting.pkl',
    'XGBoost': 'trained_model_xgboost.pkl',
    'Random Forest': 'trained_model_random_forest.pkl',
    'Logistic Regression': 'trained_model_logistic_regression.pkl'
}
results = {}
for model_name, model_file in model_files.items():
    if not os.path.exists(model_file):
        print(f"WARNING: Skipping {model_name}: model file not found")
    continue
    print(f" Testing model: {model_name}")
    print("-" * 80)
    try:
        # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    # Predictions
        y_pred = model.predict(X_test_epc_imputed)
    # Predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_epc_imputed)[:, 1]
    else:

        y_pred_proba = y_pred
    # Metrics
    accuracy = accuracy_score(y_test_epc, y_pred)
    precision = precision_score(y_test_epc, y_pred, zero_division=0)
    recall = recall_score(y_test_epc, y_pred, zero_division=0)
    f1 = f1_score(y_test_epc, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test_epc, y_pred_proba)
    except BaseException:
        roc_auc = 0.5
    # Confusion matrix
    cm = confusion_matrix(y_test_epc, y_pred)
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 score: {f1:.4f}")
    print(f" ROC-AUC: {roc_auc:.4f}")
    print()
    print(f" Confusion matrix:")
    print(f" Predicted survival Predicted death")
    print(f" Actual survival {cm[0][0]:6d} {cm[0][1]:6d}")
    print(f" Actual death {cm[1][0]:6d} {cm[1][1]:6d}")
    print()
    except Exception as e:
        print(f" ERROR: Test failed: {str(e)[:100]}")
        print()
# Step 5: result summary
        print("=" * 80)
        print(" Epcoritamab test summary")
        print("=" * 80)
        print()
        if results:
            results_df = pd.DataFrame(results).T
    results_df = results_df.drop('confusion_matrix', axis=1)
    results_df = results_df.round(4)
    results_df = results_df.sort_values('accuracy', ascending=False)
    print(results_df.to_string())
    print()
    # Best model
    best_model = results_df['f1'].idxmax()
    print(f" Best model on Epcoritamab: {best_model}")
    print(f" Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")
    print(f" F1 score: {results_df.loc[best_model, 'f1']:.4f}")
    print(f" ROC-AUC: {results_df.loc[best_model, 'roc_auc']:.4f}")
    print()
    # Save results
    results_df.to_csv("epcoritamab_test_results.csv")
    print(" Saved: epcoritamab_test_results.csv")
    print()
else:
    print("ERROR: No models successfully tested")
# Step 6: comparison with overall data
print("=" * 80)
print(" Performance comparison: overall vs Epcoritamab")
print("=" * 80)
print()
if os.path.exists("model_comparison.csv"):
    all_drugs_results = pd.read_csv("model_comparison.csv", index_col=0)
    print("Comparison notes:")
    print(" - Overall: trained/tested on all 35 drugs")
    print(" - Epcoritamab: evaluated only on target drug records")
    print()
    for model_name in results.keys():
        if model_name in all_drugs_results.index:
            print(f"{model_name}:")
        print(f" Accuracy (overall) {all_drugs_results.loc[model_name,
                                                           'accuracy']:.4f} | Epcoritamab {results_df.loc[model_name,
                                                                                                          'accuracy']:.4f}")
        print(f" F1 score (overall) {all_drugs_results.loc[model_name,
                                                           'f1']:.4f} | Epcoritamab {results_df.loc[model_name,
                                                                                                    'f1']:.4f}")
        print()
        print("=" * 80)
        print(" Epcoritamab evaluation complete")
        print("=" * 80)
        print()
        print(" Key findings:")
        print(f" 1. Epcoritamab test samples: {len(df_epc)}")
        print(
            f" 2. Mortality rate: float(
                    positive / len(y_test_epc) * 100):.1f}%")
        print(
            f" 3. Best model performance: {results_df.loc[best_model, 'f1']:.4f}{F1 score)")
        print(" Notes:")
        print(" - This represents model behaviour on the target drug cohort")
        print(" - Consider complementing with domain review for clinical deployment")