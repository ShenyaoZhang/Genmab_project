#!/usr/bin/env python3
"""
Evaluate on the true hold-out test set

Strategy:
1. Use the pre-split test set (all 35 drugs)
2. Inspect Epcoritamab performance within the test set
3. Compare performance across drugs
"""

import sys
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("=" * 80)
print("Task 5 - Hold-out Test Evaluation")
print("=" * 80)
print()

# Load test set
print("📂 Loading test data")
print("-" * 80)

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"✅ Test set: {len(X_test)} samples, {len(X_test.columns)} features")
print()

print(f"Target distribution:")
positive = y_test.sum()
negative = len(y_test) - positive
print(f"  Death: {positive} ({positive/len(y_test)*100:.1f}%)")
print(f"  Survival: {negative} ({negative/len(y_test)*100:.1f}%)")
print()

# Load raw data for drug metadata
df_all = pd.read_csv("preprocessed_data.csv")
# Attach drug names to the test set via index alignment
test_indices = X_test.index
test_drugs = df_all.loc[test_indices, 'target_drug'] if 'target_drug' in df_all.columns else None

# Load and evaluate all models
print("=" * 80)
print("🤖 Model evaluation")
print("=" * 80)
print()

model_files = {
    'Gradient Boosting': 'trained_model_gradient_boosting.pkl',
    'XGBoost': 'trained_model_xgboost.pkl',
    'Random Forest': 'trained_model_random_forest.pkl',
    'Logistic Regression': 'trained_model_logistic_regression.pkl'
}

results = {}

for model_name, model_file in model_files.items():
    if not os.path.exists(model_file):
        print(f"⚠️  Skipping {model_name} model file not found")
        continue
    
    print(f"📊 {model_name}")
    print("-" * 80)
    
    try:
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Predict labels/probabilities
        y_pred = model.predict(X_test)
        
        # Gather prediction probabilities
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
        
        # Compute core metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 score: {f1:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print()
        print(f"  Confusion matrix:")
        print(f"                Predicted survival  Predicted death")
        print(f"  Actual survival {tn:6d}    {fp:6d}")
        print(f"  Actual death    {fn:6d}    {tp:6d}")
        print()
        
        # Additional diagnostic metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"  Specificity (true negative rate): {specificity:.4f}")
        print(f"  Negative predictive value: {npv:.4f}")
        print()
        
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        print()
        import traceback
        traceback.print_exc()

# Aggregate results
print("=" * 80)
print("📊 Test results summary")
print("=" * 80)
print()

if results:
    results_df = pd.DataFrame({k: {
        'accuracy': v['accuracy'],
        'precision': v['precision'],
        'recall': v['recall'],
        'f1': v['f1'],
        'roc_auc': v['roc_auc']
    } for k, v in results.items()}).T
    
    results_df = results_df.round(4)
    results_df = results_df.sort_values('f1', ascending=False)
    
    print(results_df.to_string())
    print()
    
    best_model = results_df['f1'].idxmax()
    print(f"🏆 Best model: {best_model}")
    print(f"   F1 score: {results_df.loc[best_model, 'f1']:.4f}")
    print(f"   ROC-AUC: {results_df.loc[best_model, 'roc_auc']:.4f}")
    print()
    
    # Persist summary
    results_df.to_csv("test_set_results.csv")
    print("💾 Saved: test_set_results.csv")
    print()

# If drug metadata exists, provide per-drug analysis
if test_drugs is not None and len(test_drugs) > 0:
    print("=" * 80)
    print("📊 Per-drug analysis (Top 10)")
    print("=" * 80)
    print()
    
    # Use predictions from best model
    if results:
        best_pred = results[best_model]['y_pred']
        
        # Assemble analysis DataFrame
        analysis_df = pd.DataFrame({
            'drug': test_drugs,
            'y_true': y_test.values,
            'y_pred': best_pred
        })
        
        # Aggregate by drug
        drug_stats = []
        for drug in analysis_df['drug'].unique():
            drug_data = analysis_df[analysis_df['drug'] == drug]
            
            if len(drug_data) >= 10:  # require at least 10 samples
                correct = (drug_data['y_true'] == drug_data['y_pred']).sum()
                total = len(drug_data)
                death_rate = drug_data['y_true'].mean()
                
                drug_stats.append({
                    'drug': drug,
                    'samples': total,
                    'accuracy': correct / total,
                    'death_rate': death_rate
                })
        
        if drug_stats:
            drug_stats_df = pd.DataFrame(drug_stats)
            drug_stats_df = drug_stats_df.sort_values('samples', ascending=False)
            
            print(f"Model used: {best_model}")
            print()
            print("Top 10 drugs (by sample size):")
            print()
            for idx, row in drug_stats_df.head(10).iterrows():
                print(f"{row['drug']:20s}: "
                      f"{row['samples']:3.0f} samples, "
                      f"accuracy {row['accuracy']:.3f}, "
                      f"death rate {row['death_rate']:.3f}")
            print()
            
            # Save
            drug_stats_df.to_csv("test_by_drug.csv", index=False)
            print("💾 Saved: test_by_drug.csv")
            print()

print("=" * 80)
print("✅ Test complete")
print("=" * 80)
print()

print("🎯 Key findings:")
print(f"  1. Test sample size: {len(X_test)}")
print(f"  2. Test set mortality: {positive/len(y_test)*100:.1f}%")
if results:
    print(f"  3. Best F1 score: {results_df.loc[best_model, 'f1']:.4f}")
    print(f"  4. Best ROC-AUC: {results_df.loc[best_model, 'roc_auc']:.4f}")
print()

print("💡 Notes:")
print("  - This is performance on an independent test set")
print("  - Test set spans multiple drugs for generalisation insight")
print("  - F1 balances precision and recall")
print()


