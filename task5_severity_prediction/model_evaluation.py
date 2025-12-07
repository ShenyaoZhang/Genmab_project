#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation with Complete Feature Importance and SHAP Analysis
===================================================================

Generates comprehensive evaluation outputs for slides:
- Complete feature importance list
- SHAP summary plot
- SHAP bar plot
- SHAP waterfall plots
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not available. Install with: pip install shap")

def load_model_and_data(model_file, test_data_file=None):
    """Load trained model and test data."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model and test data
    model = model_data.get('model', model_data)  # Handle different formats
    X_test = None
    y_test = None
    
    if 'X_test' in model_data:
        X_test = model_data['X_test']
        y_test = model_data['y_test']
    elif test_data_file:
        if os.path.exists(test_data_file.replace('X_test', 'X_test.csv')):
            X_test = pd.read_csv(test_data_file.replace('X_test', 'X_test.csv'))
            y_test = pd.read_csv(test_data_file.replace('X_test', 'y_test.csv')).squeeze()
    
    return model, X_test, y_test


def generate_complete_feature_importance(model, feature_names, output_file='feature_importance_complete.csv'):
    """
    Generate complete feature importance list (not just top 15).
    
    Parameters:
    -----------
    model : trained model object
    feature_names : list of feature names
    output_file : str, output CSV file path
    """
    print("\n" + "=" * 80)
    print("📊 Complete Feature Importance List")
    print("=" * 80)
    
    importance_dict = {}
    
    # Extract feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        # Linear models - use absolute coefficients
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]
        importances = np.abs(coef)
        importance_dict = dict(zip(feature_names, importances))
    else:
        print("⚠️  Cannot extract feature importance from this model type")
        return None
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Print complete list
    print(f"\n📋 Complete Feature Importance ({len(sorted_features)} features):")
    print("-" * 80)
    for i, (feat, imp) in enumerate(sorted_features, 1):
        print(f"{i:3d}. {feat:40s} : {imp:.6f}")
    
    # Save to CSV (for slides)
    importance_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
    importance_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file} ({len(sorted_features)} features)")
    
    return importance_df


def generate_shap_plots(model, X_test, y_test, output_dir='.'):
    """
    Generate SHAP summary, bar plot, and waterfall plots.
    
    Parameters:
    -----------
    model : trained model object
    X_test : test features DataFrame
    y_test : test labels Series
    output_dir : str, output directory
    """
    if not HAS_SHAP:
        print("⚠️  SHAP not available. Skipping SHAP plots.")
        return {}
    
    print("\n" + "=" * 80)
    print("🔍 Generating SHAP Explanations")
    print("=" * 80)
    
    output_files = {}
    
    # Initialize SHAP explainer
    print("\n🔹 Initializing SHAP explainer...")
    try:
        # Use TreeExplainer for tree models
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            sample_size = min(100, len(X_test))
            background = X_test.head(sample_size)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Calculate SHAP values
        print("🔹 Calculating SHAP values...")
        if hasattr(model, 'feature_importances_'):
            shap_values = explainer.shap_values(X_test)
        else:
            shap_values = explainer.shap_values(X_test.head(200))  # Limit for speed
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Use positive class
        
        print(f"✅ SHAP values calculated (shape: {shap_values.shape})")
        
    except Exception as e:
        print(f"⚠️  SHAP initialization failed: {str(e)[:100]}")
        return output_files
    
    # 1. SHAP Summary Plot
    print("\n📊 Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.title('SHAP Summary Plot: Feature Impact on Prediction\n(Higher values push toward positive class)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_file = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    output_files['shap_summary'] = summary_file
    print(f"✅ Saved: {summary_file}")
    
    # 2. SHAP Bar Plot
    print("📊 Generating SHAP Bar Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, 
                      plot_type='bar', show=False)
    plt.title('SHAP Feature Importance: Mean |SHAP Value|', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    bar_file = os.path.join(output_dir, 'shap_bar.png')
    plt.savefig(bar_file, dpi=300, bbox_inches='tight')
    plt.close()
    output_files['shap_bar'] = bar_file
    print(f"✅ Saved: {bar_file}")
    
    # 3. SHAP Waterfall Plots (for selected cases)
    print("📊 Generating SHAP Waterfall Plots...")
    
    # Handle expected_value
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = explainer.expected_value
    
    # Death case
    death_indices = np.where(y_test == 1)[0] if y_test is not None else []
    if len(death_indices) > 0:
        try:
            case_idx = death_indices[0]
            plt.figure(figsize=(12, 8))
            explanation = shap.Explanation(
                values=shap_values[case_idx],
                base_values=base_value,
                data=X_test.iloc[case_idx].values,
                feature_names=X_test.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Waterfall: Death Case\n(Patient {case_idx})', 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            waterfall_death_file = os.path.join(output_dir, 'shap_waterfall_death.png')
            plt.savefig(waterfall_death_file, dpi=300, bbox_inches='tight')
            plt.close()
            output_files['shap_waterfall_death'] = waterfall_death_file
            print(f"✅ Saved: {waterfall_death_file}")
        except Exception as e:
            print(f"⚠️  Could not generate waterfall plot for death case: {str(e)[:50]}")
    
    # Survival case
    survival_indices = np.where(y_test == 0)[0] if y_test is not None else []
    if len(survival_indices) > 0:
        try:
            case_idx = survival_indices[0]
            plt.figure(figsize=(12, 8))
            explanation = shap.Explanation(
                values=shap_values[case_idx],
                base_values=base_value,
                data=X_test.iloc[case_idx].values,
                feature_names=X_test.columns.tolist()
            )
            shap.waterfall_plot(explanation, show=False)
            plt.title(f'SHAP Waterfall: Survival Case\n(Patient {case_idx})', 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            waterfall_survival_file = os.path.join(output_dir, 'shap_waterfall_survival.png')
            plt.savefig(waterfall_survival_file, dpi=300, bbox_inches='tight')
            plt.close()
            output_files['shap_waterfall_survival'] = waterfall_survival_file
            print(f"✅ Saved: {waterfall_survival_file}")
        except Exception as e:
            print(f"⚠️  Could not generate waterfall plot for survival case: {str(e)[:50]}")
    
    # Save SHAP values to CSV
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_csv = os.path.join(output_dir, 'shap_values.csv')
    shap_df.to_csv(shap_csv, index=False)
    output_files['shap_values'] = shap_csv
    print(f"✅ Saved: {shap_csv}")
    
    return output_files


def evaluate_model(model_file, test_data_file=None, output_dir='.'):
    """
    Complete model evaluation: feature importance + SHAP plots.
    
    Parameters:
    -----------
    model_file : str, path to saved model file
    test_data_file : str, optional path to test data
    output_dir : str, output directory for results
    """
    print("=" * 80)
    print("Model Evaluation: Complete Feature Importance + SHAP Analysis")
    print("=" * 80)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(model_file, test_data_file)
    
    if X_test is None:
        print("⚠️  Test data not available. Cannot generate SHAP plots.")
        print("   Generating feature importance only...")
        # Try to get feature names from model metadata
        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is None:
            print("❌ Cannot determine feature names")
            return
    
    # Get feature names
    if isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = list(range(X_test.shape[1]))
    
    # 1. Generate complete feature importance
    importance_df = generate_complete_feature_importance(model, feature_names, 
                                                        os.path.join(output_dir, 'feature_importance_complete.csv'))
    
    # 2. Generate SHAP plots (if test data available)
    if X_test is not None and y_test is not None:
        shap_files = generate_shap_plots(model, X_test, y_test, output_dir)
        print(f"\n✅ SHAP analysis complete. Generated {len(shap_files)} files.")
    else:
        print("\n⚠️  Skipping SHAP analysis (no test data)")
    
    print("\n" + "=" * 80)
    print("✅ Model evaluation complete")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model and generate SHAP plots')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model file (.pkl)')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data file (optional)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory (default: current directory)')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test_data, args.output_dir)

