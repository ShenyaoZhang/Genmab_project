#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 - Step 4/9: Model Training (enhanced)
- Primary metric: PR-AUC
- Threshold selected via validation split on the training data (no tuning on the test set)
- Test set used once for final evaluation only
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import sys
import os
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')


def train_general_model(X_train_path="X_train.csv", y_train_path="y_train.csv",
                        X_test_path="X_test.csv", y_test_path="y_test.csv",
                        output_dir=".", verbose=True):
                            """
    Train general models for severity prediction.

    Parameters:
        -----------
    X_train_path : str
    Path to training features CSV
    y_train_path : str
    Path to training labels CSV
    X_test_path : str
    Path to test features CSV
    y_test_path : str
    Path to test labels CSV
    output_dir : str
    Output directory for models and results
    verbose : bool
    Whether to print progress messages

    Returns:
        --------
    dict : Dictionary with model results and output file paths
    """
    if verbose:
        print("=" * 80)
        print("Task 5 - Step 4/9: Model Training (enhanced)")
        print("=" * 80)
        print()

    # Check required files
    required_files = [X_train_path, y_train_path, X_test_path, y_test_path]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}\nPlease run: python 03_preprocess_data.py")

    if verbose:
        print(" Found all required files\n")

    # Load data
        if verbose:
            print(" Loading training and test data...")
    X_train_full = pd.read_csv(X_train_path)
    y_train_full = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    if verbose:
        print(f"Train set (full): {len(X_train_full)} samples, {X_train_full.shape[1]} features")
        print(f"Test set: {len(X_test)} samples\n")

    # ---------------------------
    # Utility functions (defined inside function to avoid global scope issues)
    # ---------------------------
        def get_probabilities(model, X):
            """
 Return positive-class probabilities (or normalised scores).
 Priority order:
     1. model.predict_proba -> [:, 1]
 2. model.decision_function -> min-max scaled to [0, 1]
 3. Otherwise return None
 """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X).astype(float)
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        return (scores - smin) / (smax - smin)
        return np.full_like(scores, 0.5, dtype=float)
        return None

        def pick_best_threshold_by_f1(y_true, y_score):
            """
 Given predicted probabilities y_score, use the PR curve to choose the threshold
 that maximises F1. Returns best_threshold, best_f1, and (precision, recall).
 """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0, (0.0, 0.0)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx]), float(f1_scores[best_idx]
                                                  ), (float(precision[best_idx]), float(recall[best_idx]))

        def evaluate_at_threshold(y_true, y_score, threshold):
            """
 Compute accuracy/precision/recall/F1 and the confusion matrix at a fixed threshold.
 """
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

    # Continue with training logic inside the function...

    # ---------------------------
    # Train/validation split (for threshold tuning, early stopping, etc.)
    # ---------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    if verbose:
        print(f"Validation split: train {len(X_tr)}, validation {len(X_val)}{Stratified, 20%)\n")

    # ---------------------------
    # Class imbalance stats (for XGBoost/documentation)
    # ---------------------------
        neg_count = int((y_train_full == 0).sum())
        pos_count = int((y_train_full == 1).sum())
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        if verbose:
            print(" Class distribution (full training set)")
    print(f" Negative (0): {neg_count}{neg_count / len(y_train_full) * 100:.1f}%)")
    print(f" Positive (1): {pos_count}{pos_count / len(y_train_full) * 100:.1f}%)")
    print(f" Imbalance ratio: {scale_pos_weight:.2f}:1\n")

    # ---------------------------
    # Model collection
    # ---------------------------
    if verbose:
        print(" Initializing models...")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, random_state=42)
        }
    # ---------------------------
    # Optional XGBoost
    # ---------------------------
        try:
            import xgboost as xgb
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False
    )
    if verbose:
        print(" XGBoost available")
        except Exception:
            if verbose:
                print("WARNING: XGBoost not available")
        if verbose:
            print()
    print(f"Training {len(models)} models:")
    for i, m in enumerate(models.keys(), 1):
            print(f" {i}. {m}")
        print()

    # ---------------------------
    # Training & evaluation
    # ---------------------------
        results = {}
        for i, (name, model) in enumerate(models.items(), 1):
            if verbose:
                print("=" * 80)
        print(f"[{i}/{len(models)}] Training {name}")
        print("=" * 80)

        try:
            # Train the model
    model.fit(X_tr, y_tr)

    # ---- Threshold selection on validation set ----
    val_scores = get_probabilities(model, X_val)
    if val_scores is None:
        if verbose:
            print("WARNING: Model lacks probability/decision function; using default threshold 0.5")
    best_threshold, best_f1_val = 0.5, 0.0
    pr_val, rc_val = 0.0, 0.0
    else:

        best_threshold, best_f1_val, (pr_val, rc_val) = pick_best_threshold_by_f1(y_val, val_scores)

    if verbose:
        print(
            f" Validation best threshold (max F1): best_threshold:.3f} | F1={best_f1_val:.3f}{P={
                pr_val:.3f}, R={
                    rc_val:.3f})")

    # ---- Final evaluation on test set ----
        test_scores = get_probabilities(model, X_test)
        if test_scores is None:
            # No probabilities: use labels
    y_pred = model.predict(X_test)
    pr_auc = 0.0
    try:
            roc_auc = roc_auc_score(y_test, y_pred)
    except Exception:
        roc_auc = 0.5
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        else:

            # Threshold-independent metrics (using probabilities)
    try:
            roc_auc = roc_auc_score(y_test, test_scores)
    except Exception:
        roc_auc = 0.5
        try:
            pr_auc = average_precision_score(y_test, test_scores)
    except Exception:
        pr_auc = 0.0
    # Evaluate at fixed threshold
        acc, prec, rec, f1, cm = evaluate_at_threshold(y_test, test_scores, best_threshold)

        if verbose:
            print(
        f" Test set: PR-AUC={pr_auc:.4f} | ROC-AUC={roc_auc:.4f} | Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")
    print(f" Confusion matrix @{best_threshold:.3f}:")
    print(f" [[TN={cm[0, 0]:4d} FP={cm[0, 1]:4d}]")
    print(f" [FN={cm[1, 0]:4d} TP={cm[1, 1]:4d}]]")

    # Record results (sorted by PR-AUC)
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_threshold
    }

    # Save model
    model_filename = os.path.join(output_dir, f"trained_model_{name.lower().replace(' ', '_')}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

        except Exception as e:
            if verbose:
                print(f" ERROR: Training/evaluation failed: {str(e)[:120]}")
        results[name] = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": 0.5, "pr_auc": 0.0, "best_threshold": 0.5
        }

        if verbose:
            print("\n Model training and evaluation complete\n")

    # ---------------------------
    # Summarize and best model (by PR-AUC)
    # ---------------------------
    results_df = pd.DataFrame(results).T.round(4)
    results_df = results_df.sort_values("pr_auc", ascending=False)
    if verbose:
        print("=" * 80)
        print(" Model performance comparison (sorted by PR-AUC)")
        print("=" * 80)
        print(results_df.to_string())
        print()

        best_model_name = results_df["pr_auc"].idxmax()
        best_accuracy = float(results_df.loc[best_model_name, "accuracy"])
        if verbose:
            print(f" Best model: {best_model_name}{PR-AUC={results_df.loc[best_model_name, 'pr_auc']:.4f})")
    print(f" Acc={best_accuracy:.4f}, F1={results_df.loc[best_model_name, 'f1']:.4f}, "
          f"ROC-AUC={results_df.loc[best_model_name, 'roc_auc']:.4f}, "
          f"Best threshold={results_df.loc[best_model_name, 'best_threshold']:.3f}\n")

    # Save comparison table
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(comparison_file)
    if verbose:
        print(f" Saved: {comparison_file}\n")

    # ---------------------------
    # Probability calibration (optional: isotonic calibration for the best model)
    # - Note: Strictly speaking, calibration should be done on K-fold cross-validation or an independent validation set
    # ---------------------------
        best_model_file = os.path.join(output_dir, f"trained_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        best_model_obj = None
        try:
            with open(best_model_file, "rb") as f:
                best_model_obj = pickle.load(f)
        if hasattr(best_model_obj, "predict_proba"):
            from sklearn.calibration import CalibratedClassifierCV
    if verbose:
        print(f"Calibrating {best_model_name} probabilities (isotonic, cv=3)...")
        calibrator = CalibratedClassifierCV(best_model_obj, method="isotonic", cv=3)
        calibrator.fit(X_train_full, y_train_full)  # Calibrate on full training set
        calibrated_file = os.path.join(output_dir, "trained_model_calibrated.pkl")
        with open(calibrated_file, "wb") as f:
            pickle.dump(calibrator, f)
    if verbose:
        print(f" Saved: {calibrated_file}{calibrated)\n")
        else:

            if verbose:
                print(f"WARNING: {best_model_name} does not support probability outputs; skipping calibration\n")
        except Exception as e:
            if verbose:
                print(f"WARNING: Probability calibration failed: {str(e)[:120]}\n")

    # ---------------------------
    # Training metadata saving
    # ---------------------------
        training_meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "best_model": best_model_name,
            "n_train_full": int(len(X_train_full)),
            "n_test": int(len(X_test)),
            "pos_train_full": int(y_train_full.sum()),
            "pos_test": int(y_test.sum()),
            "pos_rate_train_full": float(y_train_full.mean()),
            "pos_rate_test": float(y_test.mean()),
            "n_features": int(X_train_full.shape[1]),
            "feature_cols": list(X_train_full.columns),
            "metric_primary": "PR-AUC",
            "threshold_source": "validation (F1-max)",
            "scale_pos_weight": float(scale_pos_weight),
            "results": {k: {mk: float(mv) if isinstance(mv, (int, float, np.floating)) else mv
                        for mk, mv in v.items()}
                        for k, v in results.items()}
        }
    meta_file = os.path.join(output_dir, "training_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(training_meta, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f" Saved: {meta_file}\n")

    # ---------------------------
    # Feature Importance Analysis (Complete List)
    # ---------------------------
    if verbose:
        print("=" * 80)
        print(" Feature Importance Analysis (Complete List)")
        print("=" * 80)

    # Get best model for feature importance (if not already loaded)
        if best_model_obj is None:
            try:
                with open(best_model_file, "rb") as f:
                    best_model_obj = pickle.load(f)
        except FileNotFoundError:
            if verbose:
                print(f"WARNING: Best model file not found: {best_model_file}")
        best_model_obj = None

        feature_importance_dict = {}

    # Extract feature importance based on model type
        if best_model_obj and hasattr(best_model_obj, 'feature_importances_'):
            # Tree-based models
    importances = best_model_obj.feature_importances_
    feature_importance_dict = dict(zip(X_train_full.columns, importances))
            elif best_model_obj and hasattr(best_model_obj, 'coef_'):
        # Linear models - use absolute coefficients
        if len(best_model_obj.coef_.shape) > 1:
            coef = best_model_obj.coef_[0]
    else:

        coef = best_model_obj.coef_
    importances = np.abs(coef)
    feature_importance_dict = dict(zip(X_train_full.columns, importances))

    if feature_importance_dict:
        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        if verbose:
            print(f"\n Complete Feature Importance List ({len(sorted_features)} features):")
    print("-" * 80)
    for i, (feat, imp) in enumerate(sorted_features, 1):
            print(f"{i:3d}. {feat:40s} : {imp:.6f}")

    # Save to CSV (for slides)
        importance_file = os.path.join(output_dir, 'feature_importance_complete.csv')
        importance_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])
        importance_df.to_csv(importance_file, index=False)
        if verbose:
            print(f"\n Saved: {importance_file}{len(sorted_features)} features)")

    # Check if cancer_stage features are present
    cancer_stage_features = [f for f in X_train_full.columns if 'cancer_stage' in f.lower()]
    if cancer_stage_features:
        if verbose:
            print(f"\n Cancer stage features found: {', '.join(cancer_stage_features)}")
    for feat in cancer_stage_features:
            if feat in feature_importance_dict:
                print(f" {feat}: importance = {feature_importance_dict[feat]:.6f}")
    else:

        if verbose:
            print("\nWARNING: Warning: No cancer_stage features found in feature list")
        else:

            if verbose:
                print("WARNING: Could not extract feature importance from best model")

    # ---------------------------
    # SHAP Values (if SHAP is available)
    # ---------------------------
        try:
            import shap
    if verbose:
        print("\n" + "=" * 80)
        print(" Generating SHAP Values for Interpretability")
        print("=" * 80)

    # Use TreeExplainer for tree models, KernelExplainer for others
        if hasattr(best_model_obj, 'feature_importances_'):
            explainer = shap.TreeExplainer(best_model_obj)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification (shap_values might be a list)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
        else:

            # Use KernelExplainer for non-tree models (sample for speed)
    sample_size = min(100, len(X_test))
    background = X_test.head(sample_size)
    explainer = shap.KernelExplainer(best_model_obj.predict_proba, background)
    shap_values = explainer.shap_values(X_test.head(200))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Save SHAP values
        shap_file = os.path.join(output_dir, 'shap_values_complete.csv')
        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        shap_df.to_csv(shap_file, index=False)
        if verbose:
            print(f" Saved: {shap_file}{shape: {shap_values.shape})")

    # Save mean absolute SHAP values (feature importance from SHAP)
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = pd.DataFrame({
        'feature': X_test.columns,
        'shap_importance': mean_shap
    }).sort_values('shap_importance', ascending=False)
    shap_importance_file = os.path.join(output_dir, 'shap_feature_importance.csv')
    shap_importance.to_csv(shap_importance_file, index=False)
    if verbose:
        print(f" Saved: {shap_importance_file}")

        except ImportError:
            if verbose:
                print("\nWARNING: SHAP not available. Install with: pip install shap")
        print(" Skipping SHAP value generation...")
        except Exception as e:
            if verbose:
                print(f"\nWARNING: SHAP value generation failed: {str(e)[:100]}")
        print(" Continuing without SHAP values...")

        if verbose:
            print("\n" + "=" * 80)
    print(" Step 4 complete - model training finished (enhanced)")
    print("=" * 80)
    print()

    # Return results dictionary
    return {
        'best_model_name': best_model_name,
        'best_model_file': best_model_file,
        'training_meta': meta_file,
        'model_comparison': comparison_file,
        'feature_importance': importance_file if 'importance_file' in locals() else None,
        'results': results,
        'best_model': best_model_obj
    }


# Main execution (for backward compatibility)
if __name__ == '__main__':
    # Locate train/test files
    required_files = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("ERROR: Error: missing required files:")
    for f in missing:
            print(" -", f)
        print("\nPlease run: python 03_preprocess_data.py")
        sys.exit(1)

    # Run training
        results = train_general_model(
            X_train_path="X_train.csv",
            y_train_path="y_train.csv",
            X_test_path="X_test.csv",
            y_test_path="y_test.csv",
            output_dir=".",
            verbose=True
        )

        print(f"\n Training complete. Best model: {results['best_model_name']}")
