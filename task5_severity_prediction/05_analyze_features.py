#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 - Step 5/9: Feature Importance Analysis (enhanced)
- Automatically selects the best model by PR-AUC
- Supports feature_importances_, coef_, with permutation importance fallback
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

print("=" * 80)
print("Task 5 - Step 5/9: Feature Importance Analysis (enhanced)")
print("=" * 80)
print()

# Pre-flight checks
# -----------------------------
need_files = ["X_train.csv", "model_comparison.csv"]
missing = [f for f in need_files if not os.path.exists(f)]
if missing:
    print("ERROR: Error: missing required files:")
    for f in missing:
        print(" -", f)
    print("\nPlease run: python 03_preprocess_data.py and python 04_train_models.py")
    sys.exit(1)

# Load feature names and pick the best model
# -----------------------------
X_train = pd.read_csv("X_train.csv")
feature_names = X_train.columns.tolist()
print(f"Number of features: {len(feature_names)}\n")

model_results = pd.read_csv("model_comparison.csv", index_col=0)
if "pr_auc" not in model_results.columns:
    print("ERROR: Error: model_comparison.csv is missing 'pr_auc' column, cannot determine best model")
    sys.exit(1)

    best_model_name = model_results["pr_auc"].idxmax()
    best_model_file = os.path.join(output_dir, f"trained_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        best_model_name.lower().replace(
            ' ', '_')}.pkl"

    print(" Selecting model for analysis")
    print(f" Best model (by PR-AUC): {best_model_name}")
    print(f" Supporting file: {best_model_file}\n")

    if not os.path.exists(best_model_file):
        print(f"ERROR: Error: model file not found {best_model_file}")
        sys.exit(1)

# Load the chosen model
# -----------------------------
print(" Loading model...")
with open(best_model_file, "rb") as f:
    model = pickle.load(f)
    print(" Model loaded successfully\n")

# Prepare evaluation data (prefer the test set for faithful interpretation)
# -----------------------------
if not os.path.exists("X_test.csv") or not os.path.exists("y_test.csv"):
    print("WARNING: Test set not found, falling back to training data for importance analysis (may be optimistic)\n")
    X_imp = X_train.copy()
    y_imp = pd.read_csv("y_train.csv").squeeze()
else:
    X_imp = pd.read_csv("X_test.csv")
    y_imp = pd.read_csv("y_test.csv").squeeze()

# Compute feature importance
# -----------------------------
print("=" * 80)
print(" Feature importance analysis")
print("=" * 80)
print()

use_method = None
if hasattr(model, "feature_importances_"):
    importances = np.array(model.feature_importances_, dtype=float)
    use_method = "feature_importances_"
    print(" Using built-in feature_importances_")
elif hasattr(model, "coef_"):
    coef = getattr(model, "coef_", None)
    if coef is not None and len(coef) > 0:
        importances = np.abs(coef[0]).astype(float)
        use_method = "coef_"
        print(" Using absolute value of coefficients as importance")
        else:

            importances = None
else:
    importances = None

if importances is None:
    print("INFO: Falling back to permutation importance (model agnostic)...")
    # Note: permutation importance calls model.score (defaults to accuracy)
    # For imbalance we can use scoring='average_precision'
    try:
        from sklearn.metrics import make_scorer, average_precision_score
    scoring = "average_precision"
    perm = permutation_importance(
        model, X_imp, y_imp,
        scoring=scoring,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    importances = perm.importances_mean
    use_method = "permutation_importance(AP)"
    print(" Permutation importance computed")
    except Exception as e:
        print(f"ERROR: Permutation importance failed: {str(e)[:120]}")
        sys.exit(1)

# Persist results table
# -----------------------------
        feature_importance_df = pd.DataFrame({
            "feature": feature_names if len(feature_names) == len(importances) else X_imp.columns[:len(importances)],
            "importance": importances
        }).sort_values("importance", ascending=False)

        print("\nTop 20 most important features:\n")
        print(feature_importance_df.head(20).to_string(index=False))
        print()

        feature_importance_df.to_csv("feature_importance.csv", index=False)
        print(" Saved: feature_importance.csv\n")

# Persist visualisation
# -----------------------------
        plt.figure(figsize=(10, 8))
        top_n = min(20, len(feature_importance_df))
        top_feat = feature_importance_df.head(top_n)
        plt.barh(range(top_n), top_feat["importance"])
        plt.yticks(range(top_n), top_feat["feature"])
        plt.xlabel("Importance")
        plt.title(
            f"Top {top_n} Feature Importances ({best_model_name} | {use_method})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        print(" Saved: feature_importance.png\n")

# Lightweight human-readable Top-5 interpretation
# -----------------------------
        print("=" * 80)
        print(" Top-5 feature interpretation (heuristic)")
        print("=" * 80)
        print()

        top5 = feature_importance_df.head(5)
        for i, row in enumerate(top5.itertuples(index=False), 1):
            fname = str(row.feature)
    val = float(row.importance)
    print(f"{i}. {fname}")
    print(f" Importance: {val:.4f}")
    fl = fname.lower()
    if "age" in fl:
        print(" Note: Patient age features - risk typically increases with age")
                elif "sex" in fl or "gender" in fl:
                    print(" Note: Gender features - risk profiles can differ by sex")
            elif "drug" in fl or "medic" in fl:
                print(" Note: Medication/comorbidity features - polypharmacy and specific classes may drive severity")
                elif "reaction" in fl or "event" in fl:
                    print(" Note: Adverse reaction features - reaction count/type may signal more severe cases")
            elif "weight" in fl or "bmi" in fl:
                print(" Note: Weight/exposure features - dose-weight balance or metabolism can influence risk")
        print()

        print("=" * 80)
        print(" Step 5 complete - feature analysis finished (enhanced)")
        print("=" * 80)
        print()
