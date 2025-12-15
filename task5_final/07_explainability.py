#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 - Step 7/9: Explainability Analysis (enhanced)
- Best model chosen dynamically via PR-AUC from model_comparison.csv
- SHAP handles list/ndarray outputs; prefer TreeExplainer with fallback to shap.Explainer
- LIME falls back to decision_function with normalised pseudo-probabilities
- y_test label column auto-detected (serious / seriousnessdeath / first column)
- explain_meta.json records version, seeds, sampling strategy for reproducibility
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

print("=" * 80)
print("Task 5 - Step 7/9: Explainability Analysis (enhanced)")
print("=" * 80)
print()

# -----------------------------
# Utility constants and helpers
# -----------------------------
RANDOM_STATE = 42
MAX_SAMPLES_SHAP = 500  # sample for SHAP speed
np_random = np.random.RandomState(RANDOM_STATE)

def load_y_auto(path="y_test.csv") -> pd.Series:
    """Auto-detect y labels:
    - Prefer serious / serious_flag / is_serious
    - Next seriousnessdeath / death / is_death
    - Otherwise first column
    - Treat non-empty non-zero as 1, else 0
    """
    ydf = pd.read_csv(path)
    for col in ydf.columns:
        lc = col.lower()
        if lc in ("serious", "serious_flag", "is_serious"):
            target = col
            break
    else:
        target = None
        for col in ydf.columns:
            lc = col.lower()
            if lc in ("seriousnessdeath", "death", "is_death"):
                target = col
                break
        if target is None:
            target = ydf.columns[0]
    y = pd.to_numeric(ydf[target], errors="coerce").fillna(0)
    y = (y.astype(float) > 0).astype(int)
    return y

def get_probabilities(model, X):
    """Return positive probabilities or normalised scores:
    - predict_proba -> proba[:,1]
    - decision_function -> min-max normalised to [0,1]
    - Otherwise None
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):  # some models return list
            proba = proba[1]
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X).astype(float)
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return (scores - smin) / (smax - smin)
        return np.full_like(scores, 0.5, dtype=float)
    return None

def ensure_dir_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Required file checks
# -----------------------------
needed = ["X_train.csv", "X_test.csv", "y_test.csv", "model_comparison.csv"]
missing = [f for f in needed if not os.path.exists(f)]
if missing:
    print("❌ Error: missing required files:")
    for f in missing:
        print("  -", f)
    print("\nPlease run: python 03_preprocess_data.py and python 04_train_models.py")
    sys.exit(1)

print("🔍 Input files ready")
print("  ✓ X_train.csv")
print("  ✓ X_test.csv")
print("  ✓ y_test.csv")
print("  ✓ model_comparison.csv\n")

# -----------------------------
# Read best model (by PR-AUC)
# -----------------------------
model_results = pd.read_csv("model_comparison.csv", index_col=0)
if "pr_auc" not in model_results.columns:
    print("❌ Error: model_comparison.csv missing pr_auc column; cannot pick best model")
    sys.exit(1)
best_model_name = model_results["pr_auc"].idxmax()
best_row = model_results.loc[best_model_name]
best_model_file = f"trained_model_{best_model_name.lower().replace(' ', '_')}.pkl"

if not os.path.exists(best_model_file):
    print(f"❌ Error: best model file not found {best_model_file}")
    print("Please ensure step4 produced the corresponding trained_model_*.pkl")
    sys.exit(1)

print(f"🏆 Best model (PR-AUC): {best_model_name}")
print(f"   Model file: {best_model_file}")
print(f"   PR-AUC  : {best_row.get('pr_auc', float('nan')):.4f}")
if "best_threshold" in best_row:
    print(f"   Threshold (F1@val): {best_row['best_threshold']:.3f}")
print()

# -----------------------------
# Load data and model
# -----------------------------
print("📂 Loading data and model...")
X_train = pd.read_csv("X_train.csv").reset_index(drop=True)
X_test  = pd.read_csv("X_test.csv").reset_index(drop=True)
y_test  = load_y_auto("y_test.csv").reset_index(drop=True)

with open(best_model_file, "rb") as f:
    model = pickle.load(f)
print("✅ Data and model loaded successfully\n")

# -----------------------------
# Check third-party dependencies: SHAP / LIME
# -----------------------------
try:
    import shap
    HAS_SHAP = True
    print("✅ SHAP installed")
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not installed. Install via: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
    print("✅ LIME installed\n")
except ImportError:
    HAS_LIME = False
    print("⚠️  LIME not installed. Install via: pip install lime\n")

if not HAS_SHAP and not HAS_LIME:
    print("❌ Error: both SHAP and LIME unavailable; cannot run explainability")
    sys.exit(1)

# -----------------------------
# Sampling (speed-up for SHAP)
# -----------------------------
if len(X_test) > MAX_SAMPLES_SHAP:
    sample_idx = np_random.choice(len(X_test), size=MAX_SAMPLES_SHAP, replace=False)
else:
    sample_idx = np.arange(len(X_test))
X_test_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_test_sample = y_test.iloc[sample_idx].reset_index(drop=True)
print(f"📌 SHAP analysis will use {len(X_test_sample)} test samples (random sample, seed={RANDOM_STATE})\n")

# -----------------------------
# SHAP analysis
# -----------------------------
shap_outputs = []
if HAS_SHAP:
    print("=" * 80)
    print("📊 SHAP explainability analysis")
    print("=" * 80)
    print()

    import matplotlib.pyplot as plt

    # Prefer TreeExplainer for tree-based models, otherwise fallback
    explainer = None
    tree_ok = False
    try:
        explainer = shap.TreeExplainer(model)
        tree_ok = True
    except Exception:
        tree_ok = False

    if not tree_ok:
        try:
            explainer = shap.Explainer(model, X_train, seed=RANDOM_STATE)
            print("ℹ️ Non-tree model or incompatible with TreeExplainer; falling back to shap.Explainer")
        except Exception as e:
            print(f"❌ Failed to initialise SHAP explainer: {str(e)[:120]}")
            HAS_SHAP = False

    if HAS_SHAP and explainer is not None:
        try:
            # Compute SHAP values
            shap_values = explainer.shap_values(X_test_sample) if hasattr(explainer, "shap_values") else explainer(X_test_sample).values

            # Handle different shapes: binary classifiers may return [neg, pos]
            if isinstance(shap_values, list):
                # Take positive class (typically index 1)
                if len(shap_values) >= 2 and shap_values[1] is not None:
                    shap_matrix = np.array(shap_values[1])
                else:
                    shap_matrix = np.array(shap_values[0])
            else:
                shap_matrix = np.array(shap_values)

            # Summary scatter plot
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_matrix, X_test_sample, max_display=15, show=False)
                plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved: shap_summary_plot.png")
                shap_outputs.append("shap_summary_plot.png")
            except Exception as e:
                print(f"⚠️  Failed to generate SHAP summary plot: {str(e)[:120]}")

            # Bar plot for mean |SHAP|
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_matrix, X_test_sample, plot_type="bar", max_display=15, show=False)
                plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved: shap_bar_plot.png")
                shap_outputs.append("shap_bar_plot.png")
            except Exception as e:
                print(f"⚠️  Failed to generate SHAP bar plot: {str(e)[:120]}")

            # Individual explanations: plot waterfall for one death/survival case
            try:
                # Calculate expected_value (TreeExplainer typically scalar or length 2)
                try:
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        # Take positive class baseline
                        expected_value = expected_value[1] if len(expected_value) >= 2 else expected_value[0]
                except Exception:
                    expected_value = None

                # Find indices
                death_pos = int(np.where(y_test_sample.values == 1)[0][0]) if (y_test_sample == 1).any() else None
                survival_pos = int(np.where(y_test_sample.values == 0)[0][0]) if (y_test_sample == 0).any() else None

                from shap import Explanation
                def save_waterfall(pos, tag):
                    if pos is None: 
                        return
                    ex = Explanation(
                        values=shap_matrix[pos],
                        base_values=expected_value,
                        data=X_test_sample.iloc[pos].values,
                        feature_names=X_test_sample.columns.tolist()
                    )
                    plt.figure(figsize=(10, 8))
                    shap.plots.waterfall(ex, max_display=15, show=False)
                    plt.title(f"SHAP Waterfall - {tag.title()} Case (Sample {pos})", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    fname = f"shap_waterfall_{tag}.png"
                    plt.savefig(fname, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Saved: {fname}")
                    shap_outputs.append(fname)

                save_waterfall(death_pos, "death")
                save_waterfall(survival_pos, "survival")

            except Exception as e:
                print(f"⚠️  Failed to generate individual SHAP waterfall plot: {str(e)[:120]}")

            # Save SHAP values and global importance
            try:
                shap_df = pd.DataFrame(shap_matrix, columns=[f"shap_{c}" for c in X_test_sample.columns])
                shap_df["y_true"] = y_test_sample.values
                shap_df.to_csv("shap_values.csv", index=False)
                print("✅ Saved: shap_values.csv")
            except Exception as e:
                print(f"⚠️  Failed to save shap_values.csv: {str(e)[:120]}")

            try:
                mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
                imp_df = pd.DataFrame({"feature": X_test_sample.columns, "mean_abs_shap": mean_abs_shap}) \
                         .sort_values("mean_abs_shap", ascending=False)
                imp_df.to_csv("shap_feature_importance.csv", index=False)
                print("✅ Saved: shap_feature_importance.csv")
            except Exception as e:
                print(f"⚠️  Failed to save shap_feature_importance.csv: {str(e)[:120]}")

            print("\n✅ SHAP analysis complete\n")

        except Exception as e:
            print(f"❌ SHAP analysis failed: {str(e)[:160]}\n")

# -----------------------------
# LIME analysis (local)
# -----------------------------
lime_outputs = []
if HAS_LIME:
    print("=" * 80)
    print("📊 LIME local explainability")
    print("=" * 80)
    print()

    import matplotlib.pyplot as plt
    try:
        # Build LIME explainer (using data in the same space as model input)
        lime_explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["Survival", "Death"],
            mode="classification",
            random_state=RANDOM_STATE
        )

        # LIME needs a predict_proba interface, if not, wrap decision_function with normalisation
        def lime_predict_proba(Xarr):
            Xdf = pd.DataFrame(Xarr, columns=X_train.columns) if Xarr.ndim == 2 else pd.DataFrame([Xarr], columns=X_train.columns)
            proba = get_probabilities(model, Xdf)
            if proba is None:
                # Worst fallback: use model's 0/1 prediction, convert to [p0,p1]
                preds = model.predict(Xdf)
                proba = preds.astype(float)
            proba = np.clip(proba, 0.0, 1.0)
            # Assemble binary probabilities: [1-p, p]
            return np.vstack([1 - proba, proba]).T

        # Select one death and one survival case
        death_pos = int(np.where(y_test.values == 1)[0][0]) if (y_test == 1).any() else None
        survival_pos = int(np.where(y_test.values == 0)[0][0]) if (y_test == 0).any() else None
        cases = []
        if death_pos is not None: cases.append(("death", death_pos))
        if survival_pos is not None: cases.append(("survival", survival_pos))

        for tag, pos in cases:
            instance = X_test.iloc[pos].values
            explanation = lime_explainer.explain_instance(
                instance, lime_predict_proba, num_features=10
            )

            fig = explanation.as_pyplot_figure()
            plt.title(f"LIME Explanation - {tag.title()} Case (Sample {pos})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname_img = f"lime_explanation_{tag}.png"
            plt.savefig(fname_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {fname_img}")
            lime_outputs.append(fname_img)

            # Text explanation
            fname_txt = f"lime_explanation_{tag}.txt"
            with open(fname_txt, "w", encoding="utf-8") as f:
                f.write(f"LIME Explanation - {tag.title()} Case (Sample {pos})\n")
                f.write("=" * 60 + "\n\n")
                # Print model probabilities
                p = lime_predict_proba(instance.reshape(1, -1))[0]
                pred_label = int(p[1] >= 0.5)
                f.write(f"Prediction (threshold=0.5): {pred_label}\n")
                f.write(f"Prediction Probabilities [p0, p1]: {p}\n\n")
                f.write("Top-10 Feature Contributions:\n")
                f.write("-" * 60 + "\n")
                for feat, weight in explanation.as_list():
                    f.write(f"{feat:50s}: {weight: .6f}\n")
            print(f"✅ Saved: {fname_txt}")

        print("\n✅ LIME analysis complete\n")

    except Exception as e:
        print(f"❌ LIME analysis failed: {str(e)[:160]}\n")

# -----------------------------
# Metadata logging (compliance/reproducibility)
# -----------------------------
print("=" * 80)
print("📝 Saving explainability metadata explain_meta.json")
print("=" * 80)
print()

meta = {
    "random_state": RANDOM_STATE,
    "max_samples_shap": MAX_SAMPLES_SHAP,
    "best_model": best_model_name,
    "best_model_file": best_model_file,
    "best_model_scores": {
        k: (float(best_row[k]) if k in best_row and pd.notna(best_row[k]) else None)
        for k in ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "best_threshold"]
    },
    "files_used": {
        "X_train.csv": os.path.exists("X_train.csv"),
        "X_test.csv": os.path.exists("X_test.csv"),
        "y_test.csv": os.path.exists("y_test.csv"),
        "model_comparison.csv": os.path.exists("model_comparison.csv"),
        best_model_file: True
    },
    "libraries": {
        "python": sys.version.split()[0],
        "numpy": None,
        "pandas": None,
        "shap": None,
        "lime": None
    },
    "artifacts": {
        "shap": shap_outputs,
        "lime": lime_outputs
    }
}
# Record version information
try:
    import numpy as _np
    meta["libraries"]["numpy"] = _np.__version__
except Exception:
    pass
try:
    import pandas as _pd
    meta["libraries"]["pandas"] = _pd.__version__
except Exception:
    pass
try:
    import shap as _sh
    meta["libraries"]["shap"] = _sh.__version__
except Exception:
    pass
try:
    import lime as _lm
    meta["libraries"]["lime"] = _lm.__version__
except Exception:
    pass

with open("explain_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("✅ Saved: explain_meta.json\n")

# -----------------------------
# Summary output
# -----------------------------
print("=" * 80)
print("✅ Step 7 complete - explainability finished (enhanced)")
print("=" * 80)
print()

print("📁 Generated files:")
if len(shap_outputs) > 0:
    print("  SHAP:")
    for p in shap_outputs:
        print("   -", p)
    print("   - shap_values.csv")
    print("   - shap_feature_importance.csv")
    print()
if len(lime_outputs) > 0:
    print("  LIME:")
    for p in lime_outputs:
        print("   -", p)
    print("   - lime_explanation_death.txt (if death samples) ")
    print("   - lime_explanation_survival.txt (if survival samples) ")
    print()

print("  Metadata:")
print("   - explain_meta.json")
print()

print("🎯 Key takeaways:")
print("  - Best model automatically aligned with Step 4/6 PR-AUC ranking")
print("  - SHAP provides global and local views; handles different return shapes")
print("  - LIME falls back to decision_function when predict_proba missing")
print("  - All artefacts saved for audit and reproducibility")
print()
