#!/usr/bin/env python3
"""
Simplified Epcoritamab CRS Analysis Script
Run directly without complex configuration

Usage:
    cd /Users/manushi/Downloads/openfda/task2
    python3 run_epcoritamab_analysis_simple.py
"""

print("=" * 80)
print("EPCORITAMAB CRS ANALYSIS - SIMPLIFIED VERSION")
print("=" * 80)
print()

# Check dependencies
print("Checking dependencies...")
try:
    import pandas as pd
    import numpy as np
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("All dependencies installed")
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease run: pip install pandas numpy lifelines scikit-learn matplotlib seaborn")
    exit(1)

print()

# Import main analysis class
print("Importing analysis module...")
try:
    from requirement2_epcoritamab_crs_analysis import EpcoritamabCRSAnalysis
    print("Analysis module imported successfully")
except ImportError as e:
    print(f"Unable to import analysis module: {e}")
    print("\nPlease ensure you run this script from the task2 directory:")
    print("  cd /Users/manushi/Downloads/openfda/task2")
    print("  python3 run_epcoritamab_analysis_simple.py")
    exit(1)

print()
print("=" * 80)
print("Starting analysis...")
print("=" * 80)
print()

# Run analysis
try:
    analyzer = EpcoritamabCRSAnalysis()
    
    # Run complete analysis (this method includes all steps)
    analyzer.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
    
except Exception as e:
    print(f"\nError during analysis:")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error message: {str(e)}")
    print("\nFor help, please check:")
    print("  1. Network connection (requires FDA API access)")
    print("  2. Package versions")
    print("  3. File write permissions")
    import traceback
    print("\nDetailed error information:")
    traceback.print_exc()
    exit(1)

