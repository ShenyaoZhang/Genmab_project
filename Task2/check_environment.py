#!/usr/bin/env python3
"""
Environment Check Script - Diagnose Runtime Issues

Run this script to check your environment configuration:
    cd /Users/manushi/Downloads/openfda/task2
    python3 check_environment.py
"""

import sys
from pathlib import Path

print("=" * 80)
print("Environment Diagnostic Check")
print("=" * 80)
print()

# 1. Check Python version
print("1. Python Version:")
print(f"   Version: {sys.version}")
print(f"   Executable: {sys.executable}")
if sys.version_info < (3, 7):
    print("   WARNING: Python version too low, recommend 3.7+")
else:
    print("   Python version OK")
print()

# 2. Check current working directory
print("2. Current Working Directory:")
current_dir = Path.cwd()
print(f"   {current_dir}")
if current_dir.name == "task2":
    print("   In correct directory")
else:
    print("   WARNING: Not in task2 directory")
    print(f"   Please run: cd {current_dir.parent}/task2")
print()

# 3. Check required script files
print("3. Check Required Files:")
required_files = [
    "requirement2_epcoritamab_crs_analysis.py",
    "run_survival_analysis.py",
    "run_epcoritamab_analysis_simple.py"
]
all_files_exist = True
for file in required_files:
    file_path = Path(file)
    if file_path.exists():
        print(f"   {file}")
    else:
        print(f"   MISSING: {file}")
        all_files_exist = False

if not all_files_exist:
    print("   WARNING: Some files missing, ensure you're in correct directory")
print()

# 4. Check Python dependencies
print("4. Check Dependencies:")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'lifelines': 'lifelines',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'requests': 'requests'
}

missing_packages = []
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        # Get version info
        try:
            if import_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'unknown')
            print(f"   {package_name:20s} (version: {version})")
        except:
            print(f"   {package_name:20s}")
    except ImportError:
        print(f"   MISSING: {package_name:20s}")
        missing_packages.append(package_name)

if missing_packages:
    print()
    print("   WARNING: Missing packages, please install:")
    print(f"   pip install {' '.join(missing_packages)}")
else:
    print()
    print("   All dependencies installed")
print()

# 5. Check Python path
print("5. Python Path (sys.path):")
for i, path in enumerate(sys.path[:5], 1):
    print(f"   {i}. {path}")
if len(sys.path) > 5:
    print(f"   ... ({len(sys.path) - 5} more paths)")
print()

# 6. Check module import
print("6. Test Module Import:")
try:
    sys.path.insert(0, str(Path.cwd()))
    from requirement2_epcoritamab_crs_analysis import EpcoritamabCRSAnalysis
    print("   requirement2_epcoritamab_crs_analysis imported successfully")
    print(f"   EpcoritamabCRSAnalysis class available")
except ImportError as e:
    print(f"   Import failed: {e}")
    print("   Please ensure you run this script from task2 directory")
except Exception as e:
    print(f"   WARNING: Other error during import: {e}")
print()

# 7. Check network connection (test FDA API)
print("7. Test FDA API Connection:")
try:
    import requests
    response = requests.get("https://api.fda.gov/drug/event.json?limit=1", timeout=5)
    if response.status_code == 200:
        print("   FDA API connection OK")
    else:
        print(f"   WARNING: FDA API returned status code: {response.status_code}")
except Exception as e:
    print(f"   Cannot connect to FDA API: {e}")
    print("   Please check network connection")
print()

# 8. Check write permissions
print("8. Check File Write Permissions:")
try:
    test_file = Path("test_write_permission.tmp")
    test_file.write_text("test")
    test_file.unlink()
    print("   Current directory has write permissions")
except Exception as e:
    print(f"   Write permission issue: {e}")
print()

# 9. Check output directories
print("9. Check Output Directories:")
output_dirs = ["output", "results"]
for dir_name in output_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   {dir_name}/ directory exists")
    else:
        print(f"   INFO: {dir_name}/ directory does not exist (will be created automatically)")
print()

# Summary
print("=" * 80)
print("Diagnostic Summary")
print("=" * 80)

issues = []
if sys.version_info < (3, 7):
    issues.append("Python version too low")
if current_dir.name != "task2":
    issues.append("Not in task2 directory")
if not all_files_exist:
    issues.append("Some required files missing")
if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")

if not issues:
    print("All checks passed! You can run the analysis scripts.")
    print()
    print("Recommended commands:")
    print("  python3 run_epcoritamab_analysis_simple.py")
    print()
    print("Or:")
    print("  python3 run_survival_analysis.py \\")
    print("      --drug epcoritamab \\")
    print("      --adverse_event 'cytokine release syndrome' \\")
    print("      --output_dir output/epcoritamab_crs \\")
    print("      --limit 1000")
else:
    print("WARNING: Found the following issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print()
    print("Please resolve these issues before running analysis scripts.")
    print("See FIX_AND_RUN.md for detailed instructions")

print("=" * 80)
