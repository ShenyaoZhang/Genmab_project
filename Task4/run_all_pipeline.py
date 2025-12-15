#!/usr/bin/env python3
"""
Run all pipeline files sequentially and save output to a text file.
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path

# Main pipeline files in order
PIPELINE_FILES = [
    "01_extract_data.py",
    "02_inspect_data.py",
    "03_preprocess_data.py",
    "04_train_models.py",
    "05_analyze_features.py",
    "06_visualize_results.py",
    "07_explainability.py",
    "08_test_models.py",
    "09_test_epcoritamab.py",
    "10_prepare_crs_dataset.py",
    "11_granular_crs_analysis.py",
    "12_crs_model_training.py",
    "13_crs_shap_analysis.py",
]

# Files to delete (temporary and utility files)
FILES_TO_DELETE = [
    # Temporary files
    "task4_data_temp_10.csv",
    "task4_data_temp_20.csv",
    "task4_data_temp_30.csv",
    "extract_log.txt",
    
    # Utility/tool files (not main pipeline)
    "pipeline.py",
    "ui_mock.py",
    "model_evaluation.py",
    "add_future_biomarkers.py",
    "generate_presentation_charts.py",
    "crs_dashboard.py",
    "run_crs_dashboard.sh",
]

def cleanup_files():
    """Delete unnecessary files."""
    print("=" * 80)
    print("Cleaning up unnecessary files...")
    print("=" * 80)
    print()
    
    deleted_count = 0
    for file_path in FILES_TO_DELETE:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    
    print(f"\nDeleted {deleted_count} files")
    print()

def run_script(script_path, output_file):
    """Run a Python script and capture output."""
    script_name = os.path.basename(script_path)
    
    print("=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)
    print()
    
    output_file.write("=" * 80 + "\n")
    output_file.write(f"Running: {script_name}\n")
    output_file.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_file.write("=" * 80 + "\n\n")
    output_file.flush()
    
    try:
        # Run the script and capture both stdout and stderr
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(script_path)) or ".",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3600  # 1 hour timeout per script
        )
        
        # Write output to file
        output_file.write(result.stdout)
        
        if result.returncode != 0:
            output_file.write(f"\n\nERROR: Script exited with code {result.returncode}\n")
            print(f"ERROR: {script_name} exited with code {result.returncode}")
        else:
            print(f"SUCCESS: {script_name} completed")
        
        output_file.write("\n\n")
        output_file.flush()
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        output_file.write(f"\n\nERROR: Script timed out after 1 hour\n")
        print(f"ERROR: {script_name} timed out")
        output_file.flush()
        return False
    except Exception as e:
        error_msg = f"ERROR running {script_name}: {str(e)}\n"
        output_file.write(error_msg)
        print(error_msg)
        output_file.flush()
        return False

def main():
    """Main function to run all pipeline files."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Cleanup first
    cleanup_files()
    
    # Output file
    output_file_path = "pipeline_execution_results.txt"
    
    print("=" * 80)
    print("Starting Pipeline Execution")
    print("=" * 80)
    print(f"Output will be saved to: {output_file_path}")
    print()
    
    start_time = datetime.datetime.now()
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Write header
        output_file.write("=" * 80 + "\n")
        output_file.write("Pipeline Execution Results\n")
        output_file.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write("=" * 80 + "\n\n")
        
        # Run each script
        success_count = 0
        failed_scripts = []
        
        for script in PIPELINE_FILES:
            script_path = Path(script)
            if not script_path.exists():
                print(f"WARNING: {script} not found, skipping...")
                output_file.write(f"WARNING: {script} not found, skipping...\n\n")
                failed_scripts.append(script)
                continue
            
            success = run_script(str(script_path), output_file)
            if success:
                success_count += 1
            else:
                failed_scripts.append(script)
            
            print()
        
        # Write summary
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        
        output_file.write("=" * 80 + "\n")
        output_file.write("Execution Summary\n")
        output_file.write("=" * 80 + "\n\n")
        output_file.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_file.write(f"Elapsed time: {elapsed}\n\n")
        output_file.write(f"Total scripts: {len(PIPELINE_FILES)}\n")
        output_file.write(f"Successful: {success_count}\n")
        output_file.write(f"Failed: {len(failed_scripts)}\n\n")
        
        if failed_scripts:
            output_file.write("Failed scripts:\n")
            for script in failed_scripts:
                output_file.write(f"  - {script}\n")
        
        output_file.write("\n" + "=" * 80 + "\n")
    
    print("=" * 80)
    print("Pipeline Execution Complete")
    print("=" * 80)
    print(f"Results saved to: {output_file_path}")
    print(f"Successful: {success_count}/{len(PIPELINE_FILES)}")
    if failed_scripts:
        print(f"Failed scripts: {', '.join(failed_scripts)}")
    print()

if __name__ == "__main__":
    main()

