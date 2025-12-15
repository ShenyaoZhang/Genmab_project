#!/usr/bin/env python3
"""
Full Pipeline Runner
Runs all steps sequentially and saves output to pipeline_full_run.txt
"""

import subprocess
import sys
import os
from datetime import datetime

def run_pipeline():
    output_file = 'pipeline_full_run.txt'
    
    # Clear previous output
    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('Full Pipeline Execution Log\n')
        f.write('=' * 80 + '\n')
        f.write(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Working directory: {os.getcwd()}\n')
        f.write('=' * 80 + '\n\n')
    
    scripts = [
        ('01_extract_data.py', 'Step 1: Data Extraction'),
        ('02_inspect_data.py', 'Step 2: Data Inspection'),
        ('03_preprocess_data.py', 'Step 3: Data Preprocessing'),
        ('04_train_models.py', 'Step 4: Model Training'),
        ('05_analyze_features.py', 'Step 5: Feature Analysis'),
        ('06_visualize_results.py', 'Step 6: Visualization'),
        ('07_explainability.py', 'Step 7: Explainability'),
        ('08_test_models.py', 'Step 8: Model Testing'),
        ('09_test_epcoritamab.py', 'Step 9: Epcoritamab Testing'),
        ('11_granular_crs_analysis.py', 'Step 11: Granular CRS Analysis'),
        ('12_crs_model_training.py', 'Step 12: CRS Model Training'),
        ('13_crs_shap_analysis.py', 'Step 13: CRS SHAP Analysis'),
        ('generate_presentation_charts.py', 'Step Optional: Generate Charts'),
    ]
    
    successful = 0
    failed = []
    
    for script, description in scripts:
        with open(output_file, 'a') as f:
            f.write('\n' + '=' * 80 + '\n')
            f.write(f'{description}\n')
            f.write(f'Script: {script}\n')
            f.write(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('=' * 80 + '\n\n')
            f.flush()
        
        print(f"\n{'='*80}")
        print(f"Running: {description}")
        print(f"Script: {script}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max per script
            )
            
            with open(output_file, 'a') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write('\n[STDERR]\n')
                    f.write(result.stderr)
                f.write(f'\n[Step completed with exit code {result.returncode}]\n')
                f.flush()
            
            if result.returncode == 0:
                successful += 1
                print(f"✓ {description} completed successfully")
            else:
                failed.append((script, description, result.returncode))
                print(f"✗ {description} failed with exit code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            with open(output_file, 'a') as f:
                f.write(f'\n[ERROR] Script timed out after 30 minutes\n')
            failed.append((script, description, 'TIMEOUT'))
            print(f"✗ {description} timed out")
        except Exception as e:
            with open(output_file, 'a') as f:
                f.write(f'\n[ERROR] {str(e)}\n')
            failed.append((script, description, str(e)))
            print(f"✗ {description} error: {e}")
    
    # Write summary
    with open(output_file, 'a') as f:
        f.write('\n' + '=' * 80 + '\n')
        f.write('Pipeline Execution Summary\n')
        f.write('=' * 80 + '\n')
        f.write(f'Completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Successful steps: {successful}/{len(scripts)}\n')
        if failed:
            f.write(f'Failed steps: {len(failed)}\n')
            for script, desc, code in failed:
                f.write(f'  - {desc} ({script}): {code}\n')
        f.write('=' * 80 + '\n')
    
    print(f"\n{'='*80}")
    print(f"Pipeline execution complete!")
    print(f"Successful: {successful}/{len(scripts)}")
    if failed:
        print(f"Failed: {len(failed)}")
    print(f"Full log saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    run_pipeline()

