#!/usr/bin/env python3
"""
Run full pipeline and capture all output to a text file
"""
import sys
import subprocess
import datetime
from pathlib import Path

OUTPUT_FILE = "pipeline_output.txt"


def run_step(step_num, script_name, description):
    """Run a pipeline step and capture output."""
    print(f"\n{'=' * 80}")
    print(f"Step {step_num}: {description}")
    print(f"{'=' * 80}\n")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Step {step_num}: {description}\n")
        f.write(f"Script: {script_name}\n")
        f.write(
            f"Time: {
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
        f.flush()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        output = result.stdout
        if result.returncode != 0:
            output += f"\n[ERROR] Script exited with code {
                result.returncode}\n"

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(output)
            f.write(
                f"\n[Step {step_num} completed with exit code {
                    result.returncode}]\n\n")
            f.flush()

        print(output[-500:] if len(output) >
              500 else output)  # Print last 500 chars
        return result.returncode == 0

    except Exception as e:
        error_msg = f"[ERROR] Failed to run {script_name}: {str(e)}\n"
        print(error_msg)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(error_msg)
        return False


def main():
    """Run the full pipeline."""
    # Initialize output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Pipeline Execution Log\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Started at: {
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Working directory: {Path.cwd()}\n")
        f.write("=" * 80 + "\n\n")

    steps = [
        (1, "01_extract_data.py", "Data Extraction"),
        (2, "02_inspect_data.py", "Data Inspection"),
        (3, "03_preprocess_data.py", "Data Preprocessing"),
        (4, "04_train_models.py", "Model Training"),
        (5, "05_analyze_features.py", "Feature Analysis"),
        (6, "06_visualize_results.py", "Visualization"),
        (7, "07_explainability.py", "Explainability Analysis"),
        (8, "08_test_models.py", "Model Testing"),
        (9, "09_test_epcoritamab.py", "Epcoritamab Testing"),
        (11, "11_granular_crs_analysis.py", "Granular CRS Analysis"),
        (12, "12_crs_model_training.py", "CRS Model Training"),
        (13, "13_crs_shap_analysis.py", "CRS SHAP Analysis"),
    ]

    # Optional: Generate presentation charts
    optional_steps = [
        ("generate_presentation_charts.py", "Generate Presentation Charts"),
    ]

    success_count = 0
    failed_steps = []

    for step_num, script, description in steps:
        if not Path(script).exists():
            print(f"[SKIP] {script} not found, skipping...")
            continue

        success = run_step(step_num, script, description)
        if success:
            success_count += 1
        else:
            failed_steps.append((step_num, script))

    # Run optional steps
    for script, description in optional_steps:
        if Path(script).exists():
            run_step("Optional", script, description)

    # Final summary
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Pipeline Execution Summary\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Completed at: {
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful steps: {success_count}/{len(steps)}\n")
        if failed_steps:
            f.write(f"Failed steps: {failed_steps}\n")
        f.write("=" * 80 + "\n")

    print(f"\n{'=' * 80}")
    print(f"Pipeline completed: {success_count}/{len(steps)} steps successful")
    print(f"Output saved to: {OUTPUT_FILE}")
    if failed_steps:
        print(f"Failed steps: {failed_steps}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
