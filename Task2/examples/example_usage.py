#!/usr/bin/env python3
"""
Example Usage of the Scalable Survival Analysis Pipeline

This script demonstrates how to use the pipeline for different drug-AE combinations
without modifying any code.
"""

import sys
sys.path.append('../task2')

from run_survival_analysis import run_pipeline

# Example 1: Epcoritamab + Cytokine Release Syndrome (original analysis)
print("=" * 80)
print("EXAMPLE 1: Epcoritamab + Cytokine Release Syndrome")
print("=" * 80)

results_crs = run_pipeline(
    drug="epcoritamab",
    adverse_event="cytokine release syndrome",
    output_dir="example_outputs/epcoritamab_crs",
    limit=1000
)

print(f"\nResults:")
print(f"  AE Rate: {results_crs['data_summary']['ae_rate']:.1f}%")
print(f"  C-index: {results_crs['cox_model']['c_index']:.4f}")

# Example 2: Tafasitamab + ICANS
print("\n" + "=" * 80)
print("EXAMPLE 2: Tafasitamab + ICANS")
print("=" * 80)

results_icans = run_pipeline(
    drug="tafasitamab",
    adverse_event="ICANS",
    output_dir="example_outputs/tafasitamab_icans",
    limit=500
)

print(f"\nResults:")
print(f"  AE Rate: {results_icans['data_summary']['ae_rate']:.1f}%")
print(f"  C-index: {results_icans['cox_model']['c_index']:.4f}")

# Example 3: Epcoritamab + Neutropenia
print("\n" + "=" * 80)
print("EXAMPLE 3: Epcoritamab + Neutropenia")
print("=" * 80)

results_neutropenia = run_pipeline(
    drug="epcoritamab",
    adverse_event="neutropenia",
    output_dir="example_outputs/epcoritamab_neutropenia",
    limit=1000
)

print(f"\nResults:")
print(f"  AE Rate: {results_neutropenia['data_summary']['ae_rate']:.1f}%")
print(f"  C-index: {results_neutropenia['cox_model']['c_index']:.4f}")

print("\n" + "=" * 80)
print("ALL EXAMPLES COMPLETE")
print("=" * 80)
print("\nCheck the example_outputs/ directory for results.")

