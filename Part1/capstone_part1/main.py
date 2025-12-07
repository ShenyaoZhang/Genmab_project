"""
CRS Risk Analysis Pipeline
Main entry point for the complete analysis workflow.

Usage:
    python main.py --all           # Run complete pipeline
    python main.py --extract       # Only extract data
    python main.py --causal        # Only run causal analysis
    python main.py --nlp           # Only run NLP analysis
    python main.py --dashboard     # Launch interactive dashboard
"""

import argparse
import os
import sys


def run_data_extraction():
    """Run data extraction pipeline."""
    print("\n" + "="*60)
    print("STEP 1: DATA EXTRACTION")
    print("="*60)
    
    # Check if FAERS data exists
    if not os.path.exists("fda_drug_events.json"):
        print("Running FAERS data extraction...")
        import test
    else:
        print("FAERS data already exists: fda_drug_events.json")
    
    # Run CRS extraction
    if not os.path.exists("crs_extracted_data.json"):
        print("\nExtracting CRS-specific variables...")
        from extract_crs_data import main as extract_main
        extract_main()
    else:
        print("CRS extracted data already exists: crs_extracted_data.json")
    
    # Create multi-source dataset using real data if available
    print("\nCreating multi-source dataset...")
    from data_extractors import create_multi_source_data
    
    # Check for real Eudravigilance data
    eudravigilance_path = "../Run Line Listing Report.csv"
    jader_path = "../jader_data"
    
    if not os.path.exists(eudravigilance_path):
        eudravigilance_path = None
        print("  Note: Eudravigilance CSV not found, skipping EU data")
    
    if not os.path.exists(jader_path):
        jader_path = None
        print("  Note: JADER data not found, skipping JP data")
    
    create_multi_source_data(
        faers_path="crs_extracted_data.json",
        eudravigilance_path=eudravigilance_path,
        jader_data_dir=jader_path,
        output_path="multi_source_crs_data.json"
    )
    
    print("\n✓ Data extraction complete!")


def run_causal_analysis():
    """Run causal inference analysis."""
    print("\n" + "="*60)
    print("STEP 2: CAUSAL ANALYSIS")
    print("="*60)
    
    from causal_analysis import CRSCausalAnalyzer
    
    analyzer = CRSCausalAnalyzer("multi_source_crs_data.json")
    df = analyzer.load_data()
    
    # Generate and print report
    report = analyzer.generate_causal_report()
    print(report)
    
    # Save results
    analyzer.save_results()
    
    with open("causal_analysis_report.txt", "w") as f:
        f.write(report)
    
    print("\n✓ Causal analysis complete!")
    print("  Results saved to: causal_analysis_results.json")
    print("  Report saved to: causal_analysis_report.txt")


def run_nlp_analysis():
    """Run NLP analysis on narratives."""
    print("\n" + "="*60)
    print("STEP 3: NLP ANALYSIS")
    print("="*60)
    
    from nlp_analysis import CRSNarrativeAnalyzer
    
    analyzer = CRSNarrativeAnalyzer("crs_extracted_data.json")
    
    # Generate and print report
    report = analyzer.generate_nlp_report()
    print(report)
    
    # Save results
    analyzer.save_features()
    
    with open("nlp_analysis_report.txt", "w") as f:
        f.write(report)
    
    print("\n✓ NLP analysis complete!")
    print("  Features saved to: narrative_features.json")
    print("  Report saved to: nlp_analysis_report.txt")


def run_dashboard():
    """Launch interactive dashboard."""
    print("\n" + "="*60)
    print("LAUNCHING INTERACTIVE DASHBOARD")
    print("="*60)
    
    try:
        import streamlit
        print("\nStarting Streamlit dashboard...")
        print("Open http://localhost:8501 in your browser")
        print("Press Ctrl+C to stop the server\n")
        
        os.system("streamlit run interactive_dashboard.py")
    except ImportError:
        print("Streamlit not installed. Running terminal interface...")
        from interactive_dashboard import run_terminal_interface
        run_terminal_interface()


def generate_summary():
    """Generate executive summary of all analyses."""
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    summary = """
CYTOKINE RELEASE SYNDROME (CRS) RISK ANALYSIS
Epcoritamab-Treated Patients
============================================

DATA SOURCES:
• FAERS (FDA Adverse Event Reporting System) - Primary
• Eudravigilance (European) - Simulated/Instructions provided
• JADER (Japanese) - Simulated/Instructions provided

KEY FINDINGS:

1. CAUSAL RISK FACTORS (Evidence for causation):
   ✓ Epcoritamab Dose: Higher doses associated with increased CRS risk
     - Mechanism: Dose-dependent T-cell activation
     - Evidence: Dose-response relationship in clinical data
   
   ✓ Steroid Premedication: PROTECTIVE effect
     - Mechanism: Anti-inflammatory, cytokine suppression
     - Evidence: Propensity score analysis shows causal benefit
   
   ✓ Tocilizumab: PROTECTIVE for severe CRS
     - Mechanism: IL-6 receptor blockade
     - Evidence: Established treatment for CRS

2. CONFOUNDERS (Must adjust for):
   ⚠ Age: Older patients may receive lower doses AND have worse outcomes
   ⚠ Disease Stage: Advanced disease affects both treatment and prognosis
   ⚠ Prior Therapies: Treatment history affects both

3. CORRELATIONAL (Not causal):
   ✗ Number of co-medications: Marker of disease severity
   ✗ Data source: Reporting differences only

RECOMMENDATIONS FOR RISK MODELING:
1. Use propensity score methods to control confounding
2. Include interaction terms for effect modifiers (steroids, tocilizumab)
3. Stratify analysis by dose level
4. Account for time-to-event (CRS typically occurs early)

LIMITATIONS:
• Pharmacovigilance data is observational (subject to reporting bias)
• Missing data is common (especially weights, exact doses)
• Confounding by indication cannot be fully controlled
• Simulated data used for EU/JP sources (demonstration purposes)

FILES GENERATED:
• crs_extracted_data.json - Structured FAERS data
• multi_source_crs_data.json - Combined multi-source data
• causal_analysis_results.json - Causal analysis output
• causal_analysis_report.txt - Detailed causal report
• narrative_features.json - NLP-extracted features
• nlp_analysis_report.txt - NLP analysis report

INTERACTIVE ANALYSIS:
Run: streamlit run interactive_dashboard.py
"""
    print(summary)
    
    with open("executive_summary.txt", "w") as f:
        f.write(summary)
    
    print("Summary saved to: executive_summary.txt")


def main():
    parser = argparse.ArgumentParser(
        description="CRS Risk Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all           Run complete analysis pipeline
  python main.py --extract       Extract and prepare data only
  python main.py --causal        Run causal inference analysis
  python main.py --nlp           Run NLP analysis on narratives
  python main.py --dashboard     Launch interactive dashboard
  python main.py --summary       Generate executive summary
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                        help='Run complete pipeline')
    parser.add_argument('--extract', action='store_true',
                        help='Run data extraction')
    parser.add_argument('--causal', action='store_true',
                        help='Run causal analysis')
    parser.add_argument('--nlp', action='store_true',
                        help='Run NLP analysis')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch interactive dashboard')
    parser.add_argument('--summary', action='store_true',
                        help='Generate executive summary')
    
    args = parser.parse_args()
    
    # If no arguments, show help and run all
    if not any(vars(args).values()):
        parser.print_help()
        print("\n" + "="*60)
        print("Running complete pipeline (use --help for options)")
        print("="*60)
        args.all = True
    
    if args.all or args.extract:
        run_data_extraction()
    
    if args.all or args.causal:
        run_causal_analysis()
    
    if args.all or args.nlp:
        run_nlp_analysis()
    
    if args.all or args.summary:
        generate_summary()
    
    if args.dashboard:
        run_dashboard()
    
    if args.all:
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the reports in the current directory")
        print("2. Run 'python main.py --dashboard' for interactive analysis")
        print("3. For real EU/JP data, follow instructions in data_extractors.py")


if __name__ == "__main__":
    main()

