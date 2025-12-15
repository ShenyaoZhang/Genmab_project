#!/usr/bin/env python3
"""
Scalable Survival Analysis Pipeline for Drug-Adverse Event Pairs

This module provides a flexible, parameterized pipeline for survival analysis
of any drug-adverse event combination using FDA FAERS data.

Usage:
    python run_survival_analysis.py --drug epcoritamab --adverse_event "cytokine release syndrome"
    
Or programmatically:
    from run_survival_analysis import run_pipeline
    results = run_pipeline(drug="tafasitamab", adverse_event="ICANS")
"""

import pandas as pd
import numpy as np
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import analysis modules
try:
    from requirement2_epcoritamab_crs_analysis import (
        EpcoritamabCRSAnalysis
    )
except ImportError as e:
    print(f"Error importing analysis module: {e}")
    print(f"Current directory: {Path(__file__).parent.absolute()}")
    print(f"Python path: {sys.path}")
    print("\nPlease ensure you run this script from the task2 directory:")
    print("  cd /Users/manushi/Downloads/openfda/task2")
    print("  python run_survival_analysis.py --drug epcoritamab --adverse_event 'cytokine release syndrome'")
    sys.exit(1)

# Survival analysis
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class ScalableSurvivalPipeline:
    """
    Scalable pipeline for drug-specific adverse event survival analysis.
    
    This class implements a fully parameterized analysis pipeline that can
    handle any drug-adverse event combination without code modification.
    """
    
    def __init__(self, drug: str, adverse_event: str, output_dir: str = "output"):
        """
        Initialize the scalable survival analysis pipeline.
        
        Parameters
        ----------
        drug : str
            Drug name to analyze (e.g., "epcoritamab", "tafasitamab")
        adverse_event : str
            Adverse event of interest (e.g., "cytokine release syndrome", "ICANS")
        output_dir : str
            Directory to save outputs
        """
        self.drug = drug
        self.adverse_event = adverse_event
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed
        
        # Analysis results storage
        self.results = {
            'drug': drug,
            'adverse_event': adverse_event,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {},
            'cox_model': {},
            'km_analysis': {},
            'feature_selection': {},
            'risk_stratification': {}
        }
        
        # AE search terms mapping
        self.ae_search_terms = self._get_ae_search_terms(adverse_event)
        
    def _get_ae_search_terms(self, adverse_event: str) -> List[str]:
        """
        Generate search terms for adverse event identification.
        
        Parameters
        ----------
        adverse_event : str
            Adverse event name
            
        Returns
        -------
        List[str]
            List of search terms to identify the AE in FAERS data
        """
        # Predefined mappings for common AEs
        ae_mappings = {
            'cytokine release syndrome': [
                'CYTOKINE RELEASE SYNDROME', 'CRS', 'CYTOKINE STORM',
                'IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY SYNDROME',
                'ICANS', 'MACROPHAGE ACTIVATION SYNDROME'
            ],
            'icans': [
                'IMMUNE EFFECTOR CELL-ASSOCIATED NEUROTOXICITY SYNDROME',
                'ICANS', 'NEUROTOXICITY', 'ENCEPHALOPATHY', 'SEIZURE'
            ],
            'neutropenia': [
                'NEUTROPENIA', 'NEUTROPHIL COUNT DECREASED',
                'FEBRILE NEUTROPENIA', 'AGRANULOCYTOSIS'
            ],
            'thrombocytopenia': [
                'THROMBOCYTOPENIA', 'PLATELET COUNT DECREASED',
                'IMMUNE THROMBOCYTOPENIA'
            ]
        }
        
        # Get predefined terms or use the AE name itself
        ae_lower = adverse_event.lower()
        if ae_lower in ae_mappings:
            return ae_mappings[ae_lower]
        else:
            return [adverse_event.upper(), adverse_event]
    
    def collect_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Step 1: Collect data from FDA FAERS for specified drug.
        
        Parameters
        ----------
        limit : int
            Maximum number of records to collect
            
        Returns
        -------
        pd.DataFrame
            Raw FAERS data
        """
        print(f"\n{'='*80}")
        print(f"STEP 1: COLLECTING DATA")
        print(f"Drug: {self.drug}")
        print(f"Adverse Event: {self.adverse_event}")
        print(f"{'='*80}")
        
        # Use existing collector (can be extended for other drugs)
        from requirement2_epcoritamab_crs_analysis import EpcoritamabCRSAnalysis
        
        analyzer = EpcoritamabCRSAnalysis()
        analyzer.drug_name = self.drug  # Override drug name
        analyzer.crs_terms = self.ae_search_terms  # Override AE terms
        
        df = analyzer.collect_epcoritamab_data(limit=limit)
        
        # Save raw data
        output_file = self.output_dir / f"{self.drug}_{self.adverse_event.replace(' ', '_')}_raw_data.csv"
        df.to_csv(output_file, index=False)
        
        self.results['data_summary'] = {
            'total_records': len(df),
            'date_collected': datetime.now().isoformat(),
            'output_file': str(output_file)
        }
        
        print(f"✓ Collected {len(df)} records")
        print(f"✓ Saved to: {output_file}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Extract and prepare features for analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw FAERS data
            
        Returns
        -------
        pd.DataFrame
            Processed data with features
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: FEATURE EXTRACTION AND PREPARATION")
        print(f"{'='*80}")
        
        df_clean = df.copy()
        
        # Identify AE occurrence
        df_clean['has_ae'] = df_clean['reactions'].str.contains(
            '|'.join(self.ae_search_terms), 
            case=False, 
            na=False
        )
        
        # Handle time variables
        df_clean['time_adjusted'] = df_clean['time_to_event_days'].fillna(0.5)
        df_clean.loc[df_clean['time_adjusted'] <= 0, 'time_adjusted'] = 0.5
        
        # Event indicator
        df_clean['event_occurred'] = df_clean['has_ae'].astype(int)
        
        # Continuous variable processing
        # Weight: z-score normalization
        if 'patient_weight' in df_clean.columns:
            weight_mean = df_clean['patient_weight'].mean()
            weight_std = df_clean['patient_weight'].std()
            df_clean['weight_zscore'] = (df_clean['patient_weight'] - weight_mean) / weight_std
            
            print(f"✓ Weight normalization: mean={weight_mean:.1f}, std={weight_std:.1f}")
        
        # Age: keep as continuous and create categorical
        if 'patient_age' in df_clean.columns:
            df_clean['age_group'] = pd.cut(
                df_clean['patient_age'],
                bins=[0, 50, 65, 100],
                labels=['<50', '50-65', '>65']
            )
            print(f"✓ Age buckets: <50, 50-65, >65")
        
        # BMI calculation and buckets (if weight and height available)
        # Note: FAERS typically doesn't have height, but showing the concept
        if 'patient_weight' in df_clean.columns:
            df_clean['weight_group'] = pd.cut(
                df_clean['patient_weight'],
                bins=[0, 60, 80, 100, 200],
                labels=['<60kg', '60-80kg', '80-100kg', '>100kg']
            )
            print(f"✓ Weight buckets: <60kg, 60-80kg, 80-100kg, >100kg")
        
        # Polypharmacy analysis: identify drug classes
        df_clean['polypharmacy'] = df_clean['total_drugs'] >= 3
        
        # Save processed data
        output_file = self.output_dir / f"{self.drug}_{self.adverse_event.replace(' ', '_')}_processed_data.csv"
        df_clean.to_csv(output_file, index=False)
        
        ae_count = df_clean['has_ae'].sum()
        ae_rate = ae_count / len(df_clean) * 100
        
        self.results['data_summary'].update({
            'ae_cases': int(ae_count),
            'ae_rate': float(ae_rate),
            'processed_file': str(output_file)
        })
        
        print(f"✓ {ae_count} {self.adverse_event} cases ({ae_rate:.1f}%)")
        print(f"✓ Processed data saved to: {output_file}")
        
        return df_clean
    
    def run_cox_model(self, df: pd.DataFrame) -> Dict:
        """
        Step 3: Cox proportional hazards model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed data
            
        Returns
        -------
        Dict
            Cox model results
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: COX PROPORTIONAL HAZARDS MODEL")
        print(f"{'='*80}")
        
        # Select features
        features = ['patient_age', 'patient_weight', 'total_drugs',
                   'concomitant_drugs', 'polypharmacy',
                   'is_lifethreatening', 'is_hospitalization']
        
        cox_data = df[features + ['time_adjusted', 'event_occurred']].copy()
        cox_data = cox_data.dropna()
        
        # Convert boolean to int
        for col in ['polypharmacy', 'is_lifethreatening', 'is_hospitalization']:
            if col in cox_data.columns:
                cox_data[col] = cox_data[col].astype(int)
        
        # Fit Cox model
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cox_data, duration_col='time_adjusted', event_col='event_occurred')
        
        # Extract results
        summary = cph.summary
        
        cox_results = {
            'c_index': float(cph.concordance_index_),
            'hazard_ratios': {},
            'feature_interpretation': {}
        }
        
        print(f"\n✓ C-index: {cph.concordance_index_:.4f}")
        print(f"\nHazard Ratios:")
        
        for idx, row in summary.iterrows():
            hr = row['exp(coef)']
            ci_lower = row['exp(coef) lower 95%']
            ci_upper = row['exp(coef) upper 95%']
            p_val = row['p']
            
            cox_results['hazard_ratios'][idx] = {
                'HR': float(hr),
                'CI_lower': float(ci_lower),
                'CI_upper': float(ci_upper),
                'p_value': float(p_val)
            }
            
            # Interpretable explanation
            if hr > 1:
                effect = "increases"
                magnitude = (hr - 1) * 100
            else:
                effect = "decreases"
                magnitude = (1 - hr) * 100
            
            interpretation = f"This feature {effect} risk by {magnitude:.1f}%"
            if p_val < 0.05:
                interpretation += " (statistically significant)"
            
            cox_results['feature_interpretation'][idx] = interpretation
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {idx:20s}: HR={hr:.3f} [{ci_lower:.3f}-{ci_upper:.3f}], p={p_val:.4f} {sig}")
            print(f"    → {interpretation}")
        
        self.results['cox_model'] = cox_results
        
        return cox_results
    
    def run_feature_selection(self, df: pd.DataFrame) -> Dict:
        """
        Step 4: Feature selection using ensemble methods.
        
        Parameters
        ----------
        df : pd.DataFrame
            Processed data
            
        Returns
        -------
        Dict
            Feature selection results
        """
        print(f"\n{'='*80}")
        print(f"STEP 4: FEATURE SELECTION")
        print(f"{'='*80}")
        
        feature_cols = ['patient_age', 'patient_weight', 'total_drugs',
                       'concomitant_drugs', 'polypharmacy',
                       'is_lifethreatening', 'is_hospitalization']
        
        df_features = df[feature_cols + ['has_ae']].copy()
        
        # Convert boolean
        for col in ['polypharmacy', 'is_lifethreatening', 'is_hospitalization', 'has_ae']:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(int)
        
        df_features = df_features.dropna()
        
        X = df_features[feature_cols]
        y = df_features['has_ae']
        
        # F-test
        selector_f = SelectKBest(score_func=f_classif, k='all')
        selector_f.fit(X, y)
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Combine results
        feature_results = {}
        
        for i, feature in enumerate(feature_cols):
            feature_results[feature] = {
                'f_test_score': float(selector_f.scores_[i]),
                'f_test_pvalue': float(selector_f.pvalues_[i]),
                'mi_score': float(mi_scores[i]),
                'rf_importance': float(rf.feature_importances_[i]),
                'interpretation': self._interpret_feature_importance(
                    feature, 
                    rf.feature_importances_[i],
                    selector_f.pvalues_[i]
                )
            }
        
        # Sort by RF importance
        sorted_features = sorted(feature_results.items(),
                                key=lambda x: x[1]['rf_importance'],
                                reverse=True)
        
        print(f"\nTop Features (by Random Forest importance):")
        for feature, scores in sorted_features[:5]:
            print(f"  {feature:20s}: {scores['rf_importance']:.3f}")
            print(f"    → {scores['interpretation']}")
        
        self.results['feature_selection'] = feature_results
        
        return feature_results
    
    def _interpret_feature_importance(self, feature: str, importance: float, p_value: float) -> str:
        """
        Generate interpretable explanation for feature importance.
        
        Example: "A positive bar means the feature pushes risk upward."
        """
        if importance > 0.2:
            strength = "strongly"
        elif importance > 0.1:
            strength = "moderately"
        else:
            strength = "weakly"
        
        interpretation = f"This feature {strength} influences {self.adverse_event} risk (importance={importance:.1%})"
        
        if p_value < 0.05:
            interpretation += " and is statistically significant"
        
        return interpretation
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the analysis.
        
        Returns
        -------
        str
            Path to summary report
        """
        print(f"\n{'='*80}")
        print(f"GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"SURVIVAL ANALYSIS REPORT")
        report_lines.append(f"Drug: {self.drug}")
        report_lines.append(f"Adverse Event: {self.adverse_event}")
        report_lines.append(f"Analysis Date: {self.results['timestamp']}")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Data Summary
        report_lines.append("DATA SUMMARY")
        report_lines.append("-"*80)
        ds = self.results['data_summary']
        report_lines.append(f"Total Records: {ds.get('total_records', 'N/A')}")
        report_lines.append(f"AE Cases: {ds.get('ae_cases', 'N/A')} ({ds.get('ae_rate', 'N/A'):.1f}%)")
        report_lines.append("")
        
        # Cox Model
        if 'cox_model' in self.results:
            report_lines.append("COX PROPORTIONAL HAZARDS MODEL")
            report_lines.append("-"*80)
            cm = self.results['cox_model']
            report_lines.append(f"C-index: {cm.get('c_index', 'N/A'):.4f}")
            report_lines.append("")
            report_lines.append("Hazard Ratios (with interpretation):")
            
            for feature, hr_data in cm.get('hazard_ratios', {}).items():
                report_lines.append(f"\n  {feature}:")
                report_lines.append(f"    HR = {hr_data['HR']:.3f} [{hr_data['CI_lower']:.3f}-{hr_data['CI_upper']:.3f}]")
                report_lines.append(f"    p-value = {hr_data['p_value']:.4f}")
                interp = cm['feature_interpretation'].get(feature, '')
                report_lines.append(f"    → {interp}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_file = self.output_dir / f"{self.drug}_{self.adverse_event.replace(' ', '_')}_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save JSON results
        json_file = self.output_dir / f"{self.drug}_{self.adverse_event.replace(' ', '_')}_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Report saved to: {report_file}")
        print(f"✓ JSON results saved to: {json_file}")
        
        return str(report_file)


def run_pipeline(drug: str, adverse_event: str, output_dir: str = "output", limit: int = 1000) -> Dict:
    """
    Run the complete survival analysis pipeline for any drug-AE pair.
    
    This is the main entry point for programmatic use.
    
    Parameters
    ----------
    drug : str
        Drug name (e.g., "epcoritamab", "tafasitamab")
    adverse_event : str
        Adverse event name (e.g., "cytokine release syndrome", "ICANS")
    output_dir : str
        Directory to save outputs
    limit : int
        Maximum records to collect
        
    Returns
    -------
    Dict
        Complete analysis results
        
    Examples
    --------
    >>> results = run_pipeline(drug="tafasitamab", adverse_event="ICANS")
    >>> print(f"C-index: {results['cox_model']['c_index']}")
    
    >>> results = run_pipeline(drug="epcoritamab", adverse_event="neutropenia")
    """
    print(f"\n{'#'*80}")
    print(f"# SCALABLE SURVIVAL ANALYSIS PIPELINE")
    print(f"# Drug: {drug}")
    print(f"# Adverse Event: {adverse_event}")
    print(f"{'#'*80}\n")
    
    # Initialize pipeline
    pipeline = ScalableSurvivalPipeline(drug=drug, adverse_event=adverse_event, output_dir=output_dir)
    
    # Step 1: Collect data
    df_raw = pipeline.collect_data(limit=limit)
    
    # Step 2: Prepare features
    df_processed = pipeline.prepare_features(df_raw)
    
    # Step 3: Cox model
    cox_results = pipeline.run_cox_model(df_processed)
    
    # Step 4: Feature selection
    feature_results = pipeline.run_feature_selection(df_processed)
    
    # Step 5: Generate report
    report_path = pipeline.generate_summary_report()
    
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    
    return pipeline.results


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Scalable Survival Analysis Pipeline for Drug-AE Pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_survival_analysis.py --drug epcoritamab --adverse_event "cytokine release syndrome"
  python run_survival_analysis.py --drug tafasitamab --adverse_event ICANS --limit 500
        """
    )
    
    parser.add_argument('--drug', type=str, required=True,
                       help='Drug name (e.g., epcoritamab, tafasitamab)')
    parser.add_argument('--adverse_event', type=str, required=True,
                       help='Adverse event name (e.g., "cytokine release syndrome", ICANS)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Maximum records to collect (default: 1000)')
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_pipeline(
        drug=args.drug,
        adverse_event=args.adverse_event,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ C-index: {results['cox_model']['c_index']:.4f}")
    print(f"✓ AE Rate: {results['data_summary']['ae_rate']:.1f}%")


if __name__ == "__main__":
    main()


