"""
Biomarker Integration Module - Conceptual Framework
Shows how biomarkers (IL-6, IL-7, IL-21, CCL17, CCL13, TGF-β1, etc.) 
could be incorporated into the pipeline if such data becomes available.

This module addresses the feedback:
"Show how similar biomarkers could be incorporated into your existing 
pipeline if such data were available. This can be conceptual rather 
than fully implemented."

Reference: CAR-T biomarker paper shared by Genmab team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BIOMARKER MAPPING TABLE
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    BIOMARKER INTEGRATION MAPPING TABLE                               │
├─────────────────────┬───────────────────┬───────────────────────────────────────────┤
│ Biomarker           │ Type              │ How It Would Be Processed                 │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ IL-6 (Interleukin-6)│ Continuous        │ - Z-score normalization                   │
│                     │ (pg/mL)           │ - Log transform if skewed                 │
│                     │                   │ - Clinical cutoff: >7 pg/mL = elevated    │
│                     │                   │ - Key CRS severity biomarker              │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ IL-7 (Interleukin-7)│ Continuous        │ - Z-score normalization                   │
│                     │ (pg/mL)           │ - Associated with T-cell homeostasis      │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ IL-21               │ Continuous        │ - Z-score normalization                   │
│ (Interleukin-21)    │ (pg/mL)           │ - T-cell activation marker                │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ CCL17               │ Continuous        │ - Log transform + Z-score                 │
│ (TARC)              │ (pg/mL)           │ - Chemokine associated with Th2 response  │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ CCL13               │ Continuous        │ - Log transform + Z-score                 │
│ (MCP-4)             │ (pg/mL)           │ - Monocyte chemoattractant                │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ TGF-β1              │ Continuous        │ - Z-score normalization                   │
│                     │ (ng/mL)           │ - Immunomodulatory cytokine               │
│                     │                   │ - Same preprocessing as other continuous  │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ IFN-γ               │ Continuous        │ - Log transform (often highly skewed)     │
│ (Interferon-gamma)  │ (pg/mL)           │ - Key marker of T-cell activation         │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ CRP                 │ Continuous        │ - Can use z-score or clinical cutoffs     │
│ (C-Reactive Protein)│ (mg/L)            │ - Normal: <10, Elevated: 10-50, High: >50 │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ Ferritin            │ Continuous        │ - Log transform + Z-score                 │
│                     │ (ng/mL)           │ - Elevated in cytokine storm              │
├─────────────────────┼───────────────────┼───────────────────────────────────────────┤
│ LDH                 │ Continuous        │ - Z-score normalization                   │
│ (Lactate DH)        │ (U/L)             │ - Tumor burden marker                     │
└─────────────────────┴───────────────────┴───────────────────────────────────────────┘

INTEGRATION INTO EXISTING PIPELINE:
===================================

Current pipeline processes:
  - Demographics (age, sex, weight) → Z-score or bucketing
  - Drug exposure (dose) → Log transform + Z-score
  - Co-medications → Binary flags for drug classes

Biomarkers would be added as:
  - Additional continuous features → Same preprocessing as weight/dose
  - Input to risk models alongside existing features
  - Potential new outcome variables (e.g., peak IL-6 level)

Example code addition to feature matrix:
  df['IL6_zscore'] = (df['IL6'] - df['IL6'].mean()) / df['IL6'].std()
  df['IL6_log'] = np.log(df['IL6'] + 1)
  df['IL6_elevated'] = (df['IL6'] > 7).astype(int)
"""

BIOMARKER_CONFIG = {
    'IL-6': {
        'full_name': 'Interleukin-6',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore'],
        'clinical_cutoffs': {'normal': 7, 'elevated': 50, 'high': 150},
        'relevance': 'Key CRS severity biomarker, IL-6 pathway is target of tocilizumab',
        'expected_association': 'Higher levels associated with more severe CRS'
    },
    'IL-7': {
        'full_name': 'Interleukin-7',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['zscore'],
        'clinical_cutoffs': None,
        'relevance': 'T-cell homeostasis and survival',
        'expected_association': 'May indicate T-cell expansion'
    },
    'IL-21': {
        'full_name': 'Interleukin-21',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['zscore'],
        'clinical_cutoffs': None,
        'relevance': 'T-cell and B-cell activation',
        'expected_association': 'Elevated in immune activation'
    },
    'CCL17': {
        'full_name': 'TARC (Thymus and Activation-Regulated Chemokine)',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore'],
        'clinical_cutoffs': None,
        'relevance': 'Th2-associated chemokine',
        'expected_association': 'Varies with immune response type'
    },
    'CCL13': {
        'full_name': 'MCP-4 (Monocyte Chemoattractant Protein-4)',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore'],
        'clinical_cutoffs': None,
        'relevance': 'Monocyte recruitment',
        'expected_association': 'May correlate with inflammation severity'
    },
    'TGF-beta1': {
        'full_name': 'Transforming Growth Factor Beta 1',
        'unit': 'ng/mL',
        'type': 'continuous',
        'preprocessing': ['zscore'],
        'clinical_cutoffs': None,
        'relevance': 'Immunomodulatory, tissue repair',
        'expected_association': 'Complex role in CRS pathophysiology'
    },
    'IFN-gamma': {
        'full_name': 'Interferon Gamma',
        'unit': 'pg/mL',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore'],
        'clinical_cutoffs': {'normal': 10, 'elevated': 100},
        'relevance': 'Key T-cell effector cytokine',
        'expected_association': 'Higher levels with T-cell activation'
    },
    'CRP': {
        'full_name': 'C-Reactive Protein',
        'unit': 'mg/L',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore', 'clinical_cutoffs'],
        'clinical_cutoffs': {'normal': 10, 'elevated': 50, 'high': 100},
        'relevance': 'General inflammation marker',
        'expected_association': 'Elevated in CRS'
    },
    'Ferritin': {
        'full_name': 'Ferritin',
        'unit': 'ng/mL',
        'type': 'continuous',
        'preprocessing': ['log_transform', 'zscore'],
        'clinical_cutoffs': {'normal': 300, 'elevated': 1000, 'high': 5000},
        'relevance': 'Elevated in cytokine storm/MAS',
        'expected_association': 'Very high levels indicate severe inflammation'
    }
}


class BiomarkerIntegrator:
    """
    Conceptual class showing how biomarkers would be integrated into the pipeline.
    
    This class demonstrates the preprocessing and integration steps that would
    be applied if biomarker data becomes available from:
    - Retrospective clinical trial data
    - Linked electronic health records
    - Future prospective studies
    """
    
    def __init__(self):
        self.biomarker_stats = {}
        self.config = BIOMARKER_CONFIG
    
    def preprocess_biomarker(
        self, 
        values: pd.Series, 
        biomarker: str,
        method: str = 'auto'
    ) -> pd.DataFrame:
        """
        Preprocess a biomarker according to its configuration.
        
        Args:
            values: Raw biomarker values
            biomarker: Biomarker name (e.g., 'IL-6')
            method: 'auto' uses configured methods, or specify 'zscore', 'log', etc.
        
        Returns:
            DataFrame with preprocessed features
        
        Example:
            >>> integrator = BiomarkerIntegrator()
            >>> il6_values = pd.Series([5, 15, 150, 8, 45])
            >>> processed = integrator.preprocess_biomarker(il6_values, 'IL-6')
            >>> print(processed)
               IL6_raw  IL6_log  IL6_zscore  IL6_category
            0      5.0    1.79      -0.72      Normal
            1     15.0    2.77      -0.45      Elevated
            2    150.0    5.02       1.89      High
            3      8.0    2.20      -0.61      Elevated
            4     45.0    3.83       0.47      Elevated
        """
        if biomarker not in self.config:
            raise ValueError(f"Unknown biomarker: {biomarker}")
        
        cfg = self.config[biomarker]
        result = pd.DataFrame()
        
        # Clean biomarker name for column names
        col_prefix = biomarker.replace('-', '').replace(' ', '_')
        
        # Raw values
        result[f'{col_prefix}_raw'] = values
        
        # Log transform if configured
        if 'log_transform' in cfg['preprocessing']:
            result[f'{col_prefix}_log'] = np.log(values + 1)
        
        # Z-score normalization
        if 'zscore' in cfg['preprocessing']:
            mean_val = values.mean()
            std_val = values.std()
            self.biomarker_stats[biomarker] = {'mean': mean_val, 'std': std_val}
            result[f'{col_prefix}_zscore'] = (values - mean_val) / std_val
        
        # Clinical cutoffs (categorical)
        if cfg['clinical_cutoffs']:
            cutoffs = cfg['clinical_cutoffs']
            categories = []
            for val in values:
                if val <= cutoffs['normal']:
                    categories.append('Normal')
                elif val <= cutoffs.get('elevated', float('inf')):
                    categories.append('Elevated')
                else:
                    categories.append('High')
            result[f'{col_prefix}_category'] = categories
        
        return result
    
    def integrate_into_feature_matrix(
        self,
        existing_features: pd.DataFrame,
        biomarker_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Integrate biomarker features into existing feature matrix.
        
        This shows how biomarkers would be added alongside existing features
        (demographics, drug exposure, etc.) for model training.
        
        Args:
            existing_features: Current feature matrix (age, weight, dose, etc.)
            biomarker_data: DataFrame with biomarker columns
        
        Returns:
            Combined feature matrix
        
        Example:
            >>> # Existing features
            >>> features = pd.DataFrame({
            ...     'age_zscore': [0.5, -0.3, 1.2],
            ...     'dose_zscore': [0.8, -0.5, 1.0],
            ...     'has_steroids': [1, 0, 1]
            ... })
            >>> 
            >>> # Biomarker data
            >>> biomarkers = pd.DataFrame({
            ...     'IL-6': [15, 150, 8],
            ...     'CRP': [25, 180, 12]
            ... })
            >>> 
            >>> # Integrate
            >>> combined = integrator.integrate_into_feature_matrix(features, biomarkers)
        """
        combined = existing_features.copy()
        
        # Process each biomarker
        for col in biomarker_data.columns:
            if col in self.config:
                processed = self.preprocess_biomarker(biomarker_data[col], col)
                for proc_col in processed.columns:
                    combined[proc_col] = processed[proc_col].values
        
        return combined
    
    def generate_integration_documentation(self) -> str:
        """Generate documentation for biomarker integration."""
        
        doc = []
        doc.append("=" * 80)
        doc.append("BIOMARKER INTEGRATION DOCUMENTATION")
        doc.append("=" * 80)
        doc.append("")
        doc.append("This document describes how biomarkers from the CAR-T paper could be")
        doc.append("incorporated into our epcoritamab CRS risk analysis pipeline.")
        doc.append("")
        
        # Mapping table
        doc.append("-" * 80)
        doc.append("BIOMARKER PREPROCESSING MAPPING")
        doc.append("-" * 80)
        doc.append("")
        doc.append(f"{'Biomarker':<15} {'Processing Method':<30} {'Integration Notes':<35}")
        doc.append("-" * 80)
        
        for biomarker, cfg in self.config.items():
            methods = ', '.join(cfg['preprocessing'])
            doc.append(f"{biomarker:<15} {methods:<30} {cfg['expected_association'][:35]}")
        
        doc.append("")
        doc.append("-" * 80)
        doc.append("INTEGRATION EXAMPLE")
        doc.append("-" * 80)
        doc.append("""
# Example: Adding IL-6 to the feature matrix

# 1. Load biomarker data (if available)
biomarker_df = pd.read_csv('biomarker_data.csv')

# 2. Preprocess IL-6
il6_preprocessed = integrator.preprocess_biomarker(
    biomarker_df['IL-6'], 
    'IL-6'
)

# 3. Add to existing features
features['IL6_zscore'] = il6_preprocessed['IL6_zscore']
features['IL6_category'] = il6_preprocessed['IL6_category']

# 4. Use in model training
model.fit(features, outcomes)

# 5. Interpret with SHAP
# "A positive SHAP value for IL-6 means higher IL-6 levels 
#  push the prediction toward higher CRS risk"
""")
        
        doc.append("")
        doc.append("-" * 80)
        doc.append("CLINICAL INTERPRETATION GUIDE")
        doc.append("-" * 80)
        
        for biomarker, cfg in self.config.items():
            doc.append(f"\n{biomarker} ({cfg['full_name']}):")
            doc.append(f"  Unit: {cfg['unit']}")
            doc.append(f"  Relevance: {cfg['relevance']}")
            if cfg['clinical_cutoffs']:
                doc.append(f"  Cutoffs: Normal ≤{cfg['clinical_cutoffs']['normal']}, "
                          f"Elevated ≤{cfg['clinical_cutoffs'].get('elevated', 'N/A')}, "
                          f"High >{cfg['clinical_cutoffs'].get('elevated', 'N/A')}")
        
        return "\n".join(doc)


def generate_biomarker_slide_content() -> str:
    """
    Generate content for presentation slide on biomarker integration.
    
    This addresses the feedback:
    "Please add one slide that shows how similar biomarkers could be 
    incorporated into your existing pipeline if such data were available."
    """
    
    content = """
================================================================================
SLIDE: FUTURE BIOMARKER INTEGRATION
================================================================================

TITLE: Integrating Biomarkers into the CRS Risk Pipeline

CURRENT PIPELINE:
┌─────────────────────────────────────────────────────────────────┐
│  Input Data          →    Preprocessing    →    Risk Model     │
│  ─────────────            ──────────────        ──────────     │
│  • Demographics           • Z-score             • Random Forest │
│  • Drug exposure          • Log transform       • Gradient Boost│
│  • Co-medications         • Categorization      • Output: Risk  │
│  • NLP features                                   Score         │
└─────────────────────────────────────────────────────────────────┘

FUTURE INTEGRATION (if biomarker data becomes available):
┌─────────────────────────────────────────────────────────────────┐
│  + Biomarker Data    →    Same Preprocessing  →  Enhanced Model │
│  ─────────────────        ──────────────────     ─────────────  │
│  • IL-6                   • Z-score              • Improved AUC │
│  • IL-7, IL-21            • Log transform        • Better risk  │
│  • CCL17, CCL13           • Clinical cutoffs       stratification│
│  • TGF-β1, IFN-γ                                                │
│  • CRP, Ferritin                                                │
└─────────────────────────────────────────────────────────────────┘

BIOMARKER PROCESSING MAPPING TABLE:
┌──────────────┬─────────────────────────────────────────────────────┐
│ Biomarker    │ Processing Method                                   │
├──────────────┼─────────────────────────────────────────────────────┤
│ IL-6         │ Log transform + Z-score (continuous lab value)      │
│              │ Same as: dose preprocessing in current pipeline     │
├──────────────┼─────────────────────────────────────────────────────┤
│ TGF-β1       │ Z-score normalization (continuous variable)         │
│              │ Same as: age/weight preprocessing                   │
├──────────────┼─────────────────────────────────────────────────────┤
│ CRP          │ Clinical cutoffs: <10, 10-50, >50 (categorical)     │
│              │ Same as: BMI bucketization                          │
├──────────────┼─────────────────────────────────────────────────────┤
│ All others   │ Standard continuous variable preprocessing          │
│              │ Flexible: log/z-score based on distribution         │
└──────────────┴─────────────────────────────────────────────────────┘

KEY POINT: Our pipeline is FLEXIBLE and can accommodate biomarker data 
without code changes - biomarkers are processed using the same methods 
we already use for continuous clinical variables.

CODE EXAMPLE:
    # Adding IL-6 to the feature matrix
    df['IL6_zscore'] = (df['IL6'] - df['IL6'].mean()) / df['IL6'].std()
    
    # Adding to existing features
    features = pd.concat([existing_features, biomarker_features], axis=1)
    
    # Retrain model with biomarkers
    model.fit(features, outcomes)

================================================================================
"""
    return content


def main():
    """Demonstrate biomarker integration concepts."""
    
    print(generate_biomarker_slide_content())
    
    # Create sample biomarker data for demonstration
    print("\n" + "="*70)
    print("BIOMARKER PREPROCESSING DEMONSTRATION")
    print("="*70)
    
    integrator = BiomarkerIntegrator()
    
    # Sample IL-6 data
    sample_il6 = pd.Series([5, 15, 150, 8, 45, 200, 12, 6])
    
    print("\nSample IL-6 values (pg/mL):", list(sample_il6))
    
    processed = integrator.preprocess_biomarker(sample_il6, 'IL-6')
    print("\nProcessed IL-6 features:")
    print(processed.round(2).to_string())
    
    # Show full documentation
    print("\n" + integrator.generate_integration_documentation())


if __name__ == "__main__":
    main()

