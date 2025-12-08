"""
Data Summary and Analysis Module
Provides comprehensive dataset summaries, missingness analysis, 
variable availability tables, and database-specific breakdowns.

This module addresses the following feedback points:
- Dataset and model summaries (counts, completeness, missingness)
- Variable availability across databases (FAERS, JADER, EudraVigilance)
- Database-specific analyses and breakdowns
- Polypharmacy analysis with drug classes
- Continuous variable preprocessing documentation
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# VARIABLE AVAILABILITY TABLE
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    VARIABLE AVAILABILITY ACROSS DATABASES                            │
├──────────────────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
│ Variable                 │ FAERS (US)  │ JADER (JP)  │ EV (EU)     │ Notes          │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ DEMOGRAPHICS                                                                         │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ Age                      │ Yes         │ Yes         │ Age Group   │ EV: bucketed   │
│ Sex                      │ Yes         │ Yes         │ Yes         │                │
│ Weight                   │ Limited     │ No          │ No          │ Often missing  │
│ Country                  │ Yes         │ JP only     │ Yes         │                │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ CLINICAL                                                                             │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ Seriousness              │ Yes         │ Yes         │ Yes         │                │
│ Outcome (recovery/death) │ Yes         │ Yes         │ Yes         │                │
│ Hospitalization          │ Yes         │ Limited     │ Yes         │                │
│ Life-threatening         │ Yes         │ Limited     │ Yes         │                │
│ Indication               │ Yes         │ Yes         │ Limited     │                │
│ Disease Stage (DLBCL)    │ No*         │ No          │ No          │ *In narratives │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ DRUG EXPOSURE                                                                        │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ Dose (numeric)           │ Yes         │ Limited     │ Limited     │                │
│ Dose unit                │ Yes         │ Limited     │ Limited     │                │
│ Route of administration  │ Yes         │ Yes         │ Yes         │                │
│ Start/stop dates         │ Limited     │ Limited     │ Limited     │                │
│ Concomitant medications  │ Yes         │ Yes         │ Yes         │                │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ COMORBIDITIES                                                                        │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ Medical history          │ Yes         │ Yes         │ Limited     │                │
│ Hypertension             │ Yes*        │ Yes*        │ No          │ *In med hx     │
│ Cardiac disease          │ Yes*        │ Yes*        │ No          │ *In med hx     │
│ Diabetes                 │ Yes*        │ Yes*        │ No          │ *In med hx     │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ NLP FEATURES                                                                         │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ Narrative text           │ Yes         │ No**        │ Limited     │ **Japanese     │
│ CRS grade mentioned      │ Yes         │ No          │ Limited     │                │
│ Time to onset            │ Yes         │ No          │ Limited     │                │
│ Severity indicators      │ Yes         │ No          │ Limited     │                │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴────────────────┘
"""

VARIABLE_AVAILABILITY = {
    'Demographics': {
        'Age': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Age Group', 'notes': 'EV provides bucketed age groups'},
        'Sex': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Yes', 'notes': ''},
        'Weight': {'faers': 'Limited', 'jader': 'No', 'eudravigilance': 'No', 'notes': 'Often missing in all sources'},
        'Country': {'faers': 'Yes', 'jader': 'JP only', 'eudravigilance': 'Yes', 'notes': ''},
    },
    'Clinical': {
        'Seriousness': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Yes', 'notes': ''},
        'Outcome': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Yes', 'notes': 'recovery, not_recovered, fatal'},
        'Hospitalization': {'faers': 'Yes', 'jader': 'Limited', 'eudravigilance': 'Yes', 'notes': ''},
        'Life-threatening': {'faers': 'Yes', 'jader': 'Limited', 'eudravigilance': 'Yes', 'notes': ''},
        'Indication': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Limited', 'notes': ''},
        'Disease Stage': {'faers': 'No*', 'jader': 'No', 'eudravigilance': 'No', 'notes': '*May appear in narratives'},
    },
    'Drug Exposure': {
        'Dose (numeric)': {'faers': 'Yes', 'jader': 'Limited', 'eudravigilance': 'Limited', 'notes': ''},
        'Dose unit': {'faers': 'Yes', 'jader': 'Limited', 'eudravigilance': 'Limited', 'notes': ''},
        'Route': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Yes', 'notes': ''},
        'Start/stop dates': {'faers': 'Limited', 'jader': 'Limited', 'eudravigilance': 'Limited', 'notes': 'Often incomplete'},
        'Concomitant medications': {'faers': 'Yes', 'jader': 'Yes', 'eudravigilance': 'Yes', 'notes': ''},
    },
    'NLP Features': {
        'Narrative text': {'faers': 'Yes', 'jader': 'No**', 'eudravigilance': 'Limited', 'notes': '**Japanese text only'},
        'CRS grade': {'faers': 'Yes*', 'jader': 'No', 'eudravigilance': 'Limited', 'notes': '*Extracted from narrative'},
        'Time to onset': {'faers': 'Yes*', 'jader': 'No', 'eudravigilance': 'Limited', 'notes': '*Extracted from narrative'},
    }
}


# =============================================================================
# DRUG CLASSES FOR POLYPHARMACY ANALYSIS
# =============================================================================
DRUG_CLASSES = {
    'Steroids': [
        'DEXAMETHASONE', 'PREDNISOLONE', 'PREDNISONE', 'METHYLPREDNISOLONE',
        'HYDROCORTISONE', 'BETAMETHASONE', 'CORTISONE'
    ],
    'IL-6 Inhibitors': [
        'TOCILIZUMAB', 'SILTUXIMAB', 'SARILUMAB'
    ],
    'Anti-CD20': [
        'RITUXIMAB', 'OBINUTUZUMAB', 'OFATUMUMAB', 'OCRELIZUMAB'
    ],
    'Chemotherapy': [
        'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE', 'BENDAMUSTINE',
        'FLUDARABINE', 'CYTARABINE', 'METHOTREXATE', 'ETOPOSIDE'
    ],
    'Antiemetics': [
        'ONDANSETRON', 'GRANISETRON', 'PALONOSETRON', 'APREPITANT'
    ],
    'Antimicrobials': [
        'ACYCLOVIR', 'VALACYCLOVIR', 'FLUCONAZOLE', 'POSACONAZOLE',
        'LEVOFLOXACIN', 'CIPROFLOXACIN', 'TRIMETHOPRIM', 'SULFAMETHOXAZOLE'
    ],
    'Growth Factors': [
        'FILGRASTIM', 'PEGFILGRASTIM', 'EPOETIN'
    ],
    'Antihistamines': [
        'DIPHENHYDRAMINE', 'CETIRIZINE', 'LORATADINE', 'FEXOFENADINE'
    ],
    'Analgesics': [
        'ACETAMINOPHEN', 'PARACETAMOL', 'IBUPROFEN', 'MORPHINE', 'OXYCODONE'
    ],
    'Antipyretics': [
        'ACETAMINOPHEN', 'PARACETAMOL', 'ASPIRIN', 'IBUPROFEN'
    ]
}


class DatasetSummary:
    """
    Comprehensive dataset summary with missingness analysis,
    variable availability, and database-specific breakdowns.
    """
    
    def __init__(self, data_path: str = "multi_source_crs_data.json"):
        self.data_path = data_path
        self.df = None
        self.raw_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        records = []
        for record in self.raw_data:
            flat_record = {
                'report_id': record.get('report_id'),
                'source': record.get('source', 'faers'),
                'is_crs': record.get('is_crs', False),
                'crs_outcome': record.get('crs_outcome'),
                'serious': record.get('serious', False),
                'hospitalized': record.get('hospitalized', False),
                'death': record.get('death', False),
                'life_threatening': record.get('life_threatening', False),
                'age': record.get('age'),
                'sex': record.get('sex'),
                'weight': record.get('weight'),
                'country': record.get('country'),
                'indication': record.get('indication'),
                'n_co_medications': len(record.get('co_medications', [])),
                'co_medications': record.get('co_medications', []),
                'narrative_text': record.get('narrative_text'),
            }
            
            # Dose info
            doses = record.get('epcoritamab_doses', [])
            if doses:
                dose_mgs = [d.get('dose_mg') for d in doses if d.get('dose_mg')]
                flat_record['max_dose_mg'] = max(dose_mgs) if dose_mgs else None
                flat_record['n_doses'] = len(doses)
            else:
                flat_record['max_dose_mg'] = None
                flat_record['n_doses'] = 0
            
            records.append(flat_record)
        
        self.df = pd.DataFrame(records)
        return self.df
    
    def generate_dataset_summary(self) -> str:
        """Generate comprehensive dataset summary."""
        if self.df is None:
            self.load_data()
        
        report = []
        report.append("=" * 80)
        report.append("DATASET SUMMARY")
        report.append("=" * 80)
        
        # Overall counts
        report.append("\n1. OVERALL COUNTS")
        report.append("-" * 60)
        
        total = len(self.df)
        by_source = self.df['source'].value_counts()
        
        report.append(f"\nTotal Records: {total}")
        report.append(f"\nBy Database:")
        for source, count in by_source.items():
            pct = count / total * 100
            report.append(f"  {source.upper():15} {count:6} ({pct:5.1f}%)")
        
        # CRS-specific counts
        report.append("\n\n2. CRS CASE DISTRIBUTION")
        report.append("-" * 60)
        
        crs_by_source = self.df.groupby('source').agg({
            'is_crs': 'sum',
            'death': 'sum',
            'report_id': 'count'
        }).rename(columns={'report_id': 'total'})
        
        crs_by_source['crs_rate'] = (crs_by_source['is_crs'] / crs_by_source['total'] * 100).round(1)
        crs_by_source['mortality_rate'] = (crs_by_source['death'] / crs_by_source['is_crs'] * 100).round(1)
        
        report.append(f"\n{'Database':<15} {'Total':>8} {'CRS Cases':>10} {'CRS Rate':>10} {'Fatal':>8} {'Mortality':>10}")
        report.append("-" * 70)
        for source in crs_by_source.index:
            row = crs_by_source.loc[source]
            report.append(
                f"{source.upper():<15} {int(row['total']):>8} {int(row['is_crs']):>10} "
                f"{row['crs_rate']:>9.1f}% {int(row['death']):>8} {row['mortality_rate']:>9.1f}%"
            )
        
        # Outcome distribution
        report.append("\n\n3. OUTCOME DISTRIBUTION BY DATABASE")
        report.append("-" * 60)
        
        outcome_dist = self.df.groupby(['source', 'crs_outcome']).size().unstack(fill_value=0)
        report.append(f"\n{outcome_dist.to_string()}")
        
        return "\n".join(report)
    
    def generate_missingness_table(self) -> str:
        """Generate missingness summary table."""
        if self.df is None:
            self.load_data()
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("MISSINGNESS SUMMARY")
        report.append("=" * 80)
        
        # Key variables to check
        variables = ['age', 'sex', 'weight', 'max_dose_mg', 'indication', 
                     'crs_outcome', 'narrative_text', 'country']
        
        report.append(f"\n{'Variable':<20} {'Overall %':>12} {'FAERS %':>12} {'JADER %':>12} {'EV %':>12}")
        report.append("-" * 70)
        
        for var in variables:
            if var not in self.df.columns:
                continue
            
            # Overall missingness
            overall_missing = self.df[var].isna().mean() * 100
            
            # By source
            faers_missing = self.df[self.df['source'] == 'faers'][var].isna().mean() * 100 if 'faers' in self.df['source'].values else float('nan')
            jader_missing = self.df[self.df['source'] == 'jader'][var].isna().mean() * 100 if 'jader' in self.df['source'].values else float('nan')
            ev_missing = self.df[self.df['source'] == 'eudravigilance'][var].isna().mean() * 100 if 'eudravigilance' in self.df['source'].values else float('nan')
            
            report.append(
                f"{var:<20} {overall_missing:>11.1f}% {faers_missing:>11.1f}% "
                f"{jader_missing:>11.1f}% {ev_missing:>11.1f}%"
            )
        
        # Completeness score
        report.append("\n\nCOMPLETENESS SCORES:")
        report.append("-" * 60)
        
        core_vars = ['age', 'sex', 'crs_outcome']
        
        for source in self.df['source'].unique():
            source_df = self.df[self.df['source'] == source]
            complete_cases = source_df[core_vars].notna().all(axis=1).sum()
            completeness = complete_cases / len(source_df) * 100
            report.append(f"  {source.upper()}: {completeness:.1f}% complete cases (age + sex + outcome)")
        
        return "\n".join(report)
    
    def generate_variable_availability_table(self) -> str:
        """Generate variable availability table across databases."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("VARIABLE AVAILABILITY ACROSS DATABASES")
        report.append("=" * 80)
        
        for category, variables in VARIABLE_AVAILABILITY.items():
            report.append(f"\n{category.upper()}")
            report.append("-" * 70)
            report.append(f"{'Variable':<25} {'FAERS':>12} {'JADER':>12} {'EV':>12}")
            report.append("-" * 70)
            
            for var, availability in variables.items():
                faers = availability['faers']
                jader = availability['jader']
                ev = availability['eudravigilance']
                notes = availability['notes']
                
                line = f"{var:<25} {faers:>12} {jader:>12} {ev:>12}"
                if notes:
                    line += f"  ({notes})"
                report.append(line)
        
        return "\n".join(report)
    
    def generate_polypharmacy_analysis(self) -> str:
        """Generate polypharmacy analysis with drug classes."""
        if self.df is None:
            self.load_data()
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("POLYPHARMACY ANALYSIS")
        report.append("=" * 80)
        
        # Classify drugs by class
        drug_class_counts = {cls: {'crs': 0, 'non_crs': 0, 'all': 0} for cls in DRUG_CLASSES}
        
        # Also track individual drugs
        drug_counts_crs = Counter()
        drug_counts_non_crs = Counter()
        
        for _, row in self.df.iterrows():
            co_meds = row.get('co_medications', [])
            if not isinstance(co_meds, list):
                continue
            
            is_severe = row.get('death', False) or row.get('life_threatening', False)
            
            for med in co_meds:
                med_upper = str(med).upper()
                
                # Track individual drugs
                if is_severe:
                    drug_counts_crs[med_upper] += 1
                else:
                    drug_counts_non_crs[med_upper] += 1
                
                # Classify into drug classes
                for cls, drugs in DRUG_CLASSES.items():
                    if any(d in med_upper for d in drugs):
                        drug_class_counts[cls]['all'] += 1
                        if is_severe:
                            drug_class_counts[cls]['crs'] += 1
                        else:
                            drug_class_counts[cls]['non_crs'] += 1
                        break
        
        # Summary statistics
        report.append("\n1. CO-MEDICATION COUNTS")
        report.append("-" * 60)
        
        n_co_meds_stats = self.df['n_co_medications'].describe()
        report.append(f"  Mean co-medications: {n_co_meds_stats['mean']:.1f}")
        report.append(f"  Median: {n_co_meds_stats['50%']:.0f}")
        report.append(f"  Range: {n_co_meds_stats['min']:.0f} - {n_co_meds_stats['max']:.0f}")
        
        # Drug class analysis
        report.append("\n\n2. DRUG CLASS USAGE: SEVERE CRS vs NON-SEVERE")
        report.append("-" * 70)
        report.append(f"{'Drug Class':<20} {'Severe CRS':>12} {'Non-Severe':>12} {'Difference':>12}")
        report.append("-" * 70)
        
        total_severe = self.df['death'].sum() + self.df['life_threatening'].sum()
        total_non_severe = len(self.df) - total_severe
        
        class_diffs = []
        for cls, counts in drug_class_counts.items():
            if counts['all'] > 0:
                crs_pct = counts['crs'] / max(total_severe, 1) * 100
                non_crs_pct = counts['non_crs'] / max(total_non_severe, 1) * 100
                diff = crs_pct - non_crs_pct
                class_diffs.append((cls, crs_pct, non_crs_pct, diff))
        
        # Sort by absolute difference
        class_diffs.sort(key=lambda x: -abs(x[3]))
        
        for cls, crs_pct, non_crs_pct, diff in class_diffs:
            sign = '+' if diff > 0 else ''
            report.append(f"{cls:<20} {crs_pct:>11.1f}% {non_crs_pct:>11.1f}% {sign}{diff:>11.1f}%")
        
        # Top individual drugs
        report.append("\n\n3. TOP CO-MEDICATIONS IN SEVERE CRS CASES")
        report.append("-" * 60)
        
        for drug, count in drug_counts_crs.most_common(10):
            non_crs_count = drug_counts_non_crs.get(drug, 0)
            report.append(f"  {drug}: {count} severe cases, {non_crs_count} non-severe")
        
        # Example table as requested
        report.append("\n\n4. USAGE DIFFERENCES TABLE")
        report.append("-" * 70)
        report.append(f"{'Drug':<25} {'Severe CRS':>15} {'Non-Severe':>15} {'Difference':>12}")
        report.append("-" * 70)
        
        # Calculate percentages for top drugs
        example_drugs = ['Steroids', 'Anti-CD20', 'IL-6 Inhibitors', 'Chemotherapy', 'Antiemetics']
        for cls in example_drugs:
            if cls in drug_class_counts:
                counts = drug_class_counts[cls]
                crs_pct = counts['crs'] / max(total_severe, 1) * 100
                non_crs_pct = counts['non_crs'] / max(total_non_severe, 1) * 100
                diff = crs_pct - non_crs_pct
                sign = '+' if diff > 0 else ''
                report.append(f"{cls:<25} {crs_pct:>14.0f}% {non_crs_pct:>14.0f}% {sign}{diff:>11.0f}%")
        
        return "\n".join(report)
    
    def generate_database_specific_analysis(self) -> str:
        """Generate database-specific analyses."""
        if self.df is None:
            self.load_data()
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("DATABASE-SPECIFIC ANALYSES")
        report.append("=" * 80)
        
        report.append("\nWhy results may differ across databases:")
        report.append("-" * 60)
        report.append("""
  • Regulatory reporting standards vary by region
  • FAERS: Voluntary reporting (US) - may underreport mild cases
  • JADER: Mandatory for serious cases (Japan) - higher severity proportion
  • EudraVigilance: Mix of mandatory/voluntary (EU) - variable completeness
  • Cultural differences in healthcare-seeking behavior
  • Different approval timelines affect case accumulation
  • Language barriers may affect narrative detail (JADER in Japanese)
""")
        
        # Detailed breakdown by source
        for source in self.df['source'].unique():
            source_df = self.df[self.df['source'] == source]
            
            report.append(f"\n{'='*60}")
            report.append(f"{source.upper()} SUMMARY")
            report.append("="*60)
            
            report.append(f"\nTotal Records: {len(source_df)}")
            report.append(f"CRS Cases: {source_df['is_crs'].sum()}")
            
            # Demographics
            report.append(f"\nDemographics:")
            if source_df['age'].notna().any():
                report.append(f"  Age: mean={source_df['age'].mean():.1f}, "
                            f"median={source_df['age'].median():.1f}")
            
            sex_dist = source_df['sex'].value_counts(normalize=True) * 100
            for sex, pct in sex_dist.items():
                report.append(f"  {sex}: {pct:.1f}%")
            
            # Outcomes
            report.append(f"\nOutcomes:")
            outcome_dist = source_df['crs_outcome'].value_counts()
            for outcome, count in outcome_dist.items():
                pct = count / len(source_df) * 100
                report.append(f"  {outcome}: {count} ({pct:.1f}%)")
            
            # Seriousness
            report.append(f"\nSeriousness Criteria:")
            report.append(f"  Hospitalized: {source_df['hospitalized'].sum()} ({source_df['hospitalized'].mean()*100:.1f}%)")
            report.append(f"  Life-threatening: {source_df['life_threatening'].sum()} ({source_df['life_threatening'].mean()*100:.1f}%)")
            report.append(f"  Fatal: {source_df['death'].sum()} ({source_df['death'].mean()*100:.1f}%)")
        
        # Comparison table
        report.append("\n\n" + "="*80)
        report.append("CROSS-DATABASE COMPARISON")
        report.append("="*80)
        
        report.append(f"\n{'Metric':<25} {'FAERS':>15} {'JADER':>15} {'EV':>15}")
        report.append("-" * 70)
        
        metrics = [
            ('CRS Prevalence', lambda df: f"{df['is_crs'].mean()*100:.1f}%"),
            ('Mean Age', lambda df: f"{df['age'].mean():.1f}" if df['age'].notna().any() else 'N/A'),
            ('Male %', lambda df: f"{(df['sex']=='male').mean()*100:.1f}%"),
            ('Mortality Rate', lambda df: f"{df['death'].mean()*100:.1f}%"),
            ('Hospitalization %', lambda df: f"{df['hospitalized'].mean()*100:.1f}%"),
        ]
        
        sources = ['faers', 'jader', 'eudravigilance']
        
        for metric_name, metric_func in metrics:
            values = []
            for source in sources:
                source_df = self.df[self.df['source'] == source]
                if len(source_df) > 0:
                    values.append(metric_func(source_df))
                else:
                    values.append('N/A')
            
            report.append(f"{metric_name:<25} {values[0]:>15} {values[1]:>15} {values[2]:>15}")
        
        return "\n".join(report)
    
    def generate_full_report(self) -> str:
        """Generate complete data summary report."""
        report_parts = [
            self.generate_dataset_summary(),
            self.generate_missingness_table(),
            self.generate_variable_availability_table(),
            self.generate_polypharmacy_analysis(),
            self.generate_database_specific_analysis()
        ]
        
        return "\n\n".join(report_parts)


def print_variable_availability_table():
    """Print the variable availability table."""
    print("""
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    VARIABLE AVAILABILITY ACROSS DATABASES                            │
├──────────────────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
│ Variable                 │ FAERS (US)  │ JADER (JP)  │ EV (EU)     │ Notes          │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ DEMOGRAPHICS             │             │             │             │                │
│   Age                    │ Yes         │ Yes         │ Age Group   │ EV: bucketed   │
│   Sex                    │ Yes         │ Yes         │ Yes         │                │
│   Weight                 │ Limited     │ No          │ No          │ Often missing  │
│   Country                │ Yes         │ JP only     │ Yes         │                │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ CLINICAL                 │             │             │             │                │
│   Seriousness            │ Yes         │ Yes         │ Yes         │                │
│   Outcome                │ Yes         │ Yes         │ Yes         │                │
│   Hospitalization        │ Yes         │ Limited     │ Yes         │                │
│   Life-threatening       │ Yes         │ Limited     │ Yes         │                │
│   Indication             │ Yes         │ Yes         │ Limited     │                │
│   Disease Stage          │ No*         │ No          │ No          │ *In narratives │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ DRUG EXPOSURE            │             │             │             │                │
│   Dose (numeric)         │ Yes         │ Limited     │ Limited     │                │
│   Route                  │ Yes         │ Yes         │ Yes         │                │
│   Concomitant meds       │ Yes         │ Yes         │ Yes         │                │
├──────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
│ NLP FEATURES             │             │             │             │                │
│   Narrative text         │ Yes         │ No**        │ Limited     │ **Japanese     │
│   CRS grade              │ Yes*        │ No          │ Limited     │ *From narrative│
│   Time to onset          │ Yes*        │ No          │ Limited     │ *From narrative│
└──────────────────────────┴─────────────┴─────────────┴─────────────┴────────────────┘
""")


def main():
    """Generate and print data summary report."""
    summary = DatasetSummary("multi_source_crs_data.json")
    report = summary.generate_full_report()
    print(report)
    
    # Save report
    with open("data_summary_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to data_summary_report.txt")


if __name__ == "__main__":
    main()
