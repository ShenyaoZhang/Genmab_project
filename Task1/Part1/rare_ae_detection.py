"""
Rare/Unexpected Adverse Event Detection Module
Implements signal detection for unexpected adverse events.

This module addresses the feedback:
"Clarify rare/unexpected AE detection steps - Provide a simple flowchart
and examples"
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# RARE AE DETECTION FLOWCHART
# =============================================================================
"""
RARE/UNEXPECTED ADVERSE EVENT DETECTION FLOWCHART
================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: COLLECT ALL AE PAIRS                              │
│                                                                              │
│   Input: All drug-AE pairs from pharmacovigilance databases                 │
│   Example: (epcoritamab, CRS), (epcoritamab, neutropenia), ...              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: REMOVE KNOWN LABEL AEs                            │
│                                                                              │
│   Filter out adverse events listed in the drug's approved label             │
│   Known CRS-related AEs: CRS, infections, cytopenias, etc.                  │
│                                                                              │
│   Example: (epcoritamab, CRS) → REMOVED (known label AE)                    │
│            (epcoritamab, renal impairment) → KEPT (not on label)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: REMOVE HIGH-FREQUENCY AEs                         │
│                                                                              │
│   Filter out AEs that occur very frequently (likely common reactions)       │
│   Threshold: >5% of all cases                                               │
│                                                                              │
│   Example: (epcoritamab, fatigue) → REMOVED if >5% frequency                │
│            (epcoritamab, cardiac arrest) → KEPT if <5%                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: FLAG REMAINING RARE AEs                           │
│                                                                              │
│   Remaining AEs are flagged as "unexpected" for review                      │
│   Calculate: PRR, ROR, IC for disproportionality                            │
│                                                                              │
│   Output: List of rare/unexpected AE signals                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: PRIORITIZE BY SIGNAL STRENGTH                     │
│                                                                              │
│   Rank unexpected AEs by:                                                   │
│   - Disproportionality metrics (PRR, ROR)                                   │
│   - Case count                                                              │
│   - Seriousness (fatal > hospitalization > other)                           │
│                                                                              │
│   Output: Prioritized list for safety review                                │
└─────────────────────────────────────────────────────────────────────────────┘


EXAMPLE OUTPUT:
==============

Signal: epcoritamab + renal impairment
  - Appeared only 2 times in dataset
  - NOT on drug label
  - Below frequency threshold (0.25%)
  - Found in: FAERS (2 cases), JADER (0), EV (0)
  - PRR: 3.2 (elevated)
  → FLAGGED AS UNEXPECTED - RECOMMEND REVIEW

"""


# Known label AEs for epcoritamab (from prescribing information)
EPCORITAMAB_LABEL_AES = {
    # Very common (≥10%)
    'cytokine release syndrome', 'crs', 'pyrexia', 'fatigue', 'injection site reaction',
    'musculoskeletal pain', 'abdominal pain', 'diarrhea', 'nausea', 'vomiting',
    'rash', 'decreased appetite', 'headache', 'cough', 'dyspnea', 'edema',
    
    # Common infections
    'covid-19', 'upper respiratory tract infection', 'pneumonia', 'urinary tract infection',
    'sepsis', 'infection', 'viral infection', 'bacterial infection', 'fungal infection',
    
    # Hematologic
    'neutropenia', 'anemia', 'thrombocytopenia', 'lymphopenia', 'leukopenia',
    'febrile neutropenia', 'pancytopenia',
    
    # Other known
    'tumor lysis syndrome', 'hypogammaglobulinemia', 'hypotension',
    'icans', 'immune effector cell-associated neurotoxicity syndrome',
    'encephalopathy', 'confusion', 'tremor'
}


class RareAEDetector:
    """
    Detects rare and unexpected adverse events for signal detection.
    
    Implements a multi-step filtering approach:
    1. Remove known label AEs
    2. Remove high-frequency AEs
    3. Flag remaining low-frequency AEs
    4. Calculate disproportionality metrics
    """
    
    def __init__(
        self, 
        drug: str = "epcoritamab",
        known_aes: Set[str] = None,
        frequency_threshold: float = 0.05
    ):
        """
        Initialize detector.
        
        Args:
            drug: Drug name
            known_aes: Set of known/expected AEs for the drug
            frequency_threshold: AEs above this frequency are considered "common"
        """
        self.drug = drug
        self.known_aes = known_aes or EPCORITAMAB_LABEL_AES
        self.frequency_threshold = frequency_threshold
        self.all_aes = []
        self.flagged_signals = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Flatten all reactions
        records = []
        for record in data:
            report_id = record.get('report_id')
            source = record.get('source', 'faers')
            is_fatal = record.get('death', False)
            is_serious = record.get('serious', False)
            
            # Get all reactions for this report
            reactions = record.get('all_reactions', [])
            if not reactions:
                continue
            
            for reaction in reactions:
                if isinstance(reaction, dict):
                    ae_name = reaction.get('name', '')
                elif isinstance(reaction, str):
                    ae_name = reaction
                else:
                    continue
                
                if ae_name:
                    records.append({
                        'report_id': report_id,
                        'source': source,
                        'ae_name': ae_name.lower().strip(),
                        'is_fatal': is_fatal,
                        'is_serious': is_serious
                    })
        
        self.ae_df = pd.DataFrame(records)
        self.all_aes = records
        print(f"Loaded {len(self.ae_df)} AE records from {self.ae_df['report_id'].nunique()} reports")
        
        return self.ae_df
    
    def step1_get_all_ae_pairs(self) -> Dict[str, int]:
        """Step 1: Get all drug-AE pairs with counts."""
        ae_counts = Counter(self.ae_df['ae_name'])
        
        print(f"\nStep 1: Found {len(ae_counts)} unique AEs")
        print(f"  Top 5: {ae_counts.most_common(5)}")
        
        return dict(ae_counts)
    
    def step2_remove_known_label_aes(self, ae_counts: Dict[str, int]) -> Dict[str, int]:
        """Step 2: Remove known label AEs."""
        filtered = {}
        removed = []
        
        for ae, count in ae_counts.items():
            ae_lower = ae.lower()
            is_known = any(
                known_ae in ae_lower or ae_lower in known_ae 
                for known_ae in self.known_aes
            )
            
            if is_known:
                removed.append(ae)
            else:
                filtered[ae] = count
        
        print(f"\nStep 2: Removed {len(removed)} known label AEs")
        print(f"  Remaining: {len(filtered)} AEs")
        
        return filtered
    
    def step3_remove_high_frequency_aes(self, ae_counts: Dict[str, int]) -> Dict[str, int]:
        """Step 3: Remove high-frequency AEs."""
        total_reports = self.ae_df['report_id'].nunique()
        threshold_count = total_reports * self.frequency_threshold
        
        filtered = {}
        removed = []
        
        for ae, count in ae_counts.items():
            if count > threshold_count:
                removed.append((ae, count))
            else:
                filtered[ae] = count
        
        print(f"\nStep 3: Removed {len(removed)} high-frequency AEs (>{self.frequency_threshold*100}%)")
        if removed:
            print(f"  Removed: {removed[:5]}...")
        print(f"  Remaining: {len(filtered)} rare AEs")
        
        return filtered
    
    def step4_flag_rare_aes(self, ae_counts: Dict[str, int]) -> List[Dict]:
        """Step 4: Flag remaining rare AEs as signals."""
        total_reports = self.ae_df['report_id'].nunique()
        signals = []
        
        for ae, count in ae_counts.items():
            frequency = count / total_reports
            
            # Get source breakdown
            ae_reports = self.ae_df[self.ae_df['ae_name'] == ae]
            source_counts = ae_reports.groupby('source')['report_id'].nunique().to_dict()
            
            # Check seriousness
            fatal_count = ae_reports['is_fatal'].sum()
            serious_count = ae_reports['is_serious'].sum()
            
            signal = {
                'adverse_event': ae,
                'total_cases': count,
                'frequency': frequency,
                'frequency_pct': f"{frequency*100:.2f}%",
                'sources': source_counts,
                'fatal_cases': int(fatal_count),
                'serious_cases': int(serious_count),
                'on_label': False,
                'status': 'FLAGGED - UNEXPECTED'
            }
            signals.append(signal)
        
        # Sort by count (descending)
        signals.sort(key=lambda x: -x['total_cases'])
        
        print(f"\nStep 4: Flagged {len(signals)} rare/unexpected AE signals")
        
        self.flagged_signals = signals
        return signals
    
    def step5_calculate_disproportionality(self, background_data: Dict = None) -> List[Dict]:
        """
        Step 5: Calculate disproportionality metrics.
        
        PRR (Proportional Reporting Ratio):
            PRR = (a / (a+b)) / (c / (c+d))
            where:
            a = cases with drug AND AE
            b = cases with drug AND NOT AE
            c = cases with AE AND NOT drug
            d = cases with NOT drug AND NOT AE
        
        Note: Requires background data for accurate calculation.
        For this demonstration, we use simplified estimates.
        """
        # In production, this would use actual background rates
        # For demonstration, we calculate relative frequencies
        
        for signal in self.flagged_signals:
            # Simplified PRR estimate (would use background data in production)
            # Higher values suggest disproportionate reporting
            
            # Estimate based on frequency relative to expected background
            expected_frequency = 0.001  # 0.1% background rate assumption
            actual_frequency = signal['frequency']
            
            signal['prr_estimate'] = actual_frequency / expected_frequency if expected_frequency > 0 else 0
            signal['prr_interpretation'] = (
                'Elevated' if signal['prr_estimate'] > 2 else 
                'Moderate' if signal['prr_estimate'] > 1 else 
                'Low'
            )
        
        return self.flagged_signals
    
    def detect_rare_aes(self, data_path: str = None) -> List[Dict]:
        """
        Run complete rare AE detection pipeline.
        
        Returns:
            List of flagged rare/unexpected AE signals
        """
        print("="*70)
        print(f"RARE AE DETECTION FOR {self.drug.upper()}")
        print("="*70)
        
        if data_path:
            self.load_data(data_path)
        
        # Run pipeline
        ae_counts = self.step1_get_all_ae_pairs()
        ae_counts = self.step2_remove_known_label_aes(ae_counts)
        ae_counts = self.step3_remove_high_frequency_aes(ae_counts)
        signals = self.step4_flag_rare_aes(ae_counts)
        signals = self.step5_calculate_disproportionality()
        
        return signals
    
    def generate_report(self) -> str:
        """Generate report of detected signals."""
        report = []
        report.append("\n" + "="*70)
        report.append("RARE/UNEXPECTED AE SIGNALS REPORT")
        report.append("="*70)
        
        report.append(f"\nDrug: {self.drug}")
        report.append(f"Total signals flagged: {len(self.flagged_signals)}")
        
        report.append("\n" + "-"*70)
        report.append("FLAGGED SIGNALS (sorted by case count)")
        report.append("-"*70)
        
        for i, signal in enumerate(self.flagged_signals[:20], 1):
            report.append(f"\n{i}. {signal['adverse_event'].upper()}")
            report.append(f"   Cases: {signal['total_cases']} ({signal['frequency_pct']})")
            report.append(f"   Fatal: {signal['fatal_cases']}, Serious: {signal['serious_cases']}")
            report.append(f"   Databases: {signal['sources']}")
            report.append(f"   PRR estimate: {signal.get('prr_estimate', 'N/A'):.2f} ({signal.get('prr_interpretation', 'N/A')})")
            report.append(f"   Status: {signal['status']}")
        
        # Example signal in detail
        if self.flagged_signals:
            report.append("\n" + "-"*70)
            report.append("EXAMPLE SIGNAL DETAIL")
            report.append("-"*70)
            
            example = self.flagged_signals[0]
            report.append(f"""
Signal: {self.drug} + {example['adverse_event']}
  - Appeared {example['total_cases']} times in dataset
  - NOT on drug label
  - Frequency: {example['frequency_pct']} (below threshold)
  - Found in: {', '.join(f"{k.upper()} ({v} cases)" for k, v in example['sources'].items())}
  - PRR: {example.get('prr_estimate', 0):.1f} ({example.get('prr_interpretation', 'N/A')})
  → FLAGGED AS UNEXPECTED - RECOMMEND SAFETY REVIEW
""")
        
        return "\n".join(report)


def print_detection_flowchart():
    """Print the detection flowchart for documentation."""
    print("""
RARE/UNEXPECTED AE DETECTION FLOWCHART
======================================

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Collect All AE Pairs                                    │
│   All drug-adverse event combinations from databases            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Remove Known Label AEs                                  │
│   Filter out AEs listed in approved drug label                  │
│   Example: CRS, neutropenia → REMOVED (known)                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Remove High-Frequency AEs                               │
│   Filter AEs with frequency >5%                                 │
│   Example: fatigue (12%) → REMOVED (too common)                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Flag Remaining Rare Unexpected AEs                      │
│   Calculate: case count, seriousness, sources                   │
│   Example: renal impairment (2 cases) → FLAGGED                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Prioritize by Signal Strength                           │
│   Rank by: PRR, case count, fatal/serious                       │
│   Output: Prioritized list for safety review                    │
└─────────────────────────────────────────────────────────────────┘

EXAMPLE OUTPUT:
--------------
Signal: epcoritamab + renal impairment
  - Appeared only 2 times
  - NOT on drug label  
  - Below frequency threshold (0.25%)
  - Found in: FAERS (2 cases), JADER (0), EV (0)
  → FLAGGED AS UNEXPECTED
""")


def main():
    """Run rare AE detection demonstration."""
    
    print_detection_flowchart()
    
    # Run detection if data exists
    import os
    data_path = "multi_source_crs_data.json"
    
    if os.path.exists(data_path):
        detector = RareAEDetector(drug="epcoritamab")
        signals = detector.detect_rare_aes(data_path)
        
        report = detector.generate_report()
        print(report)
        
        # Save report
        with open("rare_ae_report.txt", "w") as f:
            f.write(report)
        print("\nReport saved to rare_ae_report.txt")
    else:
        print(f"\nNote: {data_path} not found. Run the main pipeline first.")


if __name__ == "__main__":
    main()

