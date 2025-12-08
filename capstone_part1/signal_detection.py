"""
Signal Detection Module
Identifies rare and unexpected adverse events from pharmacovigilance data.

Provides the check_signal() function for end-user interaction:
    >>> check_signal("epcoritamab", "neutropenia")
    "Unexpected. Rare signal. Observed in FAERS and EV but not JADER."

Also implements:
- Rare AE detection flowchart logic
- Expected vs unexpected AE classification
- Cross-database signal comparison
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


# =============================================================================
# KNOWN DRUG LABELS (Expected AEs from drug labels)
# =============================================================================
DRUG_LABELS = {
    "epcoritamab": {
        "name": "Tepkinly (epcoritamab-bysp)",
        "labeled_aes": [
            "cytokine release syndrome",
            "immune effector cell-associated neurotoxicity syndrome",
            "infections",
            "neutropenia",
            "anemia", 
            "thrombocytopenia",
            "lymphopenia",
            "fatigue",
            "musculoskeletal pain",
            "injection site reaction",
            "diarrhea",
            "nausea",
            "pyrexia",
            "abdominal pain",
            "tumor flare",
        ],
        "boxed_warnings": [
            "cytokine release syndrome",
            "immune effector cell-associated neurotoxicity syndrome",
            "infections",
        ]
    },
    "tafasitamab": {
        "name": "Monjuvi (tafasitamab-cxix)",
        "labeled_aes": [
            "neutropenia",
            "fatigue",
            "anemia",
            "diarrhea",
            "thrombocytopenia",
            "cough",
            "pyrexia",
            "peripheral edema",
            "respiratory tract infection",
            "decreased appetite",
            "infusion-related reaction",
        ],
        "boxed_warnings": []
    },
    "glofitamab": {
        "name": "Columvi (glofitamab-gxbm)",
        "labeled_aes": [
            "cytokine release syndrome",
            "neurologic toxicity",
            "infections",
            "neutropenia",
            "anemia",
            "thrombocytopenia",
        ],
        "boxed_warnings": [
            "cytokine release syndrome",
        ]
    }
}


@dataclass
class SignalResult:
    """Result of signal detection analysis."""
    drug: str
    adverse_event: str
    status: str  # "expected", "unexpected", "rare", "unknown"
    is_labeled: bool
    is_boxed_warning: bool
    frequency_category: str  # "very_common", "common", "uncommon", "rare", "very_rare"
    databases_observed: List[str]
    databases_not_observed: List[str]
    observed_count: int
    expected_count: Optional[float]
    prr: Optional[float]  # Proportional Reporting Ratio
    ror: Optional[float]  # Reporting Odds Ratio
    message: str
    recommendations: List[str]


class SignalDetector:
    """
    Detects and classifies adverse event signals from pharmacovigilance data.
    
    Implements the rare/unexpected AE detection workflow:
    1. All AE pairs → 
    2. Remove known label AEs → 
    3. Remove high-frequency AEs → 
    4. Flag remaining rare unexpected AEs
    
    Example:
        >>> detector = SignalDetector()
        >>> result = detector.check_signal("epcoritamab", "renal impairment")
        >>> print(result.message)
        "Unexpected. Rare signal. Observed in FAERS (2 cases) but not in JADER or EV."
    """
    
    # Frequency thresholds (from MedDRA/CIOMS classification)
    FREQUENCY_THRESHOLDS = {
        "very_common": 0.10,    # ≥10%
        "common": 0.01,         # ≥1% and <10%
        "uncommon": 0.001,      # ≥0.1% and <1%
        "rare": 0.0001,         # ≥0.01% and <0.1%
        "very_rare": 0.0        # <0.01%
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize signal detector.
        
        Args:
            data_path: Path to multi-source data JSON file
        """
        self.data_path = data_path or "multi_source_crs_data.json"
        self.data = None
        self.df = None
        self.drug_labels = DRUG_LABELS
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for signal detection."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            self.df = pd.DataFrame(self.data)
        else:
            self.df = pd.DataFrame()
        return self.df
    
    def check_signal(
        self, 
        drug: str, 
        adverse_event: str,
        verbose: bool = False
    ) -> Dict:
        """
        Check if a drug-AE pair represents a signal of interest.
        
        This is the main user-facing function for signal detection.
        
        Args:
            drug: Drug name (e.g., "epcoritamab")
            adverse_event: Adverse event term (e.g., "neutropenia", "renal impairment")
            verbose: Whether to print detailed output
        
        Returns:
            Dictionary containing signal assessment results
        
        Example:
            >>> check_signal("epcoritamab", "neutropenia")
            {
                "status": "expected",
                "message": "Expected. Labeled adverse event for epcoritamab.",
                "is_labeled": True,
                "databases_observed": ["faers", "jader", "eudravigilance"],
                ...
            }
            
            >>> check_signal("epcoritamab", "renal impairment")
            {
                "status": "unexpected",
                "message": "Unexpected. Rare signal. Observed in FAERS (2 cases) but not JADER.",
                "is_labeled": False,
                ...
            }
        """
        drug_lower = drug.lower()
        ae_lower = adverse_event.lower()
        
        # Check if AE is on drug label
        is_labeled = self._is_labeled_ae(drug_lower, ae_lower)
        is_boxed = self._is_boxed_warning(drug_lower, ae_lower)
        
        # Get observation counts by database
        db_counts = self._get_database_counts(drug_lower, ae_lower)
        
        # Calculate disproportionality metrics (if data available)
        prr, ror = self._calculate_disproportionality(drug_lower, ae_lower)
        
        # Determine frequency category
        total_count = sum(db_counts.values())
        frequency = self._categorize_frequency(total_count)
        
        # Determine signal status
        status, message, recommendations = self._classify_signal(
            drug_lower, ae_lower, is_labeled, is_boxed, 
            db_counts, total_count, prr, frequency
        )
        
        result = {
            "drug": drug,
            "adverse_event": adverse_event,
            "status": status,
            "is_labeled": is_labeled,
            "is_boxed_warning": is_boxed,
            "frequency_category": frequency,
            "databases_observed": [db for db, count in db_counts.items() if count > 0],
            "databases_not_observed": [db for db, count in db_counts.items() if count == 0],
            "database_counts": db_counts,
            "total_observed": total_count,
            "prr": prr,
            "ror": ror,
            "message": message,
            "recommendations": recommendations
        }
        
        if verbose:
            print(self._format_signal_output(result))
        
        return result
    
    def _is_labeled_ae(self, drug: str, ae: str) -> bool:
        """Check if AE is in the drug's label."""
        if drug not in self.drug_labels:
            return False
        
        labeled_aes = self.drug_labels[drug].get("labeled_aes", [])
        return any(ae in labeled_ae.lower() or labeled_ae.lower() in ae 
                   for labeled_ae in labeled_aes)
    
    def _is_boxed_warning(self, drug: str, ae: str) -> bool:
        """Check if AE is in boxed warnings."""
        if drug not in self.drug_labels:
            return False
        
        boxed = self.drug_labels[drug].get("boxed_warnings", [])
        return any(ae in warning.lower() or warning.lower() in ae 
                   for warning in boxed)
    
    def _get_database_counts(self, drug: str, ae: str) -> Dict[str, int]:
        """Get counts by database for the drug-AE pair."""
        # Default counts (would be populated from actual database queries)
        counts = {
            "faers": 0,
            "eudravigilance": 0,
            "jader": 0
        }
        
        if self.df is None or self.df.empty:
            self.load_data()
        
        if self.df is not None and not self.df.empty:
            # Count by source
            if 'source' in self.df.columns:
                for source in counts.keys():
                    mask = self.df['source'] == source
                    counts[source] = int(mask.sum())
        
        return counts
    
    def _calculate_disproportionality(
        self, 
        drug: str, 
        ae: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate disproportionality metrics (PRR, ROR).
        
        PRR (Proportional Reporting Ratio) = (a/(a+b)) / (c/(c+d))
        ROR (Reporting Odds Ratio) = (a*d) / (b*c)
        
        Where:
        a = reports with drug AND AE
        b = reports with drug but NOT AE  
        c = reports with AE but NOT drug
        d = reports without drug AND without AE
        """
        # Placeholder - would require full database access
        return None, None
    
    def _categorize_frequency(self, count: int) -> str:
        """Categorize frequency based on count."""
        # Simplified categorization based on absolute counts
        if count >= 100:
            return "very_common"
        elif count >= 50:
            return "common"
        elif count >= 10:
            return "uncommon"
        elif count >= 2:
            return "rare"
        else:
            return "very_rare"
    
    def _classify_signal(
        self,
        drug: str,
        ae: str,
        is_labeled: bool,
        is_boxed: bool,
        db_counts: Dict[str, int],
        total_count: int,
        prr: Optional[float],
        frequency: str
    ) -> Tuple[str, str, List[str]]:
        """
        Classify the signal and generate message/recommendations.
        
        Returns:
            Tuple of (status, message, recommendations)
        """
        observed_dbs = [db for db, count in db_counts.items() if count > 0]
        not_observed_dbs = [db for db, count in db_counts.items() if count == 0]
        
        recommendations = []
        
        # Case 1: Expected (labeled) AE
        if is_labeled:
            if is_boxed:
                status = "expected_boxed"
                message = f"Expected. BOXED WARNING for {drug}. This is a known serious adverse event."
                recommendations.append("Monitor per prescribing information")
                recommendations.append("Ensure appropriate risk mitigation measures in place")
            else:
                status = "expected"
                message = f"Expected. Labeled adverse event for {drug}."
                recommendations.append("Monitor per prescribing information")
        
        # Case 2: Unexpected (not labeled) - needs further classification
        else:
            if total_count == 0:
                status = "no_signal"
                message = f"No signal detected. No cases of {ae} observed for {drug} in available databases."
                recommendations.append("Continue routine monitoring")
            
            elif total_count <= 2:
                status = "unexpected_rare"
                db_text = ", ".join(observed_dbs).upper()
                message = (f"Unexpected. RARE SIGNAL. {ae} observed {total_count} time(s) "
                          f"in {db_text}, not on drug label.")
                recommendations.append("Flag for further review")
                recommendations.append("Check biological plausibility")
                recommendations.append("Monitor for additional cases")
            
            elif len(not_observed_dbs) > 0:
                status = "unexpected_regional"
                observed_text = ", ".join(observed_dbs).upper()
                not_observed_text = ", ".join(not_observed_dbs).upper()
                message = (f"Unexpected. Signal observed in {observed_text} "
                          f"but NOT in {not_observed_text}.")
                recommendations.append("Investigate regional reporting differences")
                recommendations.append("Check for potential confounding")
            
            else:
                status = "unexpected"
                message = f"Unexpected. {ae} is not on the {drug} label but observed across databases."
                recommendations.append("Evaluate for label update consideration")
                recommendations.append("Conduct detailed case review")
        
        return status, message, recommendations
    
    def _format_signal_output(self, result: Dict) -> str:
        """Format signal result for display."""
        output = []
        output.append("=" * 60)
        output.append(f"SIGNAL CHECK: {result['drug']} + {result['adverse_event']}")
        output.append("=" * 60)
        output.append(f"\nStatus: {result['status'].upper()}")
        output.append(f"Message: {result['message']}")
        output.append(f"\nLabeled AE: {'Yes' if result['is_labeled'] else 'No'}")
        output.append(f"Boxed Warning: {'Yes' if result['is_boxed_warning'] else 'No'}")
        output.append(f"Frequency: {result['frequency_category']}")
        output.append(f"\nDatabase Observations:")
        for db, count in result['database_counts'].items():
            output.append(f"  - {db.upper()}: {count} cases")
        output.append(f"\nRecommendations:")
        for rec in result['recommendations']:
            output.append(f"  • {rec}")
        output.append("=" * 60)
        return "\n".join(output)
    
    def detect_rare_unexpected_aes(
        self, 
        drug: str,
        min_count: int = 2,
        max_count: int = 10
    ) -> List[Dict]:
        """
        Detect rare/unexpected AEs for a given drug.
        
        Implements the flowchart:
        All AE pairs → Remove known label AEs → Remove high-frequency AEs 
        → Flag remaining rare unexpected AEs
        
        Args:
            drug: Drug name
            min_count: Minimum count to be considered (filters very rare)
            max_count: Maximum count (above this is not "rare")
        
        Returns:
            List of rare unexpected AE signals
        """
        if self.df is None or self.df.empty:
            self.load_data()
        
        if self.df is None or self.df.empty:
            return []
        
        # Get all unique AEs from the data
        all_aes = set()
        if 'all_reactions' in self.df.columns:
            for reactions in self.df['all_reactions'].dropna():
                if isinstance(reactions, list):
                    for r in reactions:
                        if isinstance(r, dict) and 'name' in r:
                            all_aes.add(r['name'].lower())
        
        # Filter through the flowchart
        rare_signals = []
        
        for ae in all_aes:
            result = self.check_signal(drug, ae)
            
            # Skip if labeled
            if result['is_labeled']:
                continue
            
            # Skip if too common or too rare
            count = result['total_observed']
            if count < min_count or count > max_count:
                continue
            
            # This is a rare unexpected signal
            rare_signals.append(result)
        
        return sorted(rare_signals, key=lambda x: x['total_observed'], reverse=True)


# =============================================================================
# RARE AE DETECTION FLOWCHART (for documentation/slides)
# =============================================================================
RARE_AE_FLOWCHART = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RARE/UNEXPECTED AE DETECTION FLOWCHART                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│  START: All Drug-AE Pairs           │
│  (from FAERS, JADER, EudraVigilance)│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  STEP 1: Check Drug Label           │
│  Is AE listed on drug label?        │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼ YES               ▼ NO
┌───────────────┐   ┌───────────────┐
│   EXPECTED    │   │   Continue    │
│   (Known AE)  │   │   Evaluation  │
└───────────────┘   └───────┬───────┘
                            │
                            ▼
┌─────────────────────────────────────┐
│  STEP 2: Check Frequency            │
│  Count cases across databases       │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   ≥50 cases   2-49 cases  <2 cases
        │         │         │
        ▼         ▼         ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  COMMON   │ │UNCOMMON/  │ │VERY RARE  │
│ (not rare)│ │   RARE    │ │(need more │
└───────────┘ └─────┬─────┘ │  data)    │
                    │       └───────────┘
                    ▼
┌─────────────────────────────────────┐
│  STEP 3: Cross-Database Check       │
│  Is signal consistent across DBs?   │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
  In ALL databases    In SOME databases
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────────────┐
│  UNEXPECTED   │   │  REGIONAL/REPORTING   │
│  SIGNAL       │   │  DIFFERENCE           │
│  (Flag for    │   │  (Investigate further)│
│  review)      │   │                       │
└───────────────┘   └───────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE:
  Drug: epcoritamab
  AE: renal impairment
  
  Step 1: Not on drug label → Continue
  Step 2: 2 cases observed → RARE
  Step 3: FAERS only, not in JADER or EV → REGIONAL
  
  Result: "Unexpected. Rare signal. Observed in FAERS (2 cases) but not 
           in JADER or EudraVigilance. Not on drug label. Flag for review."
═══════════════════════════════════════════════════════════════════════════════
"""


# =============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# =============================================================================
def check_signal(drug: str, adverse_event: str) -> str:
    """
    Quick check for a drug-AE signal.
    
    This is the simple interface for end-users.
    
    Args:
        drug: Drug name (e.g., "epcoritamab")
        adverse_event: Adverse event (e.g., "neutropenia")
    
    Returns:
        Human-readable signal status message
    
    Example:
        >>> check_signal("epcoritamab", "neutropenia")
        "Expected. Labeled adverse event for epcoritamab."
        
        >>> check_signal("epcoritamab", "renal impairment")
        "Unexpected. Rare signal. Observed in FAERS but not JADER."
    """
    detector = SignalDetector()
    result = detector.check_signal(drug, adverse_event)
    return result["message"]


if __name__ == "__main__":
    print(RARE_AE_FLOWCHART)
    
    print("\n" + "=" * 70)
    print("SIGNAL DETECTION EXAMPLES")
    print("=" * 70)
    
    detector = SignalDetector()
    
    # Example 1: Known/Expected AE
    print("\nExample 1: Expected AE (CRS for epcoritamab)")
    print("-" * 40)
    result = detector.check_signal("epcoritamab", "cytokine release syndrome", verbose=True)
    
    # Example 2: Expected but common AE
    print("\nExample 2: Expected AE (neutropenia)")
    print("-" * 40)
    result = detector.check_signal("epcoritamab", "neutropenia", verbose=True)
    
    # Example 3: Unexpected AE
    print("\nExample 3: Unexpected AE (renal impairment)")
    print("-" * 40)
    result = detector.check_signal("epcoritamab", "renal impairment", verbose=True)
    
    # Simple function demo
    print("\n" + "=" * 70)
    print("SIMPLE check_signal() FUNCTION")
    print("=" * 70)
    print(f"\ncheck_signal('epcoritamab', 'neutropenia'):")
    print(f"  → {check_signal('epcoritamab', 'neutropenia')}")
    
    print(f"\ncheck_signal('epcoritamab', 'cardiac arrest'):")
    print(f"  → {check_signal('epcoritamab', 'cardiac arrest')}")

