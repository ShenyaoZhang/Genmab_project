"""
Pipeline Configuration Module
Makes the CRS analysis pipeline scalable for any drug/adverse event combination.

Example Usage:
    from pipeline_config import run_pipeline, PipelineConfig
    
    # Run for any drug and adverse event
    run_pipeline(drug="epcoritamab", adverse_event="CRS")
    run_pipeline(drug="tafasitamab", adverse_event="ICANS")
    
    # Or use configuration object for more control
    config = PipelineConfig(
        drug="epcoritamab",
        adverse_event="cytokine release syndrome",
        data_sources=["faers", "eudravigilance", "jader"]
    )
    results = config.run()
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class PipelineConfig:
    """
    Configuration class for the pharmacovigilance analysis pipeline.
    
    Attributes:
        drug: Name of the drug to analyze (e.g., "epcoritamab", "tafasitamab")
        adverse_event: Adverse event of interest (e.g., "CRS", "ICANS", "neutropenia")
        data_sources: List of data sources to include ["faers", "eudravigilance", "jader"]
        output_dir: Directory for output files
        faers_api_limit: Maximum records to fetch from FAERS API
    """
    drug: str = "epcoritamab"
    adverse_event: str = "cytokine release syndrome"
    data_sources: List[str] = field(default_factory=lambda: ["faers", "eudravigilance", "jader"])
    output_dir: str = "./output"
    faers_api_limit: int = 100
    
    # Drug name variations for searching
    drug_aliases: Dict[str, List[str]] = field(default_factory=dict)
    
    # Adverse event variations
    ae_aliases: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize drug and AE aliases."""
        # Default drug aliases (expandable)
        default_drug_aliases = {
            "epcoritamab": ["epcoritamab", "tepkinly", "エプコリタマブ", "エプキンリ"],
            "tafasitamab": ["tafasitamab", "monjuvi", "minjuvi"],
            "glofitamab": ["glofitamab", "columvi"],
            "mosunetuzumab": ["mosunetuzumab", "lunsumio"],
        }
        
        # Default AE aliases
        default_ae_aliases = {
            "cytokine release syndrome": ["cytokine release syndrome", "crs", "サイトカイン放出症候群"],
            "icans": ["icans", "immune effector cell-associated neurotoxicity syndrome", "neurotoxicity"],
            "neutropenia": ["neutropenia", "neutropenic", "好中球減少"],
            "thrombocytopenia": ["thrombocytopenia", "thrombocytopenic", "血小板減少"],
            "infection": ["infection", "sepsis", "pneumonia", "感染"],
        }
        
        if not self.drug_aliases:
            self.drug_aliases = default_drug_aliases
        if not self.ae_aliases:
            self.ae_aliases = default_ae_aliases
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_drug_search_terms(self) -> List[str]:
        """Get all search terms for the specified drug."""
        drug_lower = self.drug.lower()
        return self.drug_aliases.get(drug_lower, [drug_lower])
    
    def get_ae_search_terms(self) -> List[str]:
        """Get all search terms for the specified adverse event."""
        ae_lower = self.adverse_event.lower()
        return self.ae_aliases.get(ae_lower, [ae_lower])
    
    def build_faers_query(self) -> str:
        """Build FAERS API search query."""
        drug_terms = self.get_drug_search_terms()
        ae_terms = self.get_ae_search_terms()
        
        # Build drug part of query
        drug_query = " OR ".join([f'patient.drug.medicinalproduct:"{term}"' for term in drug_terms])
        
        # Build AE part of query
        ae_query = " OR ".join([f'patient.reaction.reactionmeddrapt:"{term}"' for term in ae_terms])
        
        return f"({drug_query}) AND ({ae_query})"


def run_pipeline(
    drug: str = "epcoritamab",
    adverse_event: str = "CRS",
    data_sources: Optional[List[str]] = None,
    output_dir: str = "./output",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete pharmacovigilance analysis pipeline for any drug/AE combination.
    
    This function is the main entry point for scalable analysis. It can be called
    for any drug and adverse event without modifying the underlying code.
    
    Args:
        drug: Name of the drug to analyze (e.g., "epcoritamab", "tafasitamab")
        adverse_event: Adverse event of interest (e.g., "CRS", "ICANS", "neutropenia")
        data_sources: List of data sources ["faers", "eudravigilance", "jader"]
        output_dir: Directory for output files
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary containing:
        - data: Combined dataset as list of records
        - summary: Dataset summary statistics
        - causal_results: Causal analysis results
        - nlp_results: NLP analysis results (if narratives available)
        - signal_detection: Rare/unexpected AE detection results
    
    Example:
        >>> # Analyze CRS for epcoritamab
        >>> results = run_pipeline(drug="epcoritamab", adverse_event="CRS")
        
        >>> # Analyze ICANS for tafasitamab
        >>> results = run_pipeline(drug="tafasitamab", adverse_event="ICANS")
        
        >>> # Check neutropenia signal
        >>> results = run_pipeline(drug="epcoritamab", adverse_event="neutropenia")
    """
    if data_sources is None:
        data_sources = ["faers", "eudravigilance", "jader"]
    
    # Normalize AE name
    ae_mapping = {
        "crs": "cytokine release syndrome",
        "icans": "immune effector cell-associated neurotoxicity syndrome",
    }
    ae_normalized = ae_mapping.get(adverse_event.lower(), adverse_event.lower())
    
    # Create configuration
    config = PipelineConfig(
        drug=drug,
        adverse_event=ae_normalized,
        data_sources=data_sources,
        output_dir=output_dir
    )
    
    results = {
        "config": {
            "drug": drug,
            "adverse_event": adverse_event,
            "data_sources": data_sources
        },
        "data": [],
        "summary": {},
        "causal_results": {},
        "nlp_results": {},
        "signal_detection": {}
    }
    
    if verbose:
        print("=" * 70)
        print(f"PHARMACOVIGILANCE ANALYSIS PIPELINE")
        print(f"Drug: {drug}")
        print(f"Adverse Event: {adverse_event}")
        print(f"Data Sources: {', '.join(data_sources)}")
        print("=" * 70)
    
    # =========================================================================
    # STEP 1: DATA EXTRACTION
    # =========================================================================
    if verbose:
        print("\n[STEP 1] Data Extraction")
        print("-" * 40)
    
    combined_data = []
    source_counts = {}
    
    # Extract from FAERS
    if "faers" in data_sources:
        faers_data = _extract_faers_data(config, verbose)
        combined_data.extend(faers_data)
        source_counts["faers"] = len(faers_data)
    
    # Extract from Eudravigilance
    if "eudravigilance" in data_sources:
        eu_data = _extract_eudravigilance_data(config, verbose)
        combined_data.extend(eu_data)
        source_counts["eudravigilance"] = len(eu_data)
    
    # Extract from JADER
    if "jader" in data_sources:
        jader_data = _extract_jader_data(config, verbose)
        combined_data.extend(jader_data)
        source_counts["jader"] = len(jader_data)
    
    results["data"] = combined_data
    results["summary"]["source_counts"] = source_counts
    results["summary"]["total_records"] = len(combined_data)
    
    if verbose:
        print(f"\nTotal records extracted: {len(combined_data)}")
        for source, count in source_counts.items():
            print(f"  - {source}: {count}")
    
    # Save extracted data
    output_file = os.path.join(output_dir, f"{drug}_{adverse_event.replace(' ', '_')}_data.json")
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    if len(combined_data) == 0:
        if verbose:
            print("\nNo data found. Pipeline complete.")
        return results
    
    # =========================================================================
    # STEP 2: DATA SUMMARY
    # =========================================================================
    if verbose:
        print("\n[STEP 2] Data Summary")
        print("-" * 40)
    
    results["summary"].update(_generate_data_summary(combined_data))
    
    if verbose:
        summary = results["summary"]
        print(f"Age: mean={summary.get('age_mean', 'N/A'):.1f}, missing={summary.get('age_missing_pct', 0):.1f}%")
        print(f"Sex: {summary.get('sex_distribution', {})}")
        print(f"Outcomes: {summary.get('outcome_distribution', {})}")
    
    # =========================================================================
    # STEP 3: CAUSAL ANALYSIS
    # =========================================================================
    if verbose:
        print("\n[STEP 3] Causal Analysis")
        print("-" * 40)
    
    try:
        from causal_analysis import CRSCausalAnalyzer
        
        # Save temp file for analyzer
        temp_file = os.path.join(output_dir, "temp_analysis_data.json")
        with open(temp_file, 'w') as f:
            json.dump(combined_data, f, default=str)
        
        analyzer = CRSCausalAnalyzer(temp_file)
        analyzer.load_data()
        
        # Run analyses
        results["causal_results"]["associations"] = analyzer.analyze_associations().to_dict('records')
        results["causal_results"]["propensity_score"] = analyzer.propensity_score_analysis()
        results["causal_results"]["sensitivity"] = analyzer.sensitivity_analysis()
        
        if verbose:
            print("Causal analysis complete.")
        
        # Clean up temp file
        os.remove(temp_file)
        
    except Exception as e:
        if verbose:
            print(f"Causal analysis error: {e}")
        results["causal_results"]["error"] = str(e)
    
    # =========================================================================
    # STEP 4: SIGNAL DETECTION
    # =========================================================================
    if verbose:
        print("\n[STEP 4] Signal Detection")
        print("-" * 40)
    
    try:
        from signal_detection import SignalDetector
        
        detector = SignalDetector()
        signal_result = detector.check_signal(drug, adverse_event)
        results["signal_detection"] = signal_result
        
        if verbose:
            print(f"Signal status: {signal_result.get('status', 'unknown')}")
            print(f"Message: {signal_result.get('message', '')}")
            
    except ImportError:
        results["signal_detection"]["note"] = "Signal detection module not available"
    except Exception as e:
        results["signal_detection"]["error"] = str(e)
    
    # =========================================================================
    # SAVE FINAL RESULTS
    # =========================================================================
    results_file = os.path.join(output_dir, f"{drug}_{adverse_event.replace(' ', '_')}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print(f"Results saved to: {results_file}")
        print("=" * 70)
    
    return results


def _extract_faers_data(config: PipelineConfig, verbose: bool = True) -> List[Dict]:
    """Extract data from FAERS API."""
    import requests
    
    if verbose:
        print("  Extracting from FAERS...")
    
    try:
        query = config.build_faers_query()
        
        url = "https://api.fda.gov/drug/event.json"
        params = {
            "search": query,
            "limit": config.faers_api_limit
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            if verbose:
                print(f"    FAERS API returned status {response.status_code}")
            return []
        
        data = response.json()
        results = data.get("results", [])
        
        # Convert to unified format
        unified_records = []
        for record in results:
            unified = _convert_faers_record(record, config)
            if unified:
                unified_records.append(unified)
        
        if verbose:
            print(f"    Found {len(unified_records)} records")
        
        return unified_records
        
    except Exception as e:
        if verbose:
            print(f"    FAERS extraction error: {e}")
        return []


def _convert_faers_record(record: Dict, config: PipelineConfig) -> Optional[Dict]:
    """Convert FAERS record to unified format."""
    try:
        patient = record.get("patient", {})
        
        # Extract age
        age = None
        age_val = patient.get("patientonsetage")
        age_unit = patient.get("patientonsetageunit", "801")
        if age_val:
            try:
                age = float(age_val)
                # Convert units (801=years, 802=months, 803=weeks, 804=days)
                if age_unit == "802":
                    age = age / 12
                elif age_unit == "803":
                    age = age / 52
                elif age_unit == "804":
                    age = age / 365
            except:
                pass
        
        # Extract sex
        sex_code = patient.get("patientsex")
        sex = {"1": "male", "2": "female"}.get(sex_code, "unknown")
        
        # Extract weight
        weight = None
        try:
            weight = float(patient.get("patientweight")) if patient.get("patientweight") else None
        except:
            pass
        
        # Extract drugs
        drugs = patient.get("drug", [])
        co_medications = []
        doses = []
        indication = None
        
        drug_terms = [t.lower() for t in config.get_drug_search_terms()]
        
        for drug in drugs:
            drug_name = drug.get("medicinalproduct", "").upper()
            
            # Check if this is our target drug
            is_target = any(term in drug_name.lower() for term in drug_terms)
            
            if is_target:
                # Extract dose
                dose_mg = None
                try:
                    dose_mg = float(drug.get("drugstructuredosagenumb"))
                except:
                    pass
                
                if dose_mg:
                    doses.append({"dose_mg": dose_mg, "date": drug.get("drugstartdate")})
                
                indication = drug.get("drugindication")
            else:
                co_medications.append(drug_name)
        
        # Extract reactions
        reactions = patient.get("reaction", [])
        ae_outcome = "unknown"
        all_reactions = []
        
        ae_terms = [t.lower() for t in config.get_ae_search_terms()]
        
        for reaction in reactions:
            reaction_name = reaction.get("reactionmeddrapt", "")
            outcome_code = reaction.get("reactionoutcome")
            
            outcome_map = {
                "1": "recovered", "2": "recovering", "3": "not_recovered",
                "4": "recovered_with_sequelae", "5": "fatal", "6": "unknown"
            }
            outcome = outcome_map.get(outcome_code, "unknown")
            
            all_reactions.append({"name": reaction_name, "outcome": outcome})
            
            # Check if this is our target AE
            if any(term in reaction_name.lower() for term in ae_terms):
                ae_outcome = outcome
        
        return {
            "report_id": record.get("safetyreportid"),
            "source": "faers",
            "is_target_ae": True,
            "ae_outcome": ae_outcome,
            "serious": record.get("serious") == "1",
            "hospitalized": record.get("seriousnesshospitalization") == "1",
            "death": record.get("seriousnessdeath") == "1",
            "life_threatening": record.get("seriousnesslifethreatening") == "1",
            "target_drug_exposure": True,
            "doses": doses,
            "co_medications": co_medications,
            "indication": indication,
            "age": age,
            "sex": sex,
            "weight": weight,
            "country": record.get("occurcountry"),
            "receive_date": record.get("receivedate"),
            "all_reactions": all_reactions,
            "narrative_text": record.get("narrative", {}).get("text") if record.get("narrative") else None
        }
        
    except Exception as e:
        return None


def _extract_eudravigilance_data(config: PipelineConfig, verbose: bool = True) -> List[Dict]:
    """Extract data from Eudravigilance CSV (if available)."""
    if verbose:
        print("  Extracting from Eudravigilance...")
    
    # Check for CSV file
    possible_paths = [
        "../Run Line Listing Report.csv",
        "Run Line Listing Report.csv",
        f"../{config.drug}_eudravigilance.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                from data_extractors import EudravigilanceExtractor
                
                extractor = EudravigilanceExtractor()
                df = extractor.parse_line_listing(path)
                
                # Filter for target AE
                ae_terms = config.get_ae_search_terms()
                
                for col in df.columns:
                    if 'reaction' in col.lower():
                        mask = df[col].str.contains('|'.join(ae_terms), case=False, na=False)
                        df = df[mask]
                        break
                
                records = extractor.to_unified_format(df)
                
                if verbose:
                    print(f"    Found {len(records)} records from {path}")
                
                return records
                
            except Exception as e:
                if verbose:
                    print(f"    Error processing {path}: {e}")
    
    if verbose:
        print("    No Eudravigilance data file found")
    
    return []


def _extract_jader_data(config: PipelineConfig, verbose: bool = True) -> List[Dict]:
    """Extract data from JADER CSV files (if available)."""
    if verbose:
        print("  Extracting from JADER...")
    
    jader_dirs = ["../jader_data", "jader_data"]
    
    for jader_dir in jader_dirs:
        if os.path.exists(jader_dir):
            try:
                from data_extractors import JADERExtractor
                
                extractor = JADERExtractor(jader_dir)
                data = extractor.load_jader_data()
                records = extractor.build_unified_dataset(data)
                
                if verbose:
                    print(f"    Found {len(records)} records from {jader_dir}")
                
                return records
                
            except Exception as e:
                if verbose:
                    print(f"    Error processing {jader_dir}: {e}")
    
    if verbose:
        print("    No JADER data directory found")
    
    return []


def _generate_data_summary(data: List[Dict]) -> Dict:
    """Generate summary statistics for the dataset."""
    df = pd.DataFrame(data)
    
    summary = {}
    
    # Age statistics
    if 'age' in df.columns:
        age_valid = df['age'].dropna()
        summary['age_mean'] = float(age_valid.mean()) if len(age_valid) > 0 else None
        summary['age_std'] = float(age_valid.std()) if len(age_valid) > 0 else None
        summary['age_missing_pct'] = float(df['age'].isna().mean() * 100)
    
    # Sex distribution
    if 'sex' in df.columns:
        summary['sex_distribution'] = df['sex'].value_counts().to_dict()
    
    # Outcome distribution
    outcome_col = 'ae_outcome' if 'ae_outcome' in df.columns else 'crs_outcome'
    if outcome_col in df.columns:
        summary['outcome_distribution'] = df[outcome_col].value_counts().to_dict()
    
    # Source distribution
    if 'source' in df.columns:
        summary['source_distribution'] = df['source'].value_counts().to_dict()
    
    # Missingness summary
    summary['missingness'] = {
        col: float(df[col].isna().mean() * 100) 
        for col in ['age', 'sex', 'weight', 'indication'] 
        if col in df.columns
    }
    
    return summary


# =============================================================================
# FLOW DIAGRAM (for documentation/slides)
# =============================================================================
PIPELINE_FLOW_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHARMACOVIGILANCE ANALYSIS PIPELINE                       │
│                                                                              │
│  Example: run_pipeline(drug="epcoritamab", adverse_event="CRS")             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: INPUT PARAMETERS                                                    │
│  ─────────────────────────                                                   │
│  • drug: "epcoritamab" → aliases: ["epcoritamab", "tepkinly", "エプコリタマブ"]│
│  • adverse_event: "CRS" → aliases: ["cytokine release syndrome", "crs"]     │
│  • data_sources: ["faers", "eudravigilance", "jader"]                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATA EXTRACTION                                                     │
│  ───────────────────────                                                     │
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │    FAERS     │   │ Eudravigilance│   │    JADER    │                     │
│  │   (US API)   │   │  (EU CSV)    │   │   (JP CSV)  │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                              │
│         └──────────────────┼──────────────────┘                              │
│                            ▼                                                 │
│                   ┌────────────────┐                                         │
│                   │ Unified Format │                                         │
│                   │   (JSON)       │                                         │
│                   └────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: FEATURE EXTRACTION                                                  │
│  ──────────────────────────                                                  │
│                                                                              │
│  Demographics: age, sex, weight, country                                     │
│  Clinical: seriousness, outcome, hospitalization                             │
│  Drug exposure: doses, co-medications, indication                            │
│  NLP features: severity indicators from narratives (FAERS only)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ANALYSIS                                                            │
│  ───────────────                                                             │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │    Causal      │  │    Signal      │  │      NLP       │                 │
│  │   Analysis     │  │   Detection    │  │   Analysis     │                 │
│  │                │  │                │  │                │                 │
│  │ • Associations │  │ • Rare AE      │  │ • Severity     │                 │
│  │ • Propensity   │  │   detection    │  │   prediction   │                 │
│  │   scores       │  │ • Expected vs  │  │ • Feature      │                 │
│  │ • E-values     │  │   unexpected   │  │   extraction   │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: OUTPUT                                                              │
│  ──────────────                                                              │
│                                                                              │
│  • {drug}_{ae}_data.json     - Combined dataset                             │
│  • {drug}_{ae}_results.json  - Analysis results                             │
│  • Summary statistics, risk factors, signal status                           │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
KEY POINT: Changing the drug or AE requires NO code changes - just pass 
different parameters to run_pipeline()
═══════════════════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    # Example usage demonstrations
    print(PIPELINE_FLOW_DIAGRAM)
    
    print("\n" + "=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    
    print("""
    # Run for epcoritamab + CRS (default)
    results = run_pipeline(drug="epcoritamab", adverse_event="CRS")
    
    # Run for tafasitamab + ICANS
    results = run_pipeline(drug="tafasitamab", adverse_event="ICANS")
    
    # Run for epcoritamab + neutropenia (to check for unexpected signals)
    results = run_pipeline(drug="epcoritamab", adverse_event="neutropenia")
    
    # Use only FAERS data
    results = run_pipeline(
        drug="epcoritamab", 
        adverse_event="CRS",
        data_sources=["faers"]
    )
    """)
    
    # Run actual demo
    print("\n" + "=" * 70)
    print("RUNNING DEMO: epcoritamab + CRS")
    print("=" * 70)
    
    results = run_pipeline(
        drug="epcoritamab",
        adverse_event="CRS",
        output_dir="./output",
        verbose=True
    )

