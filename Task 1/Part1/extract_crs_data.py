"""
FAERS Data Extractor for CRS Risk Modeling
Extracts structured, analysis-ready variables from FAERS safety reports
for the study of Cytokine Release Syndrome (CRS) caused by Epcoritamab.
"""

import json
import re
from datetime import datetime
from typing import Optional, List, Dict, Any


def normalize_drug_name(name: str) -> str:
    """Normalize drug names: remove dots, brackets, uppercase."""
    if not name:
        return None
    # Remove dots, brackets, and extra whitespace
    normalized = re.sub(r'[\.\[\]]', '', name)
    normalized = re.sub(r'\s+', ' ', normalized).strip().upper()
    return normalized


def is_epcoritamab(drug_name: str, active_substance: str = None) -> bool:
    """Check if drug is Epcoritamab."""
    name = normalize_drug_name(drug_name) or ""
    substance = normalize_drug_name(active_substance) or ""
    return "EPCORITAMAB" in name or "EPCORITAMAB" in substance


def is_crs_reaction(reaction_name: str) -> bool:
    """Check if reaction is Cytokine Release Syndrome."""
    if not reaction_name:
        return False
    normalized = reaction_name.upper()
    return "CYTOKINE RELEASE SYNDROME" in normalized or normalized == "CRS"


def parse_date(date_str: str, format_code: str = "102") -> Optional[str]:
    """Parse FAERS date format to ISO format."""
    if not date_str:
        return None
    try:
        if format_code == "102" and len(date_str) == 8:  # YYYYMMDD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        elif format_code == "610" and len(date_str) == 6:  # YYYYMM
            return f"{date_str[:4]}-{date_str[4:6]}"
        elif format_code == "602" and len(date_str) == 4:  # YYYY
            return date_str
    except:
        pass
    return date_str


def calculate_interval_days(start_date: str, end_date: str) -> Optional[int]:
    """Calculate days between two ISO dates."""
    if not start_date or not end_date:
        return None
    try:
        # Handle partial dates
        if len(start_date) == 10 and len(end_date) == 10:
            d1 = datetime.strptime(start_date, "%Y-%m-%d")
            d2 = datetime.strptime(end_date, "%Y-%m-%d")
            return (d2 - d1).days
    except:
        pass
    return None


def parse_age(age_value: str, age_unit: str) -> Optional[float]:
    """Convert age to years."""
    if not age_value:
        return None
    try:
        age = float(age_value)
        # Unit codes: 800=decade, 801=year, 802=month, 803=week, 804=day, 805=hour
        unit_map = {
            "800": 10,      # decade
            "801": 1,       # year
            "802": 1/12,    # month
            "803": 1/52,    # week
            "804": 1/365,   # day
            "805": 1/8760   # hour
        }
        multiplier = unit_map.get(age_unit, 1)
        return round(age * multiplier, 2)
    except:
        return None


def parse_weight(weight_value: str) -> Optional[float]:
    """Parse weight in kg."""
    if not weight_value:
        return None
    try:
        return float(weight_value)
    except:
        return None


def parse_sex(sex_code: str) -> Optional[str]:
    """Convert sex code to readable format."""
    sex_map = {"1": "male", "2": "female", "0": "unknown"}
    return sex_map.get(sex_code)


def parse_outcome(outcome_code: str) -> Optional[str]:
    """Convert outcome code to readable format."""
    outcome_map = {
        "1": "recovered",
        "2": "recovering",
        "3": "not_recovered",
        "4": "recovered_with_sequelae",
        "5": "fatal",
        "6": "unknown"
    }
    return outcome_map.get(outcome_code)


def extract_dose_info(drug: Dict) -> Dict:
    """Extract dose information from drug entry."""
    dose_mg = None
    dose_text = drug.get("drugdosagetext")
    
    # Try to get structured dose
    if drug.get("drugstructuredosagenumb"):
        try:
            dose_value = float(drug["drugstructuredosagenumb"])
            unit_code = drug.get("drugstructuredosageunit", "")
            # Unit 003 = mg
            if unit_code == "003":
                dose_mg = dose_value
            else:
                dose_mg = dose_value  # Store as-is, may need conversion
        except:
            pass
    
    # Parse date
    start_date = parse_date(
        drug.get("drugstartdate"),
        drug.get("drugstartdateformat", "102")
    )
    
    # Determine dose type from text
    dose_type = None
    if dose_text:
        dose_text_upper = dose_text.upper()
        if "PRIMING" in dose_text_upper or "STEP-UP" in dose_text_upper:
            dose_type = "priming"
        elif "FULL" in dose_text_upper:
            dose_type = "full"
        elif "MAINTENANCE" in dose_text_upper:
            dose_type = "maintenance"
    
    return {
        "dose_mg": dose_mg,
        "date": start_date,
        "dose_type": dose_type,
        "dose_text": dose_text
    }


def extract_report(report: Dict) -> Dict:
    """Extract structured data from a single FAERS report."""
    patient = report.get("patient", {})
    reactions = patient.get("reaction", [])
    drugs = patient.get("drug", [])
    
    # Initialize output
    output = {
        "report_id": report.get("safetyreportid"),
        
        # Outcome variables
        "is_crs": False,
        "crs_outcome": None,
        "serious": report.get("serious") == "1",
        "hospitalized": report.get("seriousnesshospitalization") == "1",
        "death": report.get("seriousnessdeath") == "1",
        "life_threatening": report.get("seriousnesslifethreatening") == "1",
        "disability": report.get("seriousnessdisabling") == "1",
        
        # Exposure variables
        "epcoritamab_exposure": False,
        "epcoritamab_suspect": False,
        "epcoritamab_doses": [],
        "co_medications": [],
        "indication": None,
        
        # Patient demographics
        "age": parse_age(
            patient.get("patientonsetage"),
            patient.get("patientonsetageunit")
        ),
        "sex": parse_sex(patient.get("patientsex")),
        "weight": parse_weight(patient.get("patientweight")),
        "country": report.get("occurcountry") or report.get("primarysourcecountry"),
        
        # Temporal variables
        "crs_onset_date": None,
        "first_epcoritamab_date": None,
        "dose_to_crs_interval_days": None,
        
        # Report metadata
        "receive_date": parse_date(
            report.get("receivedate"),
            report.get("receivedateformat", "102")
        ),
        
        # Narrative features
        "narrative_text": None,
        
        # Additional reaction info
        "all_reactions": []
    }
    
    # Process reactions
    for rxn in reactions:
        rxn_name = rxn.get("reactionmeddrapt", "")
        rxn_outcome = parse_outcome(rxn.get("reactionoutcome"))
        
        output["all_reactions"].append({
            "name": rxn_name,
            "outcome": rxn_outcome
        })
        
        if is_crs_reaction(rxn_name):
            output["is_crs"] = True
            output["crs_outcome"] = rxn_outcome
            # Try to get CRS onset date from reaction
            if rxn.get("reactionstartdate"):
                output["crs_onset_date"] = parse_date(
                    rxn.get("reactionstartdate"),
                    rxn.get("reactionstartdateformat", "102")
                )
    
    # Process drugs
    epcoritamab_dates = []
    
    for drug in drugs:
        drug_name = drug.get("medicinalproduct", "")
        active_substance = drug.get("activesubstance", {}).get("activesubstancename", "")
        drug_char = drug.get("drugcharacterization")  # 1=suspect, 2=concomitant, 3=interacting
        
        if is_epcoritamab(drug_name, active_substance):
            output["epcoritamab_exposure"] = True
            
            # Check if it's the primary suspect (drugcharacterization = 1)
            if drug_char == "1":
                output["epcoritamab_suspect"] = True
            
            # Extract dose info
            dose_info = extract_dose_info(drug)
            output["epcoritamab_doses"].append(dose_info)
            
            # Track earliest date
            if dose_info["date"]:
                epcoritamab_dates.append(dose_info["date"])
            
            # Get indication
            if drug.get("drugindication") and not output["indication"]:
                output["indication"] = drug.get("drugindication")
        
        else:
            # Co-medication
            normalized_name = normalize_drug_name(drug_name)
            if normalized_name and normalized_name not in output["co_medications"]:
                output["co_medications"].append(normalized_name)
    
    # Calculate temporal variables
    if epcoritamab_dates:
        output["first_epcoritamab_date"] = min(epcoritamab_dates)
        
        if output["crs_onset_date"]:
            output["dose_to_crs_interval_days"] = calculate_interval_days(
                output["first_epcoritamab_date"],
                output["crs_onset_date"]
            )
    
    # Extract narrative if available
    if patient.get("summary") and patient["summary"].get("narrativeincludeclinical"):
        output["narrative_text"] = patient["summary"]["narrativeincludeclinical"]
    
    return output


def main():
    # Load data
    print("Loading FAERS data...")
    with open("fda_drug_events.json", "r") as f:
        data = json.load(f)
    
    reports = data.get("results", [])
    print(f"Found {len(reports)} reports")
    
    # Extract structured data
    print("Extracting structured data...")
    extracted = []
    
    for report in reports:
        extracted_report = extract_report(report)
        extracted.append(extracted_report)
    
    # Filter to reports with Epcoritamab exposure
    epcoritamab_reports = [r for r in extracted if r["epcoritamab_exposure"]]
    print(f"Reports with Epcoritamab exposure: {len(epcoritamab_reports)}")
    
    # Filter to reports with Epcoritamab as suspect drug
    suspect_reports = [r for r in extracted if r["epcoritamab_suspect"]]
    print(f"Reports with Epcoritamab as suspect drug: {len(suspect_reports)}")
    
    # Filter to CRS cases
    crs_reports = [r for r in extracted if r["is_crs"]]
    print(f"Reports with CRS: {len(crs_reports)}")
    
    # Save all extracted data
    output_file = "crs_extracted_data.json"
    with open(output_file, "w") as f:
        json.dump(extracted, f, indent=2)
    print(f"\nAll extracted data saved to {output_file}")
    
    # Save Epcoritamab-specific data
    epcoritamab_file = "epcoritamab_crs_data.json"
    with open(epcoritamab_file, "w") as f:
        json.dump(epcoritamab_reports, f, indent=2)
    print(f"Epcoritamab reports saved to {epcoritamab_file}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # CRS outcomes
    crs_outcomes = {}
    for r in crs_reports:
        outcome = r["crs_outcome"] or "unknown"
        crs_outcomes[outcome] = crs_outcomes.get(outcome, 0) + 1
    
    print("\nCRS Outcomes:")
    for outcome, count in sorted(crs_outcomes.items()):
        print(f"  {outcome}: {count}")
    
    # Serious cases
    serious_count = sum(1 for r in extracted if r["serious"])
    hospitalized_count = sum(1 for r in extracted if r["hospitalized"])
    death_count = sum(1 for r in extracted if r["death"])
    
    print(f"\nSerious cases: {serious_count}")
    print(f"Hospitalized: {hospitalized_count}")
    print(f"Deaths: {death_count}")
    
    # Demographics
    ages = [r["age"] for r in extracted if r["age"] is not None]
    if ages:
        print(f"\nAge: mean={sum(ages)/len(ages):.1f}, min={min(ages)}, max={max(ages)}")
    
    sex_counts = {}
    for r in extracted:
        sex = r["sex"] or "unknown"
        sex_counts[sex] = sex_counts.get(sex, 0) + 1
    print(f"Sex distribution: {sex_counts}")
    
    # Country distribution
    countries = {}
    for r in extracted:
        country = r["country"] or "unknown"
        countries[country] = countries.get(country, 0) + 1
    print(f"\nTop countries: {dict(sorted(countries.items(), key=lambda x: -x[1])[:10])}")


if __name__ == "__main__":
    main()

