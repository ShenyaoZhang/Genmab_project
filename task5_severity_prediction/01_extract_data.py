#!/usr/bin/env python3
"""
Task 5 data collector for adverse event severity prediction.

Optimizations versus Task 3:
    - Collect reports for 35 common oncology drugs
- Pull 500-1000 reports per drug
- Keep report-level granularity (no drug-event explosion)
- Expected total volume: 15,000-20,000 reports

Task 5 requirements:
    - Target label is severity (death, hospitalization, etc.)
- Preserve complete patient and drug context
- Maintain one record per safety report
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

# 35 common oncology drugs covering core therapeutic classes
ONCOLOGY_DRUGS = [
    # PD-1/PD-L1 immune checkpoint inhibitors
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab", "Ipilimumab",

    # Monoclonal antibody targeted therapies
    "Trastuzumab", "Bevacizumab", "Cetuximab", "Rituximab", "Epcoritamab",
    "Pertuzumab", "Panitumumab", "Ramucirumab", "Daratumumab",

    # Small-molecule targeted therapies (TKI)
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib", "Crizotinib",
    "Palbociclib", "Ribociclib", "Abemaciclib", "Vemurafenib", "Dabrafenib",
    "Ibrutinib", "Venetoclax",

    # PARP inhibitors
    "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",

    # Common chemotherapy agents
    "Paclitaxel", "Docetaxel", "Doxorubicin",

    # Immunomodulatory agents
    "Lenalidomide", "Pomalidomide"
]

BASE_URL = "https://api.fda.gov/drug/event.json"
MAX_RETRIES = 3
RETRY_DELAY = 5


def standardize_age_to_years(age, age_unit):
    """
    Convert the incoming age to years.

    age_unit codes:
        800 = decade, 801 = year, 802 = month, 803 = week, 804 = day
    """
    if age is None or age_unit is None:
        return None

    try:
        age = float(age)
        age_unit = int(age_unit)

        # Convert to years
        unit_map = {
            800: 10,  # decade
            801: 1,  # year
            802: 1 / 12,  # month
            803: 1 / 52,  # week
            804: 1 / 365  # day
        }

        age_years = age * unit_map.get(age_unit, 1)

        # Filter unrealistic values (<0 or >120 years)
        if age_years < 0 or age_years > 120:
            return None

        return round(age_years, 2)
    except BaseException:
        return None


def collect_drug_data(drug_name, max_records=500):
    """
    Collect adverse event reports for a given drug (at report level).

    Improvements:
        - Extend the search scope with OR queries
    - Deduplicate within the same drug
    """
    print(f"\n Collecting drug: {drug_name}")

    all_records = []
    seen_ids = set()  # per-drug deduplication
    skip = 0
    limit = 100
    retries = 0
    max_retries = 3

    while len(all_records) < max_records:
        try:
            # Build OR query to broaden the search
            search = (f'(patient.drug.openfda.generic_name:"{drug_name}" OR '
                      f'patient.drug.medicinalproduct:"{drug_name}" OR '
                      f'patient.drug.activesubstance.activesubstancename:"{drug_name}" OR '
                      f'patient.drug.openfda.brand_name:"{drug_name}")')  # include brand name

            params = {
                'search': search,
                'limit': min(limit, max_records - len(all_records)),
                'skip': skip
            }

            # Custom User-Agent (rate limiting + audit trail)
            headers = {
                'User-Agent': 'task5-severity-prediction-collector/1.0 (NYU-CDS-Capstone)'}

            response = requests.get(
                BASE_URL,
                params=params,
                headers=headers,
                timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                if not results:
                    print(f" No additional data found, collected {len(all_records)} records")
                    break

                # Process each record with deduplication
                new_records_count = 0
                for record in results:
                    safety_id = record.get('safetyreportid', '')

                    # Deduplicate by safetyreportid
                    if safety_id and safety_id not in seen_ids:
                        processed = process_record(record, drug_name)
                        if processed:
                            all_records.append(processed)
                            seen_ids.add(safety_id)
                            new_records_count += 1

                # Stop early if the entire page is duplicate
                if new_records_count == 0 and len(results) > 0:
                    print(
                        " WARNING: Current page contains only duplicates, stopping collection")
                    break

                print(f" Progress: {len(all_records)} (new {new_records_count} this page)", end='\r')
                skip += len(results)
                retries = 0  # reset retry counter

                # Respect API rate limits
                time.sleep(0.3)

            elif response.status_code == 404:
                print(" WARNING: No data found")
                break
            else:

                retries += 1
                if retries >= max_retries:
                    print(f" ERROR: HTTP {response.status_code}, reached max retries")
                    break

                # Exponential backoff (handles 429/5xx)
                delay = RETRY_DELAY * (2 ** (retries - 1))  # 5s, 10s, 20s
                print(
                    f" WARNING: HTTP response.status_code}, retry {retries}/{max_retries} after {delay}s")
                time.sleep(delay)

        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f" ERROR: Error: {str(e)[:50]}, reached max retries")
                break

            # Exponential backoff on exception
            delay = RETRY_DELAY * (2 ** (retries - 1))
            print(
                f" WARNING: Error: str(e)[
                        :50]}, retry {retries}/{max_retries} after {delay}s")
            time.sleep(delay)

    print(f" Completed: {len(all_records)} records")
    return all_records


def process_record(record, target_drug):
    """
    Extract the key fields while keeping report-level integrity.
    """
    try:
        # Basic metadata
        safety_id = record.get('safetyreportid', '')
        receive_date = record.get('receivedate', '')

        # Patient information
        patient = record.get('patient', {})
        age = patient.get('patientonsetage', None)
        age_unit = patient.get('patientonsetageunit', '')
        sex = patient.get('patientsex', 0)
        weight = patient.get('patientweight', None)

        # Normalize age to years (keep original for audit)
        age_years = standardize_age_to_years(age, age_unit)

        # Normalize severity flags to binary 0/1
        def norm01(v):
            """Normalize any value to 0 or 1."""
            try:
                return 1 if int(v) > 0 else 0
            except BaseException:
                return 0

        serious = norm01(record.get('serious', 0))
        seriousness_death = norm01(record.get('seriousnessdeath', 0))
        seriousness_hosp = norm01(record.get('seriousnesshospitalization', 0))
        seriousness_life = norm01(record.get('seriousnesslifethreatening', 0))
        seriousness_disable = norm01(record.get('seriousnessdisabling', 0))
        seriousness_congenital = norm01(
            record.get('seriousnesscongenitalanomali', 0))
        seriousness_other = norm01(record.get('seriousnessother', 0))

        # Reporter information
        primary_source = record.get('primarysource', {})
        qualification = primary_source.get('qualification', '')

        # Drug information
        drugs = patient.get('drug', [])
        drug_names = []
        drug_roles = []
        drug_indications = []

        for drug in drugs:
            openfda = drug.get('openfda', {})
            generic_names = openfda.get('generic_name', [])
            drug_names.extend(generic_names)

            # Track suspect/concomitant roles
            role = drug.get('drugcharacterization', '')
            drug_roles.append(role)

            indication = drug.get('drugindication', '')
            if indication:
                drug_indications.append(indication)

        # Adverse event terms
        reactions = patient.get('reaction', [])
        adverse_events = []
        for reaction in reactions:
            ae_term = reaction.get('reactionmeddrapt', '')
            if ae_term:
                adverse_events.append(ae_term)

        # Build the processed report
        processed_record = {
            # Basic metadata
            'safetyreportid': safety_id,
            'receivedate': receive_date,
            'target_drug': target_drug,

            # Patient information
            'patientonsetage': age,
            'patientonsetageunit': age_unit,
            'age_years': age_years,  # normalized age in years
            'patientsex': sex,
            'patientweight': weight,

            # Severity indicators (target label)
            'serious': serious,
            'seriousnessdeath': seriousness_death,
            'seriousnesshospitalization': seriousness_hosp,
            'seriousnesslifethreatening': seriousness_life,
            'seriousnessdisabling': seriousness_disable,
            'seriousnesscongenitalanomali': seriousness_congenital,
            'seriousnessother': seriousness_other,

            # Drug information
            'drugname': target_drug,
            'all_drugs': '|'.join(drug_names) if drug_names else '',
            'num_drugs': len(drugs),
            'drug_indication': '|'.join(drug_indications) if drug_indications else '',

            # Adverse event information
            'reactions': '|'.join(adverse_events) if adverse_events else '',
            'num_reactions': len(adverse_events),

            # Reporter quality
            'reporter_qualification': qualification
        }

        return processed_record

    except Exception as e:
        return None


def main():
    """
    Main entry: collect data for all oncology drugs.
    """
    print("Data collection configuration:")
    print(f" Number of drugs: {len(ONCOLOGY_DRUGS)}")
    print(" Records per drug: 500")
    print(f" Expected total volume: ~{len(ONCOLOGY_DRUGS) * 500:,}")
    print(f" Estimated runtime: {len(ONCOLOGY_DRUGS) * 2} minutes")
    print()

    # Show drug list
    print(" Drug list:")
    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f" {i:2d}. {drug}")
    print()

    # Non-interactive mode: automatically start data collection
    response = 'y'  # Default to 'y' for automated runs
    # response = input("Start data collection? (y/n, default y): ").strip().lower()
    # if response in ['n', 'no']:
    #     print("Cancelled")
    #     return

    print()
    print("=" * 80)
    print("Starting data collection")
    print("=" * 80)

    start_time = time.time()
    all_data = []
    global_seen_ids = set()  # cross-drug deduplication
    success_count = 0
    failed_drugs = []

    for i, drug in enumerate(ONCOLOGY_DRUGS, 1):
        print(f"\n[{i}/{len(ONCOLOGY_DRUGS)}] {drug}")
        print("-" * 80)

        try:
            records = collect_drug_data(drug, max_records=500)

            if records:
                # Cross-drug deduplication
                before_dedup = len(records)
                unique_records = []
                for rec in records:
                    safety_id = rec.get('safetyreportid', '')
                    if safety_id and safety_id not in global_seen_ids:
                        unique_records.append(rec)
                        global_seen_ids.add(safety_id)

                after_dedup = len(unique_records)
                if before_dedup > after_dedup:
                    print(
                        f"Global dedup: {before_dedup} -> {after_dedup}{-{before_dedup - after_dedup})")

                all_data.extend(unique_records)
                success_count += 1
            else:

                failed_drugs.append(drug)

            # Show cumulative progress
            print(
                f" Total unique reports: len(all_data)} | Successful drugs: {success_count}")

            # Save checkpoints every 10 drugs
            if i % 10 == 0:
                temp_df = pd.DataFrame(all_data)
                temp_file = f'task5_data_temp_{i}.csv'
                temp_df.to_csv(temp_file, index=False)
                print(f"Checkpoint saved: {temp_file}")

        except Exception as e:
            print(f" ERROR: Processing failed: {str(e)[:50]}")
            failed_drugs.append(drug)

    elapsed_time = time.time() - start_time

    print()
    print("=" * 80)
    print(" Data collection finished")
    print("=" * 80)
    print()

    if not all_data:
        print("ERROR: Error: no data collected")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Data cleaning summary
    print(" Cleaning data...")
    print(f" Raw records: {len(df)}")

    # Deduplicate by safetyreportid
    df = df.drop_duplicates(subset=['safetyreportid'], keep='first')
    print(f" After deduplication: {len(df)}")

    # Keep rows with a target label
    df = df[df['serious'].notna()]
    print(f" Valid records: {len(df)}")
    print()

    # Persist results
    output_file = 'main_data.csv'
    df.to_csv(output_file, index=False)

    print("=" * 80)
    print(" Data collection succeeded!")
    print("=" * 80)
    print()

    # Summary statistics
    print(" Data summary:")
    print(f" Total records: {len(df):,}")
    print(f" Unique reports: {df['safetyreportid'].nunique():,}")
    print(f" Unique drugs: {df['target_drug'].nunique()}")
    print(f" Successful drugs: {success_count}/{len(ONCOLOGY_DRUGS)}")
    print()

    print(" Severity distribution:")
    # Convert to numeric (OpenFDA may return strings)
    death_count = pd.to_numeric(
        df['seriousnessdeath'],
        errors='coerce').fillna(0).sum()
    hosp_count = pd.to_numeric(
        df['seriousnesshospitalization'],
        errors='coerce').fillna(0).sum()
    life_count = pd.to_numeric(
        df['seriousnesslifethreatening'],
        errors='coerce').fillna(0).sum()
    disable_count = pd.to_numeric(
        df['seriousnessdisabling'],
        errors='coerce').fillna(0).sum()

    print(
        f" Death cases: int(death_count):,}{death_count / len(df) * 100:.1f}%)")
    print(
        f" Hospitalizations: int(hosp_count):,}{hosp_count / len(df) * 100:.1f}%)")
    print(
        f" Life-threatening: {int(life_count):,}{life_count / len(df) * 100:.1f}%)")
    print(
        f" Disability: int(disable_count):,}{disable_count / len(df) * 100:.1f}%)")
    print()

    print(" Patient demographics:")
    if 'patientsex' in df.columns:
        sex_counts = df['patientsex'].value_counts()
        sex_map = {1: 'Male', 2: 'Female', 0: 'Unknown'}
        for sex_code, count in sex_counts.items():
            label = sex_map.get(sex_code, f'Code {sex_code}')
            print(f" {label}: {count:,}{count / len(df) * 100:.1f}%)")

    if 'patientonsetage' in df.columns:
        age_data = pd.to_numeric(
            df['patientonsetage'],
            errors='coerce').dropna()
        if len(age_data) > 0:
            print(f" Mean age: {age_data.mean():.1f} years")
            print(
                f" Age range: {age_data.min():.1f} - {age_data.max():.1f} years")
    print()

    print(" Reports per drug:")
    drug_counts = df['target_drug'].value_counts()
    for drug, count in drug_counts.head(10).items():
        print(f" {drug:20s}: {count:5d}")
    if len(drug_counts) > 10:
        print(f" ... (plus {len(drug_counts) - 10} additional drugs)")
    print()

    if failed_drugs:
        print(f"WARNING: Failed drugs ({len(failed_drugs)}):")
        for drug in failed_drugs:
            print(f" - {drug}")
        print()

    print(f"Elapsed time: {elapsed_time / 60:.1f} minutes")
    print(f" Output file: {output_file}")
    print()

    # Save summary report
    with open("task5_collection_summary.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Task 5 Data Collection Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Collected at: datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Elapsed time: {elapsed_time / 60:.1f} minutes\n\n")

        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Unique drugs: {df['target_drug'].nunique()}\n")
        f.write(
            f"Success rate: {success_count}/ len(ONCOLOGY_DRUGS)}{success_count / len(ONCOLOGY_DRUGS) * 100:.1f}%)\n\n")

        f.write("Reports per drug:\n")
        for drug, count in drug_counts.items():
            f.write(f" {drug:20s}: {count:5d}\n")

        if failed_drugs:
            f.write("\nFailed drugs:\n")
            for drug in failed_drugs:
                f.write(f" - {drug}\n")

    print(" Saved summary: task5_collection_summary.txt")
    print()

    # Show sample rows
    print("=" * 80)
    print(" Data sample (first 5 rows)")
    print("=" * 80)
    print()
    sample_cols = [
        'target_drug',
        'seriousnessdeath',
        'seriousnesshospitalization',
        'patientsex',
        'patientonsetage',
        'num_drugs',
        'num_reactions']
    available_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_cols].head())
    print()

    print("=" * 80)
    print(" Next steps")
    print("=" * 80)
    print()
    print("Data is ready for downstream steps:")
    print(" 1. Run: python 02_inspect_data.py")
    print(" 2. Or inspect manually: open main_data.csv")
    print()
    print(" Tip: Later steps automatically read this file")
    print()


if __name__ == "__main__":
    main()
