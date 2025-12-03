"""
Data Extractors for Pharmacovigilance Databases
- Eudravigilance (European Medicines Agency)
- JADER (Japanese Adverse Drug Event Report)

Note: These databases have different access methods:
- Eudravigilance: Requires EMA account, data downloaded via adrreports.eu
- JADER: Publicly available CSV files from PMDA website
"""

import os
import requests
import pandas as pd
import zipfile
import io
from typing import Optional, List, Dict
import re
from datetime import datetime


class EudravigilanceExtractor:
    """
    Extractor for Eudravigilance data.
    
    Eudravigilance data can be accessed via:
    1. ADR Reports website (adrreports.eu) - Line listings by substance
    2. Bulk download (requires EMA account)
    
    This extractor handles the line listing format from adrreports.eu
    """
    
    def __init__(self, data_dir: str = "./eudravigilance_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_line_listing(self, substance: str = "epcoritamab") -> str:
        """
        Instructions for downloading Eudravigilance line listings.
        
        Manual steps required:
        1. Go to https://www.adrreports.eu/
        2. Search for substance (e.g., "epcoritamab")
        3. Click on "Line Listing" tab
        4. Download the Excel file
        5. Save to data_dir
        """
        print(f"""
        ============================================================
        EUDRAVIGILANCE DATA DOWNLOAD INSTRUCTIONS
        ============================================================
        
        Eudravigilance requires manual download:
        
        1. Visit: https://www.adrreports.eu/
        2. In the search box, enter: {substance}
        3. Click on the substance name in results
        4. Navigate to "Line Listing" tab
        5. Click "Download" to get Excel file
        6. Save the file to: {self.data_dir}/eudravigilance_{substance}.xlsx
        
        After downloading, run: extractor.parse_line_listing()
        ============================================================
        """)
        return os.path.join(self.data_dir, f"eudravigilance_{substance}.xlsx")
    
    def parse_line_listing(self, filepath: str) -> pd.DataFrame:
        """
        Parse Eudravigilance line listing Excel file.
        
        Expected columns (may vary):
        - EU Local Number
        - EV Gateway Receipt Date
        - Primary Source Country
        - Patient Age Group
        - Patient Sex
        - Reaction List (PT)
        - Suspect/interacting Drug List
        - Serious
        - Outcome
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            print("Please download the file first using download_line_listing()")
            return pd.DataFrame()
        
        df = pd.read_excel(filepath)
        
        # Standardize column names
        column_mapping = {
            'EU Local Number': 'report_id',
            'EV Gateway Receipt Date': 'receive_date',
            'Primary Source Country': 'country',
            'Patient Age Group': 'age_group',
            'Patient Sex': 'sex',
            'Reaction List (PT)': 'reactions',
            'Suspect/interacting Drug List': 'drugs',
            'Serious': 'serious',
            'Outcome': 'outcome'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        df['source'] = 'eudravigilance'
        
        return df
    
    def extract_crs_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract CRS cases from Eudravigilance data."""
        if df.empty:
            return df
        
        # Filter for CRS reactions
        crs_mask = df['reactions'].str.contains(
            'cytokine release syndrome', 
            case=False, 
            na=False
        )
        
        # Filter for Epcoritamab
        epcoritamab_mask = df['drugs'].str.contains(
            'epcoritamab', 
            case=False, 
            na=False
        )
        
        return df[crs_mask & epcoritamab_mask].copy()
    
    def to_unified_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert to unified format matching FAERS extraction."""
        records = []
        
        for _, row in df.iterrows():
            record = {
                'report_id': f"EU-{row.get('report_id', '')}",
                'source': 'eudravigilance',
                'is_crs': True,
                'crs_outcome': self._map_outcome(row.get('outcome', '')),
                'serious': str(row.get('serious', '')).lower() == 'yes',
                'hospitalized': None,  # Not always available in line listings
                'death': 'fatal' in str(row.get('outcome', '')).lower(),
                'epcoritamab_exposure': True,
                'epcoritamab_suspect': True,
                'age': self._parse_age_group(row.get('age_group', '')),
                'age_group': row.get('age_group'),
                'sex': self._map_sex(row.get('sex', '')),
                'country': row.get('country'),
                'receive_date': str(row.get('receive_date', ''))[:10],
                'co_medications': self._parse_drugs(row.get('drugs', '')),
                'all_reactions': self._parse_reactions(row.get('reactions', ''))
            }
            records.append(record)
        
        return records
    
    def _map_outcome(self, outcome: str) -> str:
        outcome = str(outcome).lower()
        if 'recover' in outcome and 'not' not in outcome:
            return 'recovered'
        elif 'not recover' in outcome:
            return 'not_recovered'
        elif 'fatal' in outcome:
            return 'fatal'
        return 'unknown'
    
    def _map_sex(self, sex: str) -> str:
        sex = str(sex).lower()
        if 'male' in sex and 'female' not in sex:
            return 'male'
        elif 'female' in sex:
            return 'female'
        return 'unknown'
    
    def _parse_age_group(self, age_group: str) -> Optional[float]:
        """Estimate age from age group."""
        age_group = str(age_group).lower()
        age_map = {
            '0-1 month': 0.04,
            '2 months - 2 years': 1,
            '3-11 years': 7,
            '12-17 years': 15,
            '18-64 years': 41,
            '65-85 years': 75,
            'more than 85 years': 90
        }
        for group, age in age_map.items():
            if group in age_group:
                return age
        return None
    
    def _parse_drugs(self, drugs_str: str) -> List[str]:
        """Parse drug list string."""
        if pd.isna(drugs_str):
            return []
        drugs = str(drugs_str).split(',')
        return [d.strip().upper() for d in drugs if 'epcoritamab' not in d.lower()]
    
    def _parse_reactions(self, reactions_str: str) -> List[Dict]:
        """Parse reactions list string."""
        if pd.isna(reactions_str):
            return []
        reactions = str(reactions_str).split(',')
        return [{'name': r.strip(), 'outcome': None} for r in reactions]


class JADERExtractor:
    """
    Extractor for JADER (Japanese Adverse Drug Event Report) database.
    
    JADER data is publicly available from PMDA:
    https://www.pmda.go.jp/safety/info-services/drugs/adr-info/suspected-adr/0003.html
    
    Data files (CSV format):
    - demo.csv: Demographics
    - drug.csv: Drug information
    - reac.csv: Adverse reactions
    - hist.csv: Medical history
    """
    
    BASE_URL = "https://www.pmda.go.jp/files"
    
    def __init__(self, data_dir: str = "./jader_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_instructions(self) -> str:
        """
        Instructions for downloading JADER data.
        """
        print(f"""
        ============================================================
        JADER DATA DOWNLOAD INSTRUCTIONS
        ============================================================
        
        JADER data can be downloaded from PMDA:
        
        1. Visit: https://www.pmda.go.jp/safety/info-services/drugs/adr-info/suspected-adr/0003.html
        2. Scroll to the download section
        3. Download the latest ZIP file (e.g., jader_all_csv_202310.zip)
        4. Extract to: {self.data_dir}
        
        Expected files after extraction:
        - demo.csv (demographics)
        - drug.csv (drug information)  
        - reac.csv (adverse reactions)
        - hist.csv (medical history)
        
        After downloading, run: extractor.load_jader_data()
        ============================================================
        """)
        return self.data_dir
    
    def load_jader_data(self) -> Dict[str, pd.DataFrame]:
        """Load JADER CSV files."""
        files = {
            'demo': 'demo.csv',
            'drug': 'drug.csv',
            'reac': 'reac.csv',
            'hist': 'hist.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                # JADER uses Shift-JIS encoding
                try:
                    data[key] = pd.read_csv(filepath, encoding='shift-jis')
                except:
                    data[key] = pd.read_csv(filepath, encoding='utf-8')
                print(f"Loaded {key}: {len(data[key])} records")
            else:
                print(f"File not found: {filepath}")
                data[key] = pd.DataFrame()
        
        return data
    
    def extract_epcoritamab_cases(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract cases involving Epcoritamab from JADER data.
        
        JADER drug.csv columns:
        - 識別番号 (Case ID)
        - 医薬品連番 (Drug sequence number)
        - 医薬品一般名 (Generic name)
        - 医薬品販売名 (Brand name)
        - 医薬品の関与 (Drug involvement: 被疑薬=suspect, 併用薬=concomitant)
        """
        if data['drug'].empty:
            return pd.DataFrame()
        
        drug_df = data['drug']
        
        # Find Epcoritamab cases (check both generic and brand names)
        epcoritamab_names = ['epcoritamab', 'エプコリタマブ', 'tepkinly', 'テプキンリー']
        
        mask = pd.Series([False] * len(drug_df))
        for col in drug_df.columns:
            if drug_df[col].dtype == 'object':
                for name in epcoritamab_names:
                    mask |= drug_df[col].str.contains(name, case=False, na=False)
        
        epcoritamab_cases = drug_df[mask]['識別番号'].unique() if '識別番号' in drug_df.columns else []
        
        if len(epcoritamab_cases) == 0:
            # Try alternative column names
            id_cols = [c for c in drug_df.columns if '番号' in c or 'id' in c.lower()]
            if id_cols:
                epcoritamab_cases = drug_df[mask][id_cols[0]].unique()
        
        print(f"Found {len(epcoritamab_cases)} cases with Epcoritamab")
        return epcoritamab_cases
    
    def extract_crs_cases(self, data: Dict[str, pd.DataFrame], case_ids: list) -> pd.DataFrame:
        """
        Extract CRS cases from reactions data.
        
        JADER reac.csv columns:
        - 識別番号 (Case ID)
        - 有害事象連番 (Reaction sequence)
        - 有害事象(PT) (Adverse event - MedDRA PT)
        - 転帰 (Outcome)
        """
        if data['reac'].empty:
            return pd.DataFrame()
        
        reac_df = data['reac']
        
        # Filter for case IDs
        id_col = '識別番号' if '識別番号' in reac_df.columns else reac_df.columns[0]
        reac_filtered = reac_df[reac_df[id_col].isin(case_ids)]
        
        # Find CRS reactions
        crs_terms = ['cytokine release syndrome', 'サイトカイン放出症候群', 'crs']
        
        mask = pd.Series([False] * len(reac_filtered))
        for col in reac_filtered.columns:
            if reac_filtered[col].dtype == 'object':
                for term in crs_terms:
                    mask |= reac_filtered[col].str.contains(term, case=False, na=False)
        
        crs_cases = reac_filtered[mask]
        print(f"Found {len(crs_cases)} CRS reactions")
        
        return crs_cases
    
    def build_unified_dataset(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Build unified dataset from JADER data."""
        records = []
        
        # Get Epcoritamab cases
        case_ids = self.extract_epcoritamab_cases(data)
        if len(case_ids) == 0:
            return records
        
        # Get CRS cases
        crs_cases = self.extract_crs_cases(data, case_ids)
        
        # Join with demographics
        demo_df = data['demo']
        drug_df = data['drug']
        
        id_col = '識別番号' if '識別番号' in demo_df.columns else demo_df.columns[0]
        
        for case_id in crs_cases[id_col].unique() if not crs_cases.empty else []:
            demo_row = demo_df[demo_df[id_col] == case_id]
            
            record = {
                'report_id': f"JADER-{case_id}",
                'source': 'jader',
                'is_crs': True,
                'crs_outcome': self._map_outcome(crs_cases, case_id),
                'serious': True,  # CRS is typically serious
                'hospitalized': None,
                'death': False,
                'epcoritamab_exposure': True,
                'epcoritamab_suspect': True,
                'age': self._get_age(demo_row),
                'sex': self._get_sex(demo_row),
                'country': 'JP',
                'co_medications': self._get_co_medications(drug_df, case_id),
                'all_reactions': self._get_reactions(data['reac'], case_id)
            }
            records.append(record)
        
        return records
    
    def _map_outcome(self, crs_df: pd.DataFrame, case_id) -> str:
        """Map JADER outcome to standardized format."""
        outcome_col = '転帰' if '転帰' in crs_df.columns else None
        if outcome_col is None:
            return 'unknown'
        
        id_col = '識別番号' if '識別番号' in crs_df.columns else crs_df.columns[0]
        outcomes = crs_df[crs_df[id_col] == case_id][outcome_col].values
        
        if len(outcomes) == 0:
            return 'unknown'
        
        outcome = str(outcomes[0]).lower()
        if '回復' in outcome:
            return 'recovered'
        elif '死亡' in outcome:
            return 'fatal'
        elif '未回復' in outcome:
            return 'not_recovered'
        return 'unknown'
    
    def _get_age(self, demo_row: pd.DataFrame) -> Optional[float]:
        """Extract age from demographics."""
        age_cols = [c for c in demo_row.columns if '年齢' in c or 'age' in c.lower()]
        if age_cols and not demo_row.empty:
            try:
                return float(demo_row[age_cols[0]].values[0])
            except:
                pass
        return None
    
    def _get_sex(self, demo_row: pd.DataFrame) -> str:
        """Extract sex from demographics."""
        sex_cols = [c for c in demo_row.columns if '性別' in c or 'sex' in c.lower()]
        if sex_cols and not demo_row.empty:
            sex = str(demo_row[sex_cols[0]].values[0]).lower()
            if '男' in sex or 'male' in sex:
                return 'male'
            elif '女' in sex or 'female' in sex:
                return 'female'
        return 'unknown'
    
    def _get_co_medications(self, drug_df: pd.DataFrame, case_id) -> List[str]:
        """Get co-medications for a case."""
        id_col = '識別番号' if '識別番号' in drug_df.columns else drug_df.columns[0]
        case_drugs = drug_df[drug_df[id_col] == case_id]
        
        name_cols = [c for c in case_drugs.columns if '名' in c]
        medications = []
        
        for col in name_cols:
            for drug in case_drugs[col].values:
                if pd.notna(drug) and 'epcoritamab' not in str(drug).lower():
                    medications.append(str(drug).upper())
        
        return list(set(medications))
    
    def _get_reactions(self, reac_df: pd.DataFrame, case_id) -> List[Dict]:
        """Get all reactions for a case."""
        id_col = '識別番号' if '識別番号' in reac_df.columns else reac_df.columns[0]
        case_reacs = reac_df[reac_df[id_col] == case_id]
        
        pt_cols = [c for c in case_reacs.columns if 'PT' in c or '有害事象' in c]
        reactions = []
        
        for col in pt_cols:
            for reac in case_reacs[col].values:
                if pd.notna(reac):
                    reactions.append({'name': str(reac), 'outcome': None})
        
        return reactions


def create_simulated_multi_source_data():
    """
    Create simulated multi-source data for demonstration.
    This simulates what the combined dataset would look like.
    """
    import random
    import json
    
    # Load FAERS data
    with open('crs_extracted_data.json', 'r') as f:
        faers_data = json.load(f)
    
    # Add source tag to FAERS data
    for record in faers_data:
        record['source'] = 'faers'
    
    # Simulate Eudravigilance data (European patterns)
    eu_countries = ['DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'PL', 'SE', 'AT', 'PT']
    eu_data = []
    
    for i in range(50):  # Simulate 50 EU cases
        record = {
            'report_id': f'EU-2024-{100000 + i}',
            'source': 'eudravigilance',
            'is_crs': True,
            'crs_outcome': random.choice(['recovered', 'recovered', 'recovered', 'not_recovered', 'fatal']),
            'serious': True,
            'hospitalized': True,
            'death': random.random() < 0.15,
            'life_threatening': random.random() < 0.3,
            'disability': random.random() < 0.05,
            'epcoritamab_exposure': True,
            'epcoritamab_suspect': True,
            'epcoritamab_doses': [
                {'dose_mg': random.choice([0.16, 0.8, 24, 48]), 'date': None, 'dose_type': 'full'}
            ],
            'co_medications': random.sample(
                ['RITUXIMAB', 'BENDAMUSTINE', 'DEXAMETHASONE', 'PREDNISOLONE', 'TOCILIZUMAB'],
                k=random.randint(1, 4)
            ),
            'indication': random.choice(['B-CELL LYMPHOMA', 'DLBCL', 'FOLLICULAR LYMPHOMA']),
            'age': random.gauss(65, 12),
            'sex': random.choice(['male', 'female', 'male']),  # Slight male predominance
            'weight': random.gauss(75, 15),
            'country': random.choice(eu_countries),
            'crs_onset_date': None,
            'first_epcoritamab_date': None,
            'dose_to_crs_interval_days': random.randint(0, 7) if random.random() > 0.3 else None,
            'receive_date': f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'narrative_text': None,
            'all_reactions': [{'name': 'Cytokine release syndrome', 'outcome': 'recovered'}]
        }
        eu_data.append(record)
    
    # Simulate JADER data (Japanese patterns)
    jader_data = []
    
    for i in range(30):  # Simulate 30 Japanese cases
        record = {
            'report_id': f'JADER-{300000 + i}',
            'source': 'jader',
            'is_crs': True,
            'crs_outcome': random.choice(['recovered', 'recovered', 'recovered', 'not_recovered']),
            'serious': True,
            'hospitalized': True,
            'death': random.random() < 0.1,  # Lower mortality in Japanese data
            'life_threatening': random.random() < 0.25,
            'disability': random.random() < 0.03,
            'epcoritamab_exposure': True,
            'epcoritamab_suspect': True,
            'epcoritamab_doses': [
                {'dose_mg': random.choice([0.16, 0.8, 24, 48]), 'date': None, 'dose_type': 'full'}
            ],
            'co_medications': random.sample(
                ['RITUXIMAB', 'CYCLOPHOSPHAMIDE', 'DOXORUBICIN', 'VINCRISTINE', 'PREDNISOLONE'],
                k=random.randint(1, 4)
            ),
            'indication': random.choice(['B-CELL LYMPHOMA', 'DLBCL', 'FOLLICULAR LYMPHOMA']),
            'age': random.gauss(68, 10),  # Slightly older in Japan
            'sex': random.choice(['male', 'female']),
            'weight': random.gauss(60, 12),  # Lower weight in Japanese population
            'country': 'JP',
            'crs_onset_date': None,
            'first_epcoritamab_date': None,
            'dose_to_crs_interval_days': random.randint(0, 5) if random.random() > 0.4 else None,
            'receive_date': f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'narrative_text': None,
            'all_reactions': [{'name': 'Cytokine release syndrome', 'outcome': 'recovered'}]
        }
        jader_data.append(record)
    
    # Combine all data
    combined_data = faers_data + eu_data + jader_data
    
    # Save combined dataset
    with open('multi_source_crs_data.json', 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Created combined dataset with {len(combined_data)} records:")
    print(f"  - FAERS: {len(faers_data)}")
    print(f"  - Eudravigilance: {len(eu_data)}")
    print(f"  - JADER: {len(jader_data)}")
    
    return combined_data


if __name__ == "__main__":
    # Show instructions for data download
    print("="*60)
    print("PHARMACOVIGILANCE DATA EXTRACTORS")
    print("="*60)
    
    # Eudravigilance
    eu_extractor = EudravigilanceExtractor()
    eu_extractor.download_line_listing("epcoritamab")
    
    # JADER
    jader_extractor = JADERExtractor()
    jader_extractor.download_instructions()
    
    # Create simulated multi-source data for demonstration
    print("\n" + "="*60)
    print("CREATING SIMULATED MULTI-SOURCE DATASET")
    print("="*60)
    create_simulated_multi_source_data()

