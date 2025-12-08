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
    
    This extractor handles the line listing CSV format from adrreports.eu
    """
    
    def __init__(self, data_dir: str = "./"):
        self.data_dir = data_dir
        
    def download_line_listing(self, substance: str = "epcoritamab") -> str:
        """
        Instructions for downloading Eudravigilance line listings.
        
        Manual steps required:
        1. Go to https://www.adrreports.eu/
        2. Search for substance (e.g., "epcoritamab")
        3. Click on "Line Listing" tab
        4. Download the CSV file
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
        5. Click "Download" to get CSV file
        6. Save the file to: {self.data_dir}
        
        After downloading, run: extractor.parse_line_listing('filename.csv')
        ============================================================
        """)
        return self.data_dir
    
    def parse_line_listing(self, filepath: str) -> pd.DataFrame:
        """
        Parse Eudravigilance line listing CSV file.
        
        Expected columns from adrreports.eu:
        - EU Local Number
        - EV Gateway Receipt Date
        - Primary Source Country for Regulatory Purposes
        - Patient Age Group
        - Patient Sex
        - Reaction List PT (Duration – Outcome - Seriousness Criteria)
        - Suspect/interacting Drug List (Drug Char - Indication PT - Action taken - [Duration - Dose - Route])
        - Concomitant/Not Administered Drug List
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            print("Please download the file first using download_line_listing()")
            return pd.DataFrame()
        
        # Try CSV first, then Excel
        try:
            df = pd.read_csv(filepath)
        except:
            df = pd.read_excel(filepath)
        
        print(f"Loaded Eudravigilance data: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        df['source'] = 'eudravigilance'
        return df
    
    def extract_crs_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract CRS cases from Eudravigilance data."""
        if df.empty:
            return df
        
        # Find the reactions column (name may vary)
        reactions_col = None
        for col in df.columns:
            if 'reaction' in col.lower() and 'pt' in col.lower():
                reactions_col = col
                break
        
        if reactions_col is None:
            print("Warning: Could not find reactions column")
            return df
        
        # Filter for CRS reactions
        crs_mask = df[reactions_col].str.contains(
            'cytokine release syndrome', 
            case=False, 
            na=False
        )
        
        crs_df = df[crs_mask].copy()
        print(f"Found {len(crs_df)} CRS cases in Eudravigilance data")
        
        return crs_df
    
    def to_unified_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert to unified format matching FAERS extraction."""
        records = []
        
        # Identify column names (they may vary slightly)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'eu local number' in col_lower:
                col_map['report_id'] = col
            elif 'gateway receipt date' in col_lower:
                col_map['receive_date'] = col
            elif 'country' in col_lower and 'regulatory' in col_lower:
                col_map['country'] = col
            elif 'age group' in col_lower and 'reporter' not in col_lower:
                col_map['age_group'] = col
            elif 'sex' in col_lower:
                col_map['sex'] = col
            elif 'reaction' in col_lower and 'pt' in col_lower:
                col_map['reactions'] = col
            elif 'suspect' in col_lower and 'drug' in col_lower:
                col_map['suspect_drugs'] = col
            elif 'concomitant' in col_lower:
                col_map['concomitant_drugs'] = col
        
        for _, row in df.iterrows():
            # Parse reactions to get CRS outcome and check for seriousness
            reactions_str = str(row.get(col_map.get('reactions', ''), ''))
            crs_outcome, is_fatal, is_hospitalized, is_life_threatening = self._parse_reaction_details(reactions_str)
            
            # Parse suspect drugs to get dose info
            suspect_drugs_str = str(row.get(col_map.get('suspect_drugs', ''), ''))
            dose_mg, indication = self._parse_drug_details(suspect_drugs_str)
            
            # Parse concomitant drugs
            concomitant_str = str(row.get(col_map.get('concomitant_drugs', ''), ''))
            co_meds = self._parse_co_medications(suspect_drugs_str, concomitant_str)
            
            record = {
                'report_id': str(row.get(col_map.get('report_id', ''), '')),
                'source': 'eudravigilance',
                'is_crs': True,
                'crs_outcome': crs_outcome,
                'serious': True,  # CRS cases are typically serious
                'hospitalized': is_hospitalized,
                'death': is_fatal,
                'life_threatening': is_life_threatening,
                'disability': False,
                'epcoritamab_exposure': True,
                'epcoritamab_suspect': True,
                'epcoritamab_doses': [{'dose_mg': dose_mg, 'date': None, 'dose_type': 'full'}] if dose_mg else [],
                'co_medications': co_meds,
                'indication': indication,
                'age': self._parse_age_group(str(row.get(col_map.get('age_group', ''), ''))),
                'age_group': str(row.get(col_map.get('age_group', ''), '')),
                'sex': self._map_sex(str(row.get(col_map.get('sex', ''), ''))),
                'weight': None,
                'country': self._parse_country(str(row.get(col_map.get('country', ''), ''))),
                'crs_onset_date': None,
                'first_epcoritamab_date': None,
                'dose_to_crs_interval_days': None,
                'receive_date': str(row.get(col_map.get('receive_date', ''), ''))[:10],
                'narrative_text': None,
                'all_reactions': self._parse_all_reactions(reactions_str)
            }
            records.append(record)
        
        return records
    
    def _parse_reaction_details(self, reactions_str: str) -> tuple:
        """
        Parse reaction string to extract CRS outcome and seriousness.
        Format: "Cytokine release syndrome (n/a - Unknown - Other Medically Important Condition)"
        """
        crs_outcome = 'unknown'
        is_fatal = False
        is_hospitalized = False
        is_life_threatening = False
        
        # Find CRS-specific entry
        reactions_str_lower = reactions_str.lower()
        
        if 'cytokine release syndrome' in reactions_str_lower:
            # Extract the part after "Cytokine release syndrome"
            import re
            crs_match = re.search(r'cytokine release syndrome\s*\([^)]+\)', reactions_str, re.IGNORECASE)
            if crs_match:
                crs_part = crs_match.group(0).lower()
                
                # Parse outcome
                if 'recovered' in crs_part and 'not' not in crs_part:
                    crs_outcome = 'recovered'
                elif 'recovering' in crs_part:
                    crs_outcome = 'recovering'
                elif 'not recovered' in crs_part or 'not resolved' in crs_part:
                    crs_outcome = 'not_recovered'
                elif 'fatal' in crs_part:
                    crs_outcome = 'fatal'
                    is_fatal = True
        
        # Check whole reactions string for seriousness criteria
        if 'fatal' in reactions_str_lower or 'results in death' in reactions_str_lower:
            is_fatal = True
            if crs_outcome == 'unknown':
                crs_outcome = 'fatal'
        if 'hospitalisation' in reactions_str_lower or 'hospitalization' in reactions_str_lower:
            is_hospitalized = True
        if 'life threatening' in reactions_str_lower:
            is_life_threatening = True
            
        return crs_outcome, is_fatal, is_hospitalized, is_life_threatening
    
    def _parse_drug_details(self, drugs_str: str) -> tuple:
        """
        Parse drug string to extract dose and indication.
        Format: "TEPKINLY [EPCORITAMAB] (S - B-cell lymphoma - Drug withdrawn - [98d - 48mg - Subcutaneous use])"
        """
        dose_mg = None
        indication = None
        
        import re
        
        # Find epcoritamab entry
        epcoritamab_match = re.search(
            r'(?:tepkinly|epcoritamab)[^\(]*\([^)]+\)', 
            drugs_str, 
            re.IGNORECASE
        )
        
        if epcoritamab_match:
            drug_part = epcoritamab_match.group(0)
            
            # Extract dose (look for patterns like "48mg", "24mg", ".16mg", "0.8mg")
            dose_match = re.search(r'[\d.]+\s*mg', drug_part, re.IGNORECASE)
            if dose_match:
                try:
                    dose_mg = float(re.search(r'[\d.]+', dose_match.group(0)).group(0))
                except:
                    pass
            
            # Extract indication (text after "S -" and before next " -")
            indication_match = re.search(r'\(S\s*-\s*([^-]+)\s*-', drug_part, re.IGNORECASE)
            if indication_match:
                indication = indication_match.group(1).strip()
        
        return dose_mg, indication
    
    def _parse_co_medications(self, suspect_str: str, concomitant_str: str) -> List[str]:
        """Parse co-medications from drug strings."""
        co_meds = []
        
        import re
        
        # From concomitant drugs
        if concomitant_str and concomitant_str.lower() != 'not reported':
            # Extract drug names in brackets or before brackets
            drug_matches = re.findall(r'([A-Z][A-Z\s]+)(?:\s*\[|\s*\()', concomitant_str)
            for drug in drug_matches:
                drug_clean = drug.strip().upper()
                if drug_clean and 'epcoritamab' not in drug_clean.lower():
                    co_meds.append(drug_clean)
        
        # Also check suspect drugs for non-epcoritamab entries
        if suspect_str:
            drug_matches = re.findall(r'([A-Z][A-Z\s]+)(?:\s*\[|\s*\()', suspect_str)
            for drug in drug_matches:
                drug_clean = drug.strip().upper()
                if drug_clean and 'epcoritamab' not in drug_clean.lower() and 'tepkinly' not in drug_clean.lower():
                    if drug_clean not in co_meds:
                        co_meds.append(drug_clean)
        
        return list(set(co_meds))
    
    def _parse_all_reactions(self, reactions_str: str) -> List[Dict]:
        """Parse all reactions from reaction string."""
        reactions = []
        
        if pd.isna(reactions_str) or not reactions_str:
            return reactions
        
        # Split by <BR><BR> or comma patterns
        import re
        parts = re.split(r'<BR><BR>|,(?=[A-Z])', str(reactions_str))
        
        for part in parts:
            part = part.strip()
            if part:
                # Extract reaction name (text before first parenthesis)
                name_match = re.match(r'^([^(]+)', part)
                if name_match:
                    name = name_match.group(1).strip()
                    
                    # Try to extract outcome from parentheses
                    outcome = None
                    if 'recovered' in part.lower() and 'not' not in part.lower():
                        outcome = 'recovered'
                    elif 'not recovered' in part.lower():
                        outcome = 'not_recovered'
                    elif 'fatal' in part.lower():
                        outcome = 'fatal'
                    
                    reactions.append({'name': name, 'outcome': outcome})
        
        return reactions
    
    def _parse_country(self, country_str: str) -> str:
        """Parse country from country string."""
        country_str = str(country_str).strip()
        if 'european economic area' in country_str.lower():
            return 'EU'
        return country_str[:2].upper() if len(country_str) >= 2 else 'EU'
    
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
        """Load JADER CSV files (handles date-suffixed filenames like demo202511-01.csv)."""
        import glob
        
        file_patterns = {
            'demo': 'demo*.csv',
            'drug': 'drug*.csv',
            'reac': 'reac*.csv',
            'hist': 'hist*.csv'
        }
        
        data = {}
        for key, pattern in file_patterns.items():
            # Find matching files
            matches = glob.glob(os.path.join(self.data_dir, pattern))
            
            if matches:
                filepath = matches[0]  # Use first match
                # JADER uses Shift-JIS (CP932) encoding
                try:
                    data[key] = pd.read_csv(filepath, encoding='cp932')
                except:
                    try:
                        data[key] = pd.read_csv(filepath, encoding='shift-jis')
                    except:
                        data[key] = pd.read_csv(filepath, encoding='utf-8')
                print(f"Loaded {key}: {len(data[key])} records from {os.path.basename(filepath)}")
            else:
                print(f"No file matching {pattern} found in {self.data_dir}")
                data[key] = pd.DataFrame()
        
        return data
    
    def extract_epcoritamab_cases(self, data: Dict[str, pd.DataFrame]) -> list:
        """
        Extract cases involving Epcoritamab from JADER data.
        
        JADER drug.csv columns (Japanese):
        - 識別番号 (Case ID)
        - 医薬品連番 (Drug sequence number)
        - 医薬品（一般名）(Generic name)
        - 医薬品（販売名）(Brand name)
        - 医薬品の関与 (Drug involvement: 被疑薬=suspect, 併用薬=concomitant)
        """
        if data['drug'].empty:
            print("JADER drug data is empty")
            return []
        
        drug_df = data['drug']
        
        # Find Epcoritamab cases (check both generic and brand names)
        # Japanese names: エプコリタマブ (epcoritamab), エプキンリ (Epkinly)
        epcoritamab_names = ['epcoritamab', 'エプコリタマブ', 'tepkinly', 'テプキンリー', 'エプキンリ']
        
        mask = pd.Series([False] * len(drug_df))
        for col in drug_df.columns:
            if drug_df[col].dtype == 'object':
                for name in epcoritamab_names:
                    col_mask = drug_df[col].astype(str).str.contains(name, case=False, na=False)
                    mask = mask | col_mask
        
        # Find the case ID column (識別番号)
        id_col = None
        for col in drug_df.columns:
            if '識別番号' in col or 'id' in col.lower():
                id_col = col
                break
        
        if id_col is None and len(drug_df.columns) > 0:
            id_col = drug_df.columns[0]  # Fallback to first column
        
        if mask.sum() > 0 and id_col:
            epcoritamab_cases = drug_df[mask][id_col].unique().tolist()
        else:
            epcoritamab_cases = []
        
        print(f"Found {len(epcoritamab_cases)} cases with Epcoritamab in JADER")
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
        
        reac_df = data['reac'].copy()
        
        # Filter for case IDs
        id_col = '識別番号' if '識別番号' in reac_df.columns else reac_df.columns[0]
        reac_filtered = reac_df[reac_df[id_col].isin(case_ids)].copy()
        reac_filtered = reac_filtered.reset_index(drop=True)
        
        # Find CRS reactions (including Japanese term)
        crs_terms = ['cytokine release syndrome', 'サイトカイン放出症候群', 'crs', 'サイトカイン']
        
        mask = pd.Series([False] * len(reac_filtered), index=reac_filtered.index)
        for col in reac_filtered.columns:
            if reac_filtered[col].dtype == 'object':
                for term in crs_terms:
                    col_mask = reac_filtered[col].astype(str).str.contains(term, case=False, na=False)
                    mask = mask | col_mask
        
        crs_cases = reac_filtered[mask].copy()
        print(f"Found {len(crs_cases)} CRS reactions in {crs_cases[id_col].nunique()} unique cases")
        
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
        """Extract age from demographics (handles Japanese age format like '60歳代')."""
        age_cols = [c for c in demo_row.columns if '年齢' in c or 'age' in c.lower()]
        if age_cols and not demo_row.empty:
            try:
                age_str = str(demo_row[age_cols[0]].values[0])
                # Try direct conversion first
                return float(age_str)
            except:
                # Handle Japanese age format like "60歳代" (60s), "70歳代" (70s)
                import re
                match = re.search(r'(\d+)', age_str)
                if match:
                    base_age = int(match.group(1))
                    # If it's a decade format, add 5 to get midpoint
                    if '歳代' in age_str or '代' in age_str:
                        return float(base_age + 5)
                    return float(base_age)
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


def create_multi_source_data(
    faers_path: str = 'crs_extracted_data.json',
    eudravigilance_path: str = None,
    jader_data_dir: str = None,
    output_path: str = 'multi_source_crs_data.json'
):
    """
    Create multi-source dataset from FAERS and real Eudravigilance/JADER data.
    
    Args:
        faers_path: Path to extracted FAERS data
        eudravigilance_path: Path to Eudravigilance CSV (None to skip)
        jader_data_dir: Path to JADER data directory (None to skip)
        output_path: Output file path
    """
    import json
    
    # Load FAERS data
    print("Loading FAERS data...")
    with open(faers_path, 'r') as f:
        faers_data = json.load(f)
    
    # Add source tag to FAERS data
    for record in faers_data:
        record['source'] = 'faers'
    
    print(f"  - FAERS: {len(faers_data)} records")
    
    # Load Eudravigilance data if provided
    eu_data = []
    if eudravigilance_path and os.path.exists(eudravigilance_path):
        print(f"Loading Eudravigilance data from {eudravigilance_path}...")
        eu_extractor = EudravigilanceExtractor()
        eu_df = eu_extractor.parse_line_listing(eudravigilance_path)
        eu_crs = eu_extractor.extract_crs_cases(eu_df)
        eu_data = eu_extractor.to_unified_format(eu_crs)
        print(f"  - Eudravigilance: {len(eu_data)} CRS records")
    else:
        print("  - Eudravigilance: No data provided (skipping)")
    
    # Load JADER data if provided
    jader_data = []
    if jader_data_dir and os.path.exists(jader_data_dir):
        print(f"Loading JADER data from {jader_data_dir}...")
        jader_extractor = JADERExtractor(jader_data_dir)
        jader_files = jader_extractor.load_jader_data()
        jader_data = jader_extractor.build_unified_dataset(jader_files)
        print(f"  - JADER: {len(jader_data)} CRS records")
    else:
        print("  - JADER: No data provided (skipping)")
    
    # Combine all data
    combined_data = faers_data + eu_data + jader_data
    
    # Save combined dataset
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Created combined dataset: {output_path}")
    print(f"Total records: {len(combined_data)}")
    print(f"  - FAERS: {len(faers_data)}")
    print(f"  - Eudravigilance: {len(eu_data)}")
    print(f"  - JADER: {len(jader_data)}")
    print(f"{'='*50}")
    
    return combined_data


def create_simulated_multi_source_data():
    """
    DEPRECATED: Use create_multi_source_data() with real data instead.
    
    This function creates simulated/fake data for demonstration only.
    """
    import random
    import json
    
    print("WARNING: Using simulated data. For real analysis, use create_multi_source_data() with real data files.")
    
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
    print(f"  - Eudravigilance (SIMULATED): {len(eu_data)}")
    print(f"  - JADER (SIMULATED): {len(jader_data)}")
    
    return combined_data


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("PHARMACOVIGILANCE DATA EXTRACTORS")
    print("="*60)
    
    # Check for real data files
    # Default paths (relative to project root)
    eudravigilance_csv = "../Run Line Listing Report.csv"
    jader_dir = "../jader_data"
    
    # Check if files exist
    eu_exists = os.path.exists(eudravigilance_csv)
    jader_exists = os.path.exists(jader_dir)
    
    print(f"\nData file check:")
    print(f"  - Eudravigilance CSV: {'FOUND' if eu_exists else 'NOT FOUND'} ({eudravigilance_csv})")
    print(f"  - JADER directory: {'FOUND' if jader_exists else 'NOT FOUND'} ({jader_dir})")
    
    if eu_exists or jader_exists:
        print("\n" + "="*60)
        print("CREATING MULTI-SOURCE DATASET WITH REAL DATA")
        print("="*60)
        
        create_multi_source_data(
            faers_path='crs_extracted_data.json',
            eudravigilance_path=eudravigilance_csv if eu_exists else None,
            jader_data_dir=jader_dir if jader_exists else None,
            output_path='multi_source_crs_data.json'
        )
    else:
        print("\nNo real data files found. Showing download instructions...")
        
        # Eudravigilance
        eu_extractor = EudravigilanceExtractor()
        eu_extractor.download_line_listing("epcoritamab")
        
        # JADER
        jader_extractor = JADERExtractor()
        jader_extractor.download_instructions()
        
        print("\n" + "="*60)
        print("CREATING SIMULATED MULTI-SOURCE DATASET (for demo)")
        print("="*60)
        create_simulated_multi_source_data()

