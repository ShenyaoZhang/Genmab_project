"""
NLP Analysis for CRS Narratives using BERT
Extracts features from narrative text that may predict CRS severity.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


class CRSNarrativeAnalyzer:
    """
    NLP-based analysis of FAERS narrative texts for CRS risk prediction.
    
    Uses:
    1. BioBERT/ClinicalBERT for medical text understanding
    2. Named Entity Recognition for symptom extraction
    3. Sentiment/severity classification
    4. Feature extraction for causal modeling
    """
    
    def __init__(self, data_path: str = "crs_extracted_data.json"):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data with narratives."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        records = []
        for record in data:
            records.append({
                'report_id': record.get('report_id'),
                'narrative_text': record.get('narrative_text'),
                'crs_outcome': record.get('crs_outcome'),
                'death': record.get('death', False),
                'severe_crs': (
                    record.get('death', False) or 
                    record.get('life_threatening', False) or
                    record.get('crs_outcome') == 'not_recovered'
                )
            })
        
        self.df = pd.DataFrame(records)
        
        # Filter to records with narratives
        self.df_with_narrative = self.df[self.df['narrative_text'].notna()].copy()
        print(f"Loaded {len(self.df)} records, {len(self.df_with_narrative)} with narratives")
        
        return self.df
    
    def setup_bert(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """
        Setup BERT model for text analysis.
        
        Options:
        - dmis-lab/biobert-base-cased-v1.1 (BioBERT)
        - emilyalsentzer/Bio_ClinicalBERT (ClinicalBERT)
        - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            print("BERT model loaded successfully")
            return True
            
        except ImportError:
            print("transformers library not installed. Using rule-based extraction instead.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_bert_features(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Extract BERT embeddings for texts."""
        import torch
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def rule_based_extraction(self, text: str) -> Dict:
        """
        Rule-based feature extraction from narrative text.
        Used when BERT is not available.
        """
        if not text or pd.isna(text):
            return {
                'has_narrative': False,
                'narrative_length': 0,
                'severity_score': 0,
                'mentions_fever': False,
                'mentions_hypotension': False,
                'mentions_hypoxia': False,
                'mentions_icu': False,
                'mentions_intubation': False,
                'mentions_vasopressor': False,
                'mentions_tocilizumab': False,
                'mentions_steroids': False,
                'crs_grade_mentioned': None,
                'time_to_onset_mentioned': None
            }
        
        text_lower = text.lower()
        
        features = {
            'has_narrative': True,
            'narrative_length': len(text),
            
            # Severity indicators
            'mentions_fever': bool(re.search(r'fever|pyrexia|temperature|febrile', text_lower)),
            'mentions_hypotension': bool(re.search(r'hypotension|low blood pressure|bp drop', text_lower)),
            'mentions_hypoxia': bool(re.search(r'hypoxia|oxygen|desaturation|spo2|o2', text_lower)),
            'mentions_icu': bool(re.search(r'icu|intensive care|critical care', text_lower)),
            'mentions_intubation': bool(re.search(r'intubat|ventilat|mechanical ventilation', text_lower)),
            'mentions_vasopressor': bool(re.search(r'vasopressor|norepinephrine|epinephrine|dopamine', text_lower)),
            
            # Treatment indicators
            'mentions_tocilizumab': bool(re.search(r'tocilizumab|actemra|il-?6', text_lower)),
            'mentions_steroids': bool(re.search(r'steroid|dexamethasone|predniso|methylpred|hydrocort', text_lower)),
            
            # CRS grade
            'crs_grade_mentioned': self._extract_crs_grade(text_lower),
            
            # Time to onset
            'time_to_onset_mentioned': self._extract_time_to_onset(text_lower)
        }
        
        # Calculate severity score
        severity_indicators = [
            'mentions_hypotension', 'mentions_hypoxia', 'mentions_icu',
            'mentions_intubation', 'mentions_vasopressor'
        ]
        features['severity_score'] = sum(features[ind] for ind in severity_indicators)
        
        return features
    
    def _extract_crs_grade(self, text: str) -> Optional[int]:
        """Extract CRS grade from text."""
        # Look for grade mentions
        grade_patterns = [
            r'grade\s*(\d)',
            r'crs\s*grade\s*(\d)',
            r'grade\s*(\d)\s*crs',
            r'g(\d)\s*crs'
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, text)
            if match:
                grade = int(match.group(1))
                if 1 <= grade <= 4:
                    return grade
        
        return None
    
    def _extract_time_to_onset(self, text: str) -> Optional[int]:
        """Extract time to CRS onset in hours."""
        # Look for time mentions
        patterns = [
            r'(\d+)\s*hours?\s*(?:after|following|post)',
            r'(\d+)\s*days?\s*(?:after|following|post)',
            r'within\s*(\d+)\s*hours?',
            r'(\d+)h\s*post'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                # Convert days to hours if needed
                if 'day' in pattern:
                    value *= 24
                return value
        
        return None
    
    def extract_all_features(self) -> pd.DataFrame:
        """Extract features from all narratives."""
        
        features_list = []
        
        for _, row in self.df.iterrows():
            features = self.rule_based_extraction(row['narrative_text'])
            features['report_id'] = row['report_id']
            features['crs_outcome'] = row['crs_outcome']
            features['death'] = row['death']
            features['severe_crs'] = row['severe_crs']
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        
        return self.features_df
    
    def train_severity_classifier(self) -> Dict:
        """
        Train a classifier to predict CRS severity from narrative features.
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import classification_report, roc_auc_score
        
        # Prepare features
        feature_cols = [
            'narrative_length', 'severity_score',
            'mentions_fever', 'mentions_hypotension', 'mentions_hypoxia',
            'mentions_icu', 'mentions_intubation', 'mentions_vasopressor',
            'mentions_tocilizumab', 'mentions_steroids'
        ]
        
        # Filter to records with narratives
        df_train = self.features_df[self.features_df['has_narrative'] == True].copy()
        
        if len(df_train) < 20:
            return {'error': 'Insufficient narratives for training'}
        
        X = df_train[feature_cols].fillna(0).astype(float)
        y = df_train['severe_crs'].astype(int)
        
        # Handle class imbalance info
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        # Fit final model
        model.fit(X, y)
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        
        results = {
            'n_samples': len(df_train),
            'n_severe': int(y.sum()),
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'feature_importance': importance,
            'top_features': sorted(importance.items(), key=lambda x: -x[1])[:5]
        }
        
        self.severity_model = model
        
        return results
    
    def analyze_severity_patterns(self) -> Dict:
        """Analyze patterns in narrative features by outcome."""
        
        if self.features_df is None:
            self.extract_all_features()
        
        # Group by outcome
        severe_df = self.features_df[self.features_df['severe_crs'] == True]
        mild_df = self.features_df[self.features_df['severe_crs'] == False]
        
        feature_cols = [
            'mentions_fever', 'mentions_hypotension', 'mentions_hypoxia',
            'mentions_icu', 'mentions_intubation', 'mentions_vasopressor',
            'mentions_tocilizumab', 'mentions_steroids', 'severity_score'
        ]
        
        patterns = {
            'severe_crs_patterns': {},
            'mild_crs_patterns': {},
            'differential_features': []
        }
        
        for col in feature_cols:
            if col in self.features_df.columns:
                severe_mean = severe_df[col].mean() if len(severe_df) > 0 else 0
                mild_mean = mild_df[col].mean() if len(mild_df) > 0 else 0
                
                patterns['severe_crs_patterns'][col] = severe_mean
                patterns['mild_crs_patterns'][col] = mild_mean
                
                diff = severe_mean - mild_mean
                if abs(diff) > 0.1:  # Meaningful difference
                    patterns['differential_features'].append({
                        'feature': col,
                        'severe_rate': severe_mean,
                        'mild_rate': mild_mean,
                        'difference': diff
                    })
        
        # Sort by difference
        patterns['differential_features'] = sorted(
            patterns['differential_features'],
            key=lambda x: -abs(x['difference'])
        )
        
        return patterns
    
    def generate_nlp_report(self) -> str:
        """Generate NLP analysis report."""
        
        report = []
        report.append("=" * 70)
        report.append("NLP ANALYSIS REPORT: CRS Narrative Features")
        report.append("=" * 70)
        report.append("")
        
        # Load and extract features
        self.load_data()
        self.extract_all_features()
        
        report.append("1. DATA OVERVIEW")
        report.append("-" * 50)
        report.append(f"Total records: {len(self.df)}")
        report.append(f"Records with narratives: {self.features_df['has_narrative'].sum()}")
        report.append(f"Average narrative length: {self.features_df['narrative_length'].mean():.0f} chars")
        report.append("")
        
        # Pattern analysis
        report.append("2. SEVERITY PATTERNS IN NARRATIVES")
        report.append("-" * 50)
        
        patterns = self.analyze_severity_patterns()
        
        report.append("\nDifferential features (severe vs mild CRS):")
        for feat in patterns['differential_features'][:10]:
            direction = "↑" if feat['difference'] > 0 else "↓"
            report.append(
                f"  {direction} {feat['feature']}: "
                f"{feat['severe_rate']*100:.1f}% vs {feat['mild_rate']*100:.1f}% "
                f"(diff: {feat['difference']*100:+.1f}%)"
            )
        
        report.append("")
        
        # Classifier results
        report.append("3. SEVERITY PREDICTION MODEL")
        report.append("-" * 50)
        
        classifier_results = self.train_severity_classifier()
        
        if 'error' not in classifier_results:
            report.append(f"\nSamples: {classifier_results['n_samples']}")
            report.append(f"Severe cases: {classifier_results['n_severe']}")
            report.append(f"Cross-validation AUC: {classifier_results['cv_auc_mean']:.3f} "
                         f"(±{classifier_results['cv_auc_std']:.3f})")
            
            report.append("\nTop predictive features:")
            for feat, imp in classifier_results['top_features']:
                report.append(f"  • {feat}: {imp:.3f}")
        else:
            report.append(f"Error: {classifier_results['error']}")
        
        report.append("")
        
        # CRS Grade distribution
        report.append("4. CRS GRADE MENTIONS")
        report.append("-" * 50)
        
        grade_counts = self.features_df['crs_grade_mentioned'].value_counts()
        report.append("\nGrades mentioned in narratives:")
        for grade, count in grade_counts.items():
            if pd.notna(grade):
                report.append(f"  Grade {int(grade)}: {count} cases")
        
        report.append("")
        
        # Time to onset
        report.append("5. TIME TO ONSET")
        report.append("-" * 50)
        
        onset_times = self.features_df['time_to_onset_mentioned'].dropna()
        if len(onset_times) > 0:
            report.append(f"\nOnset times extracted: {len(onset_times)} cases")
            report.append(f"Mean time to onset: {onset_times.mean():.1f} hours")
            report.append(f"Median time to onset: {onset_times.median():.1f} hours")
            report.append(f"Range: {onset_times.min():.0f} - {onset_times.max():.0f} hours")
        else:
            report.append("\nNo time to onset information extracted from narratives")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_features(self, output_path: str = "narrative_features.json"):
        """Save extracted features."""
        
        if self.features_df is None:
            self.extract_all_features()
        
        # Convert to records
        records = self.features_df.to_dict(orient='records')
        
        with open(output_path, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        
        print(f"Features saved to {output_path}")


def main():
    """Run NLP analysis."""
    
    analyzer = CRSNarrativeAnalyzer("crs_extracted_data.json")
    
    # Generate report
    report = analyzer.generate_nlp_report()
    print(report)
    
    # Save results
    analyzer.save_features()
    
    # Save report
    with open("nlp_analysis_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to nlp_analysis_report.txt")


if __name__ == "__main__":
    main()

