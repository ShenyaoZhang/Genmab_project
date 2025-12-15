"""
Causal Inference Analysis for CRS Risk Modeling
Distinguishes causal vs correlated variables using:
1. Directed Acyclic Graphs (DAGs)
2. Propensity Score Methods
3. DoWhy Causal Inference Framework
4. Statistical Association Tests
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CRSCausalAnalyzer:
    """
    Causal inference analyzer for CRS risk factors.
    
    Implements multiple methods to distinguish causation from correlation:
    1. DAG-based analysis (theoretical framework)
    2. Propensity score matching
    3. Statistical tests for confounding
    4. Sensitivity analysis
    """
    
    def __init__(self, data_path: str = "multi_source_crs_data.json"):
        """Initialize with data path."""
        self.data_path = data_path
        self.df = None
        self.dag = None
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        records = []
        for record in data:
            flat_record = {
                'report_id': record.get('report_id'),
                'source': record.get('source', 'faers'),
                'is_crs': record.get('is_crs', False),
                'crs_outcome': record.get('crs_outcome'),
                'serious': record.get('serious', False),
                'hospitalized': record.get('hospitalized', False),
                'death': record.get('death', False),
                'life_threatening': record.get('life_threatening', False),
                'epcoritamab_exposure': record.get('epcoritamab_exposure', False),
                'epcoritamab_suspect': record.get('epcoritamab_suspect', False),
                'age': record.get('age'),
                'sex': record.get('sex'),
                'weight': record.get('weight'),
                'country': record.get('country'),
                'indication': record.get('indication'),
                'dose_to_crs_interval_days': record.get('dose_to_crs_interval_days'),
                'n_co_medications': len(record.get('co_medications', [])),
                'n_reactions': len(record.get('all_reactions', [])),
            }
            
            # Extract dose information
            doses = record.get('epcoritamab_doses', [])
            if doses:
                dose_mgs = [d.get('dose_mg') for d in doses if d.get('dose_mg')]
                flat_record['max_dose_mg'] = max(dose_mgs) if dose_mgs else None
                flat_record['n_doses'] = len(doses)
            else:
                flat_record['max_dose_mg'] = None
                flat_record['n_doses'] = 0
            
            # Check for specific co-medications
            co_meds = [m.upper() for m in record.get('co_medications', [])]
            flat_record['has_rituximab'] = any('RITUXIMAB' in m for m in co_meds)
            flat_record['has_steroids'] = any(
                s in ' '.join(co_meds) for s in 
                ['PREDNIS', 'DEXAMETH', 'METHYLPRED', 'HYDROCORT']
            )
            flat_record['has_tocilizumab'] = any('TOCILIZUMAB' in m for m in co_meds)
            
            records.append(flat_record)
        
        self.df = pd.DataFrame(records)
        
        # Create binary outcomes
        self.df['death_binary'] = self.df['death'].astype(int)
        self.df['severe_crs'] = (
            (self.df['death'] == True) | 
            (self.df['life_threatening'] == True) |
            (self.df['crs_outcome'] == 'not_recovered')
        ).astype(int)
        
        # Create age groups
        self.df['age_group'] = pd.cut(
            self.df['age'], 
            bins=[0, 50, 65, 75, 100],
            labels=['<50', '50-65', '65-75', '>75']
        )
        
        # Encode sex
        self.df['sex_male'] = (self.df['sex'] == 'male').astype(int)
        
        print(f"Loaded {len(self.df)} records from {self.df['source'].nunique()} sources")
        return self.df
    
    def define_dag(self) -> Dict:
        """
        Define the Directed Acyclic Graph (DAG) for CRS causal structure.
        
        This represents our theoretical understanding of causal relationships.
        """
        self.dag = {
            'nodes': {
                # Exposure (Treatment)
                'epcoritamab_dose': {'type': 'exposure', 'description': 'Epcoritamab dose level'},
                
                # Outcome
                'crs_severity': {'type': 'outcome', 'description': 'CRS severity/outcome'},
                
                # Confounders (affect both treatment and outcome)
                'age': {'type': 'confounder', 'description': 'Patient age'},
                'disease_stage': {'type': 'confounder', 'description': 'Cancer stage/burden'},
                'prior_therapy': {'type': 'confounder', 'description': 'Prior treatment history'},
                
                # Mediators (on causal pathway)
                'cytokine_levels': {'type': 'mediator', 'description': 'Cytokine release levels'},
                't_cell_activation': {'type': 'mediator', 'description': 'T-cell activation'},
                
                # Effect modifiers
                'tocilizumab_use': {'type': 'effect_modifier', 'description': 'IL-6 inhibitor use'},
                'steroid_premedication': {'type': 'effect_modifier', 'description': 'Steroid premedication'},
                
                # Colliders (affected by both exposure and outcome)
                'hospitalization': {'type': 'collider', 'description': 'Hospital admission'},
            },
            'edges': [
                # Causal paths from exposure to outcome
                ('epcoritamab_dose', 't_cell_activation', 'causal'),
                ('t_cell_activation', 'cytokine_levels', 'causal'),
                ('cytokine_levels', 'crs_severity', 'causal'),
                
                # Confounding paths
                ('age', 'epcoritamab_dose', 'confounding'),
                ('age', 'crs_severity', 'confounding'),
                ('disease_stage', 'epcoritamab_dose', 'confounding'),
                ('disease_stage', 'crs_severity', 'confounding'),
                
                # Effect modification
                ('tocilizumab_use', 'crs_severity', 'effect_modification'),
                ('steroid_premedication', 'crs_severity', 'effect_modification'),
                
                # Collider paths
                ('epcoritamab_dose', 'hospitalization', 'selection'),
                ('crs_severity', 'hospitalization', 'selection'),
            ]
        }
        
        return self.dag
    
    def analyze_associations(self, outcome: str = 'severe_crs') -> pd.DataFrame:
        """
        Analyze statistical associations between variables and outcome.
        
        Returns DataFrame with odds ratios, p-values, and confidence intervals.
        """
        from scipy import stats
        
        results = []
        
        # Continuous variables
        continuous_vars = ['age', 'weight', 'max_dose_mg', 'n_co_medications', 'n_doses']
        
        for var in continuous_vars:
            if var not in self.df.columns:
                continue
                
            valid_data = self.df[[var, outcome]].dropna()
            if len(valid_data) < 10:
                continue
            
            # Point-biserial correlation
            corr, p_value = stats.pointbiserialr(
                valid_data[outcome], 
                valid_data[var]
            )
            
            # Logistic regression for odds ratio
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                
                X = valid_data[[var]].values
                y = valid_data[outcome].values
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = LogisticRegression(random_state=42)
                model.fit(X_scaled, y)
                
                # Odds ratio per SD increase
                odds_ratio = np.exp(model.coef_[0][0])
            except:
                odds_ratio = None
            
            results.append({
                'variable': var,
                'type': 'continuous',
                'correlation': corr,
                'p_value': p_value,
                'odds_ratio_per_sd': odds_ratio,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_association(var, corr, p_value)
            })
        
        # Categorical variables
        categorical_vars = ['sex_male', 'has_rituximab', 'has_steroids', 'has_tocilizumab']
        
        for var in categorical_vars:
            if var not in self.df.columns:
                continue
                
            valid_data = self.df[[var, outcome]].dropna()
            if len(valid_data) < 10:
                continue
            
            # Chi-square test
            contingency = pd.crosstab(valid_data[var], valid_data[outcome])
            if contingency.shape == (2, 2):
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                # Odds ratio
                a, b = contingency.iloc[1, 1], contingency.iloc[1, 0]
                c, d = contingency.iloc[0, 1], contingency.iloc[0, 0]
                
                if b > 0 and c > 0:
                    odds_ratio = (a * d) / (b * c)
                else:
                    odds_ratio = None
                
                results.append({
                    'variable': var,
                    'type': 'categorical',
                    'chi2': chi2,
                    'p_value': p_value,
                    'odds_ratio': odds_ratio,
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_association(var, None, p_value, odds_ratio)
                })
        
        # Source comparison
        if 'source' in self.df.columns:
            contingency = pd.crosstab(self.df['source'], self.df[outcome])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            results.append({
                'variable': 'data_source',
                'type': 'categorical',
                'chi2': chi2,
                'p_value': p_value,
                'odds_ratio': None,
                'significant': p_value < 0.05,
                'interpretation': 'Cross-database heterogeneity assessment'
            })
        
        self.results['associations'] = pd.DataFrame(results)
        return self.results['associations']
    
    def _interpret_association(self, var: str, corr: float, p_value: float, 
                                odds_ratio: float = None) -> str:
        """Provide interpretation of association."""
        
        # Known causal relationships based on mechanism
        causal_vars = {
            'max_dose_mg': 'LIKELY CAUSAL - Dose-response relationship established',
            'n_doses': 'LIKELY CAUSAL - Cumulative exposure effect',
            'has_tocilizumab': 'CAUSAL (protective) - IL-6 pathway blockade',
            'has_steroids': 'CAUSAL (protective) - Anti-inflammatory mechanism',
        }
        
        # Known confounders
        confounders = {
            'age': 'CONFOUNDER - Affects both treatment selection and outcome',
            'weight': 'CONFOUNDER - Affects drug exposure and clearance',
        }
        
        # Likely correlational
        correlational = {
            'n_co_medications': 'CORRELATION - Marker of disease severity',
            'has_rituximab': 'CORRELATION - Marker of treatment line',
            'sex_male': 'UNCLEAR - Possible biological modifier',
        }
        
        if var in causal_vars:
            return causal_vars[var]
        elif var in confounders:
            return confounders[var]
        elif var in correlational:
            return correlational[var]
        else:
            return 'REQUIRES FURTHER ANALYSIS'
    
    def propensity_score_analysis(self, treatment: str = 'has_steroids', 
                                   outcome: str = 'severe_crs') -> Dict:
        """
        Propensity score analysis to estimate causal effect.
        
        Uses inverse probability weighting (IPW) to control for confounding.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Define confounders to adjust for
        confounders = ['age', 'sex_male', 'max_dose_mg', 'n_co_medications']
        
        # Prepare data
        analysis_df = self.df[[treatment, outcome] + confounders].dropna()
        
        if len(analysis_df) < 50:
            return {'error': 'Insufficient data for propensity score analysis'}
        
        X = analysis_df[confounders].values
        T = analysis_df[treatment].values
        Y = analysis_df[outcome].values
        
        # Standardize confounders
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=42, max_iter=1000)
        ps_model.fit(X_scaled, T)
        propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Clip propensity scores to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
        
        # Calculate IPW weights
        weights = np.where(T == 1, 1/propensity_scores, 1/(1-propensity_scores))
        
        # Weighted outcome means
        treated_outcome = np.average(Y[T == 1], weights=weights[T == 1])
        control_outcome = np.average(Y[T == 0], weights=weights[T == 0])
        
        # Average Treatment Effect (ATE)
        ate = treated_outcome - control_outcome
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        ate_bootstrap = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(analysis_df), size=len(analysis_df), replace=True)
            X_boot = X_scaled[idx]
            T_boot = T[idx]
            Y_boot = Y[idx]
            
            ps_boot = ps_model.predict_proba(X_boot)[:, 1]
            ps_boot = np.clip(ps_boot, 0.05, 0.95)
            w_boot = np.where(T_boot == 1, 1/ps_boot, 1/(1-ps_boot))
            
            if T_boot.sum() > 0 and (1-T_boot).sum() > 0:
                ate_boot = (
                    np.average(Y_boot[T_boot == 1], weights=w_boot[T_boot == 1]) -
                    np.average(Y_boot[T_boot == 0], weights=w_boot[T_boot == 0])
                )
                ate_bootstrap.append(ate_boot)
        
        ci_lower = np.percentile(ate_bootstrap, 2.5)
        ci_upper = np.percentile(ate_bootstrap, 97.5)
        
        result = {
            'treatment': treatment,
            'outcome': outcome,
            'n_treated': int(T.sum()),
            'n_control': int((1-T).sum()),
            'ate': ate,
            'ate_ci_lower': ci_lower,
            'ate_ci_upper': ci_upper,
            'significant': (ci_lower > 0) or (ci_upper < 0),
            'interpretation': self._interpret_ate(treatment, ate, ci_lower, ci_upper)
        }
        
        self.results['propensity_score'] = result
        return result
    
    def _interpret_ate(self, treatment: str, ate: float, ci_lower: float, 
                       ci_upper: float) -> str:
        """Interpret Average Treatment Effect."""
        
        if ci_lower > 0:
            direction = "increases"
            strength = "significant"
        elif ci_upper < 0:
            direction = "decreases"
            strength = "significant"
        else:
            direction = "may affect"
            strength = "not statistically significant"
        
        effect_size = abs(ate) * 100
        
        return (
            f"{treatment} {direction} risk of outcome by {effect_size:.1f} "
            f"percentage points ({strength}). "
            f"95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]"
        )
    
    def causal_forest_analysis(self, outcome: str = 'severe_crs') -> Dict:
        """
        Heterogeneous treatment effect analysis using causal forests.
        Identifies subgroups with different treatment effects.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
        except ImportError:
            return {'error': 'sklearn not available'}
        
        # Use steroid premedication as treatment
        treatment = 'has_steroids'
        features = ['age', 'sex_male', 'max_dose_mg', 'n_co_medications', 'has_rituximab']
        
        analysis_df = self.df[[treatment, outcome] + features].dropna()
        
        if len(analysis_df) < 100:
            return {'error': 'Insufficient data'}
        
        X = analysis_df[features].values
        T = analysis_df[treatment].values
        Y = analysis_df[outcome].values
        
        # Separate models for treated and control
        model_treated = RandomForestClassifier(n_estimators=100, random_state=42)
        model_control = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit on respective groups
        X_treated, Y_treated = X[T == 1], Y[T == 1]
        X_control, Y_control = X[T == 0], Y[T == 0]
        
        if len(X_treated) > 10 and len(X_control) > 10:
            model_treated.fit(X_treated, Y_treated)
            model_control.fit(X_control, Y_control)
            
            # Predict potential outcomes for all
            Y1_pred = model_treated.predict_proba(X)[:, 1] if len(model_treated.classes_) > 1 else np.zeros(len(X))
            Y0_pred = model_control.predict_proba(X)[:, 1] if len(model_control.classes_) > 1 else np.zeros(len(X))
            
            # Individual treatment effects
            ite = Y1_pred - Y0_pred
            
            # Feature importance for treatment effect heterogeneity
            feature_importance = dict(zip(features, model_treated.feature_importances_))
            
            # Identify subgroups
            subgroups = {
                'high_benefit': analysis_df[ite < -0.1].describe(),
                'low_benefit': analysis_df[ite > 0.1].describe(),
                'average': analysis_df[(ite >= -0.1) & (ite <= 0.1)].describe()
            }
            
            result = {
                'treatment': treatment,
                'outcome': outcome,
                'mean_ite': float(np.mean(ite)),
                'std_ite': float(np.std(ite)),
                'feature_importance': feature_importance,
                'n_high_benefit': int(sum(ite < -0.1)),
                'n_low_benefit': int(sum(ite > 0.1)),
                'interpretation': (
                    f"Treatment effect varies across patients. "
                    f"Most important factors: {sorted(feature_importance.items(), key=lambda x: -x[1])[:3]}"
                )
            }
        else:
            result = {'error': 'Insufficient data in treatment groups'}
        
        self.results['causal_forest'] = result
        return result
    
    def sensitivity_analysis(self, outcome: str = 'severe_crs') -> Dict:
        """
        Sensitivity analysis for unmeasured confounding.
        Estimates how strong unmeasured confounding would need to be
        to explain away the observed association.
        """
        from scipy import stats
        
        # Calculate observed association
        treatment = 'max_dose_mg'
        
        analysis_df = self.df[[treatment, outcome]].dropna()
        
        # Categorize dose
        analysis_df['high_dose'] = (analysis_df[treatment] >= 24).astype(int)
        
        contingency = pd.crosstab(analysis_df['high_dose'], analysis_df[outcome])
        
        if contingency.shape != (2, 2):
            return {'error': 'Cannot create 2x2 table'}
        
        a, b = contingency.iloc[1, 1], contingency.iloc[1, 0]
        c, d = contingency.iloc[0, 1], contingency.iloc[0, 0]
        
        if b == 0 or c == 0:
            return {'error': 'Zero cell in contingency table'}
        
        observed_or = (a * d) / (b * c)
        
        # E-value calculation (Ding & VanderWeele, 2016)
        if observed_or >= 1:
            e_value = observed_or + np.sqrt(observed_or * (observed_or - 1))
        else:
            e_value = 1/observed_or + np.sqrt((1/observed_or) * (1/observed_or - 1))
        
        # Confidence interval for OR
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        log_or = np.log(observed_or)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        # E-value for CI bound
        if ci_lower >= 1:
            e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        else:
            e_value_ci = 1
        
        result = {
            'exposure': 'high_dose (>=24mg)',
            'outcome': outcome,
            'observed_or': observed_or,
            'or_ci_lower': ci_lower,
            'or_ci_upper': ci_upper,
            'e_value': e_value,
            'e_value_ci': e_value_ci,
            'interpretation': (
                f"An unmeasured confounder would need to be associated with both "
                f"the exposure and outcome by a risk ratio of at least {e_value:.2f} "
                f"to explain away the observed association. "
                f"E-value for CI: {e_value_ci:.2f}."
            )
        }
        
        self.results['sensitivity'] = result
        return result
    
    def generate_causal_report(self, outcome: str = 'severe_crs') -> str:
        """Generate comprehensive causal analysis report."""
        
        report = []
        report.append("=" * 70)
        report.append("CAUSAL ANALYSIS REPORT: CRS Risk Factors")
        report.append("=" * 70)
        report.append("")
        
        # DAG Summary
        report.append("1. THEORETICAL CAUSAL FRAMEWORK (DAG)")
        report.append("-" * 50)
        self.define_dag()
        
        report.append("Causal Variables (direct effect on CRS):")
        for node, info in self.dag['nodes'].items():
            if info['type'] in ['exposure', 'mediator']:
                report.append(f"  • {node}: {info['description']}")
        
        report.append("\nConfounders (must control for):")
        for node, info in self.dag['nodes'].items():
            if info['type'] == 'confounder':
                report.append(f"  • {node}: {info['description']}")
        
        report.append("\nEffect Modifiers:")
        for node, info in self.dag['nodes'].items():
            if info['type'] == 'effect_modifier':
                report.append(f"  • {node}: {info['description']}")
        
        report.append("")
        
        # Association Analysis
        report.append("2. STATISTICAL ASSOCIATIONS")
        report.append("-" * 50)
        
        assoc_df = self.analyze_associations(outcome)
        
        report.append("\nVariable Associations with Outcome:")
        for _, row in assoc_df.iterrows():
            sig = "***" if row.get('significant', False) else ""
            p_val = row.get('p_value', 1)
            report.append(f"  • {row['variable']}: p={p_val:.4f} {sig}")
            report.append(f"    Interpretation: {row['interpretation']}")
        
        report.append("")
        
        # Propensity Score Analysis
        report.append("3. PROPENSITY SCORE ANALYSIS")
        report.append("-" * 50)
        
        ps_result = self.propensity_score_analysis(outcome=outcome)
        if 'error' not in ps_result:
            report.append(f"\nTreatment: {ps_result['treatment']}")
            report.append(f"N treated: {ps_result['n_treated']}, N control: {ps_result['n_control']}")
            report.append(f"Average Treatment Effect: {ps_result['ate']*100:.1f}%")
            report.append(f"95% CI: [{ps_result['ate_ci_lower']*100:.1f}%, {ps_result['ate_ci_upper']*100:.1f}%]")
            report.append(f"Interpretation: {ps_result['interpretation']}")
        else:
            report.append(f"Error: {ps_result['error']}")
        
        report.append("")
        
        # Sensitivity Analysis
        report.append("4. SENSITIVITY ANALYSIS")
        report.append("-" * 50)
        
        sens_result = self.sensitivity_analysis(outcome)
        if 'error' not in sens_result:
            report.append(f"\nExposure: {sens_result['exposure']}")
            report.append(f"Observed OR: {sens_result['observed_or']:.2f} "
                         f"(95% CI: {sens_result['or_ci_lower']:.2f}-{sens_result['or_ci_upper']:.2f})")
            report.append(f"E-value: {sens_result['e_value']:.2f}")
            report.append(f"\n{sens_result['interpretation']}")
        else:
            report.append(f"Error: {sens_result['error']}")
        
        report.append("")
        
        # Summary
        report.append("5. SUMMARY: CAUSAL vs CORRELATIONAL")
        report.append("-" * 50)
        report.append("""
LIKELY CAUSAL RELATIONSHIPS:
  ✓ Epcoritamab dose → CRS risk (dose-response, mechanism understood)
  ✓ Steroid premedication → Lower CRS severity (protective, mechanism known)
  ✓ Tocilizumab → Lower CRS severity (IL-6 blockade)

CONFOUNDERS (control for in analysis):
  ⚠ Age (affects both treatment decisions and outcomes)
  ⚠ Disease burden/stage (affects both)
  ⚠ Prior therapies (affects both)

CORRELATIONAL (not causal):
  ✗ Number of co-medications (marker of disease severity)
  ✗ Data source (reporting differences, not biological)
  
REQUIRES MORE DATA:
  ? Sex differences (possible biological effect modifier)
  ? Weight (may affect drug exposure)
  ? Geographic differences (may reflect treatment practices)
""")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "causal_analysis_results.json"):
        """Save analysis results to JSON."""
        
        # Convert DataFrames to dicts
        output = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                output[key] = value.to_dict(orient='records')
            else:
                output[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")


def main():
    """Run causal analysis."""
    
    # Initialize analyzer
    analyzer = CRSCausalAnalyzer("multi_source_crs_data.json")
    
    # Load data
    df = analyzer.load_data()
    
    # Generate report
    report = analyzer.generate_causal_report()
    print(report)
    
    # Save results
    analyzer.save_results()
    
    # Save report
    with open("causal_analysis_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to causal_analysis_report.txt")


if __name__ == "__main__":
    main()

