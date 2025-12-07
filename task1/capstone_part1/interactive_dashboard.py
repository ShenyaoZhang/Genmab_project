"""
Interactive Dashboard for CRS Risk Analysis
Allows users to select any outcome of interest and explore causal relationships.

Run with: streamlit run interactive_dashboard.py
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path: str = "multi_source_crs_data.json") -> pd.DataFrame:
    """Load and prepare data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
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
            'age': record.get('age'),
            'sex': record.get('sex'),
            'weight': record.get('weight'),
            'country': record.get('country'),
            'indication': record.get('indication'),
            'n_co_medications': len(record.get('co_medications', [])),
        }
        
        # Dose info
        doses = record.get('epcoritamab_doses', [])
        if doses:
            dose_mgs = [d.get('dose_mg') for d in doses if d.get('dose_mg')]
            flat_record['max_dose_mg'] = max(dose_mgs) if dose_mgs else None
        else:
            flat_record['max_dose_mg'] = None
        
        # Co-medication flags
        co_meds = [m.upper() for m in record.get('co_medications', [])]
        flat_record['has_rituximab'] = any('RITUXIMAB' in m for m in co_meds)
        flat_record['has_steroids'] = any(
            s in ' '.join(co_meds) for s in ['PREDNIS', 'DEXAMETH', 'METHYLPRED']
        )
        flat_record['has_tocilizumab'] = any('TOCILIZUMAB' in m for m in co_meds)
        
        records.append(flat_record)
    
    df = pd.DataFrame(records)
    
    # Create derived outcomes
    df['fatal'] = df['death'].astype(int)
    df['severe_crs'] = (
        (df['death'] == True) | 
        (df['life_threatening'] == True) |
        (df['crs_outcome'] == 'not_recovered')
    ).astype(int)
    df['recovered'] = (df['crs_outcome'] == 'recovered').astype(int)
    df['sex_male'] = (df['sex'] == 'male').astype(int)
    
    return df


def run_streamlit_app():
    """Run the Streamlit interactive dashboard."""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    st.set_page_config(
        page_title="CRS Risk Analysis Dashboard",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 CRS Risk Analysis Dashboard")
    st.markdown("**Interactive analysis of Cytokine Release Syndrome risk factors in Epcoritamab-treated patients**")
    
    # Load data
    @st.cache_data
    def get_data():
        try:
            return load_data("multi_source_crs_data.json")
        except FileNotFoundError:
            return load_data("crs_extracted_data.json")
    
    df = get_data()
    
    # Sidebar for controls
    st.sidebar.header("Analysis Controls")
    
    # Outcome selection
    outcome_options = {
        'Severe CRS': 'severe_crs',
        'Fatal Outcome': 'fatal',
        'Recovery': 'recovered',
        'Hospitalization': 'hospitalized'
    }
    
    selected_outcome_name = st.sidebar.selectbox(
        "Select Outcome of Interest",
        list(outcome_options.keys())
    )
    outcome = outcome_options[selected_outcome_name]
    
    # Data source filter
    sources = ['All'] + list(df['source'].unique())
    selected_source = st.sidebar.selectbox("Data Source", sources)
    
    if selected_source != 'All':
        df_filtered = df[df['source'] == selected_source].copy()
    else:
        df_filtered = df.copy()
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['age'].min()) if df['age'].notna().any() else 0,
        int(df['age'].max()) if df['age'].notna().any() else 100,
        (40, 90)
    )
    df_filtered = df_filtered[
        (df_filtered['age'] >= age_range[0]) & 
        (df_filtered['age'] <= age_range[1])
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered)}")
    st.sidebar.markdown(f"**Outcome Rate:** {df_filtered[outcome].mean()*100:.1f}%")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cases", len(df_filtered))
    with col2:
        st.metric(f"{selected_outcome_name} Rate", f"{df_filtered[outcome].mean()*100:.1f}%")
    with col3:
        st.metric("Data Sources", df_filtered['source'].nunique())
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Descriptive Stats", 
        "🔗 Association Analysis",
        "⚗️ Causal Analysis",
        "📈 Subgroup Analysis"
    ])
    
    with tab1:
        st.header("Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(
                df_filtered, x='age', color=outcome.replace('_', ' ').title(),
                nbins=20, title="Age Distribution by Outcome",
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Sex distribution
            sex_outcome = df_filtered.groupby(['sex', outcome]).size().unstack(fill_value=0)
            fig_sex = px.bar(
                sex_outcome, barmode='group',
                title="Outcome by Sex",
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            st.plotly_chart(fig_sex, use_container_width=True)
        
        with col2:
            # Source distribution
            source_outcome = df_filtered.groupby(['source', outcome]).size().unstack(fill_value=0)
            fig_source = px.bar(
                source_outcome, barmode='stack',
                title="Outcomes by Data Source"
            )
            st.plotly_chart(fig_source, use_container_width=True)
            
            # Country distribution
            country_counts = df_filtered['country'].value_counts().head(10)
            fig_country = px.bar(
                x=country_counts.values, y=country_counts.index,
                orientation='h', title="Top 10 Countries"
            )
            st.plotly_chart(fig_country, use_container_width=True)
    
    with tab2:
        st.header("Association Analysis")
        st.markdown("*Statistical associations between variables and the selected outcome*")
        
        # Run association tests
        results = []
        
        # Continuous variables
        continuous_vars = ['age', 'weight', 'max_dose_mg', 'n_co_medications']
        
        for var in continuous_vars:
            valid_data = df_filtered[[var, outcome]].dropna()
            if len(valid_data) > 10:
                corr, p_value = stats.pointbiserialr(valid_data[outcome], valid_data[var])
                results.append({
                    'Variable': var,
                    'Type': 'Continuous',
                    'Correlation': corr,
                    'P-Value': p_value,
                    'Significant': '✓' if p_value < 0.05 else ''
                })
        
        # Categorical variables
        categorical_vars = ['sex_male', 'has_rituximab', 'has_steroids', 'has_tocilizumab']
        
        for var in categorical_vars:
            valid_data = df_filtered[[var, outcome]].dropna()
            if len(valid_data) > 10:
                contingency = pd.crosstab(valid_data[var], valid_data[outcome])
                if contingency.shape == (2, 2):
                    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                    
                    # Odds ratio
                    a, b = contingency.iloc[1, 1], contingency.iloc[1, 0]
                    c, d = contingency.iloc[0, 1], contingency.iloc[0, 0]
                    or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan
                    
                    results.append({
                        'Variable': var,
                        'Type': 'Categorical',
                        'Odds Ratio': or_val,
                        'P-Value': p_value,
                        'Significant': '✓' if p_value < 0.05 else ''
                    })
        
        results_df = pd.DataFrame(results)
        
        # Display results
        st.dataframe(results_df.style.format({
            'Correlation': '{:.3f}',
            'Odds Ratio': '{:.2f}',
            'P-Value': '{:.4f}'
        }), use_container_width=True)
        
        # Forest plot
        st.subheader("Odds Ratios (Categorical Variables)")
        
        cat_results = [r for r in results if r['Type'] == 'Categorical' and not np.isnan(r.get('Odds Ratio', np.nan))]
        if cat_results:
            cat_df = pd.DataFrame(cat_results)
            fig_forest = go.Figure()
            
            for i, row in cat_df.iterrows():
                color = '#e74c3c' if row['Odds Ratio'] > 1 else '#2ecc71'
                fig_forest.add_trace(go.Scatter(
                    x=[row['Odds Ratio']],
                    y=[row['Variable']],
                    mode='markers',
                    marker=dict(size=15, color=color),
                    name=row['Variable']
                ))
            
            fig_forest.add_vline(x=1, line_dash="dash", line_color="gray")
            fig_forest.update_layout(
                title="Odds Ratios (reference line at 1)",
                xaxis_title="Odds Ratio",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_forest, use_container_width=True)
    
    with tab3:
        st.header("Causal Analysis")
        st.markdown("*Distinguishing causal relationships from correlations*")
        
        # DAG visualization
        st.subheader("Theoretical Causal Framework")
        
        dag_text = """
        ```
        CAUSAL DIAGRAM FOR CRS:
        
        Confounders ─────────────────────┐
        (Age, Disease Stage)             │
              │                          │
              ▼                          ▼
        Epcoritamab Dose ──────► CRS Severity
              │                          ▲
              │                          │
              ▼                          │
        T-cell Activation ──► Cytokine Release
              
        Effect Modifiers: Steroids, Tocilizumab
        ```
        """
        st.markdown(dag_text)
        
        # Causal interpretation table
        st.subheader("Variable Classification")
        
        causal_classification = pd.DataFrame([
            {'Variable': 'Epcoritamab Dose', 'Classification': 'CAUSAL', 
             'Evidence': 'Dose-response relationship, known mechanism'},
            {'Variable': 'Steroid Premedication', 'Classification': 'CAUSAL (protective)', 
             'Evidence': 'Anti-inflammatory mechanism, clinical trials'},
            {'Variable': 'Tocilizumab', 'Classification': 'CAUSAL (protective)', 
             'Evidence': 'IL-6 pathway blockade'},
            {'Variable': 'Age', 'Classification': 'CONFOUNDER', 
             'Evidence': 'Affects both treatment selection and outcome'},
            {'Variable': 'Disease Stage', 'Classification': 'CONFOUNDER', 
             'Evidence': 'Affects both treatment selection and outcome'},
            {'Variable': 'Co-medications Count', 'Classification': 'CORRELATIONAL', 
             'Evidence': 'Marker of disease severity, not direct cause'},
            {'Variable': 'Data Source', 'Classification': 'NOT CAUSAL', 
             'Evidence': 'Reporting artifact'},
        ])
        
        def color_classification(val):
            if val == 'CAUSAL':
                return 'background-color: #d4edda'
            elif 'CAUSAL' in val:
                return 'background-color: #d1ecf1'
            elif val == 'CONFOUNDER':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        st.dataframe(
            causal_classification.style.applymap(
                color_classification, subset=['Classification']
            ),
            use_container_width=True
        )
        
        # Propensity score analysis
        st.subheader("Propensity Score Analysis")
        
        treatment_var = st.selectbox(
            "Select Treatment Variable",
            ['has_steroids', 'has_tocilizumab', 'has_rituximab']
        )
        
        if st.button("Run Propensity Score Analysis"):
            with st.spinner("Running analysis..."):
                # Prepare data
                confounders = ['age', 'sex_male', 'n_co_medications']
                analysis_df = df_filtered[[treatment_var, outcome] + confounders].dropna()
                
                if len(analysis_df) < 50:
                    st.warning("Insufficient data for propensity score analysis")
                else:
                    X = analysis_df[confounders].values
                    T = analysis_df[treatment_var].values
                    Y = analysis_df[outcome].values
                    
                    # Standardize
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Propensity scores
                    ps_model = LogisticRegression(random_state=42, max_iter=1000)
                    ps_model.fit(X_scaled, T)
                    ps = ps_model.predict_proba(X_scaled)[:, 1]
                    ps = np.clip(ps, 0.05, 0.95)
                    
                    # IPW
                    weights = np.where(T == 1, 1/ps, 1/(1-ps))
                    
                    treated_outcome = np.average(Y[T == 1], weights=weights[T == 1])
                    control_outcome = np.average(Y[T == 0], weights=weights[T == 0])
                    ate = treated_outcome - control_outcome
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Treated Outcome Rate", f"{treated_outcome*100:.1f}%")
                    col2.metric("Control Outcome Rate", f"{control_outcome*100:.1f}%")
                    col3.metric("Causal Effect (ATE)", f"{ate*100:+.1f}%")
                    
                    # PS distribution plot
                    fig_ps = go.Figure()
                    fig_ps.add_trace(go.Histogram(
                        x=ps[T == 1], name='Treated', 
                        opacity=0.7, nbinsx=20
                    ))
                    fig_ps.add_trace(go.Histogram(
                        x=ps[T == 0], name='Control', 
                        opacity=0.7, nbinsx=20
                    ))
                    fig_ps.update_layout(
                        title="Propensity Score Distribution",
                        xaxis_title="Propensity Score",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig_ps, use_container_width=True)
    
    with tab4:
        st.header("Subgroup Analysis")
        st.markdown("*Explore outcome rates across different patient subgroups*")
        
        # Subgroup selector
        subgroup_var = st.selectbox(
            "Select Subgroup Variable",
            ['sex', 'source', 'has_steroids', 'has_rituximab', 'has_tocilizumab']
        )
        
        # Calculate subgroup statistics
        subgroup_stats = df_filtered.groupby(subgroup_var).agg({
            outcome: ['count', 'sum', 'mean']
        }).round(3)
        subgroup_stats.columns = ['N', 'Events', 'Rate']
        subgroup_stats['Rate'] = (subgroup_stats['Rate'] * 100).round(1)
        
        st.dataframe(subgroup_stats, use_container_width=True)
        
        # Subgroup comparison plot
        fig_subgroup = px.bar(
            x=subgroup_stats.index,
            y=subgroup_stats['Rate'],
            title=f"{selected_outcome_name} Rate by {subgroup_var}",
            labels={'x': subgroup_var, 'y': 'Rate (%)'}
        )
        fig_subgroup.add_hline(
            y=df_filtered[outcome].mean() * 100,
            line_dash="dash",
            annotation_text="Overall Rate"
        )
        st.plotly_chart(fig_subgroup, use_container_width=True)
        
        # Age-stratified analysis
        st.subheader("Age-Stratified Analysis")
        
        df_filtered['age_group'] = pd.cut(
            df_filtered['age'],
            bins=[0, 50, 65, 75, 100],
            labels=['<50', '50-65', '65-75', '>75']
        )
        
        age_stats = df_filtered.groupby('age_group').agg({
            outcome: ['count', 'mean']
        }).round(3)
        age_stats.columns = ['N', 'Rate']
        age_stats['Rate'] = (age_stats['Rate'] * 100).round(1)
        
        fig_age_sub = px.bar(
            age_stats.reset_index(),
            x='age_group', y='Rate',
            title=f"{selected_outcome_name} Rate by Age Group"
        )
        st.plotly_chart(fig_age_sub, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*Dashboard for CRS Risk Analysis | "
        "Data sources: FAERS, Eudravigilance (simulated), JADER (simulated)*"
    )


def run_terminal_interface():
    """Run terminal-based interactive interface."""
    
    print("=" * 60)
    print("CRS RISK ANALYSIS - INTERACTIVE TERMINAL INTERFACE")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data("multi_source_crs_data.json")
    except FileNotFoundError:
        df = load_data("crs_extracted_data.json")
    
    print(f"\nLoaded {len(df)} records from {df['source'].nunique()} sources")
    
    while True:
        print("\n" + "-" * 40)
        print("SELECT OUTCOME OF INTEREST:")
        print("  1. Severe CRS")
        print("  2. Fatal Outcome")
        print("  3. Recovery")
        print("  4. Hospitalization")
        print("  5. Exit")
        print("-" * 40)
        
        try:
            choice = input("Enter choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        outcome_map = {
            '1': ('Severe CRS', 'severe_crs'),
            '2': ('Fatal Outcome', 'fatal'),
            '3': ('Recovery', 'recovered'),
            '4': ('Hospitalization', 'hospitalized')
        }
        
        if choice == '5':
            print("Goodbye!")
            break
        
        if choice not in outcome_map:
            print("Invalid choice. Please try again.")
            continue
        
        outcome_name, outcome_col = outcome_map[choice]
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR: {outcome_name}")
        print("=" * 60)
        
        # Basic stats
        print(f"\nOverall rate: {df[outcome_col].mean()*100:.1f}%")
        print(f"Total cases: {len(df)}")
        print(f"Events: {df[outcome_col].sum()}")
        
        # By source
        print(f"\nBy Data Source:")
        source_rates = df.groupby('source')[outcome_col].agg(['count', 'mean'])
        for source, row in source_rates.iterrows():
            print(f"  {source}: {row['mean']*100:.1f}% (n={int(row['count'])})")
        
        # Risk factors
        print(f"\nRisk Factor Analysis:")
        
        from scipy import stats
        
        # Age
        age_data = df[['age', outcome_col]].dropna()
        if len(age_data) > 10:
            corr, p = stats.pointbiserialr(age_data[outcome_col], age_data['age'])
            print(f"  Age: correlation={corr:.3f}, p={p:.4f}")
        
        # Steroids
        steroid_data = df[['has_steroids', outcome_col]].dropna()
        contingency = pd.crosstab(steroid_data['has_steroids'], steroid_data[outcome_col])
        if contingency.shape == (2, 2):
            a, b = contingency.iloc[1, 1], contingency.iloc[1, 0]
            c, d = contingency.iloc[0, 1], contingency.iloc[0, 0]
            or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan
            print(f"  Steroids: OR={or_val:.2f}")
        
        # Tocilizumab
        toci_data = df[['has_tocilizumab', outcome_col]].dropna()
        contingency = pd.crosstab(toci_data['has_tocilizumab'], toci_data[outcome_col])
        if contingency.shape == (2, 2):
            a, b = contingency.iloc[1, 1], contingency.iloc[1, 0]
            c, d = contingency.iloc[0, 1], contingency.iloc[0, 0]
            or_val = (a * d) / (b * c) if b > 0 and c > 0 else np.nan
            print(f"  Tocilizumab: OR={or_val:.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--terminal":
        run_terminal_interface()
    else:
        try:
            run_streamlit_app()
        except ImportError:
            print("Streamlit not installed. Running terminal interface...")
            print("Install streamlit with: pip install streamlit plotly")
            run_terminal_interface()

