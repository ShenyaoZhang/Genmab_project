#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRS → Death Risk Assessment Dashboard
=====================================

Interactive Streamlit dashboard for clinicians and drug safety teams.
Focused on: "Does CRS lead to death? What factors matter?"

Features:
- Risk prediction for individual patients
- Feature importance visualization
- SHAP explanations
- Plain language interpretations
- Actionable clinical insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

# SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page config
st.set_page_config(
    page_title="CRS Death Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained CRS model."""
    model_file = Path('crs_model_best.pkl')
    if not model_file.exists():
        return None
    
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

@st.cache_data
def load_metadata():
    """Load model metadata."""
    meta_file = Path('crs_model_meta.json')
    if not meta_file.exists():
        return None
    
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    return meta

@st.cache_data
def load_feature_importance():
    """Load feature importance data."""
    fi_file = Path('crs_feature_importance.csv')
    if not fi_file.exists():
        return None
    
    return pd.read_csv(fi_file)

@st.cache_data
def load_data():
    """Load CRS patient data."""
    data_file = Path('main_data.csv')
    if not data_file.exists():
        return None
    
    df = pd.read_csv(data_file)
    return df

def plain_language_feature_names():
    """Map technical names to plain language."""
    return {
        'num_drugs': 'Number of Concurrent Medications',
        'has_chemo': 'Receiving Chemotherapy',
        'age_years': 'Patient Age (years)',
        'num_reactions': 'Number of Adverse Reactions',
        'bmi': 'Body Mass Index (BMI)',
        'patientweight': 'Patient Weight (kg)',
        'age_gt_70': 'Age Over 70 Years',
        'has_antiviral': 'Receiving Antiviral Medication',
        'has_targeted': 'Receiving Targeted Therapy',
        'multiple_reactions': 'Multiple Adverse Reactions',
        'sex_female': 'Female Gender',
        'has_steroid': 'Receiving Steroids',
        'high_polypharmacy': 'High Polypharmacy (>5 drugs)',
        'steroid_plus_antibiotic': 'Steroid + Antibiotic Combination',
        'sex_male': 'Male Gender',
        'age_gt_65': 'Age Over 65 Years',
        'has_infection_ae': 'Infection-Related Adverse Event',
        'has_antibiotic': 'Receiving Antibiotics',
        'bmi_overweight': 'Overweight (BMI 25-30)',
        'bmi_obese': 'Obese (BMI >30)',
        'bmi_underweight': 'Underweight (BMI <18.5)',
        'has_antifungal': 'Receiving Antifungal Medication'
    }

def feature_descriptions():
    """Clinical descriptions of features."""
    return {
        'num_drugs': 'Total number of medications the patient is taking concurrently. Higher numbers may indicate more complex cases or polypharmacy.',
        'has_chemo': 'Whether the patient is receiving chemotherapy drugs alongside Epcoritamab. This combination may increase toxicity risk.',
        'age_years': 'Patient age in years. Older age is associated with higher risk of adverse outcomes.',
        'num_reactions': 'Total number of adverse reactions reported. More reactions may indicate more severe illness.',
        'bmi': 'Body Mass Index, calculated from weight and height. Extreme values may affect outcomes.',
        'patientweight': 'Patient weight in kilograms.',
        'age_gt_70': 'Whether patient is over 70 years old - a specific risk threshold.',
        'has_antiviral': 'Whether patient is receiving antiviral medications.',
        'has_targeted': 'Whether patient is receiving targeted cancer therapy.',
        'multiple_reactions': 'Whether patient has multiple (2+) concurrent adverse reactions.',
        'sex_female': 'Female gender indicator.',
        'has_steroid': 'Whether patient is receiving steroid medications (e.g., prednisone).',
        'high_polypharmacy': 'Whether patient is taking more than 5 medications (high polypharmacy).',
        'steroid_plus_antibiotic': 'Whether patient is receiving both steroids and antibiotics - a specific combination that may increase risk.',
        'sex_male': 'Male gender indicator.',
        'age_gt_65': 'Whether patient is over 65 years old.',
        'has_infection_ae': 'Whether patient has infection-related adverse events reported.',
        'has_antibiotic': 'Whether patient is receiving antibiotic medications.',
        'bmi_overweight': 'Whether patient BMI is in overweight range (25-30).',
        'bmi_obese': 'Whether patient BMI indicates obesity (>30).',
        'bmi_underweight': 'Whether patient BMI indicates underweight (<18.5), similar to anorexic condition. Low BMI may indicate malnutrition or cachexia.',
        'has_antifungal': 'Whether patient is receiving antifungal medications.'
    }

def predict_risk(model_data, features):
    """Predict death risk for given features."""
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    
    # Prepare feature vector
    feature_vector = np.array([features]).reshape(1, -1)
    
    if scaler:
        feature_vector = scaler.transform(feature_vector)
    
    # Predict probability
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(feature_vector)[0]
        death_risk = proba[1]  # Probability of death
    else:
        # Fallback
        prediction = model.predict(feature_vector)[0]
        death_risk = float(prediction)
    
    return death_risk

def get_risk_category(risk):
    """Categorize risk level."""
    if risk < 0.3:
        return "Low", "🟢"
    elif risk < 0.6:
        return "Moderate", "🟡"
    elif risk < 0.8:
        return "High", "🟠"
    else:
        return "Very High", "🔴"

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🏥 CRS Death Risk Assessment Tool</div>', unsafe_allow_html=True)
    st.markdown("**For Clinicians & Drug Safety Teams** | *Predicting death risk in CRS patients treated with Epcoritamab*")
    st.markdown("---")
    
    # Load data
    model_data = load_model()
    metadata = load_metadata()
    feature_importance_df = load_feature_importance()
    
    if model_data is None:
        st.error("❌ CRS model not found. Please run `python 12_crs_model_training.py` first.")
        st.stop()
    
    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Risk Assessment", "📈 Model Insights", "🔍 Feature Analysis", "📋 Plain Language Summary"]
    )
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    if metadata:
        st.sidebar.metric("CRS Patients", metadata.get('n_crs_patients', 'N/A'))
        st.sidebar.metric("Death Rate", f"{metadata.get('death_rate', 0)*100:.1f}%")
        st.sidebar.metric("Best Model", metadata.get('best_model', 'N/A'))
    
    # Main content based on page
    if page == "📊 Risk Assessment":
        show_risk_assessment(model_data, metadata)
    elif page == "📈 Model Insights":
        show_model_insights(metadata, feature_importance_df)
    elif page == "🔍 Feature Analysis":
        show_feature_analysis(feature_importance_df)
    elif page == "📋 Plain Language Summary":
        show_plain_language_summary(metadata, feature_importance_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p>CRS Death Risk Assessment Dashboard | Built for Clinical Decision Support</p>
            <p><small>⚠️ This tool is for informational purposes only. Always use clinical judgment.</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_risk_assessment(model_data, metadata):
    """Risk assessment interface."""
    st.header("🔬 Individual Patient Risk Assessment")
    
    st.markdown("""
    <div class="info-box">
        <strong>How to use:</strong> Enter patient characteristics below to estimate their death risk from CRS.
        The model considers multiple factors including age, medications, and clinical features.
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for patient inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=65)
        sex = st.selectbox("Gender", ["Male", "Female", "Unknown"])
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
        
        # Calculate BMI (assuming average height)
        height_default = 1.65  # meters
        bmi = weight / (height_default ** 2) if weight > 0 else 0
    
    with col2:
        st.subheader("Clinical Features")
        
        num_drugs = st.number_input("Number of Concurrent Medications", min_value=0, max_value=50, value=5)
        num_reactions = st.number_input("Number of Adverse Reactions", min_value=0, max_value=20, value=2)
        
        st.markdown("**Medication Types:**")
        has_chemo = st.checkbox("Receiving Chemotherapy")
        has_steroid = st.checkbox("Receiving Steroids")
        has_antibiotic = st.checkbox("Receiving Antibiotics")
        has_antifungal = st.checkbox("Receiving Antifungal")
        has_antiviral = st.checkbox("Receiving Antiviral")
        has_targeted = st.checkbox("Receiving Targeted Therapy")
        
        has_infection_ae = st.checkbox("Infection-Related Adverse Event")
    
    # Calculate derived features
    age_gt_65 = 1 if age > 65 else 0
    age_gt_70 = 1 if age > 70 else 0
    age_50_65 = 1 if 50 <= age <= 65 else 0
    bmi_obese = 1 if bmi > 30 else 0
    bmi_overweight = 1 if 25 <= bmi <= 30 else 0
    bmi_underweight = 1 if bmi < 18.5 else 0
    high_polypharmacy = 1 if num_drugs > 5 else 0
    polypharmacy = 1 if num_drugs > 1 else 0
    multiple_reactions = 1 if num_reactions > 1 else 0
    steroid_plus_antibiotic = 1 if (has_steroid and has_antibiotic) else 0
    
    sex_male = 1 if sex == "Male" else 0
    sex_female = 1 if sex == "Female" else 0
    sex_unknown = 1 if sex == "Unknown" else 0
    
    # Build feature vector (matching training features)
    feature_map = plain_language_feature_names()
    feature_order = [
        'age_years', 'age_missing', 'age_gt_65', 'age_gt_70', 'age_50_65',
        'sex_male', 'sex_female', 'sex_unknown',
        'patientweight', 'weight_missing', 'bmi', 'bmi_obese', 'bmi_overweight', 'bmi_underweight',
        'num_drugs', 'polypharmacy', 'high_polypharmacy',
        'num_reactions', 'multiple_reactions',
        'has_steroid', 'has_antibiotic', 'has_antifungal', 'has_antiviral', 'has_chemo', 'has_targeted',
        'steroid_plus_antibiotic', 'has_infection_ae'
    ]
    
    features_dict = {
        'age_years': age,
        'age_missing': 0,
        'age_gt_65': age_gt_65,
        'age_gt_70': age_gt_70,
        'age_50_65': age_50_65,
        'sex_male': sex_male,
        'sex_female': sex_female,
        'sex_unknown': sex_unknown,
        'patientweight': weight,
        'weight_missing': 0,
        'bmi': bmi,
        'bmi_obese': bmi_obese,
        'bmi_overweight': bmi_overweight,
        'bmi_underweight': bmi_underweight,
        'num_drugs': num_drugs,
        'polypharmacy': polypharmacy,
        'high_polypharmacy': high_polypharmacy,
        'num_reactions': num_reactions,
        'multiple_reactions': multiple_reactions,
        'has_steroid': 1 if has_steroid else 0,
        'has_antibiotic': 1 if has_antibiotic else 0,
        'has_antifungal': 1 if has_antifungal else 0,
        'has_antiviral': 1 if has_antiviral else 0,
        'has_chemo': 1 if has_chemo else 0,
        'has_targeted': 1 if has_targeted else 0,
        'steroid_plus_antibiotic': steroid_plus_antibiotic,
        'has_infection_ae': 1 if has_infection_ae else 0
    }
    
    # Calculate risk
    if st.button("🔍 Calculate Risk", type="primary"):
        try:
            # Create feature vector in correct order
            feature_vector = np.array([features_dict.get(f, 0) for f in feature_order])
            
            # Predict
            risk = predict_risk(model_data, feature_vector)
            risk_category, risk_emoji = get_risk_category(risk)
            
            # Display results
            st.markdown("---")
            st.markdown("### Risk Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Death Risk", f"{risk*100:.1f}%", delta=None)
            
            with col2:
                st.metric("Risk Category", f"{risk_emoji} {risk_category}")
            
            with col3:
                survival_prob = (1 - risk) * 100
                st.metric("Survival Probability", f"{survival_prob:.1f}%")
            
            # Risk interpretation
            st.markdown("---")
            st.markdown("### Clinical Interpretation")
            
            if risk >= 0.8:
                st.error(f"""
                **🔴 Very High Risk ({risk*100:.1f}%)**
                
                This patient is at very high risk of death from CRS. Consider:
                - Immediate intensive monitoring
                - Review of all medications (consider reducing if possible)
                - Early intervention for complications
                - Consider consultation with critical care team
                """)
            elif risk >= 0.6:
                st.warning(f"""
                **🟠 High Risk ({risk*100:.1f}%)**
                
                This patient is at high risk. Monitor closely and consider:
                - Frequent monitoring of vital signs
                - Review medication list for potential interactions
                - Watch for infection-related complications
                - Consider preventive measures
                """)
            elif risk >= 0.3:
                st.info(f"""
                **🟡 Moderate Risk ({risk*100:.1f}%)**
                
                This patient has moderate risk. Standard monitoring recommended:
                - Regular follow-up assessments
                - Monitor for new adverse reactions
                - Maintain current treatment plan with vigilance
                """)
            else:
                st.success(f"""
                **🟢 Low Risk ({risk*100:.1f}%)**
                
                This patient is at relatively low risk. Continue standard care:
                - Regular monitoring
                - Standard treatment protocols
                - Continue current medication regimen
                """)
            
            # Key risk factors
            st.markdown("### Key Risk Factors Identified")
            
            # Sort features by importance and show top contributing factors
            if feature_importance_df is not None:
                important_factors = []
                for _, row in feature_importance_df.head(10).iterrows():
                    feat = row['feature']
                    if feat in features_dict:
                        value = features_dict[feat]
                        if value != 0:  # Only show active factors
                            important_factors.append((feat, row['importance'], value))
                
                if important_factors:
                    for feat, imp, val in important_factors[:5]:
                        feat_name = plain_language_feature_names().get(feat, feat.replace('_', ' ').title())
                        st.markdown(f"- **{feat_name}**: Active (Importance: {imp:.3f})")
        
        except Exception as e:
            st.error(f"Error calculating risk: {str(e)}")
            st.exception(e)

def show_model_insights(metadata, feature_importance_df):
    """Model performance and insights."""
    st.header("📈 Model Performance & Insights")
    
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total CRS Patients", metadata.get('n_crs_patients', 'N/A'))
        
        with col2:
            st.metric("Deaths", metadata.get('n_deaths', 'N/A'))
        
        with col3:
            death_rate = metadata.get('death_rate', 0)
            st.metric("Overall Death Rate", f"{death_rate*100:.1f}%")
        
        with col4:
            perf = metadata.get('model_performance', {})
            pr_auc = perf.get('pr_auc', 0)
            st.metric("Model PR-AUC", f"{pr_auc:.3f}")
    
    st.markdown("---")
    st.subheader("Model Performance Metrics")
    
    if metadata and 'model_performance' in metadata:
        perf = metadata['model_performance']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ROC-AUC", f"{perf.get('roc_auc', 0):.3f}")
            st.metric("PR-AUC", f"{perf.get('pr_auc', 0):.3f}")
        
        with col2:
            st.metric("F1-Score", f"{perf.get('f1', 0):.3f}")
            st.metric("Accuracy", f"{perf.get('accuracy', 0):.3f}")
    
    st.markdown("---")
    st.subheader("Top Risk Factors")
    
    if feature_importance_df is not None:
        # Visualize top features
        top_n = st.slider("Number of features to show", 5, 15, 10)
        top_features = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_map = plain_language_feature_names()
        display_names = [feature_map.get(f, f.replace('_', ' ').title()) for f in top_features['feature']]
        
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(display_names)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Risk Factors for CRS → Death', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Feature descriptions
        st.markdown("### Feature Descriptions")
        desc_map = feature_descriptions()
        
        for _, row in top_features.iterrows():
            feat = row['feature']
            feat_name = feature_map.get(feat, feat.replace('_', ' ').title())
            desc = desc_map.get(feat, "No description available.")
            
            with st.expander(f"{feat_name} (Importance: {row['importance']:.3f})"):
                st.markdown(desc)

def show_feature_analysis(feature_importance_df):
    """Detailed feature analysis."""
    st.header("🔍 Detailed Feature Analysis")
    
    if feature_importance_df is None:
        st.warning("Feature importance data not available.")
        return
    
    st.markdown("### Feature Importance Table")
    
    # Add plain language names
    feature_map = plain_language_feature_names()
    feature_importance_df['Feature Name'] = feature_importance_df['feature'].map(
        lambda x: feature_map.get(x, x.replace('_', ' ').title())
    )
    
    # Display table
    display_df = feature_importance_df[['Feature Name', 'importance']].copy()
    display_df.columns = ['Feature', 'Importance Score']
    display_df = display_df.sort_values('Importance Score', ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Feature categories
    st.markdown("### Features by Category")
    
    categories = {
        'Demographics': ['age_years', 'age_gt_65', 'age_gt_70', 'sex_male', 'sex_female', 'bmi', 'patientweight'],
        'Medications': ['num_drugs', 'has_chemo', 'has_steroid', 'has_antibiotic', 'has_antifungal', 'has_antiviral', 'has_targeted'],
        'Clinical': ['num_reactions', 'multiple_reactions', 'has_infection_ae'],
        'Combinations': ['steroid_plus_antibiotic', 'high_polypharmacy']
    }
    
    for category, features in categories.items():
        with st.expander(f"{category} ({len(features)} features)"):
            cat_df = feature_importance_df[feature_importance_df['feature'].isin(features)]
            if len(cat_df) > 0:
                cat_df = cat_df.sort_values('importance', ascending=False)
                for _, row in cat_df.iterrows():
                    feat_name = feature_map.get(row['feature'], row['feature'].replace('_', ' ').title())
                    st.markdown(f"- **{feat_name}**: {row['importance']:.4f}")

def show_plain_language_summary(metadata, feature_importance_df):
    """Plain language summary for clinicians."""
    st.header("📋 Plain Language Summary for Clinicians")
    
    st.markdown("""
    ## What This Model Does
    
    This model predicts the **risk of death** in patients with **Cytokine Release Syndrome (CRS)** 
    who are being treated with Epcoritamab. It helps identify which factors are most important 
    in determining patient outcomes.
    """)
    
    st.markdown("---")
    st.subheader("Key Findings")
    
    if feature_importance_df is not None:
        top_5 = feature_importance_df.head(5)
        feature_map = plain_language_feature_names()
        desc_map = feature_descriptions()
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            feat = row['feature']
            feat_name = feature_map.get(feat, feat.replace('_', ' ').title())
            desc = desc_map.get(feat, "No description available.")
            
            st.markdown(f"""
            ### {i}. {feat_name}
            
            **Importance:** {row['importance']:.3f}
            
            {desc}
            """)
    
    st.markdown("---")
    st.subheader("Clinical Implications")
    
    st.markdown("""
    ### What This Means for Patient Care
    
    1. **Monitor polypharmacy:** Patients on multiple medications need close monitoring.
       - Consider medication review
       - Look for drug-drug interactions
       - Assess necessity of each medication
    
    2. **Consider chemotherapy carefully:** Combined chemo + Epcoritamab has higher risk
       - Review chemotherapy timing
       - Consider dose adjustments
       - Monitor for combined toxicity
    
    3. **Age-based risk stratification:** Older patients (>70 years) may need more intensive monitoring
       - Increase monitoring frequency
       - Consider preventive measures
       - Early intervention for complications
    
    4. **Watch for multiple reactions:** Patients with multiple concurrent adverse reactions are at higher risk
       - Closely monitor for new reactions
       - Aggressive symptom management
       - Consider escalation of care
    """)
    
    st.markdown("---")
    st.subheader("What This Means for Drug Safety Teams")
    
    st.markdown("""
    1. **Risk stratification:** Use these factors to identify high-risk patients early
    2. **Intervention opportunities:** Focus on modifiable factors (e.g., medication review, infection prevention)
    3. **Clinical decision support:** Integrate this model into clinical workflows for risk assessment
    4. **Monitoring priorities:** Allocate resources to patients with highest risk factors
    """)
    
    st.markdown("---")
    st.subheader("Limitations")
    
    st.markdown("""
    - ⚠️ Model is based on observational data (FAERS reports)
    - ⚠️ Cannot prove causation, only associations
    - ⚠️ Should be used alongside clinical judgment
    - ⚠️ Requires validation in independent datasets
    - ⚠️ Results may not apply to all patient populations
    """)
    
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <strong>Important:</strong> This tool is for informational purposes only and should not replace 
        clinical judgment. Always use professional medical expertise when making treatment decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

