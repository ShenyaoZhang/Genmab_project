# Clinical Guidelines for CRS Risk Stratification and Management

## Overview

This document provides clinical practice guidelines for managing Cytokine Release Syndrome (CRS) in patients receiving Epcoritamab, based on the survival analysis findings and established clinical protocols.

---

## 🎯 Risk Stratification System

### Weight-Based Risk Assessment (Novel Finding)

Based on our analysis, patient body weight is the primary predictor of CRS risk.

| Weight Category | CRS Risk | HR (vs. 60kg) | Clinical Classification |
|-----------------|----------|---------------|------------------------|
| **<60 kg** | **HIGH** | 1.00 (baseline) | Enhanced monitoring required |
| **60-80 kg** | **MODERATE** | 0.85 (15% ↓) | Standard monitoring |
| **>80 kg** | **LOWER** | 0.73 (27% ↓) | Standard monitoring |

### Comprehensive Risk Score

```
Risk Score Calculation:
┌─────────────────────────────────────┬────────┐
│ Risk Factor                         │ Points │
├─────────────────────────────────────┼────────┤
│ Weight <60 kg                       │   +2   │ ⭐ NEW
│ Age >65 years                       │   +1   │
│ Polypharmacy (≥3 drugs)             │   +1   │
│ Prior life-threatening event        │   +2   │
│ Prior hospitalization for AE        │   +1   │
│ ECOG status ≥2 (if available)       │   +1   │
└─────────────────────────────────────┴────────┘

Risk Classification:
  Score 0-1:  LOW risk (not observed in this cohort)
  Score 2-3:  MODERATE risk → 30.7% CRS rate
  Score ≥4:   HIGH risk → 36.9% CRS rate
```

---

## 📊 CRS Grading System

### ASTCT Consensus Grading (Lee et al., 2019)

| Grade | Fever | Hypotension | Hypoxia | Management Level |
|-------|-------|-------------|---------|------------------|
| **1** | ≥38°C | None | None | Outpatient possible after 24h |
| **2** | ≥38°C | Responsive to fluids or low-dose vasopressor | FiO₂ <40% | Inpatient monitoring |
| **3** | ≥38°C | Requiring high-dose or multiple vasopressors | FiO₂ ≥40% | ICU admission |
| **4** | ≥38°C | Life-threatening hypotension | Mechanical ventilation | ICU + aggressive intervention |

### Our Cohort Distribution

| Grade | Count | Percentage | Severity Classification |
|-------|-------|------------|------------------------|
| **Any CRS** | 344 | 34.4% | Total CRS rate |
| **Grade 1-2** | 316 | 31.6% | Mild-Moderate CRS |
| **Grade 3-4** | 28 | 2.8% | Severe CRS (requires ICU) |

---

## ⏰ Temporal Monitoring Protocol

### Critical 24-Hour Window

**Key Finding:** 100% of CRS events occurred within the first 24 hours post-dose.

```
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL MONITORING PROTOCOL                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  0-4 Hours Post-Dose:        HIGHEST RISK PERIOD           │
│  ├─ Vital signs q15min                                     │
│  ├─ Patient in observation unit                            │
│  └─ Tocilizumab immediately available                      │
│                                                             │
│  4-12 Hours Post-Dose:       HIGH RISK PERIOD              │
│  ├─ Vital signs q30min                                     │
│  ├─ Continued inpatient observation                        │
│  └─ ICU bed reserved for high-risk patients                │
│                                                             │
│  12-24 Hours Post-Dose:      MODERATE RISK PERIOD          │
│  ├─ Vital signs q1h                                        │
│  ├─ If no CRS, can consider discharge planning             │
│  └─ Patient education on delayed symptoms (rare)           │
│                                                             │
│  >24 Hours Post-Dose:        MINIMAL RISK                  │
│  ├─ CRS risk approaches zero                               │
│  ├─ Can safely discharge if no Grade ≥1 CRS               │
│  └─ Outpatient follow-up in 7 days                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏥 Risk-Stratified Management Protocols

### HIGH RISK Patients (Score ≥4 OR Weight <60kg)

#### Pre-Treatment

```
☑ Mandatory inpatient administration
☑ ICU bed reserved and available
☑ Tocilizumab (8 mg/kg) on-site in pharmacy
☑ Dexamethasone available at bedside
☑ Patient counseled on CRS symptoms
☑ Consider prophylactic:
   - Antihistamine (diphenhydramine 25-50mg)
   - Antipyretic (acetaminophen 650mg)
   - Fluid pre-hydration (500-1000 mL NS)
```

#### During Administration

```
☑ Monitor in observation unit or step-down unit
☑ Continuous pulse oximetry
☑ Cardiac monitoring
☑ Vital signs every 15 minutes × 4 hours
☑ Then vital signs every 30 minutes × 8 hours
☑ Then vital signs every 1 hour × 12 hours
```

#### Post-Dose Management

```
☑ Minimum 24-hour inpatient observation
☑ Extend to 48 hours if any Grade 1 symptoms
☑ ICU transfer if Grade ≥3
☑ Early tocilizumab for Grade 2
☑ Discharge criteria:
   - No fever for 12+ hours
   - Vital signs stable
   - Patient ambulatory
   - Reliable follow-up arranged
```

---

### MODERATE RISK Patients (Score 2-3)

#### Pre-Treatment

```
☑ Inpatient administration for first dose
☑ Tocilizumab available within 2 hours
☑ ICU bed on standby (not reserved)
☑ Patient counseled on CRS symptoms
☑ Consider prophylactic antipyretic
```

#### During Administration

```
☑ Monitor in infusion center or observation unit
☑ Vital signs every 30 minutes × 4 hours
☑ Then vital signs every 1 hour × 8 hours
☑ Pulse oximetry spot checks
```

#### Post-Dose Management

```
☑ 24-hour observation
☑ Can discharge if no Grade ≥1 CRS by 24h
☑ Patient education on self-monitoring
☑ 24/7 oncology contact provided
☑ Follow-up call at 48 hours
```

---

## 💊 CRS Treatment Algorithm

### Grade 1 CRS

**Symptoms:** Fever ≥38°C only

```
Management:
1. Supportive Care
   ├─ Acetaminophen 650-1000 mg PO q6h PRN
   ├─ Fluid bolus 500 mL NS if needed
   └─ Monitor vital signs q1h

2. Observation
   ├─ Continue monitoring for 24 hours
   ├─ Watch for progression to Grade 2
   └─ Patient remains inpatient

3. No Immunosuppression
   ├─ Do NOT give tocilizumab for Grade 1
   └─ Do NOT give corticosteroids for Grade 1

4. If persists >24h → Re-evaluate as Grade 2
```

---

### Grade 2 CRS

**Symptoms:** Fever + hypotension responsive to fluids OR hypoxia (FiO₂ <40%)

```
Management:
1. TOCILIZUMAB (IL-6 Inhibitor)
   ├─ Dose: 8 mg/kg IV (max 800 mg)
   ├─ Infuse over 1 hour
   ├─ Can repeat × 1 after 8 hours if no improvement
   └─ Maximum 3-4 doses total

2. Supportive Care
   ├─ Fluid resuscitation: 500-1000 mL NS bolus
   ├─ Supplemental oxygen to maintain SpO₂ >92%
   ├─ Acetaminophen 650-1000 mg q6h
   └─ Consider meperidine 25-50 mg for rigors

3. Monitoring
   ├─ Transfer to step-down unit or ICU
   ├─ Continuous cardiac monitoring
   ├─ Continuous pulse oximetry
   ├─ Vital signs q15min until stable
   ├─ Strict I/O monitoring
   └─ Consider arterial line if hypotension persists

4. If No Improvement at 24h
   ├─ Consider adding corticosteroids
   └─ Dexamethasone 10 mg IV q6h (off-label)
```

---

### Grade 3-4 CRS

**Symptoms:** High-dose vasopressors OR FiO₂ ≥40% OR mechanical ventilation

```
Management:
1. IMMEDIATE ICU TRANSFER

2. TOCILIZUMAB + CORTICOSTEROIDS (Combination)
   ├─ Tocilizumab 8 mg/kg IV (give first)
   ├─ PLUS Dexamethasone 10 mg IV q6h
   │  (or Methylprednisolone 1-2 mg/kg/day)
   └─ Continue steroids until CRS resolves to Grade ≤1

3. Aggressive Supportive Care
   ├─ Vasopressors (norepinephrine, vasopressin)
   ├─ Mechanical ventilation if needed
   ├─ Arterial line for BP monitoring
   ├─ Central line for vasopressor administration
   ├─ Foley catheter for strict I/O
   └─ Consider Swan-Ganz catheter if cardiogenic shock

4. Hold Epcoritamab
   ├─ Do NOT give subsequent doses until CRS resolves
   ├─ Restart at reduced dose after recovery
   └─ Consider permanent discontinuation for Grade 4

5. Multidisciplinary Management
   ├─ Oncology
   ├─ Critical care
   ├─ Infectious disease (rule out sepsis)
   └─ Cardiology (if cardiac dysfunction)
```

---

## 🔬 Laboratory Monitoring

### Baseline (Pre-Treatment)

```
☑ Complete Blood Count (CBC) with differential
☑ Comprehensive Metabolic Panel (CMP)
☑ Liver Function Tests (LFTs)
☑ Coagulation panel (PT/INR, aPTT)
☑ C-Reactive Protein (CRP)
☑ Ferritin
☑ Lactate Dehydrogenase (LDH)
☑ Fibrinogen

Optional (if biomarker study):
☑ IL-6, IL-10, IFN-γ
☑ CCL17, CCL13, MCP-1
```

### During CRS Episode

```
☑ CBC q6-12h (watch for cytopenias)
☑ CMP q12-24h (electrolytes, renal function)
☑ LFTs q24h (hepatotoxicity)
☑ CRP, ferritin q24h (inflammatory markers)
☑ Lactate q6h if Grade ≥2 (tissue perfusion)
☑ Blood cultures if fever (rule out infection)
☑ Coags if DIC suspected
```

### Biomarker Trends (if available)

| Biomarker | Grade 1 CRS | Grade 2 CRS | Grade 3-4 CRS |
|-----------|-------------|-------------|---------------|
| IL-6 | 5-50 pg/mL | 50-200 pg/mL | >200 pg/mL |
| Ferritin | 500-2000 ng/mL | 2000-10,000 ng/mL | >10,000 ng/mL |
| CRP | 10-50 mg/L | 50-150 mg/L | >150 mg/L |

---

## 📞 Decision Support Algorithm

### When to Call for Help

```
┌────────────────────────────────────────────────────────┐
│               ESCALATION CRITERIA                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  CALL PRIMARY ONCOLOGY TEAM:                           │
│  ☎ Fever ≥38°C developing                             │
│  ☎ Any Grade 1 CRS symptoms                           │
│  ☎ Patient anxiety or concern                         │
│                                                        │
│  ACTIVATE RAPID RESPONSE:                              │
│  🚨 Hypotension (SBP <90 mmHg)                        │
│  🚨 Hypoxia (SpO₂ <92% on room air)                   │
│  🚨 Altered mental status                             │
│  🚨 Respiratory distress                              │
│                                                        │
│  IMMEDIATE ICU TRANSFER:                               │
│  ⚠️ Grade 3-4 CRS confirmed                           │
│  ⚠️ Requiring vasopressors                            │
│  ⚠️ Requiring FiO₂ >40%                               │
│  ⚠️ No response to tocilizumab + fluids at 2h         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 📋 Patient Education Materials

### Pre-Treatment Patient Counseling

**Topics to Cover:**

1. **What is CRS?**
   - "Your immune system may become very active after the medication"
   - "This can cause fever, low blood pressure, and breathing problems"
   - "We will watch you very closely for 24 hours"

2. **Symptoms to Report Immediately:**
   - ✓ Fever or chills
   - ✓ Dizziness or lightheadedness
   - ✓ Shortness of breath
   - ✓ Rapid heartbeat
   - ✓ Confusion

3. **Monitoring Plan:**
   - "We will check your vital signs every 15-30 minutes at first"
   - "You will stay in the hospital for at least 24 hours"
   - "Most symptoms happen in the first day"

4. **Treatment Available:**
   - "We have medications ready to stop CRS (tocilizumab)"
   - "Most CRS is mild and responds quickly to treatment"
   - "An ICU bed is available if needed"

---

### Post-Discharge Instructions

**Give to Patient/Caregiver:**

```
┌─────────────────────────────────────────────────────┐
│        POST-EPCORITAMAB DISCHARGE INSTRUCTIONS       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ You were treated with epcoritamab today.           │
│ Your risk of serious side effects is now very low. │
│                                                     │
│ CALL YOUR DOCTOR IMMEDIATELY IF:                    │
│ • Fever >100.4°F (38°C)                            │
│ • Dizziness or fainting                            │
│ • Trouble breathing                                │
│ • Fast heartbeat                                   │
│ • Confusion or drowsiness                          │
│                                                     │
│ FOLLOW-UP:                                          │
│ • We will call you in 48 hours                     │
│ • Return to clinic in 7 days                       │
│ • Call 24/7 hotline: [PHONE NUMBER]                │
│                                                     │
│ MEDICATIONS:                                        │
│ • Take acetaminophen 650mg as needed for fever     │
│ • Stay well-hydrated (8 glasses water/day)         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Quality Improvement Metrics

### Suggested Tracking

| Metric | Target | Purpose |
|--------|--------|---------|
| **CRS incidence** | Monitor vs. 34.4% baseline | Track consistency |
| **Time to CRS onset** | Document all cases | Validate 24h window |
| **Grade 3-4 CRS rate** | <5% | Safety threshold |
| **Time to tocilizumab** | <30 min from Grade 2 | Process efficiency |
| **ICU admission rate** | <5% | Resource planning |
| **CRS-related mortality** | 0% | Safety endpoint |
| **Weight documentation** | 100% | Enable risk scoring |

---

## 📚 References

1. **Lee, D. W., et al. (2019).** "ASTCT Consensus Grading for Cytokine Release Syndrome and Neurologic Toxicity Associated with Immune Effector Cells." *Biology of Blood and Marrow Transplantation*, 25(4), 625-638.

2. **Thieblemont, C., et al. (2022).** "Epcoritamab, a Novel, Subcutaneous CD3xCD20 Bispecific T-Cell-Engaging Antibody, in Relapsed or Refractory Large B-Cell Lymphoma: Dose Escalation in an Open-Label Phase I/II Trial." *Journal of Clinical Oncology*, 40(21), 2238-2247.

3. **Hutchings, M., et al. (2021).** "Glofitamab, a Novel, Bivalent CD20-Targeting T-Cell-Engaging Bispecific Antibody, Induces Durable Complete Remissions in Relapsed or Refractory B-Cell Lymphoma." *Blood*, 137(21), 2892-2901.

4. **Hay, K. A., et al. (2017).** "Kinetics and biomarkers of severe cytokine release syndrome after CD19 chimeric antigen receptor–modified T-cell therapy for acute lymphoblastic leukemia." *Blood*, 130(21), 2295-2306.

5. **National Cancer Institute (NCI).** Common Terminology Criteria for Adverse Events (CTCAE) v5.0. https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm

---

## 📞 Emergency Contacts

### Template for Institution

```
┌─────────────────────────────────────────────┐
│      EPCORITAMAB CRS EMERGENCY CONTACTS      │
├─────────────────────────────────────────────┤
│                                             │
│ Primary Oncology Team:      [PHONE]        │
│ Hematology Fellow on Call:  [PAGER]        │
│ ICU Attending:              [PHONE]        │
│ Pharmacy (Tocilizumab):     [PHONE]        │
│ Rapid Response Team:        [CODE]         │
│                                             │
│ Epcoritamab Protocol PI:    [PHONE]        │
│ After Hours Backup:         [PHONE]        │
│                                             │
└─────────────────────────────────────────────┘
```

---

## ✅ Pre-Treatment Checklist

```
EPCORITAMAB ADMINISTRATION CHECKLIST

Patient Name: ________________  MRN: __________  Date: ________

☐ Risk score calculated: ______ (Low/Moderate/High)
☐ Patient weight documented: ______ kg
☐ Pre-medications given:
   ☐ Acetaminophen 650-1000 mg PO
   ☐ Diphenhydramine 25-50 mg PO or IV (if ordered)
☐ Baseline vitals obtained and documented
☐ Baseline labs drawn (CBC, CMP, LFTs)
☐ IV access established (18G or larger preferred)
☐ Tocilizumab 8 mg/kg available on unit
☐ Dexamethasone 10 mg vials available
☐ ICU bed reserved (if high-risk patient)
☐ Patient counseled on CRS symptoms
☐ Consent obtained and documented
☐ Monitoring plan communicated to nursing
☐ Emergency contact numbers posted in chart

Infusion Start Time: ______  Completed: ______
Administered by: ____________  RN License: ______

POST-INFUSION MONITORING SCHEDULE:
☐ Q15min × 4 hours (0-4h)
☐ Q30min × 8 hours (4-12h)
☐ Q1h × 12 hours (12-24h)
☐ 24-hour reassessment completed

Discharge Criteria Met:
☐ No fever × 12 hours
☐ Vital signs stable × 6 hours
☐ Patient ambulatory
☐ Discharge instructions given
☐ Follow-up appointment scheduled

Discharge Time: ______  Discharged by: ____________
```

---

**Last Updated:** 2025-11-18  
**Version:** 1.0  
**Review Date:** 2026-11-18

**Clinical Use:** These guidelines are based on the survival analysis findings from this study combined with established CRS management protocols. Institutional protocols may vary. Always follow your institution's policies and procedures.

---

<div align="center">

**For questions or feedback on these guidelines, contact:**  
[Your Name] • [Institution] • [Email]

</div>

