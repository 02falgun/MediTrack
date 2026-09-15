"""
MediTrack - Guideline-Grounded Retrieval Layer & Clinical Assistant
Project CP-02: Readmission Risk Prediction and Clinical Decision Support

Integrates published clinical discharge protocols:
1. ADA 2024 Standards of Care: Diabetes Care in the Hospital (Discharge Planning)
2. AHRQ Project RED (Re-Engineered Discharge) Protocol
3. CMS Transitional Care Management (TCM) Guidelines
4. ACC/AHA Inpatient Heart Failure Discharge Protocol
5. KDIGO Clinical Practice Guideline for Diabetes & Nephropathy
"""

import os
import re
import json

# Curated Published Clinical Guidelines Knowledge Base
CLINICAL_GUIDELINES = [
    {
        "id": "ADA-DISCH-01",
        "title": "ADA Standards of Care: Hospital Discharge Planning & Glycemic Transition",
        "authority": "American Diabetes Association (ADA)",
        "source_citation": "American Diabetes Association. 'Diabetes Care in the Hospital: Standards of Care in Diabetes—2024.' Diabetes Care 2024;47(Suppl. 1):S295–S306. Grade A Recommendation.",
        "keywords": ["diabetes", "insulin", "glycemic", "a1c", "metformin", "hypoglycemia", "medication change", "glucose"],
        "summary": "Structured transition from inpatient to outpatient diabetes regimen with explicit hypoglycemia safety instructions and rapid primary follow-up.",
        "key_recommendations": [
            "Medication Reconciliation: Verify outpatient medication regimen at discharge; reconcile discrepancies caused by inpatient alterations.",
            "Insulin Regimen Clarity: If discharged on insulin, provide written injection schedules, needle disposal protocols, and glucometer verification before departure.",
            "Hypoglycemia Education: Educate patient and family caregiver on Rule of 15 (15g fast-acting carbs, recheck in 15 mins) and prescribe emergency glucagon if on high-risk secretagogues/insulin.",
            "Follow-up Timing: High-risk patients with inpatient medication changes require outpatient follow-up within 7 to 14 days of discharge."
        ],
        "evidence_level": "Level A (Randomized controlled clinical trials & multi-center evidence)"
    },
    {
        "id": "AHRQ-RED-02",
        "title": "AHRQ Project RED (Re-Engineered Discharge) Checklist & Protocol",
        "authority": "Agency for Healthcare Research and Quality (AHRQ)",
        "source_citation": "Agency for Healthcare Research and Quality (AHRQ). 'Project RED (Re-Engineered Discharge) Implementation Handbook.' AHRQ Pub. No. 12(13)-0084, revised 2023. Guideline Standard.",
        "keywords": ["discharge", "red flags", "teach-back", "follow-up", "inpatient", "utilization", "transportation", "reconciliation"],
        "summary": "Evidence-based 12-step discharge bundle clinically proven to reduce 30-day hospital readmissions and emergency department visits by 30%.",
        "key_recommendations": [
            "Comprehensive AHOP: Provide an After-Hospital Care Plan (AHOP) written at 5th-to-6th grade reading level detailing medication changes and doctor appointments.",
            "Teach-Back Verification: Confirm comprehension of critical discharge self-care instructions using the teach-back method rather than passive acknowledgment.",
            "Red Flag Symptoms: Provide explicit warning symptoms ('red flags') and unambiguous contact numbers for who to call 24/7 before heading to the Emergency Room.",
            "Post-Discharge Outreach: Mandatory clinical phone call by discharge nurse or clinical pharmacist within 48 to 72 hours post-discharge."
        ],
        "evidence_level": "Level A (Clinical trial demonstration across urban tertiary medical centers)"
    },
    {
        "id": "CMS-TCM-03",
        "title": "CMS Transitional Care Management (TCM) Services Protocol",
        "authority": "Centers for Medicare & Medicaid Services (CMS)",
        "source_citation": "Centers for Medicare & Medicaid Services (CMS). 'Transitional Care Management Services.' Medicare Learning Network MLN Booklet ICN MLN908628, 2024. Federal Regulation Standard.",
        "keywords": ["tcm", "medicare", "inpatient history", "complex", "frequent", "prior", "admission", "coordination"],
        "summary": "Mandated transitional coordination for high-utilization patients to prevent rapid post-discharge relapse and readmission penalties.",
        "key_recommendations": [
            "Initial Interactive Contact: Must occur within 2 business days of discharge via telephone, electronic portal, or direct in-person coordination.",
            "Complexity Stratification: Patients with high medical decision complexity require face-to-face physician or qualified practitioner visit within 7 calendar days.",
            "Care Coordination Services: Assist with prescription refills, identify barriers to adherence (transportation, pharmacy co-pays), and coordinate durable medical equipment.",
            "Communication with Community Providers: Ensure prompt delivery of inpatient summary to the outpatient primary care physician within 48 hours."
        ],
        "evidence_level": "Federal Healthcare Quality & Payment Policy Guideline"
    },
    {
        "id": "ACC-AHA-HF-04",
        "title": "ACC/AHA Heart Failure & Cardiovascular Discharge Guidance",
        "authority": "American College of Cardiology & American Heart Association",
        "source_citation": "Heidenreich PA, et al. '2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure.' Circulation 2022;145:e895–e1032. Class I Guideline.",
        "keywords": ["circulatory", "heart failure", "cardiovascular", "hypertension", "edema", "fluid", "diuretic", "weight"],
        "summary": "Targeted clinical protocol for patients admitted with primary or secondary circulatory and cardiovascular conditions.",
        "key_recommendations": [
            "Daily Weight Monitoring: Instruct daily morning weight measurement after voiding; report weight gain > 2-3 lbs in 24h or > 5 lbs in one week.",
            "Electrolyte & Renal Panel: Schedule outpatient serum creatinine and potassium lab draw within 7 to 10 days post-discharge if adjusting ACEi/ARB/ARNI or loop diuretics.",
            "Guideline-Directed Medical Therapy (GDMT): Document optimization of beta-blocker, SGLT2i, and mineralocorticoid receptor antagonist prior to discharge.",
            "Early Ambulatory Clinic Visit: Schedule confirmed clinic follow-up within 7 days of discharge."
        ],
        "evidence_level": "Class I, Level A (High-quality evidence from multiple randomized clinical trials)"
    },
    {
        "id": "KDIGO-CKD-05",
        "title": "KDIGO Clinical Practice Guideline for Diabetes & Nephropathy Management",
        "authority": "Kidney Disease: Improving Global Outcomes (KDIGO)",
        "source_citation": "de Boer IH, et al. 'KDIGO 2023 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease.' Kidney International 2023;104(5S):S1–S127. Recommendation 1.3.",
        "keywords": ["genitourinary", "kidney", "ckd", "renal", "nephropathy", "creatinine", "metformin", "dialysis"],
        "summary": "Safety guidance on glycemic management and pharmacological dosing in patients with renal vulnerability and genitourinary conditions.",
        "key_recommendations": [
            "Renal Medication Adjustment: Re-evaluate metformin safety based on discharge eGFR (contraindicated if eGFR < 30 mL/min/1.73m²; dose halved if 30-44 mL/min/1.73m²).",
            "Hydration & Nephrotoxic Avoidance: Counsel patient to avoid over-the-counter NSAIDs (ibuprofen, naproxen) and iodinated contrast without prior hydration.",
            "Blood Pressure Target: Target systolic blood pressure < 120 mmHg when tolerated using standardized office measurement.",
            "Nephrology Co-Management: Rapid outpatient referral if proteinuria persists or rapid decline in filtration rate was observed during hospitalization."
        ],
        "evidence_level": "Level 1A Recommendation"
    }
]

# Clinical Safety Guardrail Patterns
# Strict regex matchers for diagnostic and prescriptive requests
DIAGNOSTIC_PATTERNS = [
    r"\b(do i have|does the patient have|diagnose|what diagnosis|what disease|is this cancer|is it a heart attack|what condition does|is this patient suffering from|determine diagnosis)\b",
    r"\b(what is the diagnosis|provide a diagnosis|give me a diagnosis|tell me if the patient has)\b",
    r"\b(confirm whether the patient has|clinical diagnosis of|symptom diagnosis)\b"
]

PRESCRIPTIVE_PATTERNS = [
    r"\b(prescribe|write a prescription|give prescription|start medication|prescribe \w+|dosage of \w+ should be|administer \d+ mg|change (dosage|dose) to)\b",
    r"\b(increase dose to|decrease dose to|what dose should i prescribe|order prescription|sign prescription)\b",
    r"\b(switch medication to|discontinue \w+ and start|fill prescription)\b",
    r"\b(order|administer|dispense)\s+(\d+\s*(?:mg|mcg|ml|units|tablets?)?|\w+)\b"
]

def check_guardrails(user_prompt: str) -> dict:
    """
    Evaluates user prompt against strict clinical decision support safety guardrails.
    Refuses diagnostic and prescriptive requests and defers immediately to licensed clinicians.
    """
    prompt_lower = user_prompt.lower()
    
    # 1. Test for Prescriptive Requests
    for pattern in PRESCRIPTIVE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return {
                "allowed": False,
                "violation_type": "PRESCRIPTIVE_ORDER_REQUEST",
                "refusal_message": (
                    "⛔ CLINICAL GUARDRAIL REFUSAL: Prescriptive Order Request Blocked.\n\n"
                    "MediTrack operates strictly as a post-discharge clinical decision support and risk-stratification system. "
                    "In accordance with healthcare safety standards and regulatory guidelines, this assistant does NOT have prescriptive authority "
                    "and cannot order, prescribe, or alter medication dosages. "
                    "All pharmacotherapeutic decisions must be executed directly by the licensed attending physician or clinical pharmacist."
                ),
                "defer_to_clinician": True
            }
            
    # 2. Test for Diagnostic Requests
    for pattern in DIAGNOSTIC_PATTERNS:
        if re.search(pattern, prompt_lower):
            return {
                "allowed": False,
                "violation_type": "DIAGNOSTIC_INFERENCE_REQUEST",
                "refusal_message": (
                    "⛔ CLINICAL GUARDRAIL REFUSAL: Diagnostic Request Blocked.\n\n"
                    "MediTrack is not a diagnostic instrument. The system is designed solely for post-discharge readmission risk scoring, "
                    "factor attribution, and transitional care guideline retrieval. It cannot establish primary or differential diagnoses. "
                    "Please refer patient diagnostic evaluation to the attending medical team."
                ),
                "defer_to_clinician": True
            }
            
    return {"allowed": True, "violation_type": None, "refusal_message": None, "defer_to_clinician": False}

def retrieve_relevant_guidelines(patient_profile: dict, query_text: str = "") -> list:
    """
    Retrieves the most clinically applicable published guidelines based on
    patient risk factors (diagnoses, medications, utilization) and query text.
    """
    scores = {}
    
    # Extract clinical cues
    diag1 = str(patient_profile.get('diag_1_category', '')).lower()
    diag2 = str(patient_profile.get('diag_2_category', '')).lower()
    on_insulin = patient_profile.get('on_insulin', 0)
    has_med_change = patient_profile.get('has_medication_change', 0)
    prior_inp = patient_profile.get('number_inpatient', 0)
    query_lower = query_text.lower()
    
    for g in CLINICAL_GUIDELINES:
        score = 0
        g_id = g['id']
        
        # Match keywords against profile
        if 'diabetes' in diag1 or 'diabetes' in diag2 or on_insulin or has_med_change:
            if 'diabetes' in g['keywords']:
                score += 5
                
        if 'circulatory' in diag1 or 'circulatory' in diag2:
            if 'circulatory' in g['keywords'] or 'heart failure' in g['keywords']:
                score += 5
                
        if 'genitourinary' in diag1 or 'genitourinary' in diag2:
            if 'genitourinary' in g['keywords'] or 'kidney' in g['keywords']:
                score += 5
                
        if prior_inp >= 1:
            if 'tcm' in g['keywords'] or 'inpatient history' in g['keywords']:
                score += 4
                
        # Always relevant baseline for all hospital discharges
        if g_id == 'AHRQ-RED-02':
            score += 3
            
        # Match against query text if provided
        for kw in g['keywords']:
            if kw in query_lower:
                score += 3
                
        scores[g_id] = score
        
    # Sort guidelines by relevance score
    ranked_guidelines = sorted(CLINICAL_GUIDELINES, key=lambda x: scores.get(x['id'], 0), reverse=True)
    return ranked_guidelines[:3]

def generate_discharge_planning_summary(patient_profile: dict, user_query: str = "") -> dict:
    """
    Core Assistant Generator:
    Evaluates guardrails, retrieves applicable guidelines, generates a structured
    discharge planning summary tailored to patient risk profile with MANDATORY citations.
    """
    # 1. Run safety guardrail gate
    guardrail_res = check_guardrails(user_query)
    if not guardrail_res['allowed']:
        return {
            "status": "REFUSED",
            "guardrail_triggered": True,
            "violation_type": guardrail_res['violation_type'],
            "content": guardrail_res['refusal_message'],
            "citations": []
        }
        
    # 2. Retrieve applicable guidelines
    relevant_guidelines = retrieve_relevant_guidelines(patient_profile, user_query)
    
    # 3. Assemble patient context summary
    pt_id = patient_profile.get('encounter_id', 'Unknown')
    age = patient_profile.get('age_group', 'Unknown')
    diag = patient_profile.get('diag_1_category', 'General Medical')
    los = patient_profile.get('time_in_hospital', 1)
    risk_score = patient_profile.get('predicted_risk_score', 0.0)
    risk_tier = "HIGH RISK" if risk_score >= 0.18 else ("MODERATE RISK" if risk_score >= 0.10 else "LOW RISK")
    prior_inp = patient_profile.get('number_inpatient', 0)
    med_change = "Yes" if patient_profile.get('has_medication_change', 0) else "None"
    
    # 4. Generate structured clinical guidance sections
    summary_text = (
        f"### MediTrack Clinical Decision Support: Discharge Planning Summary\n"
        f"**Patient Encounter:** #{pt_id} | **Risk Stratification:** {risk_tier} (30d Score: {risk_score*100:.1f}%)\n"
        f"**Admitting Diagnosis:** {diag} | **Length of Stay:** {los} days | **Prior Inpatient Admissions:** {prior_inp}\n\n"
        f"--- \n"
        f"#### 1. Targeted Discharge Transition Checklist\n"
    )
    
    checklist_items = []
    if patient_profile.get('on_insulin') or patient_profile.get('has_medication_change'):
        checklist_items.append("• **Insulin & Glycemic Reconciliation**: Provide written dose schedule, verify glucometer supply, and review hypoglycemia Rule of 15.")
    if prior_inp >= 1 or risk_score >= 0.18:
        checklist_items.append("• **Transitional Care Management (TCM)**: Initiate direct interactive outreach within 48 business hours post-discharge.")
        checklist_items.append("• **Accelerated Ambulatory Follow-up**: Schedule face-to-face physician appointment within 7 days.")
    checklist_items.append("• **Teach-Back Comprehension**: Utilize teach-back method to verify patient comprehension of medication schedules and red-flag symptoms.")
    checklist_items.append("• **24/7 Red-Flag Action Plan**: Provide written phone number and specific warning triggers before seeking Emergency Room care.")
    
    summary_text += "\n".join(checklist_items) + "\n\n"
    
    summary_text += "#### 2. Protocol Grounding & Clinical Citations\n"
    citations_data = []
    for g in relevant_guidelines:
        summary_text += f"**{g['title']}**\n"
        summary_text += f"- *Authority*: {g['authority']} ({g['evidence_level']})\n"
        summary_text += f"- *Mandatory Source Citation*: `{g['source_citation']}`\n"
        summary_text += f"- *Applicable Recommendation*: {g['key_recommendations'][0]}\n\n"
        citations_data.append({
            "id": g['id'],
            "title": g['title'],
            "authority": g['authority'],
            "citation": g['source_citation'],
            "evidence_level": g['evidence_level']
        })
        
    summary_text += (
        "> **Clinical Notice**: This recommendation is grounded in published national transition guidelines "
        "and is supplied to augment clinical workflow. Final medical decisions, discharge timing, and pharmacotherapy "
        "remain under the sole jurisdiction of the licensed attending medical staff."
    )
    
    return {
        "status": "SUCCESS",
        "guardrail_triggered": False,
        "violation_type": None,
        "content": summary_text,
        "citations": citations_data
    }

if __name__ == '__main__':
    # Test valid query
    sample_pt = {
        'encounter_id': 149190,
        'age_group': '[10-20)',
        'diag_1_category': 'Circulatory',
        'time_in_hospital': 3,
        'predicted_risk_score': 0.28,
        'number_inpatient': 2,
        'on_insulin': 1,
        'has_medication_change': 1
    }
    
    print("=== TEST 1: VALID CLINICAL TRANSITION QUERY ===")
    res_valid = generate_discharge_planning_summary(sample_pt, "Prepare discharge plan and transition checklist")
    print(res_valid['content'])
    
    print("\n=== TEST 2: GUARDRAIL TEST - PRESCRIPTIVE QUERY ===")
    res_rx = generate_discharge_planning_summary(sample_pt, "Prescribe 500mg Metformin twice daily")
    print(res_rx['content'])
    
    print("\n=== TEST 3: GUARDRAIL TEST - DIAGNOSTIC QUERY ===")
    res_dx = generate_discharge_planning_summary(sample_pt, "Can you diagnose what disease this patient has?")
    print(res_dx['content'])
