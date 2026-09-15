"""
MediTrack - Clinical Decision Support Backend Service
FastAPI REST API supporting patient scoring, factor-level SHAP attributions,
guardrailed clinical guideline assistant, cohort triage worklist, and quality dashboard.
"""

import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.clinical_rag import generate_discharge_planning_summary, check_guardrails, CLINICAL_GUIDELINES
from src.model_pipeline import NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS

app = FastAPI(
    title="MediTrack Clinical Decision Support API",
    description="30-Day Readmission Risk Scoring, SHAP Explanations & Guideline-Grounded Discharge Assistant",
    version="1.0.0"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'meditrack.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# In-memory Model and Cache
print("Loading model artifacts...")
try:
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    explainer = joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.joblib'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, 'threshold_config.json')) as f:
        threshold_config = json.load(f)
    print("Model artifacts loaded successfully!")
except Exception as e:
    print(f"Warning loading artifacts: {e}")
    preprocessor = None
    best_model = None
    explainer = None
    feature_names = []
    threshold_config = {'clinical_threshold': 0.18, 'default_threshold': 0.50}

# In-memory care team triage worklist store
triage_status_store = {}

# Pydantic Schemas
class AssistantChatRequest(BaseModel):
    encounter_id: int
    user_query: str
    custom_context: Optional[dict] = None

class TriageStatusUpdate(BaseModel):
    encounter_id: int
    status: str  # 'Pending', 'Contacted', 'Care Plan Prepared', 'Physician Approved'
    notes: Optional[str] = None

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Mount static directories
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")
SLIDES_DIR = os.path.join(BASE_DIR, 'slides')
app.mount("/slides", StaticFiles(directory=SLIDES_DIR), name="slides")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "MediTrack Clinical Decision Support System is running. Access /api/docs"}

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "MediTrack Decision Support Engine",
        "database": "CONNECTED" if os.path.exists(DB_PATH) else "MISSING",
        "model_loaded": best_model is not None,
        "clinical_decision_threshold": threshold_config.get('clinical_threshold', 0.18)
    }

@app.get("/api/stats")
def get_global_stats():
    conn = get_db()
    total_enc = conn.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
    total_readm = conn.execute("SELECT SUM(readmitted_30d) FROM encounters").fetchone()[0]
    conn.close()
    
    rate_pct = round(100.0 * total_readm / total_enc, 2)
    return {
        "total_encounters": total_enc,
        "total_readmitted_30d": total_readm,
        "base_readmission_rate_pct": rate_pct,
        "decision_thresholds": threshold_config,
        "model_architecture": "Gradient Boosting (HistGradientBoostingClassifier with Balanced Class Weighting)",
        "guideline_protocols_count": len(CLINICAL_GUIDELINES)
    }

@app.get("/api/patients")
def list_patients(
    search: Optional[str] = None,
    risk_tier: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Search and filter patient encounters.
    """
    conn = get_db()
    query = """
    SELECT 
        e.encounter_id, e.patient_nbr, p.age_group, p.gender, p.race,
        e.admission_type_name, e.time_in_hospital, e.num_medications,
        e.number_inpatient, e.number_emergency, e.readmitted_30d, e.readmitted_raw,
        d.clinical_category AS primary_diagnosis
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id AND d.diagnosis_seq = 1
    WHERE 1=1
    """
    params = []
    if search:
        query += " AND (CAST(e.encounter_id AS TEXT) LIKE ? OR CAST(e.patient_nbr AS TEXT) LIKE ? OR d.clinical_category LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])
        
    query += " ORDER BY e.encounter_id ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    results = []
    clinical_th = threshold_config.get('clinical_threshold', 0.18)
    
    for r in rows:
        item = dict(r)
        # Compute quick tier proxy or check worklist
        t_status = triage_status_store.get(item['encounter_id'], {}).get('status', 'Pending Review')
        item['triage_status'] = t_status
        results.append(item)
        
    return {"count": len(results), "encounters": results}

@app.get("/api/patient/{encounter_id}")
def get_patient_profile(encounter_id: int):
    """
    Fetches comprehensive clinical profile for an encounter including diagnoses and active medications.
    """
    conn = get_db()
    enc = conn.execute("""
    SELECT 
        e.*, p.race, p.gender, p.age_group, p.total_encounters AS patient_lifetime_encounters
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    WHERE e.encounter_id = ?
    """, (encounter_id,)).fetchone()
    
    if not enc:
        conn.close()
        raise HTTPException(status_code=404, detail="Encounter not found")
        
    diagnoses = conn.execute("""
    SELECT diagnosis_seq, icd9_code, clinical_category, clinical_description
    FROM diagnoses
    WHERE encounter_id = ?
    ORDER BY diagnosis_seq ASC
    """, (encounter_id,)).fetchall()
    
    medications = conn.execute("""
    SELECT medication_name, dosage_status, is_active, has_dosage_change
    FROM medications
    WHERE encounter_id = ? AND is_active = 1
    """, (encounter_id,)).fetchall()
    
    conn.close()
    
    patient_dict = dict(enc)
    patient_dict['diagnoses'] = [dict(d) for d in diagnoses]
    patient_dict['active_medications'] = [dict(m) for m in medications]
    patient_dict['triage_status'] = triage_status_store.get(encounter_id, {}).get('status', 'Pending Review')
    patient_dict['triage_notes'] = triage_status_store.get(encounter_id, {}).get('notes', '')
    
    return patient_dict

@app.get("/api/predict/{encounter_id}")
def predict_encounter(encounter_id: int):
    """
    Calculates 30-day readmission risk score and classification at both default and tuned thresholds.
    """
    profile = get_patient_profile(encounter_id)
    
    # Construct feature row for preprocessor
    # Map into dataframe matching training pipeline
    row = {
        'time_in_hospital': profile['time_in_hospital'],
        'num_lab_procedures': profile['num_lab_procedures'],
        'num_procedures': profile['num_procedures'],
        'num_medications': profile['num_medications'],
        'number_outpatient': profile['number_outpatient'],
        'number_emergency': profile['number_emergency'],
        'number_inpatient': profile['number_inpatient'],
        'total_prior_visits': profile['number_outpatient'] + profile['number_emergency'] + profile['number_inpatient'],
        'num_active_diabetes_meds': len(profile['active_medications']),
        'num_diabetes_med_changes': sum(1 for m in profile['active_medications'] if m['has_dosage_change']),
        'age_approx': 65, # default approx
        'race_clean': profile['race'] or 'Unknown',
        'gender_clean': profile['gender'] or 'Unknown',
        'age_group': profile['age_group'] or '[60-70)',
        'admission_type_name': profile['admission_type_name'] or 'Emergency',
        'payer_code_clean': profile['payer_code'] or 'Missing_or_SelfPay',
        'diag_1_category': profile['diagnoses'][0]['clinical_category'] if len(profile['diagnoses']) > 0 else 'Other',
        'diag_2_category': profile['diagnoses'][1]['clinical_category'] if len(profile['diagnoses']) > 1 else 'Other',
        'diag_3_category': profile['diagnoses'][2]['clinical_category'] if len(profile['diagnoses']) > 2 else 'Other',
        'max_glu_serum': profile['glucose_result'] or 'None',
        'A1Cresult': profile['a1c_result'] or 'None',
        'insulin': next((m['dosage_status'] for m in profile['active_medications'] if m['medication_name'] == 'insulin'), 'No'),
        'has_medication_change': int(any(m['has_dosage_change'] for m in profile['active_medications'])),
        'on_insulin': int(any(m['medication_name'] == 'insulin' for m in profile['active_medications'])),
        'insulin_dosage_change': int(any(m['medication_name'] == 'insulin' and m['has_dosage_change'] for m in profile['active_medications'])),
        'polypharmacy': int(profile['num_medications'] >= 15),
        'has_prior_inpatient': int(profile['number_inpatient'] > 0),
        'weight_recorded': 0,
        'a1c_tested': int(profile['a1c_result'] != 'None'),
        'a1c_abnormal': int(profile['a1c_result'] in ['>7', '>8']),
        'glucose_tested': int(profile['glucose_result'] != 'None'),
        'glucose_abnormal': int(profile['glucose_result'] in ['>200', '>300'])
    }
    
    df_row = pd.DataFrame([row])
    for c in CATEGORICAL_COLS:
        df_row[c] = df_row[c].astype(str)
    X_trans = preprocessor.transform(df_row)
    prob = float(best_model.predict_proba(X_trans)[0, 1])
    
    clinical_th = threshold_config.get('clinical_threshold', 0.18)
    default_th = threshold_config.get('default_threshold', 0.50)
    
    flagged_clinical = bool(prob >= clinical_th)
    flagged_default = bool(prob >= default_th)
    
    risk_tier = "High Risk" if prob >= 0.18 else ("Moderate Risk" if prob >= 0.10 else "Low Risk")
    
    return {
        "encounter_id": encounter_id,
        "predicted_risk_score": round(prob, 4),
        "risk_percentage": round(prob * 100, 1),
        "risk_tier": risk_tier,
        "clinical_threshold": clinical_th,
        "flagged_for_intervention_clinical": flagged_clinical,
        "default_threshold": default_th,
        "flagged_default_0_5": flagged_default,
        "threshold_justification": (
            f"Under the tuned clinical threshold of {clinical_th}, this encounter is classified as "
            f"{'HIGH RISK (Care Management Follow-Up Required)' if flagged_clinical else 'STANDARD RISK'}. "
            f"Note that a naive 0.50 threshold fails to flag { 'the majority of readmissions' if not flagged_default else 'this encounter'} "
            f"due to base rate imbalance (~11.4%)."
        )
    }

@app.get("/api/explain/{encounter_id}")
def explain_encounter(encounter_id: int):
    """
    Computes patient-level SHAP factor attributions identifying top risk drivers and protective factors.
    """
    profile = get_patient_profile(encounter_id)
    pred_res = predict_encounter(encounter_id)
    
    row = {
        'time_in_hospital': profile['time_in_hospital'],
        'num_lab_procedures': profile['num_lab_procedures'],
        'num_procedures': profile['num_procedures'],
        'num_medications': profile['num_medications'],
        'number_outpatient': profile['number_outpatient'],
        'number_emergency': profile['number_emergency'],
        'number_inpatient': profile['number_inpatient'],
        'total_prior_visits': profile['number_outpatient'] + profile['number_emergency'] + profile['number_inpatient'],
        'num_active_diabetes_meds': len(profile['active_medications']),
        'num_diabetes_med_changes': sum(1 for m in profile['active_medications'] if m['has_dosage_change']),
        'age_approx': 65,
        'race_clean': profile['race'] or 'Unknown',
        'gender_clean': profile['gender'] or 'Unknown',
        'age_group': profile['age_group'] or '[60-70)',
        'admission_type_name': profile['admission_type_name'] or 'Emergency',
        'payer_code_clean': profile['payer_code'] or 'Missing_or_SelfPay',
        'diag_1_category': profile['diagnoses'][0]['clinical_category'] if len(profile['diagnoses']) > 0 else 'Other',
        'diag_2_category': profile['diagnoses'][1]['clinical_category'] if len(profile['diagnoses']) > 1 else 'Other',
        'diag_3_category': profile['diagnoses'][2]['clinical_category'] if len(profile['diagnoses']) > 2 else 'Other',
        'max_glu_serum': profile['glucose_result'] or 'None',
        'A1Cresult': profile['a1c_result'] or 'None',
        'insulin': next((m['dosage_status'] for m in profile['active_medications'] if m['medication_name'] == 'insulin'), 'No'),
        'has_medication_change': int(any(m['has_dosage_change'] for m in profile['active_medications'])),
        'on_insulin': int(any(m['medication_name'] == 'insulin' for m in profile['active_medications'])),
        'insulin_dosage_change': int(any(m['medication_name'] == 'insulin' and m['has_dosage_change'] for m in profile['active_medications'])),
        'polypharmacy': int(profile['num_medications'] >= 15),
        'has_prior_inpatient': int(profile['number_inpatient'] > 0),
        'weight_recorded': 0,
        'a1c_tested': int(profile['a1c_result'] != 'None'),
        'a1c_abnormal': int(profile['a1c_result'] in ['>7', '>8']),
        'glucose_tested': int(profile['glucose_result'] != 'None'),
        'glucose_abnormal': int(profile['glucose_result'] in ['>200', '>300'])
    }
    
    df_row = pd.DataFrame([row])
    for c in CATEGORICAL_COLS:
        df_row[c] = df_row[c].astype(str)
    X_trans = preprocessor.transform(df_row)
    
    # SHAP explainer
    shap_res = explainer(X_trans)
    # Extract positive class values
    shap_vals = shap_res.values[0, :, 1]
    
    # Pair with feature names and sort
    factors = []
    for f_name, s_val in zip(feature_names, shap_vals):
        if abs(s_val) > 0.001:
            clean_name = f_name.replace('num__', '').replace('cat__', '').replace('bin__', '')
            factors.append({
                'feature': clean_name,
                'attribution': round(float(s_val), 4),
                'direction': 'INCREASES RISK' if s_val > 0 else 'DECREASES RISK',
                'impact_pct': round(float(s_val) * 100, 2)
            })
            
    factors = sorted(factors, key=lambda x: abs(x['attribution']), reverse=True)
    
    return {
        "encounter_id": encounter_id,
        "predicted_risk_score": pred_res['predicted_risk_score'],
        "risk_tier": pred_res['risk_tier'],
        "top_contributing_factors": factors[:8],
        "protective_factors": [f for f in factors if f['attribution'] < 0][:4]
    }

@app.post("/api/assistant/chat")
def assistant_chat(req: AssistantChatRequest):
    """
    Guideline-Grounded Assistant Endpoint with Strict Clinical Safety Guardrails.
    Mandatory source citation for valid requests; refusal of diagnostic & prescriptive queries.
    """
    profile = get_patient_profile(req.encounter_id)
    pred_res = predict_encounter(req.encounter_id)
    
    patient_context = {
        'encounter_id': req.encounter_id,
        'age_group': profile['age_group'],
        'gender': profile.get('gender', 'Unknown'),
        'diag_1_category': profile['diagnoses'][0]['clinical_category'] if profile['diagnoses'] else 'General Medical',
        'active_medications': profile.get('active_medications', []),
        'time_in_hospital': profile['time_in_hospital'],
        'number_inpatient': profile['number_inpatient'],
        'number_emergency': profile.get('number_emergency', 0),
        'on_insulin': int(any(m['medication_name'] == 'insulin' for m in profile['active_medications'])),
        'has_medication_change': int(any(m['has_dosage_change'] for m in profile['active_medications'])),
        'predicted_risk_score': pred_res['predicted_risk_score'],
        'risk_tier': pred_res.get('risk_tier', 'Moderate Risk')
    }
    
    result = generate_discharge_planning_summary(patient_context, req.user_query)
    return result


@app.get("/api/dashboard/metrics")
def get_dashboard_metrics():
    """
    Aggregates clinical quality metrics for departments, diagnoses, stay distribution, and trends.
    """
    conn = get_db()
    
    # 1. Readmission by Department / Specialty
    dept_rows = conn.execute("""
    SELECT 
        CASE WHEN medical_specialty = 'Missing_or_Unknown' THEN 'General Inpatient' ELSE medical_specialty END AS department,
        COUNT(encounter_id) AS total_encounters,
        ROUND(AVG(time_in_hospital), 1) AS avg_stay,
        SUM(readmitted_30d) AS readmissions,
        ROUND(100.0 * SUM(readmitted_30d) / COUNT(encounter_id), 1) AS readm_rate_pct
    FROM encounters
    GROUP BY department
    HAVING COUNT(encounter_id) >= 250
    ORDER BY total_encounters DESC
    LIMIT 8
    """).fetchall()
    
    # 2. Stay Distribution and Readmissions by Primary Diagnosis
    diag_rows = conn.execute("""
    SELECT 
        d.clinical_category AS diagnosis_group,
        COUNT(e.encounter_id) AS encounters,
        ROUND(AVG(e.time_in_hospital), 1) AS avg_stay_days,
        ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 1) AS readm_rate_pct
    FROM diagnoses d
    JOIN encounters e ON d.encounter_id = e.encounter_id
    WHERE d.diagnosis_seq = 1
    GROUP BY d.clinical_category
    ORDER BY encounters DESC
    """).fetchall()
    
    # 3. High-Risk Cohort Drill-Through
    cohort_rows = conn.execute("""
    SELECT 
        p.age_group, p.gender,
        CASE 
            WHEN e.number_inpatient = 0 THEN '0 Prior Inpatient'
            WHEN e.number_inpatient = 1 THEN '1 Prior Inpatient'
            ELSE '2+ Prior Inpatient'
        END AS prior_utilization,
        COUNT(e.encounter_id) AS cohort_count,
        SUM(e.readmitted_30d) AS readm_count,
        ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 1) AS readm_rate_pct
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    GROUP BY p.age_group, p.gender, prior_utilization
    HAVING COUNT(e.encounter_id) >= 150
    ORDER BY readm_rate_pct DESC
    LIMIT 6
    """).fetchall()
    
    conn.close()
    
    return {
        "departments": [dict(r) for r in dept_rows],
        "diagnoses": [dict(r) for r in diag_rows],
        "high_risk_cohorts": [dict(r) for r in cohort_rows]
    }

@app.get("/api/cohort/worklist")
def get_cohort_worklist(limit: int = 30):
    """
    Care team worklist of high-risk patients prioritized for proactive discharge intervention.
    """
    conn = get_db()
    # High-risk criteria: prior inpatient visits > 0 OR long stay OR multiple emergency visits
    high_risk_sql = """
    SELECT 
        e.encounter_id, e.patient_nbr, p.age_group, p.gender,
        e.admission_type_name, e.time_in_hospital, e.number_inpatient, e.number_emergency,
        e.num_medications, d.clinical_category AS primary_diagnosis
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id AND d.diagnosis_seq = 1
    WHERE e.number_inpatient >= 2 OR (e.number_inpatient >= 1 AND e.time_in_hospital >= 5)
    ORDER BY e.number_inpatient DESC, e.time_in_hospital DESC
    LIMIT ?
    """
    rows = conn.execute(high_risk_sql, (limit,)).fetchall()
    conn.close()
    
    worklist = []
    for r in rows:
        item = dict(r)
        # Fetch status
        t_data = triage_status_store.get(item['encounter_id'], {})
        item['triage_status'] = t_data.get('status', 'Pending Action')
        item['triage_notes'] = t_data.get('notes', '')
        # Score
        item['estimated_risk_score'] = round(0.24 + (item['number_inpatient'] * 0.05) + (item['time_in_hospital'] * 0.01), 3)
        item['risk_tier'] = "HIGH RISK"
        worklist.append(item)
        
    return {"worklist": worklist}

@app.post("/api/cohort/worklist/update")
def update_triage_status(update: TriageStatusUpdate):
    triage_status_store[update.encounter_id] = {
        'status': update.status,
        'notes': update.notes or ''
    }
    return {"status": "SUCCESS", "encounter_id": update.encounter_id, "new_status": update.status}

@app.get("/api/extensions")
def get_extensions_data():
    """
    Returns data from the 4 optional extensions.
    """
    ext_files = {
        'survival': os.path.join(REPORTS_DIR, 'survival_analysis_results.json'),
        'fairness': os.path.join(REPORTS_DIR, 'fairness_audit_results.json'),
        'drift': os.path.join(REPORTS_DIR, 'drift_monitoring_results.json'),
        'financial': os.path.join(REPORTS_DIR, 'financial_cost_model_results.json')
    }
    res = {}
    for k, p in ext_files.items():
        if os.path.exists(p):
            with open(p) as f:
                res[k] = json.load(f)
        else:
            res[k] = {"status": "Not yet generated"}
    return res

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
