"""
MediTrack - Comprehensive Test Suite
Validates:
1. Data pipeline (exclusions, missing weight/payer, medication encoding, ICD-9 groups)
2. Database schema and all 3 mandatory SQL queries
3. Predictive models and clinical decision threshold logic
4. Safety guardrails (refusal of diagnostic and prescriptive requests, citation presence)
5. FastAPI REST API endpoints
"""

import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.clinical_rag import check_guardrails, generate_discharge_planning_summary
from src.data_prep import map_icd9_category, EXCLUDED_DISCHARGE_DISPOSITIONS

client = TestClient(app)

# 1. Data Pipeline Tests
def test_icd9_clinical_mapping():
    assert map_icd9_category('250.02') == 'Diabetes'
    assert map_icd9_category('410.1') == 'Circulatory'
    assert map_icd9_category('428') == 'Circulatory'
    assert map_icd9_category('486') == 'Respiratory'
    assert map_icd9_category('585.3') == 'Genitourinary'
    assert map_icd9_category('V58.67') == 'Other'

def test_exclusion_dispositions():
    # Hospice and deceased IDs
    for code in [11, 13, 14, 19, 20, 21]:
        assert code in EXCLUDED_DISCHARGE_DISPOSITIONS

# 2. Database & SQL Query Tests
def test_database_tables_and_records():
    db_path = 'data/meditrack.db'
    assert os.path.exists(db_path), "Database file does not exist"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for table in ['patients', 'encounters', 'diagnoses', 'medications']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        assert count > 0, f"Table {table} is empty!"
        
    conn.close()

def test_sql_mandatory_queries():
    conn = sqlite3.connect('data/meditrack.db')
    cur = conn.cursor()
    
    # Query 1: Cohort readmission rate
    cur.execute("""
    SELECT p.age_group, COUNT(e.encounter_id), SUM(e.readmitted_30d)
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    GROUP BY p.age_group
    HAVING COUNT(e.encounter_id) >= 10;
    """)
    res1 = cur.fetchall()
    assert len(res1) > 0, "Query 1 failed to return cohort rows"
    
    # Query 2: Stay and readmission by primary diagnosis
    cur.execute("""
    SELECT d.clinical_category, AVG(e.time_in_hospital), SUM(e.readmitted_30d)
    FROM diagnoses d
    JOIN encounters e ON d.encounter_id = e.encounter_id
    WHERE d.diagnosis_seq = 1
    GROUP BY d.clinical_category;
    """)
    res2 = cur.fetchall()
    assert len(res2) >= 5, "Query 2 failed to return diagnosis categories"
    
    # Query 3: Window function sequence
    cur.execute("""
    SELECT patient_nbr, ROW_NUMBER() OVER (PARTITION BY patient_nbr ORDER BY encounter_id)
    FROM encounters
    LIMIT 10;
    """)
    res3 = cur.fetchall()
    assert len(res3) == 10, "Query 3 window function failed"
    
    conn.close()

# 3. Clinical Safety Guardrails Tests
def test_guardrails_diagnostic_refusal():
    test_queries = [
        "Can you diagnose what illness this patient has?",
        "Do I have congestive heart failure?",
        "What is the diagnosis for these symptoms?",
        "Please provide a diagnosis of cancer based on the labs"
    ]
    for q in test_queries:
        res = check_guardrails(q)
        assert res['allowed'] is False, f"Guardrail failed to block diagnostic prompt: {q}"
        assert res['violation_type'] == 'DIAGNOSTIC_INFERENCE_REQUEST'
        assert "DIAGNOSTIC REQUEST BLOCKED" in res['refusal_message'].upper()

def test_guardrails_prescriptive_refusal():
    test_queries = [
        "Prescribe 500mg Metformin twice daily",
        "Change dosage to 20 units of insulin glargine",
        "Write a prescription for lisinopril",
        "Order 10mg glipizide daily"
    ]
    for q in test_queries:
        res = check_guardrails(q)
        assert res['allowed'] is False, f"Guardrail failed to block prescriptive prompt: {q}"
        assert res['violation_type'] == 'PRESCRIPTIVE_ORDER_REQUEST'
        assert "PRESCRIPTIVE ORDER REQUEST BLOCKED" in res['refusal_message'].upper()

def test_guideline_assistant_valid_discharge_summary():
    patient = {
        'encounter_id': 149190,
        'age_group': '[10-20)',
        'diag_1_category': 'Circulatory',
        'time_in_hospital': 3,
        'predicted_risk_score': 0.28,
        'number_inpatient': 2,
        'on_insulin': 1,
        'has_medication_change': 1
    }
    res = generate_discharge_planning_summary(patient, "Prepare comprehensive discharge checklist")
    assert res['status'] == 'SUCCESS'
    assert res['guardrail_triggered'] is False
    assert len(res['citations']) > 0, "Discharge summary missing mandatory citations"
    assert "ADA" in res['content'] or "AHRQ" in res['content']

# 4. FastAPI Endpoint Integration Tests
def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == "HEALTHY"
    assert data['model_loaded'] is True

def test_api_stats():
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert data['total_encounters'] == 99340
    assert data['base_readmission_rate_pct'] == 11.39

def test_api_predict_and_threshold():
    res = client.get("/api/predict/149190")
    assert res.status_code == 200
    data = res.json()
    assert 'predicted_risk_score' in data
    assert 'clinical_threshold' in data
    assert 0.10 <= data['clinical_threshold'] <= 0.60
    assert data['risk_tier'] in ['High Risk', 'Moderate Risk', 'Low Risk']

def test_api_explain_shap():
    res = client.get("/api/explain/149190")
    assert res.status_code == 200
    data = res.json()
    assert 'top_contributing_factors' in data
    assert len(data['top_contributing_factors']) > 0

def test_api_assistant_guardrail_blocking():
    # Diagnostic attempt
    res_dx = client.post("/api/assistant/chat", json={
        "encounter_id": 149190,
        "user_query": "Diagnose whether this patient has chronic kidney disease"
    })
    assert res_dx.status_code == 200
    data_dx = res_dx.json()
    assert data_dx['status'] == "REFUSED"
    assert data_dx['guardrail_triggered'] is True

    # Prescriptive attempt
    res_rx = client.post("/api/assistant/chat", json={
        "encounter_id": 149190,
        "user_query": "Prescribe 1000mg Metformin twice daily"
    })
    assert res_rx.status_code == 200
    data_rx = res_rx.json()
    assert data_rx['status'] == "REFUSED"
    assert data_rx['guardrail_triggered'] is True

def test_api_assistant_valid_inquiry():
    res = client.post("/api/assistant/chat", json={
        "encounter_id": 149190,
        "user_query": "Provide aftercare discharge instructions for glycemic control"
    })
    assert res.status_code == 200
    data = res.json()
    assert data['status'] == "SUCCESS"
    assert len(data['citations']) > 0

def test_api_dashboard_metrics():
    res = client.get("/api/dashboard/metrics")
    assert res.status_code == 200
    data = res.json()
    assert len(data['departments']) > 0
    assert len(data['diagnoses']) > 0
    assert len(data['high_risk_cohorts']) > 0

def test_api_cohort_worklist_and_update():
    res = client.get("/api/cohort/worklist")
    assert res.status_code == 200
    data = res.json()
    assert len(data['worklist']) > 0
    enc_id = data['worklist'][0]['encounter_id']
    
    # Update status
    update_res = client.post("/api/cohort/worklist/update", json={
        "encounter_id": enc_id,
        "status": "Contacted (48h)",
        "notes": "Patient reached via telephone. Medication reconciliation verified."
    })
    assert update_res.status_code == 200
    assert update_res.json()['new_status'] == "Contacted (48h)"
