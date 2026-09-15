"""
MediTrack - Relational Database Loader & Query Runner
Project CP-02: Readmission Risk Prediction and Clinical Decision Support
"""

import os
import sqlite3
import pandas as pd
from src.data_prep import DIABETES_MEDICATIONS

def build_database(csv_path='data/processed/cleaned_encounters.csv', db_path='data/meditrack.db', schema_path='sql/schema.sql'):
    """
    Initializes the SQLite database from schema.sql and populates it with cleaned encounters.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Executing schema DDL from {schema_path}...")
    with open(schema_path, 'r') as f:
        cursor.executescript(f.read())
    conn.commit()
    
    print(f"Reading cleaned dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} records for database insertion.")
    
    # 1. Populate Patients Table (unique patients)
    patient_grp = df.groupby('patient_nbr').agg({
        'race_clean': 'first',
        'gender_clean': 'first',
        'age_group': 'last',
        'encounter_id': 'count'
    }).reset_index()
    patient_grp.columns = ['patient_nbr', 'race', 'gender', 'age_group', 'total_encounters']
    
    print(f"Inserting {len(patient_grp):,} unique patients...")
    patient_grp.to_sql('patients', conn, if_exists='append', index=False)
    
    # 2. Populate Encounters Table
    encounters_df = df[[
        'encounter_id', 'patient_nbr', 'admission_type_id', 'admission_type_name',
        'discharge_disposition_id', 'time_in_hospital', 'payer_code_clean',
        'medical_specialty_clean', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'A1Cresult', 'max_glu_serum', 'readmission_category', 'readmitted_30d'
    ]].copy()
    encounters_df.columns = [
        'encounter_id', 'patient_nbr', 'admission_type_id', 'admission_type_name',
        'discharge_disposition_id', 'time_in_hospital', 'payer_code',
        'medical_specialty', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'a1c_result', 'glucose_result', 'readmitted_raw', 'readmitted_30d'
    ]
    print(f"Inserting {len(encounters_df):,} encounters...")
    encounters_df.to_sql('encounters', conn, if_exists='append', index=False)
    
    # 3. Populate Diagnoses Table (unpivoted primary, secondary, tertiary)
    diag_rows = []
    for seq, col_prefix in [(1, 'diag_1'), (2, 'diag_2'), (3, 'diag_3')]:
        sub_df = df[['encounter_id', f'{col_prefix}', f'{col_prefix}_category', f'{col_prefix}_desc']].dropna(subset=[col_prefix]).copy()
        sub_df['diagnosis_seq'] = seq
        sub_df.columns = ['encounter_id', 'icd9_code', 'clinical_category', 'clinical_description', 'diagnosis_seq']
        diag_rows.append(sub_df[['encounter_id', 'diagnosis_seq', 'icd9_code', 'clinical_category', 'clinical_description']])
    
    diagnoses_df = pd.concat(diag_rows, ignore_index=True)
    print(f"Inserting {len(diagnoses_df):,} diagnosis records...")
    diagnoses_df.to_sql('diagnoses', conn, if_exists='append', index=False)
    
    # 4. Populate Medications Table (active or changed medications)
    med_rows = []
    for med in DIABETES_MEDICATIONS:
        # Include records where med is either active or tracked
        med_sub = df[['encounter_id', med]].copy()
        med_sub['medication_name'] = med
        med_sub['dosage_status'] = med_sub[med]
        med_sub['is_active'] = (med_sub[med] != 'No').astype(int)
        med_sub['has_dosage_change'] = med_sub[med].isin(['Up', 'Down']).astype(int)
        # Filter for active medications to keep table lean and indexed
        med_active = med_sub[med_sub['is_active'] == 1][['encounter_id', 'medication_name', 'dosage_status', 'is_active', 'has_dosage_change']]
        med_rows.append(med_active)
        
    medications_df = pd.concat(med_rows, ignore_index=True)
    print(f"Inserting {len(medications_df):,} active medication entries...")
    medications_df.to_sql('medications', conn, if_exists='append', index=False)
    
    conn.commit()
    print("Database built successfully!")
    return conn

def test_queries(db_path='data/meditrack.db'):
    """
    Executes each of the 3 required queries against the database and prints outputs.
    """
    conn = sqlite3.connect(db_path)
    
    queries = [
        ("Query 1: Readmission Rate by Demographic & Prior Inpatient Cohort", """
        SELECT 
            p.age_group,
            p.gender,
            CASE 
                WHEN e.number_inpatient = 0 THEN '0 Prior Inpatient Visits'
                WHEN e.number_inpatient = 1 THEN '1 Prior Inpatient Visit'
                WHEN e.number_inpatient = 2 THEN '2 Prior Inpatient Visits'
                ELSE '3+ Prior Inpatient Visits'
            END AS inpatient_utilization_tier,
            COUNT(e.encounter_id) AS total_encounters,
            SUM(e.readmitted_30d) AS readmitted_30d_count,
            ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 2) AS readmission_rate_pct,
            ROUND(AVG(e.time_in_hospital), 2) AS avg_length_of_stay_days
        FROM encounters e
        JOIN patients p ON e.patient_nbr = p.patient_nbr
        GROUP BY p.age_group, p.gender, inpatient_utilization_tier
        HAVING COUNT(e.encounter_id) >= 50
        ORDER BY readmission_rate_pct DESC
        LIMIT 6;
        """),
        ("Query 2: Average Stay & Readmission by Primary Diagnosis Group", """
        SELECT 
            d.clinical_category AS primary_diagnosis_group,
            COUNT(DISTINCT e.encounter_id) AS encounter_count,
            ROUND(AVG(e.time_in_hospital), 2) AS avg_stay_days,
            ROUND(MIN(e.time_in_hospital), 1) AS min_stay_days,
            ROUND(MAX(e.time_in_hospital), 1) AS max_stay_days,
            SUM(e.readmitted_30d) AS readmissions_30d_count,
            ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 2) AS readmission_rate_pct
        FROM diagnoses d
        JOIN encounters e ON d.encounter_id = e.encounter_id
        WHERE d.diagnosis_seq = 1
        GROUP BY d.clinical_category
        ORDER BY avg_stay_days DESC;
        """),
        ("Query 3: Patient Encounter Sequences (Window Functions)", """
        WITH patient_journey AS (
            SELECT 
                e.patient_nbr,
                p.age_group,
                p.gender,
                e.encounter_id,
                e.time_in_hospital AS stay_duration_days,
                e.admission_type_name,
                e.readmitted_30d,
                e.readmitted_raw,
                d.clinical_category AS primary_diagnosis,
                ROW_NUMBER() OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS encounter_sequence_num,
                COUNT(e.encounter_id) OVER (PARTITION BY e.patient_nbr) AS total_patient_admissions,
                LAG(e.encounter_id) OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS prior_encounter_id,
                LAG(e.time_in_hospital) OVER (PARTITION BY e.patient_nbr ORDER BY e.encounter_id ASC) AS prior_stay_duration_days
            FROM encounters e
            JOIN patients p ON e.patient_nbr = p.patient_nbr
            LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id AND d.diagnosis_seq = 1
        )
        SELECT 
            patient_nbr,
            encounter_sequence_num,
            total_patient_admissions,
            encounter_id,
            admission_type_name,
            primary_diagnosis,
            stay_duration_days,
            prior_encounter_id,
            prior_stay_duration_days,
            readmitted_raw,
            readmitted_30d
        FROM patient_journey
        WHERE total_patient_admissions > 1
        ORDER BY patient_nbr, encounter_sequence_num
        LIMIT 6;
        """)
    ]
    
    for name, sql in queries:
        print(f"\n=======================================================")
        print(f"RUNNING: {name}")
        print("=======================================================")
        res = pd.read_sql_query(sql, conn)
        print(res.to_string(index=False))
        
    conn.close()

if __name__ == '__main__':
    build_database()
    test_queries()
