-- ==============================================================================
-- MediTrack - Analytical SQL Queries
-- Project CP-02: Readmission Risk Prediction and Clinical Decision Support
-- Required Queries:
-- 1. Readmission rate by cohort
-- 2. Average stay by diagnosis group
-- 3. Patient encounter sequences
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- QUERY 1: Readmission Rate by Demographic & Prior Inpatient Cohort
-- Identifies high-risk patient cohorts by combining age group, gender, and prior hospitalizations
-- ------------------------------------------------------------------------------
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
    ROUND(AVG(e.time_in_hospital), 2) AS avg_length_of_stay_days,
    ROUND(AVG(e.num_medications), 1) AS avg_medications_count
FROM encounters e
JOIN patients p ON e.patient_nbr = p.patient_nbr
GROUP BY p.age_group, p.gender, inpatient_utilization_tier
HAVING COUNT(e.encounter_id) >= 50
ORDER BY readmission_rate_pct DESC;


-- ------------------------------------------------------------------------------
-- QUERY 2: Average Length of Stay and Readmission Rate by Primary Diagnosis Group
-- Evaluates inpatient resource utilization and post-discharge vulnerability across clinical categories
-- ------------------------------------------------------------------------------
SELECT 
    d.clinical_category AS primary_diagnosis_group,
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    ROUND(AVG(e.time_in_hospital), 2) AS avg_stay_days,
    ROUND(MIN(e.time_in_hospital), 1) AS min_stay_days,
    ROUND(MAX(e.time_in_hospital), 1) AS max_stay_days,
    SUM(e.readmitted_30d) AS readmissions_30d_count,
    ROUND(100.0 * SUM(e.readmitted_30d) / COUNT(e.encounter_id), 2) AS readmission_rate_pct,
    ROUND(AVG(e.num_procedures), 2) AS avg_procedures_performed,
    ROUND(AVG(e.num_medications), 2) AS avg_medications_prescribed
FROM diagnoses d
JOIN encounters e ON d.encounter_id = e.encounter_id
WHERE d.diagnosis_seq = 1 -- Filter for primary admitting diagnosis
GROUP BY d.clinical_category
ORDER BY avg_stay_days DESC;


-- ------------------------------------------------------------------------------
-- QUERY 3: Patient Encounter Sequences Tracing Readmission Trajectory
-- Uses SQL window functions (ROW_NUMBER, LAG) to analyze consecutive admissions per patient
-- ------------------------------------------------------------------------------
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
        ROW_NUMBER() OVER (
            PARTITION BY e.patient_nbr 
            ORDER BY e.encounter_id ASC
        ) AS encounter_sequence_num,
        COUNT(e.encounter_id) OVER (
            PARTITION BY e.patient_nbr
        ) AS total_patient_admissions,
        LAG(e.encounter_id) OVER (
            PARTITION BY e.patient_nbr 
            ORDER BY e.encounter_id ASC
        ) AS prior_encounter_id,
        LAG(e.time_in_hospital) OVER (
            PARTITION BY e.patient_nbr 
            ORDER BY e.encounter_id ASC
        ) AS prior_stay_duration_days,
        LAG(d.clinical_category) OVER (
            PARTITION BY e.patient_nbr 
            ORDER BY e.encounter_id ASC
        ) AS prior_primary_diagnosis
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
    prior_primary_diagnosis,
    prior_stay_duration_days,
    (stay_duration_days - prior_stay_duration_days) AS stay_duration_shift_days,
    readmitted_raw,
    readmitted_30d
FROM patient_journey
WHERE total_patient_admissions > 1
ORDER BY patient_nbr, encounter_sequence_num
LIMIT 100;
