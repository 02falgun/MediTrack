-- MediTrack Relational Database Schema
-- Project CP-02: Readmission Risk Prediction and Clinical Decision Support
-- Relational model covering patients, encounters, diagnoses, and medications

DROP TABLE IF EXISTS medications;
DROP TABLE IF EXISTS diagnoses;
DROP TABLE IF EXISTS encounters;
DROP TABLE IF EXISTS patients;

-- 1. Patients Entity
CREATE TABLE patients (
    patient_nbr BIGINT PRIMARY KEY,
    race VARCHAR(50),
    gender VARCHAR(20),
    age_group VARCHAR(20),
    total_encounters INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patients_race ON patients(race);
CREATE INDEX idx_patients_age ON patients(age_group);

-- 2. Encounters Entity
CREATE TABLE encounters (
    encounter_id BIGINT PRIMARY KEY,
    patient_nbr BIGINT NOT NULL,
    admission_type_id INTEGER,
    admission_type_name VARCHAR(50),
    discharge_disposition_id INTEGER,
    time_in_hospital INTEGER NOT NULL,
    payer_code VARCHAR(50),
    medical_specialty VARCHAR(100),
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    a1c_result VARCHAR(20),
    glucose_result VARCHAR(20),
    readmitted_raw VARCHAR(20),
    readmitted_30d INTEGER NOT NULL CHECK (readmitted_30d IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr) ON DELETE CASCADE
);

CREATE INDEX idx_encounters_patient ON encounters(patient_nbr);
CREATE INDEX idx_encounters_readm30 ON encounters(readmitted_30d);
CREATE INDEX idx_encounters_specialty ON encounters(medical_specialty);
CREATE INDEX idx_encounters_adm_type ON encounters(admission_type_name);

-- 3. Diagnoses Entity (Normalized 1:N with encounters)
CREATE TABLE diagnoses (
    diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id BIGINT NOT NULL,
    diagnosis_seq INTEGER NOT NULL CHECK (diagnosis_seq IN (1, 2, 3)),
    icd9_code VARCHAR(20),
    clinical_category VARCHAR(50) NOT NULL,
    clinical_description TEXT,
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE
);

CREATE INDEX idx_diagnoses_encounter ON diagnoses(encounter_id);
CREATE INDEX idx_diagnoses_seq_cat ON diagnoses(diagnosis_seq, clinical_category);

-- 4. Medications Entity (Normalized 1:N with encounters for 23 diabetes medications)
CREATE TABLE medications (
    medication_entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    encounter_id BIGINT NOT NULL,
    medication_name VARCHAR(50) NOT NULL,
    dosage_status VARCHAR(20) NOT NULL CHECK (dosage_status IN ('No', 'Steady', 'Up', 'Down')),
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    has_dosage_change INTEGER NOT NULL CHECK (has_dosage_change IN (0, 1)),
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id) ON DELETE CASCADE
);

CREATE INDEX idx_meds_encounter ON medications(encounter_id);
CREATE INDEX idx_meds_name_active ON medications(medication_name, is_active);
