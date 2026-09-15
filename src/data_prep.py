"""
MediTrack - Clinical Data Preparation & Preprocessing Pipeline
Project CP-02: Readmission Risk Prediction and Clinical Decision Support
"""

import os
import re
import pandas as pd
import numpy as np

# Standard clinical exclusions for readmission studies
# IDs corresponding to expired/deceased or hospice care
EXCLUDED_DISCHARGE_DISPOSITIONS = [11, 13, 14, 19, 20, 21]

# 23 specific diabetes medications in the UCI 130-US Hospitals dataset
DIABETES_MEDICATIONS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

# ICD-9 Diagnosis categorization mapping based on clinical reference
def map_icd9_category(code):
    """
    Categorizes ICD-9 codes into standardized clinical categories.
    Follows established healthcare informatics conventions (e.g. Strack et al., 2014).
    """
    if pd.isna(code) or code == '?' or str(code).strip() == '':
        return 'Missing_or_Other'
    
    code_str = str(code).strip()
    
    # Handle V and E codes (supplementary classifications)
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'Other'
    
    # Try converting numeric portion
    try:
        # Extract numeric prefix before decimal point
        numeric_part = float(re.findall(r"^\d+\.?\d*", code_str)[0])
    except (IndexError, ValueError):
        return 'Other'
    
    int_code = int(numeric_part)
    
    if (390 <= int_code <= 459) or int_code == 785:
        return 'Circulatory'
    elif (460 <= int_code <= 519) or int_code == 786:
        return 'Respiratory'
    elif (520 <= int_code <= 579) or int_code == 787:
        return 'Digestive'
    elif int_code == 250 or code_str.startswith('250.'):
        return 'Diabetes'
    elif 800 <= int_code <= 999:
        return 'Injury'
    elif 710 <= int_code <= 739:
        return 'Musculoskeletal'
    elif (580 <= int_code <= 629) or int_code == 788:
        return 'Genitourinary'
    elif 140 <= int_code <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

def get_icd9_description(code):
    """
    Returns clinical text description for common ICD-9 codes/ranges for text processing.
    """
    cat = map_icd9_category(code)
    code_str = str(code).strip()
    
    if cat == 'Diabetes':
        return f"Diabetes mellitus complication (ICD-9 {code_str})"
    elif cat == 'Circulatory':
        if code_str.startswith('428'):
            return f"Congestive heart failure (ICD-9 {code_str})"
        elif code_str.startswith('410'):
            return f"Acute myocardial infarction (ICD-9 {code_str})"
        elif code_str.startswith('414'):
            return f"Chronic ischemic heart disease (ICD-9 {code_str})"
        elif code_str.startswith('401'):
            return f"Essential hypertension (ICD-9 {code_str})"
        return f"Circulatory cardiovascular system disease (ICD-9 {code_str})"
    elif cat == 'Respiratory':
        if code_str.startswith('486'):
            return f"Pneumonia organism unspecified (ICD-9 {code_str})"
        elif code_str.startswith('491') or code_str.startswith('496'):
            return f"Chronic obstructive pulmonary disease (ICD-9 {code_str})"
        return f"Respiratory disease pulmonary distress (ICD-9 {code_str})"
    elif cat == 'Digestive':
        return f"Digestive gastrointestinal disorder (ICD-9 {code_str})"
    elif cat == 'Genitourinary':
        if code_str.startswith('585'):
            return f"Chronic kidney disease (ICD-9 {code_str})"
        elif code_str.startswith('599'):
            return f"Urinary tract infection (ICD-9 {code_str})"
        return f"Genitourinary renal tract condition (ICD-9 {code_str})"
    elif cat == 'Musculoskeletal':
        return f"Musculoskeletal system connective tissue disorder (ICD-9 {code_str})"
    elif cat == 'Neoplasms':
        return f"Neoplasm malignant or benign tumor (ICD-9 {code_str})"
    elif cat == 'Injury':
        return f"Trauma injury or poisoning complication (ICD-9 {code_str})"
    else:
        return f"Medical symptom observation or other condition (ICD-9 {code_str})"

def clean_and_prepare_data(raw_csv_path, output_dir=None):
    """
    Performs end-to-end data cleaning, exclusion filtering, and feature engineering.
    """
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, low_memory=False)
    initial_count = len(df)
    print(f"Initial encounters loaded: {initial_count}")
    
    # 1. Apply documented exclusion rules for deceased and hospice discharges
    df = df[~df['discharge_disposition_id'].isin(EXCLUDED_DISCHARGE_DISPOSITIONS)].copy()
    post_exclusion_count = len(df)
    excluded_count = initial_count - post_exclusion_count
    print(f"Excluded {excluded_count} deceased/hospice records. Remaining: {post_exclusion_count}")
    
    # 2. Target Variable: 30-Day Readmission (<30 days vs NO / >30 days)
    # Binary classification target: 1 = readmitted < 30 days, 0 = not readmitted < 30 days
    df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
    
    # Also create standard readmission category for analysis: '<30', '>30', 'NO'
    df['readmission_category'] = df['readmitted'].copy()
    
    # 3. Handle Missing Weight and Payer Code
    # Weight is missing in >96% of records ('?') -> create binary indicator and drop raw
    df['weight_recorded'] = (df['weight'] != '?').astype(int)
    
    # Payer code is missing in ~40% of records -> impute as 'Missing_or_SelfPay'
    df['payer_code_clean'] = df['payer_code'].replace('?', 'Missing_or_SelfPay')
    
    # Medical specialty missing imputation
    df['medical_specialty_clean'] = df['medical_specialty'].replace('?', 'Missing_or_Unknown')
    
    # Race missing imputation
    df['race_clean'] = df['race'].replace('?', 'Unknown')
    
    # Gender clean
    df['gender_clean'] = df['gender'].replace('Unknown/Invalid', np.nan)
    df = df.dropna(subset=['gender_clean'])
    
    # 4. Map High-Cardinality Diagnosis Codes to Standardized Clinical Categories
    df['diag_1_category'] = df['diag_1'].apply(map_icd9_category)
    df['diag_2_category'] = df['diag_2'].apply(map_icd9_category)
    df['diag_3_category'] = df['diag_3'].apply(map_icd9_category)
    
    # Create diagnosis textual descriptions for NLP / text processing
    df['diag_1_desc'] = df['diag_1'].apply(get_icd9_description)
    df['diag_2_desc'] = df['diag_2'].apply(get_icd9_description)
    df['diag_3_desc'] = df['diag_3'].apply(get_icd9_description)
    df['clinical_diagnosis_text'] = df['diag_1_desc'] + " | " + df['diag_2_desc'] + " | " + df['diag_3_desc']
    
    # 5. Inconsistent Medication Entries & Feature Extraction
    # Map each of the 23 medications:
    active_meds = []
    dosage_changes = []
    
    for med in DIABETES_MEDICATIONS:
        df[f'{med}_prescribed'] = (df[med] != 'No').astype(int)
        df[f'{med}_changed'] = df[med].isin(['Up', 'Down']).astype(int)
        active_meds.append(f'{med}_prescribed')
        dosage_changes.append(f'{med}_changed')
    
    df['num_active_diabetes_meds'] = df[active_meds].sum(axis=1)
    df['num_diabetes_med_changes'] = df[dosage_changes].sum(axis=1)
    df['has_medication_change'] = (df['num_diabetes_med_changes'] > 0).astype(int)
    
    # Specific insulin indicator and trajectory
    df['on_insulin'] = (df['insulin'] != 'No').astype(int)
    df['insulin_dosage_change'] = df['insulin'].isin(['Up', 'Down']).astype(int)
    
    # Polypharmacy flag (clinical threshold: 15+ medications)
    df['polypharmacy'] = (df['num_medications'] >= 15).astype(int)
    
    # Overall prior utilization history
    df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
    
    # Clean Age Groups
    df['age_group'] = df['age']
    age_midpoint_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age_approx'] = df['age'].map(age_midpoint_map).fillna(65)
    
    # Clean Admission Type Name
    admission_type_map = {
        1: 'Emergency',
        2: 'Urgent',
        3: 'Elective',
        4: 'Newborn',
        5: 'Not Available',
        6: 'NULL',
        7: 'Trauma Center',
        8: 'Not Mapped'
    }
    df['admission_type_name'] = df['admission_type_id'].map(admission_type_map).fillna('Other')
    
    # Clean A1C and Glucose Test Results
    df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int)
    df['a1c_abnormal'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)
    df['glucose_tested'] = (df['max_glu_serum'] != 'None').astype(int)
    df['glucose_abnormal'] = df['max_glu_serum'].isin(['>200', '>300']).astype(int)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        processed_path = os.path.join(output_dir, 'cleaned_encounters.csv')
        df.to_csv(processed_path, index=False)
        print(f"Cleaned dataset saved to {processed_path} (Records: {len(df)})")
        
    return df

if __name__ == '__main__':
    raw_path = 'data/raw/diabetic_data.csv'
    out_dir = 'data/processed'
    clean_and_prepare_data(raw_path, out_dir)
